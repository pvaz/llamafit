"""``llamafit list``, ``search``, ``info`` and the ``catalog`` maintenance group.

These commands put the catalog services in front of a person: ``list`` and
``search`` narrow it down, ``info`` shows one model in full, and ``catalog
validate|refresh|show`` maintain the underlying YAML files. Nothing here computes
a memory budget for a quant on this host; that is phase 1C's job, once it exists
to compute it (see ``llamafit.services.catalog.ModelDetail``).
"""

from __future__ import annotations

import dataclasses
import functools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast, get_args

import httpx
import typer
import yaml
from rich.console import Console
from rich.text import Text

from llamafit.catalog.hf import HttpHfClient
from llamafit.catalog.loader import bundled_catalog_dir, custom_models_path, load_catalog
from llamafit.catalog.refresh import RefreshResult, refresh_file
from llamafit.catalog.validate import validate_files
from llamafit.cli.app import CliState, app
from llamafit.cli.render import render_catalog_list, render_model_facts, render_quants
from llamafit.errors import CatalogError
from llamafit.gguf.cache import HeaderCache, read_facts
from llamafit.models.catalog import Capability, Catalog, CatalogModel, UseCase
from llamafit.paths import get_paths
from llamafit.services.catalog import ModelFilters, describe, filter_models, summarise

_VALID_CAPABILITIES = get_args(Capability)
_VALID_USE_CASES = get_args(UseCase)

catalog_app = typer.Typer(
    name="catalog", help="Maintain the catalog: validate its files, refresh volatile fields."
)
app.add_typer(catalog_app, name="catalog")

# Module-level singletons rather than inline `typer.Option(...)`/`typer.Argument(...)` calls:
# ruff's B008 does not recognise every annotation shape (a ``Literal`` union, a ``list``, a
# ``Path``) as safe to call in a default position, even though Typer only ever reads these
# once, at import time, the same as any other default. A singleton sidesteps the warning
# without arguing with it.
_USE_CASE_OPTION: str | None = typer.Option(None, "--use-case", help="Keep only this use case.")
_CAPABILITY_OPTION: list[str] = typer.Option(
    [], "--capability", help="Keep only models with this capability (repeatable)."
)
_LICENSE_OPTION: list[str] = typer.Option(
    [], "--license", help="Keep only these licence identifiers (repeatable)."
)
_VALIDATE_FILE_ARGUMENT: Path | None = typer.Argument(
    None, help="Validate only this file instead of the bundled catalog and custom overrides."
)


def _closest_ids(
    catalog_ids: Sequence[str], target: str, *, min_prefix: int = 2, limit: int = 3
) -> list[str]:
    """The catalog ids that share the longest prefix with ``target``.

    "Closest" is deliberately narrow: the longest common prefix, case-insensitive,
    counted from the first character. A typo like ``qwen3-coder`` for
    ``qwen3-coder-next`` shares an eleven-character prefix and is an obvious match;
    unrelated input shares nothing worth naming. Below ``min_prefix`` characters of
    overlap, nothing is suggested at all, so nonsense input gets an honest "not
    found" instead of a handful of unrelated ids picked only because they had to
    come from somewhere. At most ``limit`` ids are returned, alphabetically, so a
    genuine tie does not turn one bad guess into a wall of suggestions.
    """

    def prefix_len(candidate: str) -> int:
        length = 0
        for a, b in zip(target.casefold(), candidate.casefold(), strict=False):
            if a != b:
                break
            length += 1
        return length

    scored = [(prefix_len(candidate), candidate) for candidate in catalog_ids]
    best = max((score for score, _ in scored), default=0)
    if best < min_prefix:
        return []
    return sorted(candidate for score, candidate in scored if score == best)[:limit]


def _find_model(catalog: Catalog, model_id: str) -> CatalogModel:
    """The model with ``model_id``, or a :class:`CatalogError` naming close ids instead.

    An id is a name, not a password: matching strips surrounding whitespace (from
    a pasted value) and folds case, so ``" QWEN3-CODER-NEXT"`` finds
    ``qwen3-coder-next`` the same way the closest-id suggestion already would.
    Every catalog id is lowercase by construction, so folding the input is enough;
    no separate case-insensitive index is needed.
    """
    normalized = model_id.strip().casefold()
    model = catalog.by_id.get(normalized)
    if model is not None:
        return model
    suggestions = _closest_ids(sorted(catalog.by_id), normalized)
    hint = (
        f"Did you mean: {', '.join(suggestions)}?"
        if suggestions
        else "Run `llamafit list` to see every model id."
    )
    raise CatalogError(f"no model named {model_id!r} in the catalog", hint=hint)


def _load_catalog(state: CliState) -> Catalog:
    """Load the catalog, warning to stderr once when any file had a problem.

    A user who adds a model to their custom catalog and mistypes a field would
    otherwise just be told the model does not exist, with a hint pointing at
    ``list``, which silently does not show it either: the loader already knows
    exactly what went wrong, and this is the one place every command reaches it
    from. The warning is one line, not fatal, and never dumped into a table:
    browsing keeps working with whatever loaded, and ``catalog validate`` is
    where the detail lives.
    """
    catalog, problems = load_catalog()
    if problems:
        word = "problem" if len(problems) == 1 else "problems"
        stderr = Console(stderr=True, no_color=state.no_color, highlight=False)
        stderr.print(
            Text(
                f"{len(problems)} catalog {word} found; "
                "run `llamafit catalog validate` for details.",
                style="yellow",
            )
        )
    return catalog


def _print_list(ctx: typer.Context, filters: ModelFilters, limit: int | None) -> None:
    """Filter the catalog, summarise the results, and print them as a table or as JSON."""
    state: CliState = ctx.obj
    catalog = _load_catalog(state)
    models = filter_models(catalog, filters)
    if limit is not None:
        models = models[:limit]
    summaries = [summarise(model) for model in models]
    if state.json_output:
        payload = [s.model_dump(mode="json", by_alias=True) for s in summaries]
        typer.echo(json.dumps(payload, indent=2))
        return
    console = state.console
    if not summaries:
        console.print(
            "No models matched. Try a broader --use-case, --capability, --license, "
            "--vendor or --search."
        )
        return
    console.print(render_catalog_list(summaries, console_width=console.width))


def _build_filters(
    *,
    use_case: str | None,
    capability: list[str],
    license_: list[str],
    vendor: str | None,
    search: str | None,
) -> ModelFilters:
    """Build filters from raw CLI values, rejecting an unknown use case or capability by name.

    Typer can enumerate a single ``Literal`` option's choices itself, but doing
    that for ``--use-case`` would make an unknown value a bare Click usage error:
    a different exit code and no list of what would have worked, for the same
    kind of mistake ``--capability`` already reports as a clean, listed
    :class:`CatalogError`. Typer also cannot enumerate a ``list[Literal[...]]``
    option at all (a list of a "complex" sub-type is one of the few shapes it
    refuses outright), so ``--capability`` has to be validated by hand regardless;
    checking ``--use-case`` the same way here, instead of leaning on Typer for it,
    means the two now fail alike.
    """
    if use_case is not None and use_case not in _VALID_USE_CASES:
        raise CatalogError(
            f"invalid --use-case {use_case!r}",
            hint=f"Valid use cases: {', '.join(_VALID_USE_CASES)}",
        )
    invalid_capabilities = [value for value in capability if value not in _VALID_CAPABILITIES]
    if invalid_capabilities:
        raise CatalogError(
            f"invalid --capability value(s): {', '.join(invalid_capabilities)}",
            hint=f"Valid capabilities: {', '.join(_VALID_CAPABILITIES)}",
        )
    return ModelFilters(
        use_case=cast("UseCase | None", use_case),
        capabilities=cast("list[Capability]", capability),
        licenses=license_,
        vendor=vendor,
        search=search,
    )


@app.command(name="list")
def list_command(
    ctx: typer.Context,
    use_case: str | None = _USE_CASE_OPTION,
    capability: list[str] = _CAPABILITY_OPTION,
    license_: list[str] = _LICENSE_OPTION,
    vendor: str | None = typer.Option(
        None, "--vendor", help="Keep only this vendor, case-insensitive."
    ),
    search: str | None = typer.Option(
        None, "--search", help="Case-insensitive substring match on id, name, vendor or family."
    ),
    limit: int | None = typer.Option(None, "--limit", min=1, help="Show at most this many."),
) -> None:
    """List the model catalog, narrowed by any filters given.

    The table shows at most three capabilities per model, plus a ``+N`` marker for
    the rest, and is sorted by the Quality column shown; ``info`` or ``--json`` has
    every capability.
    """
    filters = _build_filters(
        use_case=use_case, capability=capability, license_=license_, vendor=vendor, search=search
    )
    _print_list(ctx, filters, limit)


@app.command(name="search")
def search_command(
    ctx: typer.Context,
    text: str = typer.Argument(..., help="Text to match against id, name, vendor or family."),
) -> None:
    """Shorthand for ``llamafit list --search TEXT``."""
    _print_list(ctx, ModelFilters(search=text), limit=None)


@app.command(name="info")
def info_command(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., metavar="MODEL", help="A catalog model id."),
) -> None:
    """Show one model's facts, sources and every quant it publishes."""
    state: CliState = ctx.obj
    catalog = _load_catalog(state)
    model = _find_model(catalog, model_id)
    detail = describe(model)
    if state.json_output:
        typer.echo(detail.model_dump_json(indent=2, by_alias=True))
        return
    console = state.console
    console.print(render_model_facts(detail.model))
    console.print()
    console.print(render_quants(detail.quants))


def _default_catalog_paths() -> list[Path]:
    """The bundled catalog files, in order, plus the custom file when it exists."""
    directory = bundled_catalog_dir()
    paths = sorted(directory.glob("*.yaml")) if directory.is_dir() else []
    custom = custom_models_path()
    if custom.is_file():
        paths.append(custom)
    return paths


@catalog_app.command("validate")
def validate_command(
    ctx: typer.Context,
    file: Path | None = _VALIDATE_FILE_ARGUMENT,
) -> None:
    """Validate catalog YAML files against the schema, printing one line per problem."""
    state: CliState = ctx.obj
    paths = [file] if file is not None else _default_catalog_paths()
    problems = validate_files(paths)
    if state.json_output:
        typer.echo(json.dumps([dataclasses.asdict(p) for p in problems], indent=2))
    else:
        console = state.console
        if not problems:
            console.print(f"{len(paths)} file(s) checked, no problems found.")
        for problem in problems:
            where = f" ({problem.model_id})" if problem.model_id else ""
            console.print(Text(f"{problem.file}: {problem.location}{where}: {problem.message}"))
    if problems:
        raise typer.Exit(code=1)


@catalog_app.command("refresh")
def refresh_command(
    ctx: typer.Context,
    model: str | None = typer.Option(None, "--model", help="Refresh only this model id."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Compute the refresh but write nothing."),
    check: bool = typer.Option(
        False,
        "--check",
        help="Exit 1 if anything would change; changes nothing, for a scheduled CI job.",
    ),
) -> None:
    """Refresh file names, sizes, checksums and GGUF facts for the catalog from Hugging Face."""
    state: CliState = ctx.obj
    if model is not None:
        catalog = _load_catalog(state)
        _find_model(catalog, model)

    results: list[RefreshResult] = []
    with httpx.Client(follow_redirects=True) as client:
        hf = HttpHfClient(client=client)
        cache = HeaderCache(get_paths().cache_dir / "gguf-headers")
        read_facts_fn = functools.partial(read_facts, client=client, cache=cache)
        for path in _default_catalog_paths():
            results.extend(
                refresh_file(
                    path,
                    hf=hf,
                    read_facts_fn=read_facts_fn,
                    dry_run=dry_run or check,
                    only=model,
                )
            )

    if state.json_output:
        typer.echo(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
    else:
        console = state.console
        printed = False
        for result in results:
            # ``warnings`` is read defensively: it is not yet a field of the
            # ``RefreshResult`` this branch was built against, but is landing in a
            # sibling change and should start printing the moment it exists,
            # without another edit here. A quant that matched no files in its
            # repository looks exactly like a quant needing no update unless this
            # says otherwise.
            warnings: list[str] = getattr(result, "warnings", [])
            for warning in warnings:
                console.print(Text(f"{result.model_id}: warning: {warning}", style="yellow"))
                printed = True
            if result.error:
                console.print(Text(f"{result.model_id}: error: {result.error}", style="red"))
                printed = True
            elif result.changed:
                console.print(Text(f"{result.model_id}: {', '.join(result.fields)}"))
                printed = True
        if not printed:
            console.print("No changes.")

    if any(r.error for r in results) or (check and any(r.changed for r in results)):
        raise typer.Exit(code=1)


def _dump_one(model: CatalogModel) -> str:
    """Serialize one catalog entry to YAML for ``catalog show --yaml``.

    This renders the single, already-loaded entry handed to it; it does not read or
    write any file, so it has no opinion on how (or whether) a whole catalog file
    gets rewritten. Field order follows declaration order, aliases are used so the
    output matches hand-written catalog files (``bytes`` rather than ``bytes_``,
    ``class`` rather than ``class_``), and unset fields are omitted.
    """
    data = model.model_dump(by_alias=True, exclude_none=True, mode="json")
    return yaml.safe_dump(data, sort_keys=False, indent=2, allow_unicode=True, width=100)


@catalog_app.command("show")
def show_command(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., metavar="MODEL", help="A catalog model id."),
    yaml_output: bool = typer.Option(False, "--yaml", help="Print YAML instead of JSON."),
) -> None:
    """Print one model's raw catalog entry, as JSON by default or as YAML with ``--yaml``."""
    catalog = _load_catalog(ctx.obj)
    model = _find_model(catalog, model_id)
    if yaml_output:
        typer.echo(_dump_one(model))
    else:
        typer.echo(model.model_dump_json(indent=2, by_alias=True))
