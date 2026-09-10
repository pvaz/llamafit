# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
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
from pathlib import Path
from typing import cast

import httpx
import typer
import yaml
from rich.text import Text

from llamafit.catalog.hf import HttpHfClient
from llamafit.catalog.loader import bundled_catalog_dir, custom_models_path
from llamafit.catalog.refresh import RefreshResult, refresh_file
from llamafit.catalog.validate import validate_files
from llamafit.cli.app import CliState, app
from llamafit.cli.common import (
    check_capabilities,
    check_use_case,
    find_model,
    load_catalog_or_warn,
)
from llamafit.cli.render import render_catalog_list, render_model_facts, render_quants
from llamafit.gguf.cache import HeaderCache, read_facts
from llamafit.i18n import _, lazy_gettext, ngettext
from llamafit.models.catalog import CatalogModel
from llamafit.paths import get_paths
from llamafit.services.catalog import ModelFilters, describe, filter_models, summarise

catalog_app = typer.Typer(
    name="catalog",
    help=cast(
        str,
        lazy_gettext("Maintain the catalog: validate its files, refresh volatile fields."),
    ),
)
app.add_typer(catalog_app, name="catalog")

# Module-level singletons rather than inline `typer.Option(...)`/`typer.Argument(...)` calls:
# ruff's B008 does not recognise every annotation shape (a ``Literal`` union, a ``list``, a
# ``Path``) as safe to call in a default position, even though Typer only ever reads these
# once, at import time, the same as any other default. A singleton sidesteps the warning
# without arguing with it.
#
# Every help= below is deferred for the same reason: these are built while the module is
# imported, long before any language is chosen, so an eager _() would freeze them in
# English for the life of the process without anything failing.
_USE_CASE_OPTION: str | None = typer.Option(
    None, "--use-case", help=cast(str, lazy_gettext("Keep only this use case."))
)
_CAPABILITY_OPTION: list[str] = typer.Option(
    [],
    "--capability",
    help=cast(str, lazy_gettext("Keep only models with this capability (repeatable).")),
)
_LICENSE_OPTION: list[str] = typer.Option(
    [],
    "--license",
    help=cast(str, lazy_gettext("Keep only these licence identifiers (repeatable).")),
)
_VALIDATE_FILE_ARGUMENT: Path | None = typer.Argument(
    None,
    help=cast(
        str,
        lazy_gettext(
            "Validate only this file instead of the bundled catalog and custom overrides."
        ),
    ),
)


def _print_list(ctx: typer.Context, filters: ModelFilters, limit: int | None) -> None:
    """Filter the catalog, summarise the results, and print them as a table or as JSON."""
    state: CliState = ctx.obj
    catalog = load_catalog_or_warn(state)
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
            _(
                "No models matched. Try a broader --use-case, --capability, --license, "
                "--vendor or --search."
            )
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

    Both checks live in :mod:`llamafit.cli.common`, because ``recommend`` takes the same
    two values and a typo has to fail the same way whichever command met it.
    """
    return ModelFilters(
        use_case=None if use_case is None else check_use_case(use_case),
        capabilities=list(check_capabilities(capability)),
        licenses=license_,
        vendor=vendor,
        search=search,
    )


@app.command(
    name="list",
    help=cast(
        str,
        lazy_gettext(
            "List the model catalog, narrowed by any filters given.\n\n"
            "The table shows at most three capabilities per model, plus a +N marker for "
            "the rest, and is sorted by the Quality column shown; info or --json has "
            "every capability."
        ),
    ),
)
def list_command(
    ctx: typer.Context,
    use_case: str | None = _USE_CASE_OPTION,
    capability: list[str] = _CAPABILITY_OPTION,
    license_: list[str] = _LICENSE_OPTION,
    vendor: str | None = typer.Option(
        None,
        "--vendor",
        help=cast(str, lazy_gettext("Keep only this vendor, case-insensitive.")),
    ),
    search: str | None = typer.Option(
        None,
        "--search",
        help=cast(
            str,
            lazy_gettext("Case-insensitive substring match on id, name, vendor or family."),
        ),
    ),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help=cast(str, lazy_gettext("Show at most this many."))
    ),
) -> None:
    """List the model catalog, narrowed by any filters given.

    The help a reader sees is the ``help=`` argument above, not this docstring: Typer
    reads a docstring as a literal, which no wrapper can translate.
    """
    filters = _build_filters(
        use_case=use_case, capability=capability, license_=license_, vendor=vendor, search=search
    )
    _print_list(ctx, filters, limit)


@app.command(
    name="search",
    help=cast(str, lazy_gettext("Shorthand for llamafit list --search TEXT.")),
)
def search_command(
    ctx: typer.Context,
    text: str = typer.Argument(
        ...,
        help=cast(str, lazy_gettext("Text to match against id, name, vendor or family.")),
    ),
) -> None:
    """Shorthand for llamafit list --search TEXT."""
    _print_list(ctx, ModelFilters(search=text), limit=None)


@app.command(
    name="info",
    help=cast(str, lazy_gettext("Show one model's facts, sources and every quant it publishes.")),
)
def info_command(
    ctx: typer.Context,
    model_id: str = typer.Argument(
        ..., metavar="MODEL", help=cast(str, lazy_gettext("A catalog model id."))
    ),
) -> None:
    """Show one model's facts, sources and every quant it publishes."""
    state: CliState = ctx.obj
    catalog = load_catalog_or_warn(state)
    model = find_model(catalog, model_id)
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


@catalog_app.command(
    "validate",
    help=cast(
        str,
        lazy_gettext(
            "Validate catalog YAML files against the schema, printing one line per problem."
        ),
    ),
)
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
            console.print(
                ngettext(
                    "%(count)d file checked, no problems found.",
                    "%(count)d files checked, no problems found.",
                    len(paths),
                )
                % {"count": len(paths)}
            )
        for problem in problems:
            where = f" ({problem.model_id})" if problem.model_id else ""
            console.print(Text(f"{problem.file}: {problem.location}{where}: {problem.message}"))
    if problems:
        raise typer.Exit(code=1)


def _dedupe_repeated_prefix(message: str) -> str:
    """Collapse "X: ...X..." into "...X..." when the leading label reappears verbatim.

    A failed source's error is built elsewhere as "<repo>: <what went wrong>", and
    what went wrong already names the repository in every case that matters here,
    so the plain concatenation says it twice in the same line. This does not
    change what was raised, only whether its own repeated leading label is shown
    once or twice.
    """
    prefix, separator, rest = message.partition(": ")
    if separator and prefix and prefix in rest:
        return rest
    return message


@catalog_app.command(
    "refresh",
    help=cast(
        str,
        lazy_gettext(
            "Refresh file names, sizes, checksums and GGUF facts for the catalog from Hugging Face."
        ),
    ),
)
def refresh_command(
    ctx: typer.Context,
    model: str | None = typer.Option(
        None, "--model", help=cast(str, lazy_gettext("Refresh only this model id."))
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help=cast(str, lazy_gettext("Compute the refresh but write nothing.")),
    ),
    check: bool = typer.Option(
        False,
        "--check",
        help=cast(
            str,
            lazy_gettext(
                "Exit 1 if anything would change; changes nothing, for a scheduled CI job."
            ),
        ),
    ),
) -> None:
    """Refresh file names, sizes, checksums and GGUF facts for the catalog from Hugging Face."""
    state: CliState = ctx.obj
    if model is not None:
        catalog = load_catalog_or_warn(state)
        find_model(catalog, model)

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

        # Grouped by text rather than printed once per model: a facts file that
        # fails to parse loses the same recorded facts for every model that reads
        # it, and five identical lines tell a reader less than one line and a
        # count.
        warned_by_text: dict[str, list[str]] = {}
        for result in results:
            for warning in result.warnings:
                warned_by_text.setdefault(warning, []).append(result.model_id)
        for warning, model_ids in warned_by_text.items():
            if len(model_ids) == 1:
                console.print(
                    Text(
                        _("%(model)s: warning: %(warning)s")
                        % {"model": model_ids[0], "warning": warning},
                        style="yellow",
                    )
                )
            else:
                console.print(
                    Text(
                        ngettext(
                            "warning: %(warning)s (%(count)d model: %(models)s)",
                            "warning: %(warning)s (%(count)d models: %(models)s)",
                            len(model_ids),
                        )
                        % {
                            "warning": warning,
                            "count": len(model_ids),
                            "models": ", ".join(model_ids),
                        },
                        style="yellow",
                    )
                )
            printed = True

        for result in results:
            if result.error:
                message = _dedupe_repeated_prefix(result.error)
                console.print(
                    Text(
                        _("%(model)s: error: %(error)s")
                        % {"model": result.model_id, "error": message},
                        style="red",
                    )
                )
                printed = True
            elif result.changed:
                console.print(Text(f"{result.model_id}: {', '.join(result.fields)}"))
                printed = True
        if any(r.error for r in results):
            console.print(
                Text(
                    _("Hint: %(hint)s")
                    % {
                        "hint": _(
                            "refresh needs network access to Hugging Face; check your "
                            "connection and that the repository id is correct."
                        )
                    },
                    style="dim",
                )
            )
        if not printed:
            console.print(_("No changes."))

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


@catalog_app.command(
    "show",
    help=cast(
        str,
        lazy_gettext(
            "Print one model's raw catalog entry, as JSON by default or as YAML with --yaml."
        ),
    ),
)
def show_command(
    ctx: typer.Context,
    model_id: str = typer.Argument(
        ..., metavar="MODEL", help=cast(str, lazy_gettext("A catalog model id."))
    ),
    yaml_output: bool = typer.Option(
        False, "--yaml", help=cast(str, lazy_gettext("Print YAML instead of JSON."))
    ),
) -> None:
    """Print one model's raw catalog entry, as JSON by default or as YAML with --yaml."""
    catalog = load_catalog_or_warn(ctx.obj)
    model = find_model(catalog, model_id)
    if yaml_output:
        typer.echo(_dump_one(model))
    else:
        typer.echo(model.model_dump_json(indent=2, by_alias=True))
