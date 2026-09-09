"""Fill the catalog's volatile fields from Hugging Face and each file's GGUF header.

A curated catalog entry never carries the exact file names, sizes, checksums, bits
per weight or architecture facts of the quants it lists: those numbers change every
time a repository is re-uploaded, and nobody should type a SHA-256 by hand. This
module reads them from the Hugging Face Hub and from the first file of each quant's
GGUF header, then rewrites the catalog file deterministically so that a diff shows
only what actually changed.

Nothing here touches a curated field. A repository whose file listing cannot be
trusted leaves its whole model untouched, rather than risk blanking out fields that
took a real read to fill in.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from llamafit.catalog.hf import HfClient, RepoFile, assign_files_to_quants
from llamafit.catalog.loader import load_models_from_file
from llamafit.errors import LlamaFitError
from llamafit.models.catalog import CatalogModel, Extra, ModelSource, Quant
from llamafit.models.gguf import GgufFacts


@dataclass(frozen=True)
class RefreshResult:
    """The outcome of refreshing one model.

    Attributes:
        model_id: The model's id.
        file: The catalog file the model was read from and, when it changed,
            written back to.
        changed: Whether any volatile field's value was different from what the
            file already held.
        fields: The volatile fields that changed, each named by its position, for
            example ``sources[0].quants[0].bytes``.
        error: Why this model could not be refreshed, when it could not. When set,
            ``changed`` is always ``False`` and the model was left untouched.
    """

    model_id: str
    file: str
    changed: bool
    fields: list[str]
    error: str | None = None


def _total_bytes(files: Sequence[RepoFile]) -> int | None:
    """Sum every file's size, or ``None`` when any file's size is unknown."""
    total = 0
    for repo_file in files:
        if repo_file.size is None:
            return None
        total += repo_file.size
    return total


def _sha256_list(files: Sequence[RepoFile]) -> list[str] | None:
    """One checksum per file, in order, or ``None`` when any checksum is unknown."""
    checksums: list[str] = []
    for repo_file in files:
        if repo_file.sha256 is None:
            return None
        checksums.append(repo_file.sha256)
    return checksums


def _find_extra_file(files: Sequence[RepoFile], name: str) -> RepoFile | None:
    """Find the repository file matching an extra's file name, by path or basename."""
    for repo_file in files:
        if repo_file.path == name or repo_file.path.rsplit("/", 1)[-1] == name:
            return repo_file
    return None


def _refresh_quant(
    quant: Quant,
    files: Sequence[RepoFile],
    total_b: float,
    read_facts_fn: Callable[..., GgufFacts],
    prefix: str,
    changed_fields: list[str],
    file_url: Callable[[str], str],
) -> None:
    """Fill one quant's volatile fields from its matched files, recording what changed."""
    if not files:
        return

    new_files = [repo_file.path for repo_file in files]
    if new_files != quant.files:
        quant.files = new_files
        changed_fields.append(f"{prefix}.files")

    total_bytes = _total_bytes(files)
    if total_bytes is not None and total_bytes != quant.bytes_:
        quant.bytes_ = total_bytes
        changed_fields.append(f"{prefix}.bytes")

    sha256 = _sha256_list(files)
    if sha256 is not None and sha256 != quant.sha256:
        quant.sha256 = sha256
        changed_fields.append(f"{prefix}.sha256")

    if total_bytes is not None:
        bpw = round(total_bytes * 8 / (total_b * 1e9), 2)
        if bpw != quant.bpw:
            quant.bpw = bpw
            changed_fields.append(f"{prefix}.bpw")

    facts = read_facts_fn(file_url(files[0].path))
    if facts != quant.gguf_facts:
        quant.gguf_facts = facts
        changed_fields.append(f"{prefix}.gguf_facts")


def _refresh_extra(
    extra: Extra, files: Sequence[RepoFile], prefix: str, changed_fields: list[str]
) -> None:
    """Fill one extra's size and checksum from the repository file that matches its name."""
    match = _find_extra_file(files, extra.file)
    if match is None:
        return
    if match.size is not None and match.size != extra.bytes_:
        extra.bytes_ = match.size
        changed_fields.append(f"{prefix}.bytes")
    if match.sha256 is not None and match.sha256 != extra.sha256:
        extra.sha256 = match.sha256
        changed_fields.append(f"{prefix}.sha256")


def _refresh_source(
    source: ModelSource,
    index: int,
    total_b: float,
    hf: HfClient,
    read_facts_fn: Callable[..., GgufFacts],
    changed_fields: list[str],
) -> str | None:
    """Refresh one source's quants and extras in place.

    Returns:
        An error message when the source's repository could not be listed, so the
        caller can abandon the whole model; ``None`` on success.
    """
    if source.kind != "gguf" or not source.repo:
        return None
    repo = source.repo

    try:
        files = hf.list_files(repo)
    except LlamaFitError as exc:
        return f"{repo}: {exc}"
    if not files:
        return f"{repo}: the repository listing returned no files"

    def file_url(path: str) -> str:
        return hf.file_url(repo, path)

    assigned = assign_files_to_quants(files, [quant.name for quant in source.quants])
    for qi, quant in enumerate(source.quants):
        _refresh_quant(
            quant,
            assigned.get(quant.name, []),
            total_b,
            read_facts_fn,
            f"sources[{index}].quants[{qi}]",
            changed_fields,
            file_url,
        )
    for ei, extra in enumerate(source.extras):
        _refresh_extra(extra, files, f"sources[{index}].extras[{ei}]", changed_fields)
    return None


def _refresh_model(
    model: CatalogModel, hf: HfClient, read_facts_fn: Callable[..., GgufFacts]
) -> tuple[CatalogModel, list[str], str | None]:
    """Refresh one model's volatile fields on a deep copy, leaving the original untouched.

    Returns:
        The refreshed model (or the original, unchanged, on error), the fields that
        changed, and an error message when the model could not be refreshed.
    """
    updated = model.model_copy(deep=True)
    changed_fields: list[str] = []
    for index, source in enumerate(updated.sources):
        error = _refresh_source(
            source, index, model.params.total_b, hf, read_facts_fn, changed_fields
        )
        if error is not None:
            return model, [], error
    return updated, changed_fields, None


def refresh_file(
    path: Path,
    *,
    hf: HfClient,
    read_facts_fn: Callable[..., GgufFacts],
    dry_run: bool = False,
    only: str | None = None,
) -> list[RefreshResult]:
    """Refresh every model's volatile fields in one catalog file.

    Each model is refreshed independently: its repositories are listed once each,
    its quants and extras are matched against that listing, and the resulting
    fields are compared with what the file already holds. A model whose repository
    listing fails is left exactly as it was and reported with ``error`` set; it
    never blanks out fields that were already filled in.

    The file is rewritten, through :func:`dump_models`, only when at least one
    model actually changed and ``dry_run`` is ``False``. Passing ``only`` limits
    the refresh to the model with that id; every other model in the file is left
    untouched.

    Args:
        path: The catalog YAML file to refresh.
        hf: The Hugging Face metadata client to list repository files with.
        read_facts_fn: Reads a GGUF file's architecture facts, given its URL.
        dry_run: When ``True``, compute the refresh but never write the file.
        only: When set, refresh only the model with this id.

    Returns:
        One :class:`RefreshResult` per model that was considered, in file order.

    Raises:
        LlamaFitError: The file could not be parsed or a model in it failed
            validation, so refreshing it could not be done safely.
    """
    models, problems = load_models_from_file(path)
    if problems:
        details = "; ".join(f"{p.location}: {p.message}" for p in problems)
        raise LlamaFitError(
            f"{path} has {len(problems)} problem(s) and cannot be refreshed: {details}",
            hint="Run `llamafit catalog validate` and fix the file first.",
        )

    results: list[RefreshResult] = []
    final_models: list[CatalogModel] = []
    any_changed = False

    for model in models:
        if only is not None and model.id != only:
            final_models.append(model)
            continue

        updated, changed_fields, error = _refresh_model(model, hf, read_facts_fn)
        final_models.append(updated)
        changed = bool(changed_fields)
        any_changed = any_changed or changed
        results.append(
            RefreshResult(
                model_id=model.id,
                file=str(path),
                changed=changed,
                fields=changed_fields,
                error=error,
            )
        )

    if any_changed and not dry_run:
        path.write_text(dump_models(final_models), encoding="utf-8")

    return results


def dump_models(models: Sequence[CatalogModel]) -> str:
    """Serialize catalog models to YAML deterministically.

    Field order follows each model's declaration order, aliases are used so the
    output matches hand-written catalog files (``bytes`` rather than ``bytes_``,
    ``class`` rather than ``class_``), and unset fields are omitted entirely. Two
    calls on unchanged data always produce byte-identical text, so a diff shows
    only what actually changed.

    Args:
        models: The models to serialize, in the order they should appear.

    Returns:
        The YAML text, as it should be written to a catalog file.
    """
    data = [model.model_dump(by_alias=True, exclude_none=True, mode="json") for model in models]
    return yaml.safe_dump(data, sort_keys=False, indent=2, allow_unicode=True, width=100)
