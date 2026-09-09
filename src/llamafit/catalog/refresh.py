"""Fill the catalog's volatile fields from Hugging Face and each file's GGUF header.

A curated catalog entry never carries the exact file names, sizes, checksums, bits
per weight or architecture facts of the quants it lists: those numbers change every
time a repository is re-uploaded, and nobody should type a SHA-256 by hand. This
module reads them from the Hugging Face Hub and from the first file of each quant's
GGUF header, then writes them to ``<family>.facts.json``, a sibling of the curated
``<family>.yaml`` file.

The curated YAML is never written by this module, or by anything else: a machine
that can rewrite hand-written YAML can also silently discard a hand-written comment,
and a catalog whose whole claim is that every fact has a source cannot afford to
lose the notes that record those sources. Splitting the generated facts into their
own file makes that failure impossible rather than merely tested. See
:mod:`llamafit.catalog.loader` for how the two files are merged back together when
the catalog is read.

Nothing here touches a curated field, and nothing here writes a truncated file: the
facts file is written to a sibling temporary path and moved into place with
``os.replace``, so a crash or a full disk leaves either the previous file or the new
one, never a partial one. A repository whose file listing cannot be trusted leaves
its whole model untouched, rather than risk blanking out fields that took a real
read to fill in.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llamafit.catalog.hf import HfClient, RepoFile, assign_files_to_quants
from llamafit.catalog.loader import facts_path_for, load_models_from_file
from llamafit.errors import CatalogError, LlamaFitError
from llamafit.models.catalog import CatalogModel, Extra, ModelSource, Quant
from llamafit.models.gguf import GgufFacts

FACTS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RefreshResult:
    """The outcome of refreshing one model.

    Attributes:
        model_id: The model's id.
        file: The catalog file the model was read from.
        changed: Whether any volatile field's value was different from what the
            sibling facts file already held.
        fields: The volatile fields that changed, each named by its position, for
            example ``sources[0].quants[0].bytes``.
        error: Why this model could not be refreshed, when it could not. When set,
            ``changed`` is always ``False`` and the model was left untouched.
        warnings: Things worth a curator's attention that are not themselves
            failures, for example a quant name that matched no files at all.
    """

    model_id: str
    file: str
    changed: bool
    fields: list[str]
    error: str | None = None
    warnings: list[str] = field(default_factory=list)


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
    warnings: list[str],
) -> str | None:
    """Refresh one source's quants and extras in place.

    A quant name that matched no files is not an error: the repository listing was
    read successfully, the name is simply not (or no longer) published there. It is
    recorded in ``warnings`` instead, so a curator can catch a typo or a renamed
    quant rather than have it silently do nothing.

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
        matched = assigned.get(quant.name, [])
        if not matched:
            warnings.append(
                f"sources[{index}].quants[{qi}] ({quant.name!r}): no files in {repo} "
                "matched this quant name"
            )
            continue
        _refresh_quant(
            quant,
            matched,
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
) -> tuple[CatalogModel, list[str], list[str], str | None]:
    """Refresh one model's volatile fields on a deep copy, leaving the original untouched.

    Returns:
        The refreshed model (or the original, unchanged, on error), the fields that
        changed, any warnings worth a curator's attention, and an error message when
        the model could not be refreshed.
    """
    updated = model.model_copy(deep=True)
    changed_fields: list[str] = []
    warnings: list[str] = []
    for index, source in enumerate(updated.sources):
        error = _refresh_source(
            source, index, model.params.total_b, hf, read_facts_fn, changed_fields, warnings
        )
        if error is not None:
            return model, [], [], error
    return updated, changed_fields, warnings, None


def _facts_entry_for_model(model: CatalogModel) -> dict[str, Any] | None:
    """Build one model's facts entry from its current quant and extra state.

    Returns:
        A mapping with ``quants`` and/or ``extras`` sub-mappings, or ``None`` when
        the model has nothing refreshed at all, so untouched models add nothing to
        the facts file.
    """
    quants: dict[str, Any] = {}
    for source in model.sources:
        for quant in source.quants:
            has_data = (
                quant.files
                or quant.bytes_ is not None
                or quant.sha256
                or quant.bpw is not None
                or quant.gguf_facts is not None
            )
            if has_data:
                quants[quant.name] = {
                    "files": quant.files,
                    "bytes": quant.bytes_,
                    "sha256": quant.sha256,
                    "bpw": quant.bpw,
                    "gguf_facts": (
                        quant.gguf_facts.model_dump(mode="json")
                        if quant.gguf_facts is not None
                        else None
                    ),
                }

    extras: dict[str, Any] = {}
    for source in model.sources:
        for extra in source.extras:
            if extra.bytes_ is not None or extra.sha256 is not None:
                extras[extra.file] = {"bytes": extra.bytes_, "sha256": extra.sha256}

    if not quants and not extras:
        return None
    entry: dict[str, Any] = {}
    if quants:
        entry["quants"] = quants
    if extras:
        entry["extras"] = extras
    return entry


def _build_facts_document(models: Sequence[CatalogModel]) -> dict[str, Any]:
    """Build the whole facts document from every model's current volatile fields."""
    by_model: dict[str, Any] = {}
    for model in models:
        entry = _facts_entry_for_model(model)
        if entry is not None:
            by_model[model.id] = entry
    return {
        "schema_version": FACTS_SCHEMA_VERSION,
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "models": by_model,
    }


def _dump_facts_json(document: dict[str, Any]) -> str:
    """Serialize a facts document deterministically: sorted keys, two-space indent."""
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _write_facts_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a sibling temporary file and an atomic replace.

    A crash or a full disk between writing the temporary file and replacing the
    target leaves either the previous file or the new one in place, never a
    truncated one.

    Raises:
        CatalogError: The temporary file could not be written, or could not be
            moved into place.
    """
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    except OSError as exc:
        raise CatalogError(
            f"could not write {path}: {exc}",
            hint="Check that the directory is writable and has room.",
        ) from exc


def refresh_file(
    path: Path,
    *,
    hf: HfClient,
    read_facts_fn: Callable[..., GgufFacts],
    dry_run: bool = False,
    only: str | None = None,
) -> list[RefreshResult]:
    """Refresh every model's volatile fields, writing only the sibling facts file.

    Each model is refreshed independently: its repositories are listed once each,
    its quants and extras are matched against that listing, and the resulting
    fields are compared with what the sibling facts file already holds (already
    merged onto the model by :func:`~llamafit.catalog.loader.load_models_from_file`).
    A model whose repository listing fails is left exactly as it was and reported
    with ``error`` set; it never blanks out fields that were already filled in.

    The curated YAML file named by ``path`` is only ever read, never written. The
    facts file, named by :func:`~llamafit.catalog.loader.facts_path_for`, is
    rewritten, atomically, only when at least one model actually changed and
    ``dry_run`` is ``False``; it always reflects every model currently in the YAML,
    including those left untouched by an ``only`` filter or a failed source, so a
    partial refresh never drops facts a previous run already recorded.

    Args:
        path: The curated catalog YAML file to refresh.
        hf: The Hugging Face metadata client to list repository files with.
        read_facts_fn: Reads a GGUF file's architecture facts, given its URL.
        dry_run: When ``True``, compute the refresh but never write the facts file.
        only: When set, refresh only the model with this id.

    Returns:
        One :class:`RefreshResult` per model that was considered, in file order.

    Raises:
        CatalogError: The YAML file, or its sibling facts file, could not be parsed
            or failed validation, so refreshing it could not be done safely; or the
            facts file could not be written.
    """
    models, problems = load_models_from_file(path)
    if problems:
        details = "; ".join(f"{p.location}: {p.message}" for p in problems)
        raise CatalogError(
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

        updated, changed_fields, warnings, error = _refresh_model(model, hf, read_facts_fn)
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
                warnings=warnings,
            )
        )

    if any_changed and not dry_run:
        document = _build_facts_document(final_models)
        _write_facts_atomically(facts_path_for(path), _dump_facts_json(document))

    return results
