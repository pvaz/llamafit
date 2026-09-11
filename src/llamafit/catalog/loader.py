# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Read the catalog from YAML: the bundled files and a user's custom overrides.

Nothing here raises on malformed input. A YAML syntax error, a file that is not a list
of entries, or an entry that fails validation each becomes a :class:`Problem` naming
the file and, where it applies, the field; the caller decides what to do about it.

A curated ``<family>.yaml`` file never carries the volatile fields ``llamafit catalog
refresh`` fills in: those live beside it, in ``<family>.facts.json``, written only by
that command. This module merges the sibling facts file into the models it loads, so
the rest of LlamaFit never has to know the two files exist. A facts file that is
missing simply means nothing has been refreshed yet, not a problem.

Everything else about that file is checked before any of it is believed. A document
whose schema version this build does not know is refused whole; a model, a quant or
an extra the YAML does not have, an entry that is not an object, and a field that is
not the shape or the value it claims to be each become a :class:`Problem` naming
exactly where the trouble is. No field is ever quietly dropped, because a field that
failed to merge would then look exactly like a field that was never refreshed.

None of it is fatal to the caller. ``llamafit catalog refresh`` rewrites the whole
facts document, so it treats these as warnings and refreshes as though the fields had
never been filled; only a problem in the hand-written YAML stops it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, TypeGuard

import yaml
from pydantic import ValidationError

from llamafit.data import packaged_dir
from llamafit.errors import CatalogError
from llamafit.gguf.facts import accounts_for_file
from llamafit.i18n import _, ngettext
from llamafit.models.catalog import (
    MAX_BPW,
    Catalog,
    CatalogModel,
    Extra,
    ModelSource,
    Quant,
)
from llamafit.models.gguf import GgufFacts
from llamafit.paths import get_paths

FACTS_SCHEMA_VERSION = 1
"""The layout of the facts document this build writes and reads.

It lives with the reader, since the reader is what has to refuse a document it
cannot read; :mod:`llamafit.catalog.refresh` imports it and stamps it into every
file it writes.
"""

_CUSTOM_MODELS_FILENAME = "custom_models.yaml"

_REWRITE_HINT = "`llamafit catalog refresh` rewrites it"

_EXPECTED_SHAPE = "expected an object with a 'models' mapping"

_DISCARDING_LOCATIONS = frozenset({"file", "schema_version", "(root)", "quants", "extras"})
"""Where a problem costs a whole document, model entry or section rather than one field.

These are the locations :func:`_merge_facts`, :func:`_merge_model_facts` and
:func:`_facts_section` report when they give up on more than a single field, and
:func:`discards_recorded_facts` is the name a caller should ask by.
"""

_JSON_TYPE_NAMES: dict[type[object], str] = {
    type(None): "null",
    bool: "a boolean",
    int: "a number",
    float: "a number",
    str: "a string",
    list: "a list",
    dict: "an object",
}


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a catalog file.

    Attributes:
        file: The file the problem was found in.
        model_id: The entry's id, when it could be read; ``None`` otherwise.
        location: Where in the entry the problem is, dotted, for example
            ``quality.baseline``, or ``"file"`` for a problem with the file itself.
        message: A human-readable description of the problem.
    """

    file: str
    model_id: str | None
    location: str
    message: str


def discards_recorded_facts(problem: Problem) -> bool:
    """Whether a facts-file problem cost more than the one field it names.

    A field is only a field: everything else in its entry still merged, and a caller
    that rewrites that entry loses nothing else. A document, a model entry or a whole
    ``quants``/``extras`` section that could not be read takes every fact recorded
    under it, which is the difference a partial refresh has to care about: it exists
    to leave other models exactly as they were, and it cannot preserve what nobody
    could read.

    Returns:
        ``True`` when the problem discarded a document, an entry or a section.
    """
    return problem.location in _DISCARDING_LOCATIONS


def load_models_from_file(path: Path) -> tuple[list[CatalogModel], list[Problem]]:
    """Parse and validate every entry in one catalog YAML file, facts file merged in.

    The file must hold a list of entries at its top level. A syntax error, a
    top-level shape that is not a list, or an entry that fails validation each
    becomes a :class:`Problem`; this function never raises for malformed input.

    A model whose sources reuse a quant name or an extra file name is returned with
    its curated fields and no facts at all: the facts file keys on those names, so it
    cannot say which source a fact belongs to, and merging it would fabricate data.

    Returns:
        Every model the file declares, and every problem found in it or in its
        sibling facts file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [Problem(file=str(path), model_id=None, location="file", message=str(exc))]

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [], [
            Problem(file=str(path), model_id=None, location="file", message=f"invalid YAML: {exc}")
        ]

    if raw is None:
        return [], []
    if not isinstance(raw, list):
        return [], [
            Problem(
                file=str(path),
                model_id=None,
                location="file",
                message="expected a list of entries at the top level",
            )
        ]

    models: list[CatalogModel] = []
    problems: list[Problem] = []
    unmergeable_ids: set[str] = set()
    for entry in raw:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        try:
            model = CatalogModel.model_validate(entry)
        except ValidationError as exc:
            for error in exc.errors():
                location = ".".join(str(part) for part in error["loc"]) or "(root)"
                problems.append(
                    Problem(
                        file=str(path),
                        model_id=model_id,
                        location=location,
                        message=error["msg"],
                    )
                )
            continue
        models.append(model)
        reused_names = _duplicate_name_problems(model, str(path))
        if reused_names:
            unmergeable_ids.add(model.id)
        problems.extend(reused_names)
    problems.extend(_merge_facts(models, facts_path_for(path), unmergeable_ids))
    return models, problems


def facts_path_for(path: Path) -> Path:
    """The generated facts file that sits beside one curated catalog file."""
    return path.with_name(f"{path.stem}.facts.json")


def _source_identifier(source: ModelSource) -> str:
    """A human-readable identifier for one source: its repository, or its local path."""
    if source.repo is not None:
        return source.repo
    return source.path if source.path is not None else source.kind


def _duplicate_name_problems(model: CatalogModel, file: str) -> list[Problem]:
    """Report a quant name or an extra file name defined by more than one of a model's sources.

    The facts file keys volatile fields by the model id and the quant name or extra
    file name alone; it has no notion of which source published them. A name reused
    across sources (an official repository and a community one often carry the same
    quant names, and ``mmproj-F16.gguf`` is a common vendor default) would have both
    sources merged from that one key, so this is checked rather than merely assumed.

    Returns:
        One :class:`Problem` per reused name; empty when every name is unique.
    """
    quants = [
        (_source_identifier(source), quant.name)
        for source in model.sources
        for quant in source.quants
    ]
    extras = [
        (_source_identifier(source), extra.file)
        for source in model.sources
        for extra in source.extras
    ]
    problems = _reused_name_problems(model, file, "quants", "quant name", quants)
    problems.extend(_reused_name_problems(model, file, "extras", "extra file name", extras))
    return problems


def _reused_name_problems(
    model: CatalogModel,
    file: str,
    section: str,
    label: str,
    named_by: list[tuple[str, str]],
) -> list[Problem]:
    """Report every name in ``named_by`` claimed by a second source, once each."""
    first_source: dict[str, str] = {}
    reported: set[str] = set()
    problems: list[Problem] = []
    for identifier, name in named_by:
        earlier = first_source.get(name)
        if earlier is None:
            first_source[name] = identifier
        elif name not in reported:
            reported.add(name)
            problems.append(
                Problem(
                    file=file,
                    model_id=model.id,
                    location=f"{section}.{name}",
                    message=(
                        f"{label} {name!r} is defined by more than one source "
                        f"({earlier!r} and {identifier!r}); {label}s must be unique "
                        "within a model because the facts file keys on them"
                    ),
                )
            )
    return problems


def _describe(value: object) -> str:
    """Say what a JSON value is, the way a sentence can read it.

    A number is named by its value, since a number in the wrong range is wrong for
    what it says rather than for what it is; everything else is named by its type,
    for example ``a string``.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    return _JSON_TYPE_NAMES.get(type(value), "a value of an unexpected type")


def _count(number: int, noun: str) -> str:
    """A count with its noun in the right number: ``1 file``, ``3 files``."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _wrong_shape(subject: str, found: object, expected: str) -> str:
    """The message for a facts file entry or field that is not the shape it must be."""
    return f"the facts file gives {subject} as {_describe(found)}, not {expected}; {_REWRITE_HINT}"


def _facts_file_problem(facts_path: Path, location: str, message: str) -> list[Problem]:
    """One problem with the facts file as a whole, ready to be returned on its own."""
    return [Problem(file=str(facts_path), model_id=None, location=location, message=message)]


def _merge_facts(
    models: list[CatalogModel], facts_path: Path, unmergeable_ids: set[str]
) -> list[Problem]:
    """Fill each model's volatile fields from its sibling facts file, when there is one.

    Args:
        models: The models just read from the curated YAML file, filled in place.
        facts_path: The sibling facts file, which need not exist.
        unmergeable_ids: Models the facts file structurally cannot describe, because
            their sources reuse a quant name or an extra file name. They are left
            with their curated fields alone; the reused name is already a problem of
            its own, so nothing more is reported here.

    Returns:
        One :class:`Problem` per thing the facts file gets wrong; empty when it
        merged cleanly, or when there is no facts file at all.
    """
    if not facts_path.is_file():
        return []
    try:
        raw = json.loads(facts_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return _facts_file_problem(facts_path, "file", str(exc))
    except json.JSONDecodeError as exc:
        return _facts_file_problem(facts_path, "file", f"invalid JSON: {exc}")

    if not isinstance(raw, dict):
        return _facts_file_problem(facts_path, "file", _EXPECTED_SHAPE)

    version = raw.get("schema_version")
    if isinstance(version, bool) or version != FACTS_SCHEMA_VERSION:
        found = "no schema version" if version is None else f"schema version {version!r}"
        return _facts_file_problem(
            facts_path,
            "schema_version",
            f"the facts file declares {found}, but this build reads version "
            f"{FACTS_SCHEMA_VERSION}; {_REWRITE_HINT}",
        )

    by_model = raw.get("models")
    if not isinstance(by_model, dict):
        return _facts_file_problem(facts_path, "file", _EXPECTED_SHAPE)

    by_id = {model.id: model for model in models}
    problems: list[Problem] = []
    for model_id, model_facts in by_model.items():
        model = by_id.get(model_id)
        if model is None:
            problems.append(
                Problem(
                    file=str(facts_path),
                    model_id=model_id,
                    location="id",
                    message=f"the facts file names an unknown model {model_id!r}",
                )
            )
            continue
        if model_id in unmergeable_ids:
            continue
        if not isinstance(model_facts, dict):
            problems.append(
                Problem(
                    file=str(facts_path),
                    model_id=model_id,
                    location="(root)",
                    message=_wrong_shape(f"model {model_id!r}", model_facts, "an object"),
                )
            )
            continue
        problems.extend(_merge_model_facts(model, model_id, model_facts, facts_path))
    return problems


def _merge_model_facts(
    model: CatalogModel, model_id: str, model_facts: Mapping[str, Any], facts_path: Path
) -> list[Problem]:
    """Merge one model's quant and extra facts, reporting everything the file gets wrong.

    The model's sources are known to use each quant name and extra file name once,
    so every name here names at most one quant or extra.

    Returns:
        One :class:`Problem` per unknown name, malformed entry or malformed field.
    """
    quants_by_name = {quant.name: quant for source in model.sources for quant in source.quants}
    extras_by_name = {extra.file: extra for source in model.sources for extra in source.extras}

    problems: list[Problem] = []

    quants_facts, quants_problems = _facts_section(model_facts, "quants", model_id, facts_path)
    problems.extend(quants_problems)
    for quant_name, quant_facts in quants_facts.items():
        quant = quants_by_name.get(quant_name)
        if quant is None:
            problems.append(
                Problem(
                    file=str(facts_path),
                    model_id=model_id,
                    location=f"quants.{quant_name}",
                    message=f"the facts file names an unknown quant {quant_name!r}",
                )
            )
            continue
        problems.extend(_apply_quant_facts(quant, quant_facts, facts_path, model_id, quant_name))

    extras_facts, extras_problems = _facts_section(model_facts, "extras", model_id, facts_path)
    problems.extend(extras_problems)
    for extra_name, extra_facts in extras_facts.items():
        extra = extras_by_name.get(extra_name)
        if extra is None:
            problems.append(
                Problem(
                    file=str(facts_path),
                    model_id=model_id,
                    location=f"extras.{extra_name}",
                    message=f"the facts file names an unknown extra {extra_name!r}",
                )
            )
            continue
        problems.extend(_apply_extra_facts(extra, extra_facts, facts_path, model_id, extra_name))

    return problems


def _facts_section(
    model_facts: Mapping[str, Any], key: str, model_id: str, facts_path: Path
) -> tuple[Mapping[str, Any], list[Problem]]:
    """Read one model entry's ``quants`` or ``extras`` sub-mapping.

    Returns:
        The sub-mapping, empty when the entry has none or when it is not an object,
        and one :class:`Problem` when it is there but is not an object.
    """
    section = model_facts.get(key)
    if section is None:
        return {}, []
    if not isinstance(section, dict):
        return {}, [
            Problem(
                file=str(facts_path),
                model_id=model_id,
                location=key,
                message=_wrong_shape(f"{key!r}", section, f"an object keyed by {key[:-1]} name"),
            )
        ]
    return section, []


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    """Whether a value is a list of strings, the shape ``files`` and ``sha256`` need."""
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_size(value: object) -> TypeGuard[int]:
    """Whether a value is a size a file could really have: a whole number, zero or more.

    Booleans are excluded, since JSON's ``true`` decodes as the integer 1.
    """
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_bits_per_weight(value: object) -> TypeGuard[float]:
    """Whether a value is a bits-per-weight figure a real quant could have.

    Finite, above zero and no wider than :data:`~llamafit.models.catalog.MAX_BPW`,
    the same bounds :class:`~llamafit.models.catalog.Quant` enforces on curated
    entries. ``json.loads`` decodes ``Infinity`` and ``NaN``, so both are excluded
    here rather than assumed away.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isfinite(value) and 0 < value <= MAX_BPW


def _validation_detail(exc: ValidationError) -> str:
    """The first validation error, as ``field: message``, short enough for one line."""
    error = exc.errors()[0]
    location = ".".join(str(part) for part in error["loc"]) or "(root)"
    return f"{location}: {error['msg']}"


def _apply_quant_facts(
    quant: Quant, facts: object, facts_path: Path, model_id: str, quant_name: str
) -> list[Problem]:
    """Copy the fields a facts file carries for one quant onto it, reporting the bad ones.

    A field that is absent, or ``null`` because nothing was ever refreshed for it, is
    left alone. A field that is present but malformed is left unset and reported, so
    one bad field neither poisons its neighbours nor passes for an unrefreshed one.
    Malformed covers the value as well as the type: a negative size and a
    bits-per-weight figure no real quant could have are refused here exactly as
    :class:`~llamafit.models.catalog.Quant` refuses them in a curated entry. The
    checksums are dropped, and reported, when they cannot be paired with the files
    one to one, since then nobody can say which checksum covers which file. The GGUF
    facts are dropped, and reported, when their tensors do not account for the files'
    own size (see :func:`~llamafit.gguf.facts.accounts_for_file`): those are facts
    about some other file, and a budget built on them would be wrong by the difference.
    The planner then has no facts for the quant and says so, which is the honest answer
    until a refresh reads the file again.

    Returns:
        One :class:`Problem` per malformed field; empty when everything applied.
    """

    def problem(field: str, found: object, expected: str) -> Problem:
        return Problem(
            file=str(facts_path),
            model_id=model_id,
            location=f"quants.{quant_name}.{field}",
            message=_wrong_shape(f"{field!r} for quant {quant_name!r}", found, expected),
        )

    if not isinstance(facts, dict):
        return [
            Problem(
                file=str(facts_path),
                model_id=model_id,
                location=f"quants.{quant_name}",
                message=_wrong_shape(f"quant {quant_name!r}", facts, "an object"),
            )
        ]

    problems: list[Problem] = []

    files = facts.get("files")
    if _is_string_list(files):
        quant.files = files
    elif files is not None:
        problems.append(problem("files", files, "a list of file names"))

    size = facts.get("bytes")
    if _is_size(size):
        quant.bytes_ = size
    elif size is not None:
        problems.append(problem("bytes", size, "a whole number of bytes, zero or more"))

    sha256 = facts.get("sha256")
    if _is_string_list(sha256):
        quant.sha256 = sha256
    elif sha256 is not None:
        problems.append(problem("sha256", sha256, "a list of checksums"))

    bpw = facts.get("bpw")
    if _is_bits_per_weight(bpw):
        quant.bpw = float(bpw)
    elif bpw is not None:
        problems.append(
            problem(
                "bpw", bpw, f"a finite number of bits per weight, above 0 and at most {MAX_BPW:g}"
            )
        )

    gguf_facts = facts.get("gguf_facts")
    if isinstance(gguf_facts, dict):
        try:
            quant.gguf_facts = GgufFacts.model_validate(gguf_facts)
        except ValidationError as exc:
            problems.append(
                Problem(
                    file=str(facts_path),
                    model_id=model_id,
                    location=f"quants.{quant_name}.gguf_facts",
                    message=(
                        f"the facts file's GGUF facts for quant {quant_name!r} are not "
                        f"valid ({_validation_detail(exc)}); {_REWRITE_HINT}"
                    ),
                )
            )
    elif gguf_facts is not None:
        problems.append(problem("gguf_facts", gguf_facts, "an object"))

    if (
        quant.gguf_facts is not None
        and quant.bytes_ is not None
        and not accounts_for_file(quant.gguf_facts, quant.bytes_)
    ):
        problems.append(
            Problem(
                file=str(facts_path),
                model_id=model_id,
                location=f"quants.{quant_name}.gguf_facts",
                message=(
                    f"the facts file's GGUF facts for quant {quant_name!r} sum its tensors "
                    f"to {quant.gguf_facts.bytes_total:,} bytes, but its files hold "
                    f"{quant.bytes_:,}; a tensor was sized wrong, and a memory budget built "
                    f"on these facts would be wrong by the difference; {_REWRITE_HINT}"
                ),
            )
        )
        quant.gguf_facts = None

    if quant.files and quant.sha256 and len(quant.files) != len(quant.sha256):
        problems.append(
            Problem(
                file=str(facts_path),
                model_id=model_id,
                location=f"quants.{quant_name}.sha256",
                message=(
                    f"the facts file gives {_count(len(quant.files), 'file')} but "
                    f"{_count(len(quant.sha256), 'checksum')} for quant {quant_name!r}, so no "
                    f"checksum can be paired with its file; {_REWRITE_HINT}"
                ),
            )
        )
        quant.sha256 = []

    return problems


def _apply_extra_facts(
    extra: Extra, facts: object, facts_path: Path, model_id: str, extra_name: str
) -> list[Problem]:
    """Copy the fields a facts file carries for one extra onto it, reporting the bad ones.

    Returns:
        One :class:`Problem` per malformed field; empty when everything applied.
    """

    def problem(field: str, found: object, expected: str) -> Problem:
        return Problem(
            file=str(facts_path),
            model_id=model_id,
            location=f"extras.{extra_name}.{field}",
            message=_wrong_shape(f"{field!r} for extra {extra_name!r}", found, expected),
        )

    if not isinstance(facts, dict):
        return [
            Problem(
                file=str(facts_path),
                model_id=model_id,
                location=f"extras.{extra_name}",
                message=_wrong_shape(f"extra {extra_name!r}", facts, "an object"),
            )
        ]

    problems: list[Problem] = []

    size = facts.get("bytes")
    if _is_size(size):
        extra.bytes_ = size
    elif size is not None:
        problems.append(problem("bytes", size, "a whole number of bytes, zero or more"))

    sha256 = facts.get("sha256")
    if isinstance(sha256, str):
        extra.sha256 = sha256
    elif sha256 is not None:
        problems.append(problem("sha256", sha256, "a checksum string"))

    return problems


def bundled_catalog_dir() -> Path:
    """The packaged directory holding the curated catalog YAML files.

    Raises:
        PackagedDataError: The directory did not ship in this installation, or shipped
            empty. An empty one is checked for because the loader would read it as a
            catalog with no models in it and report nothing wrong, which is the worst
            way for a tool whose job is reading a catalog to be broken.
    """
    return packaged_dir("llamafit.data.catalog", what="its model catalog", contains="*.yaml")


def custom_models_path(env: Mapping[str, str] | None = None) -> Path:
    """Where the user's custom catalog overrides live.

    ``LLAMAFIT_CUSTOM_MODELS`` names an exact path when set; otherwise the file lives
    in the data directory as ``custom_models.yaml``.
    """
    env = os.environ if env is None else env
    override = env.get("LLAMAFIT_CUSTOM_MODELS")
    if override:
        return Path(override).expanduser()
    return get_paths(env).data_dir / _CUSTOM_MODELS_FILENAME


def load_catalog(
    *,
    bundled_dir: Path | None = None,
    custom_path: Path | None = None,
    strict: bool = False,
) -> tuple[Catalog, list[Problem]]:
    """Load the catalog: bundled files first in sorted order, then the custom file.

    A custom entry whose id already exists replaces the bundled one in place; a new
    id is appended. A duplicate id inside the bundled set is a :class:`Problem`, and
    the later file's entry is discarded. With ``strict=True``, any problem raises a
    :class:`CatalogError` listing every one of them instead of returning them.
    """
    directory = bundled_dir if bundled_dir is not None else bundled_catalog_dir()
    path = custom_path if custom_path is not None else custom_models_path()

    problems: list[Problem] = []
    by_id: dict[str, CatalogModel] = {}
    order: list[str] = []

    if directory.is_dir():
        for file in sorted(directory.glob("*.yaml")):
            models, file_problems = load_models_from_file(file)
            problems.extend(file_problems)
            for model in models:
                if model.id in by_id:
                    problems.append(
                        Problem(
                            file=str(file),
                            model_id=model.id,
                            location="id",
                            message=f"duplicate id {model.id!r}",
                        )
                    )
                    continue
                by_id[model.id] = model
                order.append(model.id)

    if path.is_file():
        models, file_problems = load_models_from_file(path)
        problems.extend(file_problems)
        for model in models:
            if model.id not in by_id:
                order.append(model.id)
            by_id[model.id] = model

    if strict and problems:
        # The count is a sentence and goes through ngettext; the colon that joins it to
        # the detail is punctuation, and the details themselves name YAML fields, which
        # is a catalog author's language rather than a reader's. See docs/translations.md.
        details = "; ".join(f"{p.file}: {p.location}: {p.message}" for p in problems)
        counted = ngettext(
            "the catalog has %(count)d problem",
            "the catalog has %(count)d problems",
            len(problems),
        ) % {"count": len(problems)}
        raise CatalogError(
            f"{counted}: {details}",
            hint=_("Run `llamafit catalog validate` for details."),
        )

    return Catalog(models=[by_id[model_id] for model_id in order]), problems
