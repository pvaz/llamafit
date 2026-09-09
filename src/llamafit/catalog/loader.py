"""Read the catalog from YAML: the bundled files and a user's custom overrides.

Nothing here raises on malformed input. A YAML syntax error, a file that is not a list
of entries, or an entry that fails validation each becomes a :class:`Problem` naming
the file and, where it applies, the field; the caller decides what to do about it.

A curated ``<family>.yaml`` file never carries the volatile fields ``llamafit catalog
refresh`` fills in: those live beside it, in ``<family>.facts.json``, written only by
that command. This module merges the sibling facts file into the models it loads, so
the rest of LlamaFit never has to know the two files exist. A facts file that is
missing simply means nothing has been refreshed yet, not a problem; one that names a
model or a quant the YAML does not have is a :class:`Problem` naming both.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llamafit.errors import CatalogError
from llamafit.models.catalog import Catalog, CatalogModel, Extra, Quant
from llamafit.models.gguf import GgufFacts
from llamafit.paths import get_paths

_CUSTOM_MODELS_FILENAME = "custom_models.yaml"


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


def load_models_from_file(path: Path) -> tuple[list[CatalogModel], list[Problem]]:
    """Parse and validate every entry in one catalog YAML file.

    The file must hold a list of entries at its top level. A syntax error, a
    top-level shape that is not a list, or an entry that fails validation each
    becomes a :class:`Problem`; this function never raises for malformed input.
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
    for entry in raw:
        model_id = entry.get("id") if isinstance(entry, dict) else None
        try:
            models.append(CatalogModel.model_validate(entry))
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
    problems.extend(_merge_facts(models, facts_path_for(path)))
    return models, problems


def facts_path_for(path: Path) -> Path:
    """The generated facts file that sits beside one curated catalog file."""
    return path.with_name(f"{path.stem}.facts.json")


def _merge_facts(models: list[CatalogModel], facts_path: Path) -> list[Problem]:
    """Fill each model's volatile fields from its sibling facts file, when there is one."""
    if not facts_path.is_file():
        return []
    try:
        raw = json.loads(facts_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [Problem(file=str(facts_path), model_id=None, location="file", message=str(exc))]
    except json.JSONDecodeError as exc:
        return [
            Problem(
                file=str(facts_path), model_id=None, location="file", message=f"invalid JSON: {exc}"
            )
        ]

    by_model = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(by_model, dict):
        return [
            Problem(
                file=str(facts_path),
                model_id=None,
                location="file",
                message="expected an object with a 'models' mapping",
            )
        ]

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
        if not isinstance(model_facts, dict):
            continue
        problems.extend(_merge_model_facts(model, model_id, model_facts, facts_path))
    return problems


def _merge_model_facts(
    model: CatalogModel, model_id: str, model_facts: Mapping[str, Any], facts_path: Path
) -> list[Problem]:
    """Merge one model's quant and extra facts, reporting names the model does not have."""
    problems: list[Problem] = []

    quants_by_name: dict[str, list[Quant]] = {}
    extras_by_name: dict[str, list[Extra]] = {}
    for source in model.sources:
        for quant in source.quants:
            quants_by_name.setdefault(quant.name, []).append(quant)
        for extra in source.extras:
            extras_by_name.setdefault(extra.file, []).append(extra)

    quants_facts = model_facts.get("quants")
    if isinstance(quants_facts, dict):
        for quant_name, quant_facts in quants_facts.items():
            matches = quants_by_name.get(quant_name)
            if not matches:
                problems.append(
                    Problem(
                        file=str(facts_path),
                        model_id=model_id,
                        location=f"quants.{quant_name}",
                        message=f"the facts file names an unknown quant {quant_name!r}",
                    )
                )
                continue
            for quant in matches:
                _apply_quant_facts(quant, quant_facts)

    extras_facts = model_facts.get("extras")
    if isinstance(extras_facts, dict):
        for extra_name, extra_facts in extras_facts.items():
            extra_matches = extras_by_name.get(extra_name)
            if not extra_matches:
                problems.append(
                    Problem(
                        file=str(facts_path),
                        model_id=model_id,
                        location=f"extras.{extra_name}",
                        message=f"the facts file names an unknown extra {extra_name!r}",
                    )
                )
                continue
            for extra in extra_matches:
                _apply_extra_facts(extra, extra_facts)

    return problems


def _apply_quant_facts(quant: Quant, facts: object) -> None:
    """Copy the fields a facts file may carry for one quant onto it, ignoring the rest."""
    if not isinstance(facts, dict):
        return
    files = facts.get("files")
    if isinstance(files, list) and all(isinstance(item, str) for item in files):
        quant.files = files
    size = facts.get("bytes")
    if isinstance(size, int) and not isinstance(size, bool):
        quant.bytes_ = size
    sha256 = facts.get("sha256")
    if isinstance(sha256, list) and all(isinstance(item, str) for item in sha256):
        quant.sha256 = sha256
    bpw = facts.get("bpw")
    if isinstance(bpw, (int, float)) and not isinstance(bpw, bool):
        quant.bpw = float(bpw)
    gguf_facts = facts.get("gguf_facts")
    if isinstance(gguf_facts, dict):
        with suppress(ValidationError):
            quant.gguf_facts = GgufFacts.model_validate(gguf_facts)


def _apply_extra_facts(extra: Extra, facts: object) -> None:
    """Copy the fields a facts file may carry for one extra onto it, ignoring the rest."""
    if not isinstance(facts, dict):
        return
    size = facts.get("bytes")
    if isinstance(size, int) and not isinstance(size, bool):
        extra.bytes_ = size
    sha256 = facts.get("sha256")
    if isinstance(sha256, str):
        extra.sha256 = sha256


def bundled_catalog_dir() -> Path:
    """The packaged directory holding the curated catalog YAML files."""
    return Path(str(resources.files("llamafit.data.catalog")))


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
        details = "; ".join(f"{p.file}: {p.location}: {p.message}" for p in problems)
        raise CatalogError(
            f"the catalog has {len(problems)} problem(s): {details}",
            hint="Run `llamafit catalog validate` for details.",
        )

    return Catalog(models=[by_id[model_id] for model_id in order]), problems
