"""Read the catalog from YAML: the bundled files and a user's custom overrides.

Nothing here raises on malformed input. A YAML syntax error, a file that is not a list
of entries, or an entry that fails validation each becomes a :class:`Problem` naming
the file and, where it applies, the field; the caller decides what to do about it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml
from pydantic import ValidationError

from llamafit.errors import CatalogError
from llamafit.models.catalog import Catalog, CatalogModel
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
    return models, problems


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
