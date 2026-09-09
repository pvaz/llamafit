"""Validate catalog files directly, without composing them into one catalog."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llamafit.catalog.loader import Problem, load_models_from_file


def validate_files(paths: Sequence[Path]) -> list[Problem]:
    """Validate each file in ``paths`` independently and report every problem.

    Every model each file declares is parsed and validated on its own terms; a
    syntax error or a failed validation is never an exception, only a
    :class:`Problem`. A duplicate id across the given files is reported too, since
    two files that both claim the same id would otherwise silently shadow one
    another once loaded into a catalog.
    """
    problems: list[Problem] = []
    seen: dict[str, str] = {}
    for path in paths:
        models, file_problems = load_models_from_file(path)
        problems.extend(file_problems)
        for model in models:
            first_seen_in = seen.get(model.id)
            if first_seen_in is not None:
                problems.append(
                    Problem(
                        file=str(path),
                        model_id=model.id,
                        location="id",
                        message=f"duplicate id {model.id!r}, first seen in {first_seen_in}",
                    )
                )
            else:
                seen[model.id] = str(path)
    return problems
