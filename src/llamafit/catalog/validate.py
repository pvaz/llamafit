# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Validate catalog files directly, without composing them into one catalog."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llamafit.catalog.loader import Problem, load_models_from_file
from llamafit.models.catalog import CatalogModel


def _declared_contexts(model: CatalogModel) -> list[tuple[str, int]]:
    """Every refreshed quant, with the context length its own GGUF header declares.

    Only ``gguf_facts.context_length`` counts. A ``measured[]`` entry carries a
    ``context`` of its own, but that is the context one measurement was taken at, not
    a claim about what the model supports, and comparing it here would report a
    benchmark run at 16K as a contradiction of a 262K model.

    Returns:
        One ``(quant name, declared context)`` pair per refreshed quant whose header
        declares a context length, in catalog order.
    """
    declared: list[tuple[str, int]] = []
    for source in model.sources:
        for quant in source.quants:
            facts = quant.gguf_facts
            if facts is not None and facts.context_length is not None:
                declared.append((quant.name, facts.context_length))
    return declared


def _context_problem(path: Path, model: CatalogModel) -> Problem | None:
    """Report a curated ``context.native`` that the model's own files contradict.

    A curator types ``context.native`` by hand from a model card; the GGUF header says
    what the file itself permits. Only a curated value *larger* than the file's is
    reported, because that is a claim the file contradicts. A smaller one is a
    deliberate conservative choice, and is often exactly what a vendor documents as
    the supported context even where the file permits more — Qwen3-0.6B is curated at
    its documented 32,768 tokens against a header that declares 40,960 — so flagging
    it would train curators to ignore the check.

    The quant with the smallest declared context is the one named, since that is the
    file the curated value contradicts hardest, and one line per model says more than
    one line per quant when every quant of a model shares the header that carries the
    figure.

    Returns:
        The problem, or ``None`` when nothing has been refreshed yet or the curated
        value is one the files support.
    """
    declared = _declared_contexts(model)
    if not declared:
        return None
    quant_name, smallest = min(declared, key=lambda pair: pair[1])
    if model.context.native <= smallest:
        return None
    return Problem(
        file=str(path),
        model_id=model.id,
        location="context.native",
        message=(
            f"context.native claims {model.context.native:,} tokens, but the GGUF header of "
            f"quant {quant_name!r} declares {smallest:,}. Lower context.native to what the "
            "vendor documents, or check that this quant is the file you meant."
        ),
    )


def validate_files(paths: Sequence[Path]) -> list[Problem]:
    """Validate each file in ``paths`` independently and report every problem.

    Every model each file declares is parsed and validated on its own terms; a
    syntax error or a failed validation is never an exception, only a
    :class:`Problem`. A duplicate id across the given files is reported too, since
    two files that both claim the same id would otherwise silently shadow one
    another once loaded into a catalog.

    A model whose quants have been refreshed is also checked against its own files: a
    hand-typed ``context.native`` longer than the context the GGUF header declares is
    a claim the file contradicts. See :func:`_context_problem` for why the reverse is
    not reported.

    Returns:
        One :class:`Problem` per thing wrong, in the order the files were given.
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
            context_problem = _context_problem(path, model)
            if context_problem is not None:
                problems.append(context_problem)
    return problems
