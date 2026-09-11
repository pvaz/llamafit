# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The estimate beside the measurement, with the difference between them left in.

This is the part of ``bench`` that decides whether the rest of LlamaFit can be trusted, and
the temptation it exists to resist is a specific one: once a measurement is in hand it is
the easiest thing in the world to print the measurement, mark it ``measured``, and never
mention that the formula had said something else. That would leave a user better informed
about one model and no better informed about the next one, which is the only thing the
comparison is really for.

So the estimate a row shows is the one that was made **before** the run, from the formula
on its defaults, and it is carried on the stored result rather than recomputed afterwards.
Recomputing it once the measurement is in the database would hand back the measurement:
:func:`llamafit.speed.estimate_speed` would find a benchmark of exactly this configuration,
label it ``measured``, and every ratio would come out at 1.00 for as long as anybody cared
to look.

**Before the run, and for the conditions the run turned out to have.** Those are two
requirements and this module used to meet only the first. Every row of a benchmark was put
beside the plan's single estimate -- one context, one micro-batch -- while the rows
underneath it ran at four different depths of key-value cache and, with ``--sweep``, at
every micro-batch on the ladder. The published sample reported generation at 6.77 times the
estimate for a measurement that matched this project's own calibration record to a tenth of
a percent; what the 6.77 measured was 32,768 tokens of key-value cache against 128.
:func:`llamafit.bench.run.run_benchmark` now runs the formula once per row at that row's
own conditions, and every row here carries them, so the ratio is the estimator's error and
nothing else. :class:`~llamafit.bench.types.ComparisonRow` refuses a ratio that cannot say
what context its two halves were taken at.
"""

from __future__ import annotations

from collections.abc import Sequence

from llamafit.bench.types import BenchRun, ComparisonRow
from llamafit.i18n import _

_ROWS: dict[str, tuple[str, str]] = {
    "llama-bench-tg": ("generation-bench", "tok/s"),
    "llama-bench-pp": ("prompt-bench", "tok/s"),
    "server-short": ("generation-short", "tok/s"),
    "server-1k": ("generation-1k", "tok/s"),
}
"""Which comparison row each kind of run produces, and what its figures are counted in.

One row per kind and never a shared one, because the same configuration generates at three
different speeds depending on whether it is being measured by ``llama-bench``, by a fresh
server on its first request, or by a warm one. Averaging those would turn a real and
explainable spread into apparent noise.
"""


def compare(
    runs: Sequence[BenchRun],
    *,
    predicted_vram_bytes: int | None = None,
    predicted_at_context: int | None = None,
) -> list[ComparisonRow]:
    """Put every measurement beside the estimate that was made before it.

    Args:
        runs: The results of one benchmark, in the order they were taken.
        predicted_vram_bytes: What the budget said the configuration would need on the
            card, so the memory prediction is checked as well as the speed one.
        predicted_at_context: The context that prediction was made for, used when no run
            reports one of its own -- ``--no-server`` leaves a budget figure with nothing
            to have been measured against, and a figure with no conditions is the thing
            this module exists to stop printing.

    Returns:
        One row per figure, ending with peak VRAM when there is a reading for it. Every row
        names the context and the micro-batch that both of its figures are for; peak VRAM,
        which is an allocation rather than a run, names only a context.
    """
    rows: list[ComparisonRow] = []
    for run in runs:
        entry = _ROWS.get(run.kind)
        if entry is None:
            continue
        metric, unit = entry
        context = run.conditions.measured_context
        micro_batch = run.conditions.micro_batch
        if run.kind == "llama-bench-pp":
            rows.append(_row(metric, run.estimated_pp_tps, run.pp_tps, unit, context, micro_batch))
            continue
        # A generation row carries its micro-batch too, though section 10.1 has no term for
        # one. `--sweep` runs `tg128` once per rung of the ladder, and four rows reading
        # "generation (llama-bench), 128 tokens" with four different measurements and one
        # repeated estimate is the same unlabelled table in miniature. Repeated on purpose:
        # the estimate not moving while the measurement does is the formula's claim that
        # generation does not depend on the micro-batch, put where it can be checked.
        rows.append(_row(metric, run.estimated_gen_tps, run.gen_tps, unit, context, micro_batch))
        if run.kind == "server-1k" and run.pp_tps:
            rows.append(
                _row("prompt-1k", run.estimated_pp_tps, run.pp_tps, unit, context, micro_batch)
            )
    peak = max((run.peak_vram_bytes or 0) for run in runs) if runs else 0
    if peak or predicted_vram_bytes:
        rows.append(
            _row(
                "peak-vram",
                float(predicted_vram_bytes) if predicted_vram_bytes else None,
                float(peak) if peak else None,
                "bytes",
                _allocated_context(runs, predicted_at_context),
                None,
            )
        )
    return rows


def _allocated_context(runs: Sequence[BenchRun], planned: int | None) -> int | None:
    """The context the server was started at, which is the one a memory prediction is for.

    The other rows compare speeds and take the context a run *filled*; this one compares a
    cache that was allocated whole at load time, whatever any later request went on to use
    of it. A server's own report of the context it came up at wins over the planned figure,
    because llama.cpp clamps a context it cannot honour and the prediction is then about a
    configuration nobody ran. The planned figure is the fallback for a benchmark that
    started no server and so has nobody to ask.
    """
    return next((run.conditions.context for run in runs if run.conditions.context), planned)


def _row(
    metric: str,
    estimated: float | None,
    measured: float | None,
    unit: str,
    context: int | None,
    micro_batch: int | None,
) -> ComparisonRow:
    """One row, with the ratio filled in only when both halves of it exist and agree.

    "Agree" is the context: two figures taken at different depths of key-value cache are
    two answers to two questions, and dividing one by the other produces a number that
    looks like an error and is not. Without a context there is no ratio, which is the
    honest shape of "these were not compared".
    """
    ratio: float | None = None
    if estimated and measured and context is not None:
        ratio = measured / estimated
    return ComparisonRow(
        metric=metric,
        estimated=estimated,
        measured=measured,
        ratio=ratio,
        unit=unit,
        context=context,
        micro_batch=micro_batch,
    )


def metric_label(metric: str) -> str:
    """The name of one comparison row, in the reader's language.

    Args:
        metric: The stored key.

    Returns:
        A short label naming what was measured and how, because how it was measured is
        half of what the number means.
    """
    labels = {
        "generation-bench": _("generation (llama-bench)"),
        "prompt-bench": _("prompt (llama-bench)"),
        "generation-short": _("generation, first request"),
        "generation-1k": _("generation, 1K prompt"),
        "prompt-1k": _("prompt, 1K request"),
        "peak-vram": _("peak VRAM"),
    }
    return labels.get(metric, metric)


def within_tolerance(ratio: float | None, *, low: float = 0.8, high: float = 1.25) -> bool | None:
    """Whether a ratio is the ordinary kind of wrong, from ``docs/benchmarking.md``.

    Args:
        ratio: Measured over estimated.
        low: The bottom of the band an uncalibrated formula is expected to land in.
        high: The top of it.

    Returns:
        True inside the band, false outside it, and ``None`` when there is no ratio. A
        formula on default constants lands between 0.8 and 1.25 on a machine it was not
        fitted to; outside that band something is worth looking at, and the two directions
        mean different things -- far below with a full card is paging, far below with a
        micro-batch involved is usually the link assumption.
    """
    if ratio is None:
        return None
    return low <= ratio <= high
