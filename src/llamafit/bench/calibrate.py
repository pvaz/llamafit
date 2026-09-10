# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Fitting section 10's constants to one machine, and refusing to fit the rest.

Section 16.3 lists five things to calibrate. This module fits the ones the data can
determine and says, by name and with a reason, why it did not fit the others. That is the
whole design, and it comes from a specific failure this project has already met once: a
number that looks precise and is arbitrary is worse than an admitted default, because the
label attached to it says somebody measured something.

**Generation** is linear in the four constants, which is what makes identifiability a
question with an answer rather than a matter of taste. Section 10.1 says::

    t_token = device_bytes / (device_bw x eff_vram)
            + sequential_bytes / (ram_bw x eff_ram_sequential)
            + scattered_bytes / (ram_bw x eff_ram_scattered)
            + fixed_overhead

Write ``A = 1/eff_vram``, ``B = 1/eff_ram_sequential``, ``C = 1/eff_ram_scattered`` and
``F = fixed_overhead``, and every run is one equation::

    t = (device_bytes/device_bw) A + (sequential_bytes/ram_bw) B
      + (scattered_bytes/ram_bw) C + F

Four unknowns. Two runs give two equations, and two equations do not determine three
unknowns however carefully they were measured -- which is exactly why the reference
machine's calibration set has four runs in it and why two of them are a small dense model
held first entirely in system memory and then entirely on the card. Those two exist to put
a non-zero in a column that would otherwise be empty.

So the rule is arithmetic and not judgement: a constant is fitted when its column carries
signal, when there are at least as many distinct configurations as free parameters, and
when the columns are independent enough for the system to have one solution. Anything else
is refused by name.

**And a fit that comes out impossible is discarded whole.** An efficiency is a fraction of
a bandwidth; a value above one says the card moved more bytes than it has bytes to move.
When that happens the measurements are mutually inconsistent under section 10.1's model --
either one of them is wrong or the model is missing a term -- and the honest report is that
sentence, not four numbers three of which happen to look reasonable. This is not
hypothetical: the reference machine's own four published runs, solved exactly, put
``eff_vram`` above 1.6, which is why the shipped constants were chosen as round figures
consistent with all four rather than read off a solve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median

from llamafit import __version__
from llamafit.bench.lstsq import least_squares
from llamafit.bench.parse import device_compute_bytes
from llamafit.bench.types import (
    BenchKind,
    BenchRun,
    Calibration,
    ComputeBufferPoint,
    Refusal,
    RunTraffic,
)
from llamafit.i18n import _

GENERATION_KINDS: tuple[BenchKind, ...] = ("llama-bench-tg", "server-1k")
"""Which measurements the generation fit may use.

``server-short`` is deliberately not here. The calibration record is explicit that the
first request after a cold start is slower than every later one because the model's data
pages in from disk while it runs -- 8.3 tokens per second against 13.7 for the same
configuration a moment later. That is a real measurement of a real thing, and the thing it
measures is a disk, not a memory system. Fitting a memory constant to it would move every
estimate on the machine towards a state that lasts one request.
"""

GENERATION_PARAMETERS = (
    "eff_vram",
    "eff_ram_sequential",
    "eff_ram_scattered",
    "fixed_overhead_s",
)
"""The four constants of section 10.1, in the order their columns are built."""

PROMPT_PARAMETERS = ("eff_pp", "pcie_effective_gbps")
"""The two constants of section 10.2 that a prompt measurement can reach."""

_MIN_CONTEXTS_FOR_A_SLOPE = 2
"""Distinct contexts needed before a straight line through them means anything."""


@dataclass(frozen=True)
class _Point:
    """One distinct configuration, with its repeats already reduced to a median."""

    traffic: RunTraffic
    gen_tps: float | None
    pp_tps: float | None
    context: int | None
    micro_batch: int | None
    compute_bytes: int | None


def calibrate(
    runs: Sequence[BenchRun], *, host_fingerprint: str, now: datetime | None = None
) -> Calibration:
    """Fit what this machine's measurements determine, and name what they do not.

    Args:
        runs: Every stored run. Runs from other machines, and runs the paging detector
            caught, are dropped here rather than by the caller.
        host_fingerprint: The machine being fitted.
        now: The moment to stamp the result with; the current time when omitted.

    Returns:
        The calibration, with a refusal for every constant that was not fitted.

    A calibration with no fitted constants is still returned rather than raising. "Nothing
    here could be determined, and here is what each of them was missing" is the useful
    answer to somebody who has just run one benchmark and expected five numbers.
    """
    mine = [
        run
        for run in runs
        if run.conditions.host_fingerprint == host_fingerprint
        and run.trustworthy
        and run.traffic is not None
    ]
    refusals: list[Refusal] = []
    generation = _fit_generation(mine, refusals)
    prompt = _fit_prompt(mine, refusals)
    buffers = _fit_compute_buffers(mine, refusals)
    refusals.append(Refusal(parameter="layer_overhead_ms", reason="no-term-in-the-model"))
    return Calibration(
        host_fingerprint=host_fingerprint,
        fitted_at=now or datetime.now(timezone.utc),
        llamafit_version=__version__,
        eff_vram=generation.values.get("eff_vram"),
        eff_ram_sequential=generation.values.get("eff_ram_sequential"),
        eff_ram_scattered=generation.values.get("eff_ram_scattered"),
        fixed_overhead_s=generation.values.get("fixed_overhead_s"),
        eff_pp=prompt.values.get("eff_pp"),
        pcie_effective_gbps=prompt.values.get("pcie_effective_gbps"),
        compute_buffer=buffers,
        generation_runs=generation.points,
        prompt_runs=prompt.points,
        generation_residual=generation.residual,
        prompt_residual=prompt.residual,
        refusals=tuple(refusals),
    )


@dataclass(frozen=True)
class _Outcome:
    """What one fit produced: the constants it determined, and how well."""

    values: dict[str, float]
    points: int
    residual: float | None


def _distinct(runs: Sequence[BenchRun], kinds: Sequence[BenchKind]) -> list[_Point]:
    """Group runs of the given kinds by conditions and reduce each group to one point.

    Repeats are reduced to their median, not their mean and not their best. A benchmark is
    a claim about what a machine does; the fastest of five runs is a claim about what it
    did once, and one slow run out of five should not drag a constant with it either.
    """
    grouped: dict[str, list[BenchRun]] = {}
    for run in runs:
        if run.kind in kinds:
            grouped.setdefault(run.conditions_hash, []).append(run)
    points: list[_Point] = []
    for repeats in grouped.values():
        newest = max(repeats, key=lambda item: item.recorded_at)
        if newest.traffic is None:
            continue
        gen = [run.gen_tps for run in repeats if run.gen_tps]
        prompt = [run.pp_tps for run in repeats if run.pp_tps]
        compute = [
            value
            for run in repeats
            if (value := device_compute_bytes(run.buffer_bytes)) is not None
        ]
        points.append(
            _Point(
                traffic=newest.traffic,
                gen_tps=float(median(gen)) if gen else None,
                pp_tps=float(median(prompt)) if prompt else None,
                context=newest.conditions.context,
                micro_batch=newest.conditions.micro_batch,
                compute_bytes=int(median(compute)) if compute else None,
            )
        )
    return points


def _fit_generation(runs: Sequence[BenchRun], refusals: list[Refusal]) -> _Outcome:
    """Fit section 10.1's four constants, or refuse each of them by name."""
    points = [p for p in _distinct(runs, GENERATION_KINDS) if p.gen_tps]
    rows: list[list[float]] = []
    targets: list[float] = []
    for point in points:
        traffic = point.traffic
        device = traffic.device_gbps * 1e9 if traffic.device_gbps else 0.0
        ram = traffic.ram_gbps * 1e9
        rows.append(
            [
                traffic.device_bytes / device if device else 0.0,
                traffic.sequential_bytes / ram,
                traffic.scattered_bytes / ram,
                1.0,
            ]
        )
        targets.append(1.0 / float(point.gen_tps or 1.0))
    solution = _solve(GENERATION_PARAMETERS, rows, targets, refusals)
    if solution is None:
        return _Outcome(values={}, points=len(points), residual=None)
    coefficients, residual = solution
    values = {
        "eff_vram": _reciprocal(coefficients.get("eff_vram")),
        "eff_ram_sequential": _reciprocal(coefficients.get("eff_ram_sequential")),
        "eff_ram_scattered": _reciprocal(coefficients.get("eff_ram_scattered")),
        "fixed_overhead_s": coefficients.get("fixed_overhead_s"),
    }
    kept = _check_bounds(values, refusals, len(points))
    return _Outcome(values=kept, points=len(points), residual=residual)


def _fit_prompt(runs: Sequence[BenchRun], refusals: list[Refusal]) -> _Outcome:
    """Fit section 10.2's compute efficiency and effective link bandwidth.

    Only ``llama-bench`` prompt rows are used. A prompt figure from a real request carries
    the chat template, the tokeniser and whatever the server's prompt cache had left over,
    none of which section 10.2 models; ``llama-bench`` runs a fixed prompt three times and
    reports the median, which is the measurement this formula is about.
    """
    points = [
        p
        for p in _distinct(runs, ("llama-bench-pp",))
        if p.pp_tps and p.traffic.compute_flops > 0 and p.traffic.micro_batch > 0
    ]
    rows: list[list[float]] = []
    targets: list[float] = []
    for point in points:
        traffic = point.traffic
        rows.append(
            [
                2.0 * traffic.active_params * traffic.micro_batch / traffic.compute_flops,
                float(traffic.streamed_expert_bytes),
            ]
        )
        targets.append(traffic.micro_batch / float(point.pp_tps or 1.0))
    solution = _solve(PROMPT_PARAMETERS, rows, targets, refusals)
    if solution is None:
        return _Outcome(values={}, points=len(points), residual=None)
    coefficients, residual = solution
    per_byte = coefficients.get("pcie_effective_gbps")
    values = {
        "eff_pp": _reciprocal(coefficients.get("eff_pp")),
        # The column is bytes and the coefficient is seconds per byte, so its reciprocal is
        # bytes per second and the constant this fills is quoted in GB/s.
        "pcie_effective_gbps": _reciprocal(per_byte, scale=1e-9),
    }
    kept = _check_bounds(values, refusals, len(points))
    return _Outcome(values=kept, points=len(points), residual=residual)


def _fit_compute_buffers(
    runs: Sequence[BenchRun], refusals: list[Refusal]
) -> tuple[ComputeBufferPoint, ...]:
    """Fit a base and a per-1K slope for the compute buffer, one micro-batch at a time.

    Section 8.2 models the buffer as ``base + slope x context``, and the two cannot be
    separated from a single context: one point determines a line only if you already know
    its gradient. So a micro-batch measured at one context is refused rather than given a
    base equal to whatever was allocated there.
    """
    by_batch: dict[int, list[_Point]] = {}
    for point in _distinct(runs, ("server-short", "server-1k", "server-toolcall")):
        if point.compute_bytes is None or not point.micro_batch or not point.context:
            continue
        by_batch.setdefault(point.micro_batch, []).append(point)
    fitted: list[ComputeBufferPoint] = []
    for micro_batch, points in sorted(by_batch.items()):
        contexts = {point.context for point in points}
        if len(contexts) < _MIN_CONTEXTS_FOR_A_SLOPE:
            refusals.append(
                Refusal(
                    parameter=f"compute_buffer[{micro_batch}]",
                    reason="too-few-measurements",
                    measurements=len(contexts),
                    parameters=_MIN_CONTEXTS_FOR_A_SLOPE,
                )
            )
            continue
        rows = [[1.0, (point.context or 0) / 1024.0] for point in points]
        targets = [float(point.compute_bytes or 0) for point in points]
        solution = least_squares(rows, targets)
        if solution.values is None:
            refusals.append(
                Refusal(
                    parameter=f"compute_buffer[{micro_batch}]",
                    reason="not-identifiable",
                    measurements=len(points),
                    parameters=2,
                )
            )
            continue
        base, slope = solution.values
        if base < 0 or slope < 0:
            refusals.append(
                Refusal(
                    parameter=f"compute_buffer[{micro_batch}]",
                    reason="unphysical",
                    measurements=len(points),
                    parameters=2,
                    value=min(base, slope),
                )
            )
            continue
        fitted.append(
            ComputeBufferPoint(
                micro_batch=micro_batch,
                base_bytes=int(base),
                per_1k_context_bytes=int(slope),
                measurements=len(points),
            )
        )
    return tuple(fitted)


def _solve(
    names: Sequence[str],
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    refusals: list[Refusal],
) -> tuple[dict[str, float], float | None] | None:
    """Fit only the columns that carry signal, refusing every parameter that is not fitted.

    Returns the fitted coefficients keyed by name and the residual, or ``None`` when
    nothing could be fitted. The residual is ``None`` for an exactly determined system,
    where it is zero by construction and says nothing about agreement.
    """
    free = [index for index in range(len(names)) if any(abs(row[index]) > 0.0 for row in rows)]
    for index, name in enumerate(names):
        if index not in free:
            refusals.append(Refusal(parameter=name, reason="no-data", measurements=len(rows)))
    if not free:
        return None
    if len(rows) < len(free):
        for index in free:
            refusals.append(
                Refusal(
                    parameter=names[index],
                    reason="too-few-measurements",
                    measurements=len(rows),
                    parameters=len(free),
                )
            )
        return None
    reduced = [[row[index] for index in free] for row in rows]
    solution = least_squares(reduced, targets)
    if solution.values is None:
        for index in free:
            refusals.append(
                Refusal(
                    parameter=names[index],
                    reason="not-identifiable",
                    measurements=len(rows),
                    parameters=len(free),
                )
            )
        return None
    coefficients = {names[index]: solution.values[slot] for slot, index in enumerate(free)}
    residual = None if solution.exactly_determined else solution.residual
    return coefficients, residual


def refusal_text(refusal: Refusal) -> str:
    """Why one constant was not fitted, in the reader's language.

    Args:
        refusal: The stored refusal.

    Returns:
        One sentence. The reason is stored as a key and turned into prose here, so a
        calibration fitted by somebody working in one language reads correctly to somebody
        working in another.
    """
    if refusal.reason == "no-data":
        return _("nothing measured touches this term, so there is no equation it appears in")
    if refusal.reason == "too-few-measurements":
        return _(
            "%(measurements)d distinct configurations for %(parameters)d free parameters;"
            " fewer points than parameters gives numbers that look precise and are"
            " arbitrary"
        ) % {"measurements": refusal.measurements, "parameters": refusal.parameters}
    if refusal.reason == "not-identifiable":
        return _(
            "the %(measurements)d configurations do not vary independently enough to"
            " separate %(parameters)d parameters"
        ) % {"measurements": refusal.measurements, "parameters": refusal.parameters}
    if refusal.reason == "unphysical":
        return _(
            "the fit put it at %(value).3f, which its own definition does not allow; the"
            " measurements disagree with each other under section 10.1's model"
        ) % {"value": refusal.value if refusal.value is not None else 0.0}
    if refusal.reason == "discarded-with-the-fit":
        return _(
            "it came out at %(value).3f, but another parameter of the same fit was"
            " impossible, and least squares chose them against each other"
        ) % {"value": refusal.value if refusal.value is not None else 0.0}
    return _(
        "the estimator has no such term: section 10.1 removed the per-layer overhead,"
        " which for a 28-layer model exceeded the whole measured token"
    )


def _reciprocal(coefficient: float | None, *, scale: float = 1.0) -> float | None:
    """Turn a fitted coefficient back into the constant it is the reciprocal of.

    A coefficient of exactly zero comes back as infinity rather than as ``None``, and a
    negative one comes back negative. Both are impossible values that :func:`_check_bounds`
    then refuses by name. Swallowing them here as "not fitted" would lose the difference
    between a constant the data said nothing about and one the data said something
    impossible about, and those want telling apart.
    """
    if coefficient is None:
        return None
    if coefficient == 0.0:
        return float("inf")
    return 1.0 / (coefficient / scale)


def _within_bounds(name: str, value: float) -> bool:
    """Whether a fitted constant is a number its own definition allows.

    An efficiency is a fraction of a bandwidth, so it lies in ``(0, 1]``. An overhead is a
    duration and cannot be negative. An effective link bandwidth has to be a finite
    positive rate -- and is deliberately *not* capped at the link's rated speed, because
    the calibration record measures an expert set streaming at 12.8 GB/s across a link
    rated at 12: what is being measured there is residency in the page cache, not the wire.
    """
    if value != value or value in (float("inf"), float("-inf")):  # NaN or infinite
        return False
    if name.startswith("eff_"):
        return 0.0 < value <= 1.0
    if name == "fixed_overhead_s":
        return value >= 0.0
    return value > 0.0


def _check_bounds(
    values: dict[str, float | None], refusals: list[Refusal], points: int
) -> dict[str, float]:
    """Discard the whole fit when any parameter of it landed somewhere impossible.

    When one parameter breaks its bounds the others are not thereby right -- they are the
    numbers least squares chose to compensate for an impossible one -- so every parameter
    of the fit is refused and the value each of them reached is reported, so a reader can
    see which one was impossible and by how much.
    """
    present = {name: value for name, value in values.items() if value is not None}
    if not present:
        return {}
    impossible = [name for name, value in present.items() if not _within_bounds(name, value)]
    if not impossible:
        return present
    for name, value in present.items():
        refusals.append(
            Refusal(
                parameter=name,
                reason="unphysical" if name in impossible else "discarded-with-the-fit",
                measurements=points,
                value=value,
            )
        )
    return {}
