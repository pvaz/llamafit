# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Turning a placement into tokens per second, and saying how much to trust the number.

This is section 10 of the design specification. Generation is memory traffic divided by
the bandwidth of the pool it comes out of, plus fixed overheads; prompt processing is
arithmetic on the card plus whatever the experts cost to get there. Both formulas are
written out in :func:`formula_estimate`, and every constant they use lives in
:mod:`llamafit.constants` with the measurement it came from.

Two things about this module are more important than the arithmetic.

**System memory has two effective bandwidths, not one.** A dense model's weights are read
in long contiguous runs and reach 0.70 of the machine's measured read bandwidth. A routed
expert set is not read that way: each token selects a different handful of experts, so a
layer's read is a scatter of small blocks across tens of gigabytes and the memory system
never gets to stream. That is 0.57, and every estimate says in its notes which figure it
applied to which bytes. The three efficiencies and the fixed overhead were identified
together from four runs on the reference machine, two of them a small dense model held
first entirely in system memory and then entirely on the card so that each pool could be
isolated; :mod:`llamafit.constants` carries the arithmetic.

**A formula and a measurement must never look alike.** Section 10.3's four labels are a
precedence, and :func:`estimate_speed` walks it: a benchmark taken on this machine for
this model, quant and flags wins outright and carries the date it was taken; a benchmark
for a different configuration of the same model calibrates the formula; otherwise the
formula runs on defaults and says so; and a placement with nowhere to put the weights
produces no number at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from llamafit.constants import (
    ASSUMED_BITS_PER_WEIGHT,
    DEFAULT_WORKING_CONTEXT,
    EFF_PP,
    EFF_RAM_SCATTERED,
    EFF_RAM_SEQUENTIAL,
    FIXED_OVERHEAD_S,
)
from llamafit.i18n import _
from llamafit.models.catalog import Measured
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Host
from llamafit.models.plan import Confidence, Placement, SpeedEstimate
from llamafit.speed.bandwidths import EffectiveBandwidths, resolve_bandwidths
from llamafit.speed.traffic import (
    TokenTraffic,
    expert_bytes_in_ram,
    per_token_traffic,
    streamed_expert_fraction,
)

_NGL = re.compile(r"(?:^|\s)-ngl\s+(\d+)")
_N_CPU_MOE = re.compile(r"(?:^|\s)--n-cpu-moe\s+(\d+)")
_UBATCH = re.compile(r"(?:^|\s)-ub\s+(\d+)")

_RANK: dict[Confidence, int] = {"measured": 0, "calibrated": 1, "estimated": 2, "unsupported": 3}


@dataclass(frozen=True)
class Formula:
    """What the formula produced, before any measurement was allowed to correct it.

    Attributes:
        gen_tps: Generated tokens per second.
        pp_tps: Prompt tokens per second.
        vram_seconds: Share of a token spent reading the graphics card.
        ram_seconds: Share spent reading system memory.
        overhead_seconds: Per-layer and sampling overheads.
        traffic: The bytes the token reads, itemised.
        notes: What the formula applied, and to which bytes.
    """

    gen_tps: float
    pp_tps: float
    vram_seconds: float
    ram_seconds: float
    overhead_seconds: float
    traffic: TokenTraffic
    notes: tuple[str, ...]


def _weight_bytes(traffic: TokenTraffic) -> int:
    """Bytes of weights the token reads, leaving out the KV cache it also reads."""
    return sum(line.bytes_ for line in traffic.lines if line.component != "kv-cache")


def _active_params(traffic: TokenTraffic, given: float | None) -> float:
    """Parameters active per token: the caller's figure, or one inferred from the bytes."""
    if given:
        return given
    return _weight_bytes(traffic) * 8 / ASSUMED_BITS_PER_WEIGHT


def formula_estimate(
    placement: Placement,
    facts: GgufFacts,
    bandwidths: EffectiveBandwidths,
    *,
    working_context: int,
    micro_batch: int,
    active_params: float | None = None,
) -> Formula:
    """Run section 10's two formulas for one configuration.

    Generation, from section 10.1::

        t_token = bytes_vram / (vram_bw x eff_vram)
                + scattered_bytes / (ram_bw x eff_ram_scattered)
                + sequential_bytes / (ram_bw x eff_ram_sequential)
                + fixed_overhead

    The two system-memory terms are section 10.1's correction: the specification's single
    ``bytes_ram / (ram_bw x eff_ram)`` is split in two because a routed expert read and a
    contiguous weight read do not reach the same fraction of the same memory controller.
    The overhead is one fixed term and not a per-layer one: a per-layer term large enough
    to matter on a 48-layer model exceeds the whole measured token of a 28-layer one.

    Prompt processing, from section 10.2::

        t_ubatch = 2 x active_params x ub / (tflops_fp16 x eff_pp)
                 + streamed_expert_bytes / pcie_bw

    with the expert term charged to system memory instead of the link when there is no
    card to stream to, which is section 10.2's third term.

    Args:
        placement: Where the bytes go and at what settings.
        facts: The file's derived facts.
        bandwidths: What this host's pools achieve.
        working_context: Tokens of KV cache the attention reads per generated token.
        micro_batch: The micro-batch prompt processing is sized for.
        active_params: Parameters active per token; inferred from the bytes when absent.

    Returns:
        The two speeds and the breakdown they came from.
    """
    traffic = per_token_traffic(placement, facts, working_context=working_context)
    notes: list[str] = []

    device_bw = bandwidths.device
    t_vram = traffic.device_bytes / device_bw if device_bw and traffic.device_bytes else 0.0
    t_scattered = traffic.scattered_bytes / bandwidths.scattered if traffic.scattered_bytes else 0.0
    t_sequential = (
        traffic.sequential_bytes / bandwidths.sequential if traffic.sequential_bytes else 0.0
    )
    t_overhead = FIXED_OVERHEAD_S
    t_token = t_vram + t_scattered + t_sequential + t_overhead

    if traffic.scattered_bytes:
        notes.append(
            _(
                "%(gb).2f GB of routed experts is read from system memory per token at %(eff).2f"
                " of its %(raw).0f GB/s, because each token selects a different handful of"
                " experts and the read is a scatter of small blocks rather than a stream."
            )
            % {
                "gb": traffic.scattered_bytes / 1e9,
                "eff": EFF_RAM_SCATTERED,
                "raw": bandwidths.ram_gbps,
            }
        )
    if traffic.sequential_bytes:
        notes.append(
            _(
                "%(gb).2f GB of contiguous weights is read from system memory per token at"
                " %(eff).2f of its %(raw).0f GB/s."
            )
            % {
                "gb": traffic.sequential_bytes / 1e9,
                "eff": EFF_RAM_SEQUENTIAL,
                "raw": bandwidths.ram_gbps,
            }
        )
    if not active_params:
        notes.append(
            _("No active parameter count was supplied; it was inferred from the weight bytes.")
        )

    params = _active_params(traffic, active_params)
    t_compute = 2 * params * micro_batch / (bandwidths.compute_flops * EFF_PP)
    streamed = expert_bytes_in_ram(placement, facts) * streamed_expert_fraction(facts, micro_batch)
    stream_bw = bandwidths.pcie if device_bw else bandwidths.sequential
    t_ubatch = t_compute + (streamed / stream_bw if streamed else 0.0)

    return Formula(
        gen_tps=1.0 / t_token if t_token > 0 else 0.0,
        pp_tps=micro_batch / t_ubatch if t_ubatch > 0 else 0.0,
        vram_seconds=t_vram,
        ram_seconds=t_scattered + t_sequential,
        overhead_seconds=t_overhead,
        traffic=traffic,
        notes=tuple(notes),
    )


def _int_flag(flags: str | None, pattern: re.Pattern[str]) -> int | None:
    """Read one integer flag out of a recorded command line, or ``None`` if absent."""
    if not flags:
        return None
    found = pattern.search(flags)
    return int(found.group(1)) if found else None


def _flags_agree(
    measurement: Measured, placement: Placement, n_layer: int, *, match_micro_batch: bool
) -> bool:
    """Whether every flag the measurement records agrees with this placement.

    A flag the measurement does not record cannot disagree, so a benchmark whose command
    line only mentions the micro-batch still counts as a benchmark of any offload that
    used that micro-batch. ``-ngl`` is compared after clamping to the layer count, because
    99 and 48 mean the same thing on a 48-layer model. The micro-batch is compared only
    for prompt processing: generation on the reference machine varies by four percent
    across micro-batch sizes and by a factor of two across offloads.
    """
    ngl = _int_flag(measurement.flags, _NGL)
    if ngl is not None and n_layer and min(ngl, n_layer) != min(placement.gpu_layers, n_layer):
        return False
    moe = _int_flag(measurement.flags, _N_CPU_MOE)
    if moe is not None and moe != (placement.cpu_moe_layers or 0):
        return False
    if match_micro_batch:
        ubatch = _int_flag(measurement.flags, _UBATCH)
        if ubatch is not None and ubatch != placement.micro_batch:
            return False
    return True


def _as_placement(measurement: Measured, placement: Placement) -> Placement:
    """The placement a benchmark was taken under, as far as its command line records it.

    Everything the flags do not mention is taken from ``placement``, since a benchmark of
    the same model on the same machine differs from it only in what it says it does.
    """
    updates: dict[str, object] = {}
    ngl = _int_flag(measurement.flags, _NGL)
    if ngl is not None:
        updates["gpu_layers"] = ngl
    moe = _int_flag(measurement.flags, _N_CPU_MOE)
    if moe is not None:
        updates["cpu_moe_layers"] = moe
    ubatch = _int_flag(measurement.flags, _UBATCH)
    if ubatch is not None:
        updates["micro_batch"] = ubatch
        updates["batch"] = max(2 * ubatch, 2048)
    if measurement.context:
        updates["context"] = measurement.context
    return placement.model_copy(update=updates) if updates else placement


def _pick(
    measurements: Sequence[Measured],
    placement: Placement,
    *,
    quant: str | None,
    n_layer: int,
    working_context: int,
    prompt: bool,
) -> tuple[Measured | None, Measured | None]:
    """Find the benchmark that matches this configuration, and the closest one that does not.

    Args:
        measurements: Benchmarks taken on this host for this model.
        placement: The configuration being estimated.
        quant: The quantisation being estimated, or ``None`` to accept any.
        n_layer: The model's layer count, for comparing ``-ngl``.
        working_context: The context being estimated, for ranking benchmarks by closeness.
        prompt: True to look for a prompt figure, false for a generation figure.

    Returns:
        The matching benchmark (or ``None``), and the nearest benchmark of the same model
        to calibrate against (or ``None``). Benchmarks are ranked by whether they used
        this micro-batch and then by how close their context is; one that records neither
        is treated as the model's canonical run and ranks first, which is what a
        ``llama-bench tg128`` row is.
    """

    def rank(measurement: Measured) -> tuple[int, int]:
        ubatch = _int_flag(measurement.flags, _UBATCH)
        wrong_ubatch = ubatch is not None and ubatch != placement.micro_batch
        return int(wrong_ubatch), abs((measurement.context or working_context) - working_context)

    usable = [
        m
        for m in measurements
        if (m.pp_tps if prompt else m.gen_tps) and (quant is None or m.quant == quant)
    ]
    usable.sort(key=rank)
    exact = [m for m in usable if _flags_agree(m, placement, n_layer, match_micro_batch=prompt)]
    return (exact[0] if exact else None), (usable[0] if usable else None)


def estimate_speed(
    placement: Placement,
    facts: GgufFacts,
    host: Host,
    *,
    working_context: int | None = None,
    active_params: float | None = None,
    quant: str | None = None,
    measurements: Sequence[Measured] = (),
) -> SpeedEstimate:
    """Estimate how fast ``placement`` will run, and label how the figure was arrived at.

    The label follows section 10.3's precedence. ``measurements`` are benchmarks taken *on
    this host* for this model -- the caller decides that, because a catalog entry records
    where its numbers came from and this module cannot tell one machine from another. A
    benchmark whose recorded flags agree with the placement is used as it stands and the
    estimate is ``measured``, carrying the date. A benchmark of another configuration of
    the same model corrects the formula by the ratio it shows there, and the estimate is
    ``calibrated``. With neither, the formula runs on the project's defaults and the
    estimate is ``estimated``. A placement with nowhere to put the weights is
    ``unsupported`` and carries no number at all.

    Generation and prompt processing walk the ladder separately, because a machine can
    have a measured generation figure and no prompt figure at the micro-batch in question.
    The estimate then carries the weaker of the two labels, since one label has to stand
    for both numbers, and the notes say which figure was which.

    Args:
        placement: Where the bytes go and at what settings.
        facts: The file's derived facts.
        host: The scanned machine.
        working_context: Tokens of KV cache to size the attention read for. Defaults to
            8K, the figure the board compares candidates at, clamped to the placement's
            own context.
        active_params: Parameters active per token, from the catalog. Inferred from the
            weight bytes when absent, which the notes say.
        quant: The quantisation being estimated, so a benchmark of another quant of the
            same model is not mistaken for this one.
        measurements: Benchmarks taken on this host for this model.

    Returns:
        The estimate, with the share of a token spent in each pool and notes saying which
        effective bandwidth was applied to which bytes. The three shares always add up to
        one over ``gen_tps``, whichever rung of the ladder that figure came from: a
        benchmark overturns the level the formula predicted, not the proportions.
    """
    if placement.mode == "unsupported":
        return SpeedEstimate(
            gen_tps=0.0,
            pp_tps=0.0,
            confidence="unsupported",
            notes=(_("No placement of this model on this machine could run it."),),
        )

    context = min(working_context or DEFAULT_WORKING_CONTEXT, placement.context)
    n_layer = facts.n_layer or 0
    bandwidths = resolve_bandwidths(host)
    formula = formula_estimate(
        placement,
        facts,
        bandwidths,
        working_context=context,
        micro_batch=placement.micro_batch,
        active_params=active_params,
    )
    notes = list(formula.notes) + list(bandwidths.notes)

    def rerun(measurement: Measured) -> Formula:
        """Run the formula for the configuration a benchmark was actually taken at.

        This is what makes a calibration a correction rather than a substitution: the
        factor is what the formula got wrong *where the benchmark was taken*, and it is
        then applied to the configuration being asked about. Reusing the target placement
        here would make every factor come out at exactly the measured figure, which would
        relabel a measurement as a calibration and predict nothing.
        """
        anchor = _as_placement(measurement, placement)
        return formula_estimate(
            placement=anchor,
            facts=facts,
            bandwidths=bandwidths,
            working_context=min(measurement.context or context, anchor.context),
            micro_batch=anchor.micro_batch,
            active_params=active_params,
        )

    gen_exact, gen_near = _pick(
        measurements,
        placement,
        quant=quant,
        n_layer=n_layer,
        working_context=context,
        prompt=False,
    )
    pp_exact, pp_near = _pick(
        measurements, placement, quant=quant, n_layer=n_layer, working_context=context, prompt=True
    )

    gen_tps = formula.gen_tps
    gen_label: Confidence = "estimated"
    measured_on: date | None = None
    if gen_exact is not None and gen_exact.gen_tps:
        gen_tps, gen_label, measured_on = gen_exact.gen_tps, "measured", gen_exact.date
        notes.append(_("Generation is a benchmark of this configuration, not a prediction."))
    elif gen_near is not None and gen_near.gen_tps:
        baseline = rerun(gen_near).gen_tps
        if baseline > 0:
            factor = gen_near.gen_tps / baseline
            gen_tps, gen_label = formula.gen_tps * factor, "calibrated"
            notes.append(
                _(
                    "Generation is the formula corrected by %(factor).2f, from a benchmark of"
                    " %(profile)s on this machine taken on %(date)s."
                )
                % {
                    "factor": factor,
                    "profile": gen_near.profile,
                    "date": gen_near.date.isoformat() if gen_near.date else "?",
                }
            )

    pp_tps = formula.pp_tps
    pp_label: Confidence = "estimated"
    if pp_exact is not None and pp_exact.pp_tps:
        pp_tps, pp_label = pp_exact.pp_tps, "measured"
        notes.append(_("Prompt processing is a benchmark of this configuration."))
    elif pp_near is not None and pp_near.pp_tps:
        baseline = rerun(pp_near).pp_tps
        if baseline > 0:
            factor = pp_near.pp_tps / baseline
            pp_tps, pp_label = formula.pp_tps * factor, "calibrated"
            notes.append(
                _("Prompt processing is the formula corrected by %(factor).2f from a benchmark.")
                % {"factor": factor}
            )

    confidence: Confidence = gen_label if _RANK[gen_label] >= _RANK[pp_label] else pp_label
    if bandwidths.assumed and _RANK[confidence] < _RANK["estimated"]:
        confidence = "estimated"
    if placement.budget.verdict == "too-tight":
        notes.append(
            _("This figure assumes nothing pages: a placement this tight often does, silently.")
        )

    # The breakdown is the formula's proportions, rescaled to the token time actually
    # reported. A reader who doubts the figure is owed three numbers that add up to it and
    # say which term dominates; three that add up to something else would be worse than
    # none, and the proportions are the part of the formula a benchmark does not overturn.
    scale = (formula.gen_tps / gen_tps) if gen_tps > 0 and formula.gen_tps > 0 else 1.0

    return SpeedEstimate(
        gen_tps=max(gen_tps, 0.0),
        pp_tps=max(pp_tps, 0.0),
        confidence=confidence,
        measured_on=measured_on if confidence == "measured" else None,
        vram_seconds_per_token=formula.vram_seconds * scale,
        ram_seconds_per_token=formula.ram_seconds * scale,
        overhead_seconds_per_token=formula.overhead_seconds * scale,
        notes=tuple(notes),
    )
