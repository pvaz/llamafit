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
    DEFAULT_MICRO_BATCH,
    DEFAULT_WORKING_CONTEXT,
    EFF_PP,
    EFF_RAM_SCATTERED,
    EFF_RAM_SEQUENTIAL,
    FIXED_OVERHEAD_S,
    KV_TYPE_DEFAULT,
)
from llamafit.i18n import _
from llamafit.models.catalog import Measured
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Host
from llamafit.models.plan import Confidence, Placement, Pool, SpeedEstimate
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
_KV_CACHE_TYPE = re.compile(r"(?:^|\s)-ct[kv]\s+(\S+)")
_SHARED_EXPERTS_TO_CPU = re.compile(
    r"(?:^|\s)(?:-ot|--override-tensor)\s+[\"']?[^\s\"']*shexp[^\s\"']*=CPU", re.IGNORECASE
)
_PROJECTOR_IN_RAM = re.compile(r"(?:^|\s)--no-mmproj-offload(?=\s|$)")
_PROJECTOR_LEFT_OUT = re.compile(r"(?:^|\s)--no-mmproj(?=\s|$)")

_RANK: dict[Confidence, int] = {"measured": 0, "calibrated": 1, "estimated": 2, "unsupported": 3}


@dataclass(frozen=True)
class Formula:
    """What the formula produced, before any measurement was allowed to correct it.

    The two speeds each keep their own terms, and each set adds up to one over its own
    figure: the generation trio to a token, the prompt trio to a micro-batch.

    Attributes:
        gen_tps: Generated tokens per second.
        pp_tps: Prompt tokens per second.
        vram_seconds: Share of a token spent reading the graphics card.
        ram_seconds: Share spent reading system memory.
        overhead_seconds: Per-layer and sampling overheads.
        pp_compute_seconds: Share of a micro-batch spent on arithmetic, on the card and
            on the CPU together in the proportion the placement splits the layers.
        pp_link_seconds: Share spent streaming the expert set across the link to the card.
        pp_ram_seconds: Share spent reading the expert set out of system memory, which is
            section 10.2's third term and happens instead of the second when there is no
            card to stream to.
        traffic: The bytes the token reads, itemised.
        notes: What the formula applied, and to which bytes.
    """

    gen_tps: float
    pp_tps: float
    vram_seconds: float
    ram_seconds: float
    overhead_seconds: float
    pp_compute_seconds: float
    pp_link_seconds: float
    pp_ram_seconds: float
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


def card_share_of_compute(placement: Placement, n_layer: int, *, card_flops: float) -> float:
    """The share of the prompt arithmetic that runs on the graphics card.

    Prompt processing multiplies matrices where the weights are. A layer offloaded to the
    card multiplies on the card; a layer left in system memory multiplies on the CPU, at a
    rate the two are nowhere near agreeing on. So the share is the share of the layers,
    and ``-ngl`` is what says it.

    ``--n-cpu-moe`` is deliberately not read here. It moves a layer's routed experts into
    system memory, but prompt processing streams them back across the link a micro-batch
    at a time and multiplies them on the card, which is what section 10.2's second term
    charges for; counting those layers as CPU layers would charge the same bytes twice.

    Args:
        placement: Where the bytes go and at what settings.
        n_layer: The model's transformer blocks, or 0 when the file did not say.
        card_flops: What the card reaches, so that a host with no card figure at all
            answers zero rather than dividing by one.

    Returns:
        A fraction from 0 to 1. With no layer count to go on it is all-or-nothing on
        whether ``-ngl`` offloads anything, which is the honest reading of a file that
        does not say how many blocks it has.
    """
    if card_flops <= 0:
        return 0.0
    if n_layer <= 0:
        return 1.0 if placement.gpu_layers > 0 else 0.0
    return min(placement.gpu_layers, n_layer) / n_layer


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

        flops    = 2 x active_params x ub
        t_ubatch = flops x on_card / (tflops_fp16 x eff_pp)      # the card's share
                 + flops x (1 - on_card) / cpu_pp_flops          # the CPU's share
                 + streamed_expert_bytes / pcie_bw

    with the expert term charged to system memory instead of the link when there is no
    card to stream to, which is section 10.2's third term. The three are kept apart rather
    than summed, because the link term is the one this project knows least about: its
    constant was fitted on the single model whose expert set does not fit in system
    memory, and a model whose set stays in the page cache streams about three times
    faster. A reader who is shown the terms can see how much of the answer rests on it.

    **The compute term is split between the card and the CPU**, by
    :func:`card_share_of_compute`, and the specification's single term is not. Prompt
    arithmetic runs where the weights are, and a hybrid placement leaves most of them in
    system memory: charging all of it to the card put Gemma 3 27B, which this machine can
    offload nine layers of out of sixty-two, at a prompt rate that needed six times what
    the same machine's CPU has ever been measured doing.

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
    card_flops = bandwidths.compute_flops * EFF_PP
    on_card = card_share_of_compute(placement, facts.n_layer or 0, card_flops=card_flops)
    prompt_flops = 2 * params * micro_batch
    t_compute = prompt_flops * (1.0 - on_card) / bandwidths.cpu_compute_flops
    if on_card:
        t_compute += prompt_flops * on_card / card_flops
    if on_card < 1.0:
        notes.append(
            _(
                "%(pct).0f percent of the layers sit in system memory, so that share of the"
                " prompt arithmetic is charged to the CPU at %(tflops).2f TFLOP/s rather than"
                " to the card. Both rates come from one small dense model on one machine."
            )
            % {
                "pct": 100.0 * (1.0 - on_card),
                "tflops": bandwidths.cpu_compute_flops / 1e12,
            }
        )
    streamed = expert_bytes_in_ram(placement, facts) * streamed_expert_fraction(facts, micro_batch)
    t_link = streamed / bandwidths.pcie if streamed and device_bw else 0.0
    t_prompt_ram = streamed / bandwidths.sequential if streamed and not device_bw else 0.0
    t_ubatch = t_compute + t_link + t_prompt_ram

    if t_link:
        notes.append(
            _(
                "%(gb).1f GB of the expert set is streamed across the link once per"
                " micro-batch at %(gbps).1f GB/s. That rate was fitted on the one model"
                " whose expert set does not fit in system memory, so part of its read comes"
                " off the disk; an expert set that stays in the page cache streams about"
                " three times faster, and for one of those this term is that much too slow."
            )
            % {"gb": streamed / 1e9, "gbps": bandwidths.pcie / 1e9}
        )
    if t_prompt_ram:
        notes.append(
            _(
                "%(gb).1f GB of the expert set is read out of system memory once per"
                " micro-batch at %(gbps).1f GB/s, because there is no card to stream it to."
            )
            % {"gb": streamed / 1e9, "gbps": bandwidths.sequential / 1e9}
        )

    return Formula(
        gen_tps=1.0 / t_token if t_token > 0 else 0.0,
        pp_tps=micro_batch / t_ubatch if t_ubatch > 0 else 0.0,
        vram_seconds=t_vram,
        ram_seconds=t_scattered + t_sequential,
        overhead_seconds=t_overhead,
        pp_compute_seconds=t_compute,
        pp_link_seconds=t_link,
        pp_ram_seconds=t_prompt_ram,
        traffic=traffic,
        notes=tuple(notes),
    )


def _int_flag(flags: str | None, pattern: re.Pattern[str]) -> int | None:
    """Read one integer flag out of a recorded command line, or ``None`` if absent."""
    if not flags:
        return None
    found = pattern.search(flags)
    return int(found.group(1)) if found else None


def _has_flag(flags: str | None, pattern: re.Pattern[str]) -> bool:
    """Whether a recorded command line carries one switch."""
    return flags is not None and pattern.search(flags) is not None


def _micro_batch(flags: str | None) -> int:
    """The micro-batch a recorded run used: what it names, or llama.cpp's own default."""
    return _int_flag(flags, _UBATCH) or DEFAULT_MICRO_BATCH


def _kv_type(flags: str | None) -> str:
    """The cache type a recorded run used: what it names, or llama.cpp's own default.

    ``llamafit.placement.flags`` does not render ``-ctk``/``-ctv`` for ``f16`` either, so
    a line this project prints and a line somebody recorded agree on what silence means.
    """
    found = _KV_CACHE_TYPE.search(flags) if flags else None
    return found.group(1).lower() if found else KV_TYPE_DEFAULT


def _projector_agrees(flags: str | None, pool: Pool | None) -> bool:
    """Whether a recorded run put the vision projector where this placement puts it.

    Only one direction of this is decidable, and it is the one that matters. A run that
    passed ``--no-mmproj-offload`` put the projector in system memory, and a run that
    passed ``--no-mmproj`` had none loaded at all; both say so. A run that passed neither
    either had no projector or left it on the card, and a catalog entry's quoted flags
    cannot tell those two apart, because the ``--mmproj`` path that would is not part of
    what gets quoted. What silence does settle is that the run was not one that moved a
    projector into system memory, which on an eight-gigabyte card is the difference
    between 1.9 GB of VRAM spent and 1.9 GB free.
    """
    if _has_flag(flags, _PROJECTOR_IN_RAM):
        return pool == "ram"
    if _has_flag(flags, _PROJECTOR_LEFT_OUT):
        return pool is None
    return pool != "ram"


def _flags_agree(
    measurement: Measured, placement: Placement, n_layer: int, *, match_micro_batch: bool
) -> bool:
    """Whether the run this benchmark records is a run of *this* placement.

    Six flags are load-bearing, and they are exactly the ones that decide which bytes are
    read out of which pool: ``-ngl`` and ``--n-cpu-moe`` for the split of layers and of
    routed experts, ``-ot ffn_.*_shexp=CPU`` for the always-on shared experts,
    ``--no-mmproj-offload`` for the vision projector, ``-ctk``/``-ctv`` for the size of a
    cached token, and ``-ub`` for prompt processing, whose whole formula is per
    micro-batch. Everything else a command line carries -- a sampling temperature, a
    thread count, ``--jinja``, ``--fit off`` -- moves no byte between pools and is not
    compared. ``-ngl`` is compared after clamping to the layer count, because 99 and 48
    mean the same thing on a 48-layer model. ``-ub`` is compared only for prompt
    processing: generation on the reference machine varies by four percent across
    micro-batch sizes and by a factor of two across offloads.

    **An unrecorded flag is not agreement.** The switches above have a knowable meaning
    when absent -- no override, no offload refused, ``f16``, ``-ub 512`` -- and are read
    that way. ``-ngl`` has none: this project's own calibration record keeps the layer
    split in a base line the catalog rows then quote only the delta of, so an absent
    ``-ngl`` here means "recorded elsewhere, or not at all", not "none". Since that one
    flag is what separates 78 tokens per second from 279.5 for the same file on the same
    machine, a run that does not name it describes no placement, and this returns False
    rather than matching every placement in sight. Such a run can still calibrate the
    formula; it cannot be called ``measured``, which is the label a reader trusts above
    all the others and the one a false claim does the most damage to.
    """
    flags = measurement.flags
    ngl = _int_flag(flags, _NGL)
    if ngl is None:
        return False
    if n_layer and min(ngl, n_layer) != min(placement.gpu_layers, n_layer):
        return False
    if (_int_flag(flags, _N_CPU_MOE) or 0) != (placement.cpu_moe_layers or 0):
        return False
    if _has_flag(flags, _SHARED_EXPERTS_TO_CPU) != (placement.shared_experts_pool == "ram"):
        return False
    if not _projector_agrees(flags, placement.projector_pool):
        return False
    if _kv_type(flags) != placement.kv_type.lower():
        return False
    return not (match_micro_batch and _micro_batch(flags) != placement.micro_batch)


def _as_placement(measurement: Measured, placement: Placement) -> Placement:
    """The placement a benchmark was taken under, as far as its command line records it.

    Every load-bearing flag is read back the way :func:`_flags_agree` compares it, so the
    anchor a calibration is computed at is the configuration the run actually describes
    rather than the one being asked about. What the flags cannot settle -- how many layers
    went to the card when ``-ngl`` is missing, and where a projector went when nothing
    says -- is taken from ``placement``, since a benchmark of the same model on the same
    machine differs from it only in what it says it does. That residue is why a run
    missing ``-ngl`` calibrates against itself and the estimate says so.
    """
    updates: dict[str, object] = {
        "micro_batch": _micro_batch(measurement.flags),
        "batch": max(2 * _micro_batch(measurement.flags), 2048),
        "kv_type": _kv_type(measurement.flags),
        "cpu_moe_layers": _int_flag(measurement.flags, _N_CPU_MOE) or 0,
        "shared_experts_pool": (
            "ram" if _has_flag(measurement.flags, _SHARED_EXPERTS_TO_CPU) else None
        ),
    }
    ngl = _int_flag(measurement.flags, _NGL)
    if ngl is not None:
        updates["gpu_layers"] = ngl
    if measurement.context:
        updates["context"] = measurement.context
    return placement.model_copy(update=updates)


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
        this micro-batch and then by how close their context is; a run that names no
        micro-batch is ranked as the ``-ub 512`` it was, and one that names no context is
        taken at the context being asked about, which is what a ``llama-bench tg128`` row
        deserves -- it was run at a context small enough not to matter.
    """

    def rank(measurement: Measured) -> tuple[int, int]:
        wrong_ubatch = _micro_batch(measurement.flags) != placement.micro_batch
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
        The estimate, with the share of a token spent in each pool, the share of a prompt
        token spent in each of section 10.2's three terms, and notes saying which
        effective bandwidth was applied to which bytes. Each trio always adds up to one
        over its own figure, whichever rung of the ladder that figure came from: a
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

    unpinned = False

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
            unpinned = unpinned or _int_flag(gen_near.flags, _NGL) is None
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
            unpinned = unpinned or _int_flag(pp_near.flags, _NGL) is None
            notes.append(
                _("Prompt processing is the formula corrected by %(factor).2f from a benchmark.")
                % {"factor": factor}
            )

    confidence: Confidence = gen_label if _RANK[gen_label] >= _RANK[pp_label] else pp_label
    if bandwidths.assumed and _RANK[confidence] < _RANK["estimated"]:
        confidence = "estimated"
    if unpinned:
        notes.append(
            _(
                "That benchmark's command line does not record how many layers went to the"
                " card, so the correction assumes it ran at this placement."
            )
        )
    if placement.budget.verdict == "too-tight":
        notes.append(
            _("This figure assumes nothing pages: a placement this tight often does, silently.")
        )

    # Both breakdowns are the formula's proportions, rescaled to the time actually
    # reported. A reader who doubts a figure is owed terms that add up to it and say which
    # one dominates; terms that add up to something else would be worse than none, and the
    # proportions are the part of the formula a benchmark does not overturn. The prompt
    # terms are divided by the micro-batch they were computed for, so that they add up to
    # one over ``pp_tps`` exactly as the generation terms add up to one over ``gen_tps``
    # and the two tables can be read the same way.
    scale = (formula.gen_tps / gen_tps) if gen_tps > 0 and formula.gen_tps > 0 else 1.0
    pp_scale = (formula.pp_tps / pp_tps) if pp_tps > 0 and formula.pp_tps > 0 else 1.0
    per_prompt_token = pp_scale / placement.micro_batch

    return SpeedEstimate(
        gen_tps=max(gen_tps, 0.0),
        pp_tps=max(pp_tps, 0.0),
        confidence=confidence,
        measured_on=measured_on if confidence == "measured" else None,
        vram_seconds_per_token=formula.vram_seconds * scale,
        ram_seconds_per_token=formula.ram_seconds * scale,
        overhead_seconds_per_token=formula.overhead_seconds * scale,
        prompt_compute_seconds_per_token=formula.pp_compute_seconds * per_prompt_token,
        prompt_link_seconds_per_token=formula.pp_link_seconds * per_prompt_token,
        prompt_ram_seconds_per_token=formula.pp_ram_seconds * per_prompt_token,
        notes=tuple(notes),
    )
