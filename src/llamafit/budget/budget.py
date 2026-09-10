# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Compose one configuration's memory budget and say how well it fits.

This is the function every later stage rests on: the planner searches placements by
computing a budget for each, the estimator turns the placement it picks into a speed, and
the score turns that into a ranking. If this is wrong, everything after it is wrong in the
same direction and just as confidently.

Two decisions here are worth reading before the code.

The first is that a budget never speaks in one voice. Weights and the key-value cache are
arithmetic on tensor sizes the file itself reports; the compute buffer is a formula fitted
to ten measurements on one graphics card. Both end up as byte figures in the same table,
and without the ``exact`` flag on each line the second would borrow the authority of the
first. Section 8.1's own components divide about evenly between them.

The second is what happens above the line. An NVIDIA driver does not refuse an allocation
larger than the card: it pages the excess to system memory, the server starts, the log
looks healthy, and generation runs at a third of its speed with nothing anywhere saying
why. That is not "does not fit" -- it fits, in the sense that it runs -- and it is the
failure a person cannot diagnose for themselves, which is why section 8.4 gives it a name
of its own. A configuration one byte over the card's free memory is reported as paging.
"""

from __future__ import annotations

from math import inf

from llamafit.constants import (
    CUDA_CONTEXT_BYTES,
    PROCESS_OVERHEAD_BYTES,
    UTILISATION_COMFORTABLE,
    UTILISATION_FITS,
    UTILISATION_TIGHT,
    VRAM_RESERVE_BYTES,
)
from llamafit.errors import BudgetError
from llamafit.i18n import _
from llamafit.models.catalog import CatalogModel, Extra, Quant
from llamafit.models.host import Host
from llamafit.models.plan import Budget, BudgetLine, Pool, RunMode, Verdict

from .compute_buffer import batch_for, buffer_lines
from .kv import cache_lines
from .projector import projector_lines
from .weights import weight_lines

_COMPONENT_ORDER = (
    "dense-weights",
    "shared-expert-weights",
    "expert-weights",
    "token-embedding",
    "output-head",
    "global-weights",
    "lazy-tables",
    "kv-cache",
    "kv-cache-unaccounted",
    "recurrent-state",
    "compute-buffer",
    "output-buffer",
    "vision-projector",
    "vision-projector-compute",
    "cuda-context",
    "process-overhead",
)
"""The order a reader should meet the components in, following section 8.1's own table."""

_SEVERITY: dict[Verdict, int] = {
    "comfortable": 0,
    "fits": 1,
    "tight": 2,
    "too-tight": 3,
    "does-not-fit": 4,
}
"""How bad each verdict is, so the worse of two pools can be picked.

``too-tight`` sits between ``tight`` and ``does-not-fit`` because a configuration that
pages does still run: badly, silently, but it runs, which is more than one that cannot be
loaded at all does.
"""


def utilisation(required: int, available: int) -> float:
    """The share of a pool a requirement takes up, with nothing dividing by zero.

    A pool with nothing in it is at zero however small it is, and a pool with something in
    it and no room at all is infinitely over -- which is exactly what asking a machine with
    no graphics card to hold weights on one amounts to.
    """
    if required <= 0:
        return 0.0
    if available <= 0:
        return inf
    return required / available


def pool_verdict(share: float, pool: Pool) -> Verdict:
    """How well one pool fits, from section 8.3's thresholds and section 8.4's paging rule.

    Args:
        share: The pool's utilisation, required over available.
        pool: Which pool, since what happens past the line differs by pool. On the
            graphics card the driver pages and the server starts anyway; in system memory
            there is nothing left to page to and the load fails.

    Returns:
        The verdict for that pool alone.
    """
    if share <= UTILISATION_COMFORTABLE:
        return "comfortable"
    if share <= UTILISATION_FITS:
        return "fits"
    if share <= UTILISATION_TIGHT:
        return "tight"
    return "too-tight" if pool == "vram" else "does-not-fit"


def _verdict(
    *,
    vram_required: int,
    vram_available: int,
    ram_required: int,
    ram_available: int,
    has_vram_pool: bool,
    mode: RunMode,
) -> Verdict:
    """The verdict for a whole budget: the worse pool, with two rules laid over it.

    A configuration whose overflow off the card is larger than what is left of system
    memory has nowhere to page to, so it does not run at all rather than running slowly.
    And a placement with everything on the processor is capped at ``fits``, because
    measured generation there is rarely comfortable however much memory is spare; a
    mixture-of-experts placement is deliberately *not* capped, since with attention and the
    cache on the card its measured speeds sit close to the all-GPU case.
    """
    worst: Verdict = pool_verdict(utilisation(ram_required, ram_available), "ram")
    if has_vram_pool:
        card = pool_verdict(utilisation(vram_required, vram_available), "vram")
        if _SEVERITY[card] > _SEVERITY[worst]:
            worst = card
        overflow = vram_required - vram_available
        if overflow > max(ram_available - ram_required, 0):
            worst = "does-not-fit"
    if mode in ("cpu", "unsupported") and _SEVERITY[worst] < _SEVERITY["fits"]:
        worst = "fits"
    return worst


def _layers_for(
    mode: RunMode, n_layer: int, gpu_layers: int | None, cpu_moe: int | None
) -> tuple[int, int]:
    """The layer counts a mode implies, when the caller has not given them explicitly."""
    if gpu_layers is None:
        if mode == "hybrid":
            raise BudgetError(
                _("A hybrid placement has to say how many layers are on the graphics card."),
                hint=_("Pass gpu_layers, which is what -ngl would be."),
            )
        gpu_layers = 0 if mode in ("cpu", "unsupported") else n_layer
    if cpu_moe is None:
        cpu_moe = n_layer if mode in ("moe-offload", "cpu", "unsupported") else 0
    return gpu_layers, cpu_moe


def compute(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    *,
    context: int,
    mode: RunMode = "gpu",
    micro_batch: int = 2048,
    batch: int | None = None,
    kv_type: str = "f16",
    gpu_layers: int | None = None,
    cpu_moe_layers: int | None = None,
    projector: Extra | None = None,
    projector_pool: Pool | None = None,
    stream_lazy_tables: bool = True,
    shared_experts_pool: Pool | None = None,
) -> Budget:
    """What one configuration of one model needs, component by component.

    Args:
        model: The catalog entry, for the architecture and the layer count.
        quant: The quantisation, whose ``gguf_facts`` carry every exact byte figure.
        host: The scanned machine, for what is free now.
        context: The context length in tokens.
        mode: How the weights are divided. It sets the layer counts when they are not
            given, and it caps a processor-only placement at ``fits``.
        micro_batch: What ``-ub`` would be, which drives the compute buffer.
        batch: What ``-b`` would be, defaulting to section 8.1's ``max(2 x ub, 2048)``.
        kv_type: The cache quantisation, ``f16``, ``q8_0`` or ``q4_0``.
        gpu_layers: What ``-ngl`` would be. Required for a hybrid placement; taken from
            the mode otherwise.
        cpu_moe_layers: What ``--n-cpu-moe`` would be. Taken from the mode when omitted.
        projector: The catalog's ``mmproj`` entry, when the model has one.
        projector_pool: Where the projector was placed, or ``None`` for vision off.
        stream_lazy_tables: Whether tensors the catalog marks streamable are read from
            disk rather than held in memory.
        shared_experts_pool: Where the always-on shared experts go, which is what an
            ``-ot ffn_.*_shexp=CPU`` override decides. ``None`` leaves them with their
            layers. It has no effect until the GGUF facts carry those bytes as a bucket of
            their own; see :data:`~llamafit.budget.weights.SHARED_EXPERT_BUCKET`.

    Returns:
        The budget: every line, both totals, both pools' utilisation and the verdict.

    Raises:
        BudgetError: If the quant has no GGUF facts, if the cache cannot be sized, or if a
            hybrid placement does not say how many layers are on the card.
    """
    facts = quant.gguf_facts
    if facts is None:
        raise BudgetError(
            _("LlamaFit has not read %(model)s's %(quant)s file, so it cannot size it.")
            % {"model": model.id, "quant": quant.name},
            hint=_("Run `llamafit catalog refresh` to read the file's header."),
        )
    n_layer = facts.n_layer or 0
    ngl, cpu_moe = _layers_for(mode, n_layer, gpu_layers, cpu_moe_layers)

    lines: list[BudgetLine] = [
        *weight_lines(
            facts,
            gpu_layers=ngl,
            cpu_moe_layers=cpu_moe,
            shared_experts_pool=shared_experts_pool,
            stream_lazy_tables=stream_lazy_tables,
        ),
        *cache_lines(facts, context=context, kv_type=kv_type, gpu_layers=ngl),
        *buffer_lines(
            micro_batch=micro_batch,
            context=context,
            n_vocab=facts.n_vocab or 0,
            batch=batch if batch is not None else batch_for(micro_batch),
            projector_on_gpu=projector_pool == "vram",
            pool="vram" if ngl > 0 else "ram",
        ),
        *projector_lines(projector, pool=projector_pool),
    ]

    # Apple Silicon has one pool with two names. Charging the same bytes to a card that is
    # the same memory would report every model as needing twice what it needs, so on a
    # unified machine every line lands in system memory and there is no card to be over.
    if host.unified_memory:
        lines = [line.model_copy(update={"pool": "ram"}) for line in lines]

    if any(line.pool == "vram" for line in lines):
        lines.append(
            BudgetLine(
                component="cuda-context",
                pool="vram",
                bytes=CUDA_CONTEXT_BYTES,
                exact=False,
                note=_("what the GPU backend costs before it allocates anything"),
            )
        )
    lines.append(
        BudgetLine(
            component="process-overhead",
            pool="ram",
            bytes=PROCESS_OVERHEAD_BYTES,
            exact=False,
            note=_("the server process before any model buffer"),
        )
    )
    lines.sort(key=lambda line: _COMPONENT_ORDER.index(line.component))

    vram_required = sum(line.bytes_ for line in lines if line.pool == "vram")
    ram_required = sum(line.bytes_ for line in lines if line.pool == "ram")

    gpu = host.primary_gpu
    free = None if gpu is None or host.unified_memory else gpu.vram_free_bytes
    vram_available = 0 if free is None else max(free - VRAM_RESERVE_BYTES, 0)
    ram_available = host.memory.available_bytes
    has_vram_pool = free is not None or vram_required > 0

    return Budget(
        lines=tuple(lines),
        vram_required=vram_required,
        ram_required=ram_required,
        vram_available=vram_available,
        ram_available=ram_available,
        vram_utilisation=utilisation(vram_required, vram_available) if has_vram_pool else None,
        ram_utilisation=utilisation(ram_required, ram_available),
        verdict=_verdict(
            vram_required=vram_required,
            vram_available=vram_available,
            ram_required=ram_required,
            ram_available=ram_available,
            has_vram_pool=has_vram_pool,
            mode=mode,
        ),
    )
