# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The run modes, the settings one is made of, and the ladders the search walks.

A placement is a set of knobs — a mode, a context, a micro-batch, a KV type, how many
layers are on the card, where the vision projector lives — together with a verdict on
what those knobs need. This module holds the knobs (:class:`PlacementSettings`), the seam
to the thing that costs them (:class:`BudgetFn`), and the ordered ladders section 9.2
walks.

The budget itself is deliberately not here. Sizing a configuration is section 8's job and
another module's code; the planner takes it as an injected callable so a test can hand it
a fake and so the two can be read, changed and trusted separately.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from llamafit.constants import (
    ALL_GPU_LAYERS,
    KV_TYPE_DEFAULT,
    KV_TYPE_QUANTISED,
    MIN_BATCH_TOKENS,
    MIN_CONTEXT_TOKENS,
)
from llamafit.models.catalog import CatalogModel, Extra, Quant
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Cpu, Host
from llamafit.models.plan import Budget, ContextTier, Placement, Pool, RunMode, Verdict

ACCEPTABLE_VERDICTS: frozenset[Verdict] = frozenset({"comfortable", "fits", "tight"})
"""The verdicts a placement may be recommended with (section 9.2).

Everything better than Too Tight. The two that are missing are not the same failure and
neither is recommendable: ``does-not-fit`` would refuse to load, and ``too-tight`` would
load and then silently page to system memory at a fraction of the speed.
"""

MARGIN_VERDICTS: frozenset[Verdict] = frozenset({"comfortable", "fits"})
"""Acceptable *with margin*, which is what section 9.2 asks of the vision projector.

The projector is the most expensive optional thing on a small card — 1.9 GB on the
reference machine, the equivalent of 52,000 tokens of context — so it earns its place
only when the configuration still has room afterwards. "With margin" is spelled in the
design's own vocabulary rather than as a new number: a verdict of Tight already means "a
browser or a second process can push it over", and that is not room to spare.
"""

MODE_ORDER: tuple[RunMode, ...] = ("gpu", "moe-offload", "hybrid", "cpu")
"""The modes section 9.2 evaluates, in the order it evaluates them.

The order is also a speed order, which is why it doubles as the rank used to choose
between two placements that reach the same context.
"""

_MODE_RANK: dict[RunMode, int] = {mode: len(MODE_ORDER) - i for i, mode in enumerate(MODE_ORDER)}
_MODE_RANK["unsupported"] = 0

_VERDICT_RANK: dict[Verdict, int] = {
    "comfortable": 3,
    "fits": 2,
    "tight": 1,
    "too-tight": 0,
    "does-not-fit": 0,
}

_PROJECTOR_RANK: dict[Pool | None, int] = {"vram": 2, "ram": 1, "disk": 0, None: 0}

_SHARED_EXPERT_RANK: dict[Pool | None, int] = {None: 1, "vram": 1, "ram": 0, "disk": 0}
"""Shared experts kept with their layers or sent to system memory; kept is better.

``None`` and ``vram`` rank alike because in the mode where the question arises they
mean the same thing: every layer is on the card, so leaving the shared experts with
their layers leaves them on it.
"""


@dataclass(frozen=True)
class PlacementSettings:
    """The knobs a budget is computed for: everything section 8.1 asks about a placement.

    This is the input side of :class:`~llamafit.models.plan.Placement` — the same fields
    minus the ones that are results (the budget, the tier table, the thread count, the
    notes). It exists so the planner can cost a configuration before deciding it is the
    one, and so the seam to the budget is one small value rather than a growing argument
    list.

    Attributes:
        mode: How the weights are divided.
        context: The context length in tokens.
        micro_batch: ``-ub``, which drives the compute buffer.
        batch: ``-b``, always :func:`batch_for` of the micro-batch.
        kv_type: The KV cache type, ``f16`` or a quantised one.
        gpu_layers: What ``-ngl`` would be; :data:`~llamafit.constants.ALL_GPU_LAYERS`
            when every layer goes to the card.
        cpu_moe_layers: What ``--n-cpu-moe`` would be, or ``None`` when no expert
            weights are being held in system memory.
        projector_pool: Where the vision projector goes, or ``None`` when the model has
            none or it was left out.
        shared_experts_pool: Where the always-on shared experts go, or ``None`` to leave
            them with their layers. ``ram`` is ``-ot ffn_.*_shexp=CPU``.
    """

    mode: RunMode
    context: int
    micro_batch: int
    batch: int
    kv_type: str
    gpu_layers: int
    cpu_moe_layers: int | None = None
    projector_pool: Pool | None = None
    shared_experts_pool: Pool | None = None

    def with_context(self, context: int) -> PlacementSettings:
        """Return a copy sized for another context."""
        return replace(self, context=context)

    def with_micro_batch(self, micro_batch: int) -> PlacementSettings:
        """Return a copy with another micro-batch, and the batch that goes with it."""
        return replace(self, micro_batch=micro_batch, batch=batch_for(micro_batch))

    def with_projector(self, pool: Pool | None) -> PlacementSettings:
        """Return a copy with the vision projector somewhere else."""
        return replace(self, projector_pool=pool)

    def with_shared_experts(self, pool: Pool | None) -> PlacementSettings:
        """Return a copy with the always-on shared experts somewhere else."""
        return replace(self, shared_experts_pool=pool)

    def with_layer_choice(self, value: int) -> PlacementSettings:
        """Return a copy with one rung of :func:`layer_ladder` applied.

        The rung means a different thing in each mode — ``--n-cpu-moe`` for
        ``moe-offload``, ``-ngl`` for ``hybrid`` — and nothing at all for the two modes
        that have only one configuration, so the mode decides which field it lands in.
        """
        if self.mode == "moe-offload":
            return replace(self, gpu_layers=ALL_GPU_LAYERS, cpu_moe_layers=value)
        if self.mode == "hybrid":
            return replace(self, gpu_layers=value, cpu_moe_layers=None)
        return self

    def to_placement(
        self,
        budget: Budget,
        *,
        threads: int,
        max_context_fit: int,
        tiers: Sequence[ContextTier] = (),
        notes: Sequence[str] = (),
    ) -> Placement:
        """Combine these knobs with what they cost into the model everything else reads.

        Args:
            budget: What this configuration needs.
            threads: The thread count from section 9.4.
            max_context_fit: The largest context this placement holds.
            tiers: The context ladder for a launch script to choose from.
            notes: Anything a reader should know.

        Returns:
            The finished :class:`~llamafit.models.plan.Placement`.
        """
        return Placement(
            mode=self.mode,
            context=self.context,
            micro_batch=self.micro_batch,
            batch=self.batch,
            kv_type=self.kv_type,
            gpu_layers=self.gpu_layers,
            cpu_moe_layers=self.cpu_moe_layers,
            projector_pool=self.projector_pool,
            shared_experts_pool=self.shared_experts_pool,
            threads=threads,
            budget=budget,
            max_context_fit=max_context_fit,
            tiers=tuple(tiers),
            notes=tuple(notes),
        )


class BudgetFn(Protocol):
    """Turns a model, a quant, a host and a set of placement settings into a budget.

    The one seam the planner does not own. Section 8's module supplies the real
    implementation and tests supply a fake, which is why every parameter is
    positional-only: any callable of four arguments satisfies this, whatever it happens
    to name them.
    """

    def __call__(
        self, model: CatalogModel, quant: Quant, host: Host, settings: PlacementSettings, /
    ) -> Budget:
        """Cost one configuration of one model on one machine."""
        ...


def batch_for(micro_batch: int) -> int:
    """The logical batch that goes with a micro-batch: ``max(2 x ub, 2048)`` (section 8.1)."""
    return max(2 * micro_batch, MIN_BATCH_TOKENS)


def thread_count(cpu: Cpu) -> int:
    """How many threads to run: the performance cores (section 9.4).

    Args:
        cpu: The scanned processor.

    Returns:
        The performance-core count, the physical-core count when the operating system
        does not distinguish, and never less than one.

    The smaller number is deliberate, and it looks like a bug. On a hybrid Intel part —
    the reference machine is an i9-14900KF with 8 performance and 16 efficiency cores —
    this returns 8 on a machine that advertises 24 cores and 32 threads. Efficiency cores
    are excluded because measured generation is *slower* with them: llama.cpp splits each
    layer evenly across its threads and then waits for the slowest one, so a thread
    sitting on an efficiency core holds up every thread on a performance core, once per
    layer, for every token. The reference machine measured 23.8 tokens per second at 8
    threads and 22.8 at 24; asking for every core took speed away.
    """
    return max(cpu.performance_cores or cpu.physical_cores, 1)


def is_moe(model: CatalogModel, facts: GgufFacts | None) -> bool:
    """Whether this model has routed experts that could be held in system memory.

    The file has the last word once it has been read: a header reporting experts is a
    fact, while the catalog's architecture class is a curator's label. Before a file has
    been read, the label is all there is.
    """
    if facts is not None and facts.n_expert:
        return True
    return model.architecture.class_ in ("moe", "moe-hybrid")


def layer_count(facts: GgufFacts | None) -> int | None:
    """The number of transformer blocks, or ``None`` when no file has been read."""
    return None if facts is None or not facts.n_layer else facts.n_layer


def native_context(model: CatalogModel) -> int:
    """The longest context the planner will size for.

    The vendor's native length, not the extended one. Reaching the extended length needs
    the rope-scaling flags that go with ``context.extended_method``, and section 9.5's
    flag list does not render them, so planning for a context the rendered command line
    could not actually reach would be planning a configuration nobody can launch.
    """
    return model.context.native


def has_gpu(host: Host) -> bool:
    """Whether there is a card with free memory to place anything on."""
    gpu = host.primary_gpu
    return gpu is not None and (gpu.vram_free_bytes or 0) > 0


def available_modes(model: CatalogModel, quant: Quant, host: Host) -> tuple[RunMode, ...]:
    """The modes worth evaluating for this candidate on this host, in search order.

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        host: The scanned machine.

    Returns:
        A subset of :data:`MODE_ORDER`. Without a graphics card only ``cpu`` survives;
        ``moe-offload`` needs routed experts to move; ``hybrid`` needs a layer count,
        because ``-ngl n`` is a number of layers and a number nobody has read out of the
        file cannot be chosen.
    """
    facts = quant.gguf_facts
    layers = layer_count(facts)
    modes: list[RunMode] = []
    if has_gpu(host):
        modes.append("gpu")
        if is_moe(model, facts):
            modes.append("moe-offload")
        if layers is not None and layers > 1:
            modes.append("hybrid")
    modes.append("cpu")
    return tuple(modes)


def context_ladder(requested: int, minimum: int) -> tuple[int, ...]:
    """The contexts to try, largest first: the request, then halves (section 9.2).

    Args:
        requested: What to size for.
        minimum: The smallest context worth having.

    Returns:
        The requested context and each halving of it down to ``minimum``, which is itself
        never allowed below :data:`~llamafit.constants.MIN_CONTEXT_TOKENS` — except by an
        explicit request for less, which is a user saying they know what they want. Never
        empty, so the planner always has something to cost.
    """
    floor = min(max(minimum, MIN_CONTEXT_TOKENS), requested)
    contexts: list[int] = []
    context = requested
    while context >= floor:
        contexts.append(context)
        context //= 2
    return tuple(contexts) or (floor,)


def kv_ladder(model: CatalogModel, *, allow_kv_quant: bool) -> tuple[str, ...]:
    """The KV cache types to try, unquantised first (section 9.2).

    Args:
        model: The catalog entry, whose ``llama_cpp.kv_types_allowed`` has the last word.
        allow_kv_quant: Whether the user will accept a quantised cache at all.

    Returns:
        ``f16`` alone, or ``f16`` and then ``q8_0``. An empty ``kv_types_allowed`` means
        the curator recorded nothing, not that nothing works: llama.cpp quantises the
        cache of most architectures happily, and the list exists for the ones that assert
        on it, which name what they do accept.
    """
    allowed = model.llama_cpp.kv_types_allowed
    if not allow_kv_quant:
        return (KV_TYPE_DEFAULT,)
    if allowed and KV_TYPE_QUANTISED not in allowed:
        return (KV_TYPE_DEFAULT,)
    return (KV_TYPE_DEFAULT, KV_TYPE_QUANTISED)


def projector_of(model: CatalogModel, quant: Quant) -> Extra | None:
    """The vision projector published beside this quant, when there is one.

    Looked up in the source that publishes the quant first, and only then anywhere in the
    entry: a model published twice over can carry two projectors, and the one that goes
    with these weights is the one from the same repository.
    """
    for source in model.sources:
        if any(published.name == quant.name for published in source.quants):
            for extra in source.extras:
                if extra.role == "mmproj":
                    return extra
    for source in model.sources:
        for extra in source.extras:
            if extra.role == "mmproj":
                return extra
    return None


def projector_ladder(model: CatalogModel, quant: Quant, *, vision: bool) -> tuple[Pool | None, ...]:
    """Where to try putting the vision projector, best first (section 9.2).

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        vision: Whether the user wants vision at all.

    Returns:
        ``(None,)`` when there is no projector or the user asked for none; otherwise the
        card, then system memory, then leaving it out altogether.
    """
    if not vision or projector_of(model, quant) is None:
        return (None,)
    return ("vram", "ram", None)


def shared_expert_ladder(mode: RunMode, facts: GgufFacts | None) -> tuple[Pool | None, ...]:
    """Where to try putting the always-on shared experts, best first (section 9.2).

    Args:
        mode: The run mode. Only ``moe-offload`` asks the question: it is the mode that
            already holds the routed experts in system memory, and the shared experts are
            what is left on the card that could follow them without the attention path
            going too.
        facts: The quant's GGUF facts, which say how many bytes there are to move.

    Returns:
        ``(None,)`` when there is nothing to move or no mode that would move it; otherwise
        keeping them with their layers, and then sending them to system memory.

    A shared expert runs for every token, so moving it off the card costs generation
    speed on every one; it is tried second and never preferred. What makes it worth trying
    at all is that it is a *whole* bucket, movable on its own with one tensor override,
    and on the reference machine those 239 MiB are the difference between the best
    measured configuration being offered and being withheld.
    """
    if mode != "moe-offload" or facts is None or not facts.bytes_shared_expert_weights:
        return (None,)
    return (None, "ram")


def layer_ladder(mode: RunMode, layers: int | None) -> tuple[int, ...]:
    """The mode's own parameter, in the order section 9.2 tries it.

    Args:
        mode: The run mode.
        layers: The model's block count, or ``None`` when no file has been read.

    Returns:
        For ``moe-offload``, the ``--n-cpu-moe`` values from every layer downwards, since
        the search starts with all routed experts in system memory and moves them back
        while VRAM allows. For ``hybrid``, the ``-ngl`` values from the largest that is
        not the whole model downwards. For ``gpu`` and ``cpu``, the single value that
        mode means.
    """
    if mode == "gpu":
        return (ALL_GPU_LAYERS,)
    if mode == "cpu":
        return (0,)
    if mode == "moe-offload":
        if layers is None:
            # Without a block count, "all of them" is still expressible: llama.cpp reads
            # any --n-cpu-moe at or above the layer count as every layer. Stepping down
            # from it is not, so the mode offers its one configuration and no more.
            return (ALL_GPU_LAYERS,)
        return tuple(range(layers, 0, -1))
    if layers is None:
        return ()
    return tuple(range(layers - 1, 0, -1))


def initial_settings(
    mode: RunMode, *, context: int, micro_batch: int, kv_type: str, projector_pool: Pool | None
) -> PlacementSettings:
    """One point of the search, with the mode's own defaults filled in."""
    return PlacementSettings(
        mode=mode,
        context=context,
        micro_batch=micro_batch,
        batch=batch_for(micro_batch),
        kv_type=kv_type,
        gpu_layers=ALL_GPU_LAYERS if mode in ("gpu", "moe-offload") else 0,
        cpu_moe_layers=None,
        projector_pool=projector_pool,
        shared_experts_pool=None,
    )


def is_acceptable(verdict: Verdict) -> bool:
    """Whether a verdict may be recommended at all (section 9.2)."""
    return verdict in ACCEPTABLE_VERDICTS


def has_margin(verdict: Verdict) -> bool:
    """Whether a verdict leaves room to spare, which the vision projector has to earn."""
    return verdict in MARGIN_VERDICTS


def rank(settings: PlacementSettings, budget: Budget, *, requested_context: int) -> tuple[int, ...]:
    """How good a placement is, as a tuple compared left to right; bigger is better.

    Args:
        settings: The configuration being judged.
        budget: What it needs.
        requested_context: What the user asked to be sized for.

    Returns:
        A sort key.

    The order of the parts is the whole argument of section 9.2's "keeps the first
    verdict better than Too Tight, continuing to find the best one when several fit", and
    the search's pruning is only sound because it matches this order exactly:

    1. **The mode**, which is section 9.2's own order and the standing proxy for speed.
       It comes first, and that is the single most consequential decision in this file.
       Section 10.1 puts the difference between reading a weight from the card and
       reading it over the memory bus at roughly eightfold on the reference machine, so a
       longer context bought by dropping a mode is a bad trade almost every time it is
       offered: taken the other way round, a 125-billion-parameter model would be
       recommended on the processor alone, at about a token a second, because that is
       where its whole context fits. What keeps this from running away is the floor under
       the context ladder — :data:`~llamafit.constants.MIN_CONTEXT_TOKENS`, or the user's
       own ``min_context`` — which no mode is allowed to go below. Speed only breaks ties
       between configurations that are all usable.
    2. **The context the user asked to be sized for**, capped at the request. Capped, so
       a mode that could hold more does not win on a length nobody wanted; below the
       request, more context is still better.
    3. **An unquantised KV cache**, which costs no quality.
    4. **The micro-batch**, which is prompt-processing speed.
    5. **The vision projector**: on the card, then in system memory, then absent.
    6. **The shared experts kept with their layers**, since they run for every token and
       reading them over the memory bus costs speed on all of them. Below the projector
       because vision is something the user asked for and this is only speed: give up the
       239 MiB before giving up a capability.
    7. **Layers on the card**, and fewer experts in system memory, which is generation
       speed within one mode. This is the one part the search cannot reach by taking the
       first acceptable answer, because ``moe-offload`` starts from the configuration
       with *every* expert in system memory and improves on it from there — which is
       exactly what "continuing to find the best one" is asking for.
    8. **The verdict**, last among the things that differ, because parts 3 to 7 are each
       decided by a ladder that stops at its first acceptable rung; a verdict ranked
       above them would contradict those ladders rather than refine them.
    9. **Context again**, uncapped, to separate two placements that both reached the
       request.
    """
    return (
        _MODE_RANK[settings.mode],
        min(settings.context, requested_context),
        int(settings.kv_type == KV_TYPE_DEFAULT),
        settings.micro_batch,
        _PROJECTOR_RANK[settings.projector_pool],
        _SHARED_EXPERT_RANK[settings.shared_experts_pool],
        settings.gpu_layers - (settings.cpu_moe_layers or 0),
        _VERDICT_RANK[budget.verdict],
        settings.context,
    )
