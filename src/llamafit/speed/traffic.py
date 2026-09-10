"""How many bytes a placement reads to produce one token, and out of which pool.

Generation with llama.cpp is a memory-traffic problem: every weight a token touches has to
be read once, from wherever the placement put it, and the arithmetic on top of those bytes
is small enough to hide behind them. This module turns a :class:`~llamafit.models.plan.Placement`
and a file's :class:`~llamafit.models.gguf.GgufFacts` into that traffic, itemised.

The itemisation matters because system memory does not have one speed. A read is filed
under one of three access patterns:

``device``
    Bytes read by the graphics card.
``sequential``
    Bytes read from system memory in long contiguous runs -- dense block weights left on
    the CPU, an output head that did not fit on the card, a KV cache in RAM.
``scattered``
    Routed expert weights read from system memory. Each token picks a different handful of
    experts out of hundreds, so a layer's read is a scatter of small blocks across tens of
    gigabytes and the memory system never gets to stream. This is the term that decides
    whether a mixture-of-experts recommendation is right or twice as optimistic.

What is deliberately *not* counted: the token embedding table, because a token reads one
row of it; the vision projector, because it runs on images rather than on every token; and
the recurrent state of a hybrid architecture, which section 10.1's formula omits and which
is under one millisecond a token on both reference models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llamafit.constants import KV_TYPE_BITS
from llamafit.models.gguf import GgufFacts
from llamafit.models.plan import Placement, Pool

Access = Literal["device", "sequential", "scattered"]
"""How a read reaches memory, which is what decides the bandwidth it gets."""


@dataclass(frozen=True)
class TrafficLine:
    """One component's contribution to the bytes read per generated token.

    Attributes:
        component: What this is, for example ``routed-experts`` or ``kv-cache``.
        pool: Where the bytes are read from.
        bytes_: How many bytes are read per token.
        access: The access pattern, which selects the effective bandwidth.
    """

    component: str
    pool: Pool
    bytes_: int
    access: Access


@dataclass(frozen=True)
class TokenTraffic:
    """Everything one generated token reads, split by the bandwidth it gets.

    Attributes:
        lines: Every component, in the order a reader should meet them.
        device_bytes: Bytes read by the graphics card.
        sequential_bytes: Bytes read from system memory in contiguous runs.
        scattered_bytes: Routed expert bytes read from system memory.
    """

    lines: tuple[TrafficLine, ...]
    device_bytes: int
    sequential_bytes: int
    scattered_bytes: int

    @property
    def total_bytes(self) -> int:
        """Every byte the token reads, wherever it comes from."""
        return self.device_bytes + self.sequential_bytes + self.scattered_bytes


def kv_bytes_per_token(facts: GgufFacts, kv_type: str) -> int:
    """Bytes of KV cache one token occupies at ``kv_type``.

    A GGUF file's facts report the cache at f16; every other type is that figure scaled by
    its bits per element (:data:`~llamafit.constants.KV_TYPE_BITS`).

    Args:
        facts: The file's derived facts.
        kv_type: The cache type, for example ``f16`` or ``q8_0``.

    Returns:
        Bytes per token, or 0 when the file reports no cache size.
    """
    at_f16 = facts.kv_bytes_per_token_f16
    if not at_f16:
        return 0
    bits = KV_TYPE_BITS.get(kv_type.lower(), KV_TYPE_BITS["f16"])
    return round(at_f16 * bits / KV_TYPE_BITS["f16"])


def _fraction(part: int | None, whole: int | None) -> float:
    """``part / whole`` clamped to 0..1, and 0.0 when either figure is missing."""
    if not whole or part is None:
        return 0.0
    return min(max(part / whole, 0.0), 1.0)


def active_expert_bytes(facts: GgufFacts) -> int:
    """Bytes of routed expert weights one token activates.

    ``bytes_expert_weights x n_expert_used / n_expert``, per section 10.1, and 0 for a
    model with no routed experts.
    """
    if not facts.n_expert or not facts.n_expert_used:
        return 0
    return round(facts.bytes_expert_weights * facts.n_expert_used / facts.n_expert)


def per_token_traffic(
    placement: Placement, facts: GgufFacts, *, working_context: int
) -> TokenTraffic:
    """Work out what one generated token reads under ``placement``.

    Layers are split by proportion rather than by identity: the facts say how many layers
    a model has and the placement says how many the graphics card holds, but neither says
    *which* ones, so a component that spans the layers is prorated. Routed experts are in
    system memory when either their layer was not offloaded at all or it was offloaded
    with its experts pinned to the CPU, which is the larger of the two fractions.

    Args:
        placement: Where the bytes go and at what settings.
        facts: The file's derived facts.
        working_context: Tokens of KV cache the attention reads per token.

    Returns:
        The traffic, itemised and totalled by access pattern.
    """
    n_layer = facts.n_layer or 0
    on_gpu = _fraction(placement.gpu_layers, n_layer) if n_layer else 0.0
    experts_on_cpu = max(_fraction(placement.cpu_moe_layers, n_layer), 1.0 - on_gpu)
    lines: list[TrafficLine] = []

    def add(component: str, pool: Pool, byte_count: float, access: Access) -> None:
        if byte_count >= 1:
            lines.append(TrafficLine(component, pool, round(byte_count), access))

    dense = facts.bytes_dense_block_weights
    add("block-weights", "vram", dense * on_gpu, "device")
    add("block-weights", "ram", dense * (1.0 - on_gpu), "sequential")

    experts = active_expert_bytes(facts)
    add("routed-experts", "ram", experts * experts_on_cpu, "scattered")
    add("routed-experts", "vram", experts * (1.0 - experts_on_cpu), "device")

    # llama.cpp keeps the output head on the card only when every layer is already there.
    head_on_gpu = on_gpu >= 1.0
    add(
        "output-head",
        "vram" if head_on_gpu else "ram",
        facts.bytes_output_head,
        "device" if head_on_gpu else "sequential",
    )

    kv = kv_bytes_per_token(facts, placement.kv_type) * max(working_context, 0)
    add("kv-cache", "vram", kv * on_gpu, "device")
    add("kv-cache", "ram", kv * (1.0 - on_gpu), "sequential")

    by_access = {"device": 0, "sequential": 0, "scattered": 0}
    for line in lines:
        by_access[line.access] += line.bytes_
    return TokenTraffic(
        lines=tuple(lines),
        device_bytes=by_access["device"],
        sequential_bytes=by_access["sequential"],
        scattered_bytes=by_access["scattered"],
    )


def expert_bytes_in_ram(placement: Placement, facts: GgufFacts) -> int:
    """Bytes of routed expert weights this placement leaves in system memory.

    Not the active subset: prompt processing touches nearly every expert once per
    micro-batch, so this is the whole set that lives off the card.
    """
    n_layer = facts.n_layer or 0
    on_gpu = _fraction(placement.gpu_layers, n_layer) if n_layer else 0.0
    on_cpu = max(_fraction(placement.cpu_moe_layers, n_layer), 1.0 - on_gpu)
    return round(facts.bytes_expert_weights * on_cpu)


def streamed_expert_fraction(facts: GgufFacts, micro_batch: int) -> float:
    """Fraction of the expert set a micro-batch of ``micro_batch`` tokens touches.

    An expert is read once if any token in the micro-batch routes to it, so the fraction
    is ``1 - (1 - used/total) ** micro_batch``. Section 10.2 says this is "nearly all of
    them" for a micro-batch of 256 or more, and it is: at 10 experts of 512 it reaches
    99.4 percent by 256 tokens. Below that the curve matters, which is why it is written
    out rather than pinned at one.
    """
    if not facts.n_expert or not facts.n_expert_used or micro_batch <= 0:
        return 0.0
    missed = 1.0 - min(facts.n_expert_used / facts.n_expert, 1.0)
    return 1.0 - missed**micro_batch
