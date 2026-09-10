"""Where a model's weight tensors land, pool by pool.

Every figure this module produces is a sum of tensor sizes the GGUF header already
reports, prorated by whole layers, so every line it returns is marked exact. What is
*not* exact is the pool each line lands in: which tensors llama.cpp puts on the card for
a given ``-ngl`` is a property of llama.cpp, not of the file, and the rules below are
written from what it was observed to do rather than from what it promises. Two of them
are worth stating out loud because they are surprising.

The input embedding table stays in system memory even at full offload. llama.cpp keeps it
in a CPU buffer and gathers rows there, which is why a fully offloaded model still prints
a CPU buffer size. The calibration record settles it arithmetically for
Qwen3.8-Flash-Next: its block weights and output head come to 4,607 MiB and llama.cpp
reported 4,606, leaving no room on the card for the 644 MiB embedding table. The exception
is a model with tied embeddings, where the output projection *is* the embedding table;
there it has to follow the output layer, and a file says so by carrying no separate
``output.weight`` at all.

``--n-cpu-moe N`` is treated as claiming layers out of the set that is on the card. That
is exactly true in the one placement that combines the two flags, ``-ngl 99 --n-cpu-moe
N``, where every layer is offloaded and the flag takes N of them back; it is an
approximation only for a hybrid placement that also offloads experts, which the planner
does not currently produce.
"""

from __future__ import annotations

from llamafit.i18n import _
from llamafit.models.gguf import GgufFacts
from llamafit.models.plan import BudgetLine, Pool


def split_by_layers(total: int, layers_on_gpu: int, n_layer: int) -> tuple[int, int]:
    """Split a per-block byte total into the part on the card and the part in memory.

    The remainder always goes to system memory rather than being distributed, so the two
    halves add up to ``total`` exactly however the division falls. A budget whose parts do
    not add up to the file is not a budget.

    Args:
        total: Bytes of the tensors being split.
        layers_on_gpu: How many blocks are on the graphics card.
        n_layer: How many blocks the model has.

    Returns:
        The bytes on the card and the bytes in system memory, in that order.
    """
    if n_layer <= 0:
        return (total, 0) if layers_on_gpu > 0 else (0, total)
    on_gpu = total * min(max(layers_on_gpu, 0), n_layer) // n_layer
    return on_gpu, total - on_gpu


def _line(component: str, pool: Pool, size: int, note: str | None = None) -> BudgetLine:
    """One exact weight line, since every figure in this module comes from the header."""
    return BudgetLine(component=component, pool=pool, bytes=size, exact=True, note=note)


def weight_lines(
    facts: GgufFacts,
    *,
    gpu_layers: int,
    cpu_moe_layers: int = 0,
    stream_lazy_tables: bool = True,
) -> tuple[BudgetLine, ...]:
    """Every weight tensor of a model, in the pool llama.cpp would put it in.

    Args:
        facts: The quant's GGUF facts, whose byte buckets account for the whole file.
        gpu_layers: How many transformer blocks are on the card, i.e. what ``-ngl``
            resolves to. Values above the model's block count mean all of them.
        cpu_moe_layers: How many blocks have their routed experts in system memory, i.e.
            what ``--n-cpu-moe`` says. Zero leaves every expert with its block.
        stream_lazy_tables: Whether the tensors the catalog marks streamable are read from
            disk as they are needed rather than held in memory. They are marked precisely
            because llama.cpp can stream them, so this defaults to true; passing false
            charges them to system memory instead.

    Returns:
        One line per bucket and pool, skipping the buckets this model has none of.
    """
    n_layer = facts.n_layer or 0
    on_gpu = min(max(gpu_layers, 0), n_layer) if n_layer else max(gpu_layers, 0)
    whole_model_on_gpu = on_gpu >= n_layer if n_layer else gpu_layers > 0
    tail_pool: Pool = "vram" if whole_model_on_gpu else "ram"

    lines: list[BudgetLine] = []

    dense_vram, dense_ram = split_by_layers(facts.bytes_dense_block_weights, on_gpu, n_layer)
    dense_note = _("attention projections, feed-forward weights, norms and any shared experts")
    if dense_vram:
        lines.append(_line("dense-weights", "vram", dense_vram, dense_note))
    if dense_ram:
        lines.append(_line("dense-weights", "ram", dense_ram, dense_note))

    expert_layers_on_gpu = max(on_gpu - min(max(cpu_moe_layers, 0), n_layer), 0)
    expert_vram, expert_ram = split_by_layers(
        facts.bytes_expert_weights, expert_layers_on_gpu, n_layer
    )
    if expert_vram:
        lines.append(_line("expert-weights", "vram", expert_vram))
    if expert_ram:
        lines.append(
            _line(
                "expert-weights",
                "ram",
                expert_ram,
                _("routed experts, read across the bus once per micro-batch"),
            )
        )

    if facts.bytes_token_embd:
        tied = facts.bytes_output_head == 0
        lines.append(
            _line(
                "token-embedding",
                tail_pool if tied else "ram",
                facts.bytes_token_embd,
                _("this model ties its output projection to its embedding table")
                if tied
                else _("llama.cpp keeps the embedding table in memory even at full offload"),
            )
        )
    if facts.bytes_output_head:
        lines.append(_line("output-head", tail_pool, facts.bytes_output_head))
    if facts.bytes_global_weights:
        lines.append(_line("global-weights", tail_pool, facts.bytes_global_weights))

    if facts.bytes_lazy_tables:
        streamed = stream_lazy_tables
        lines.append(
            _line(
                "lazy-tables",
                "disk" if streamed else "ram",
                facts.bytes_lazy_tables,
                _("streamed from disk as it is needed, not held in memory")
                if streamed
                else _("held in memory, since streaming from disk was turned off"),
            )
        )

    return tuple(lines)
