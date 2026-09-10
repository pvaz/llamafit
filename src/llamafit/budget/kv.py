"""The key-value cache and the recurrent state: what a model remembers, and where.

The cache is arithmetic on the file's own shapes -- attention layers, key/value heads and
the two head dimensions -- so its bytes are exact for the tensors llama.cpp allocates for
it, and it grows strictly with the context. The recurrent state is not: it is derived from
the architecture's state dimensions by a formula that was calibrated against one
measurement, so it is marked modelled even though it does not move with the context.

The cache is the one line with no fallback. A file that does not say how many key/value
heads it has cannot have its cache sized, and a budget that left the line out would report
a model fitting a card that it would page off by a gigabyte or more. That is a
:class:`~llamafit.errors.BudgetError`, not a zero.
"""

from __future__ import annotations

from llamafit.errors import BudgetError
from llamafit.gguf.facts import kv_bytes_per_token
from llamafit.i18n import _
from llamafit.models.gguf import GgufFacts
from llamafit.models.plan import BudgetLine, Pool

from .weights import split_by_layers

KV_TYPES = ("f16", "q8_0", "q4_0")
"""The cache quantisations llama.cpp's ``-ctk``/``-ctv`` accept and this module can size."""


def kv_cache_bytes(facts: GgufFacts, *, context: int, kv_type: str) -> int:
    """Bytes both caches occupy at ``context`` tokens.

    Args:
        facts: The quant's GGUF facts.
        context: The context length in tokens.
        kv_type: The cache quantisation, one of :data:`KV_TYPES`.

    Returns:
        Bytes the key and value caches occupy together.

    Raises:
        BudgetError: If the cache type is not one this build can size, or if the file does
            not carry the attention shape the size is computed from.
    """
    if kv_type not in KV_TYPES:
        raise BudgetError(
            _("LlamaFit cannot size a %(kv_type)s key-value cache.") % {"kv_type": kv_type},
            hint=_("Use one of: %(types)s.") % {"types": ", ".join(KV_TYPES)},
        )
    per_token = kv_bytes_per_token(facts, kv_type)
    if per_token is None:
        raise BudgetError(
            _("This model's file does not say how large its key-value cache is per token."),
            hint=_("Refresh the catalog so the file's header is read again."),
        )
    return per_token * max(context, 0)


def cache_lines(
    facts: GgufFacts,
    *,
    context: int,
    kv_type: str,
    gpu_layers: int,
) -> tuple[BudgetLine, ...]:
    """The key-value cache and the recurrent state, in the pool their layers are in.

    Both follow their layers, so both are prorated by the share of blocks on the card. The
    proration is by block count rather than by which blocks actually hold a cache: a file
    says how many of its layers have full attention but not which, and on a hybrid
    placement that is the difference between two figures nobody has measured.

    Args:
        facts: The quant's GGUF facts.
        context: The context length in tokens.
        kv_type: The cache quantisation, one of :data:`KV_TYPES`.
        gpu_layers: How many transformer blocks are on the card.

    Returns:
        The cache line, and the recurrent-state line for an architecture that has one.

    Raises:
        BudgetError: If the cache cannot be sized; see :func:`kv_cache_bytes`.
    """
    n_layer = facts.n_layer or 0
    on_gpu = min(max(gpu_layers, 0), n_layer) if n_layer else max(gpu_layers, 0)

    lines: list[BudgetLine] = []
    counted = facts.attention_layers_source == "tensors"
    note = None if counted else _("the attention-layer count was assumed, not counted")
    cache_vram, cache_ram = split_by_layers(
        kv_cache_bytes(facts, context=context, kv_type=kv_type), on_gpu, n_layer
    )
    cache_split: tuple[tuple[Pool, int], ...] = (("vram", cache_vram), ("ram", cache_ram))
    for pool, size in cache_split:
        if size:
            lines.append(
                BudgetLine(component="kv-cache", pool=pool, bytes=size, exact=counted, note=note)
            )

    state_vram, state_ram = split_by_layers(facts.recurrent_state_bytes or 0, on_gpu, n_layer)
    state_note = _("from the architecture's state dimensions; it does not grow with context")
    state_split: tuple[tuple[Pool, int], ...] = (("vram", state_vram), ("ram", state_ram))
    for pool, size in state_split:
        if size:
            lines.append(
                BudgetLine(
                    component="recurrent-state",
                    pool=pool,
                    bytes=size,
                    exact=False,
                    note=state_note,
                )
            )

    return tuple(lines)
