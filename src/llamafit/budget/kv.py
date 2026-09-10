# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The key-value cache and the recurrent state: what a model remembers, and where.

The cache the file describes is arithmetic on the file's own shapes -- attention layers,
key/value heads and the two head dimensions -- so those bytes are exact, and they grow
strictly with the context. The recurrent state is not: it is derived from the
architecture's state dimensions by a formula calibrated against one measurement, so it is
marked modelled even though it does not move with the context.

Some architectures allocate more cache than their header describes. Qwen4exp allocates
about 9 MiB per 1,024 tokens beyond the key and value caches its shapes account for, which
at 32K is 27 percent of that model's cache and at 128K is 1.1 GB. That memory is real
whether or not a header mentions it, so it is carried here as a line of its own,
:data:`~llamafit.constants.UNACCOUNTED_KV_CACHE_BYTES_PER_1K`, named for the architecture
that allocates it and never mixed into the derived figure. Reporting a cache smaller than
the one llama.cpp will allocate is the single way of being wrong that turns into "this
fits" when it does not.

The cache is also the one component with no fallback. A file that does not say how many
key/value heads it has cannot have its cache sized at all, and that is a
:class:`~llamafit.errors.BudgetError` rather than a zero.
"""

from __future__ import annotations

from llamafit.constants import UNACCOUNTED_KV_CACHE_BYTES_PER_1K
from llamafit.errors import BudgetError
from llamafit.gguf.facts import kv_bytes_per_token
from llamafit.i18n import _
from llamafit.models.gguf import GgufFacts
from llamafit.models.plan import BudgetLine, Pool

from .weights import split_by_layers

KV_TYPES = ("f16", "q8_0", "q4_0")
"""The cache quantisations llama.cpp's ``-ctk``/``-ctv`` accept and this module can size."""


def derived_kv_cache_bytes(facts: GgufFacts, *, context: int, kv_type: str) -> int:
    """Bytes the key and value caches the file describes occupy at ``context`` tokens.

    Args:
        facts: The quant's GGUF facts.
        context: The context length in tokens.
        kv_type: The cache quantisation, one of :data:`KV_TYPES`.

    Returns:
        Bytes the two caches occupy together, from the shapes the file declares.

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


def unaccounted_kv_cache_bytes(facts: GgufFacts, *, context: int) -> int:
    """Cache this architecture is known to allocate that its header does not describe.

    Zero for every architecture but the few measured to allocate more than their shapes
    declare, which is what makes this safe to add to every budget.

    Args:
        facts: The quant's GGUF facts, whose ``arch`` is what this is looked up by.
        context: The context length in tokens.

    Returns:
        Bytes beyond the derived cache, measured rather than derived.
    """
    per_1k = UNACCOUNTED_KV_CACHE_BYTES_PER_1K.get(facts.arch, 0)
    return per_1k * max(context, 0) // 1024


def kv_cache_bytes(facts: GgufFacts, *, context: int, kv_type: str) -> int:
    """Every byte of cache llama.cpp will allocate: the derived part and the measured part.

    This is the figure a caller who wants to know how large the cache is should ask for.
    :func:`derived_kv_cache_bytes` and :func:`unaccounted_kv_cache_bytes` are the two halves
    for a caller that needs to show them apart, which a budget does.

    Args:
        facts: The quant's GGUF facts.
        context: The context length in tokens.
        kv_type: The cache quantisation, one of :data:`KV_TYPES`.

    Returns:
        Bytes of cache at that context.

    Raises:
        BudgetError: If the derived part cannot be sized.
    """
    return derived_kv_cache_bytes(
        facts, context=context, kv_type=kv_type
    ) + unaccounted_kv_cache_bytes(facts, context=context)


def _cache_lines(
    component: str,
    total: int,
    *,
    on_gpu: int,
    n_layer: int,
    exact: bool,
    note: str | None,
) -> list[BudgetLine]:
    """One component's bytes, split across the pools its layers are in."""
    on_card, in_memory = split_by_layers(total, on_gpu, n_layer)
    split: tuple[tuple[Pool, int], ...] = (("vram", on_card), ("ram", in_memory))
    return [
        BudgetLine(component=component, pool=pool, bytes=size, exact=exact, note=note)
        for pool, size in split
        if size
    ]


def cache_lines(
    facts: GgufFacts,
    *,
    context: int,
    kv_type: str,
    gpu_layers: int,
) -> tuple[BudgetLine, ...]:
    """The key-value cache and the recurrent state, in the pool their layers are in.

    All of them follow their layers, so all are prorated by the share of blocks on the
    card. The proration is by block count rather than by which blocks actually hold a
    cache: a file says how many of its layers have full attention but not which, and on a
    hybrid placement that is the difference between two figures nobody has measured.

    Args:
        facts: The quant's GGUF facts.
        context: The context length in tokens.
        kv_type: The cache quantisation, one of :data:`KV_TYPES`.
        gpu_layers: How many transformer blocks are on the card.

    Returns:
        The cache the file describes; the cache this architecture is known to allocate
        beyond it, when there is any; and the recurrent state, when the architecture has
        one.

    Raises:
        BudgetError: If the cache cannot be sized; see :func:`derived_kv_cache_bytes`.
    """
    n_layer = facts.n_layer or 0
    on_gpu = min(max(gpu_layers, 0), n_layer) if n_layer else max(gpu_layers, 0)

    unaccounted = unaccounted_kv_cache_bytes(facts, context=context)
    notes = []
    if facts.attention_layers_source != "tensors":
        notes.append(_("the attention-layer count was assumed, not counted"))
    if unaccounted:
        notes.append(
            _("%(arch)s allocates more cache than its header describes; see the line below")
            % {"arch": facts.arch}
        )
    lines = _cache_lines(
        "kv-cache",
        derived_kv_cache_bytes(facts, context=context, kv_type=kv_type),
        on_gpu=on_gpu,
        n_layer=n_layer,
        exact=not notes,
        note="; ".join(notes) or None,
    )
    lines += _cache_lines(
        "kv-cache-unaccounted",
        unaccounted,
        on_gpu=on_gpu,
        n_layer=n_layer,
        exact=False,
        note=_("what %(arch)s allocates beyond its header, measured rather than derived")
        % {"arch": facts.arch},
    )
    lines += _cache_lines(
        "recurrent-state",
        facts.recurrent_state_bytes or 0,
        on_gpu=on_gpu,
        n_layer=n_layer,
        exact=False,
        note=_("from the architecture's state dimensions; it does not grow with context"),
    )
    return tuple(lines)
