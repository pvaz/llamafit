# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Turn a parsed GGUF header into the facts a memory budget needs.

Every number here is read from the file, not estimated: tensor sizes come from the
dimensions and type in the tensor table, and the layer counts come from the keys the
architecture writes. The one derivation that is not a direct read is the number of
full-attention layers, which is counted from the tensors that only such a layer owns.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from llamafit.models.gguf import GgufFacts, GgufHeader

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")
_ATTENTION_MARKERS = ("attn_k.weight", "attn_v.weight")
_KV_TYPE_BYTES: dict[str, tuple[int, int]] = {"f16": (1, 2), "q8_0": (32, 34), "q4_0": (32, 18)}
_AttentionSource = Literal["tensors", "all-layers", "unknown"]


def _int(header: GgufHeader, *keys: str) -> int | None:
    """The first key present in the metadata whose value is an integer."""
    for key in keys:
        value = header.metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


_F32_BYTES = 4


def _recurrent_state_bytes(
    header: GgufHeader, arch: str, n_layer: int | None, attention_layers: int | None
) -> int | None:
    """Bytes of fixed recurrent/SSM state, when the shape and layer split are known.

    llama.cpp keeps recurrent state in F32 regardless of the model's own
    quantisation, and its layout follows the Mamba2/gated-DeltaNet state cache: a
    convolution state over the concatenated ``(x, B, C)`` vector, of width
    ``inner_size + 2 * group_count * state_size``, held for ``conv_kernel - 1``
    steps, plus an SSM state of ``state_size * inner_size`` elements, once per
    recurrent (non-attention) layer. ``group_count`` defaults to 0 when the
    architecture does not report one, which correctly drops the ``B``/``C`` term for
    a plain Mamba1-style layer whose convolution only covers ``x``.

    Calibrated against a real measurement recorded in
    ``docs/calibration/2026-09-09-reference-machine.md``: llama.cpp reported a 112.57
    MiB (118,036,480-byte) recurrent buffer for Qwen3.8-Flash-Next (48 layers, 12 of
    them full-attention, so 36 recurrent), which this formula reproduces within 0.4
    percent.
    """
    state_size = _int(header, f"{arch}.ssm.state_size")
    inner_size = _int(header, f"{arch}.ssm.inner_size")
    conv_kernel = _int(header, f"{arch}.ssm.conv_kernel")
    if (
        state_size is None
        or inner_size is None
        or conv_kernel is None
        or n_layer is None
        or attention_layers is None
    ):
        return None
    group_count = _int(header, f"{arch}.ssm.group_count")
    if group_count is None:
        group_count = 0
    recurrent_layers = n_layer - attention_layers
    conv_state_elements = (conv_kernel - 1) * (inner_size + 2 * group_count * state_size)
    ssm_state_elements = state_size * inner_size
    return _F32_BYTES * recurrent_layers * (conv_state_elements + ssm_state_elements)


def derive_facts(header: GgufHeader, *, lazy_tensor_names: Sequence[str] = ()) -> GgufFacts:
    """Derive architecture facts and a byte-size breakdown from a parsed header."""
    arch = str(header.metadata.get("general.architecture", "unknown"))

    n_layer = _int(header, f"{arch}.block_count")
    n_embd = _int(header, f"{arch}.embedding_length")
    n_head = _int(header, f"{arch}.attention.head_count")
    n_head_kv = _int(header, f"{arch}.attention.head_count_kv", f"{arch}.attention.head_count")
    sliding_window = _int(header, f"{arch}.attention.sliding_window")
    context_length = _int(header, f"{arch}.context_length")
    n_expert = _int(header, f"{arch}.expert_count")
    n_expert_used = _int(header, f"{arch}.expert_used_count")

    head_dim = _int(header, f"{arch}.attention.key_length")
    if head_dim is None and n_embd is not None and n_head:
        head_dim = n_embd // n_head
    # The value vector is sized separately, because nothing says it matches the key.
    value_head_dim = _int(header, f"{arch}.attention.value_length")
    if value_head_dim is None:
        value_head_dim = head_dim

    tokens = header.metadata.get("tokenizer.ggml.tokens")
    n_vocab = len(tokens) if isinstance(tokens, list) else _int(header, f"{arch}.vocab_size")

    bytes_total = 0
    bytes_token_embd = 0
    bytes_output_head = 0
    bytes_lazy_tables = 0
    bytes_expert_weights = 0
    bytes_dense_block_weights = 0
    bytes_global_weights = 0
    has_shared_experts = False
    full_attention_blocks: set[int] = set()

    for tensor in header.tensors:
        bytes_total += tensor.bytes_
        if any(tensor.name.startswith(prefix) for prefix in lazy_tensor_names):
            bytes_lazy_tables += tensor.bytes_
        elif tensor.name == "token_embd.weight":
            bytes_token_embd += tensor.bytes_
        elif tensor.name == "output.weight":
            bytes_output_head += tensor.bytes_
        elif "_exps" in tensor.name:
            bytes_expert_weights += tensor.bytes_
        elif _BLOCK_RE.match(tensor.name):
            bytes_dense_block_weights += tensor.bytes_
        else:
            bytes_global_weights += tensor.bytes_

        if "_shexp" in tensor.name:
            has_shared_experts = True

        block_match = _BLOCK_RE.match(tensor.name)
        if block_match and tensor.name.endswith(_ATTENTION_MARKERS):
            full_attention_blocks.add(int(block_match.group(1)))

    attention_layers: int | None
    attention_layers_source: _AttentionSource
    if full_attention_blocks:
        attention_layers = len(full_attention_blocks)
        attention_layers_source = "tensors"
    elif n_layer is not None:
        attention_layers = n_layer
        attention_layers_source = "all-layers"
    else:
        attention_layers = None
        attention_layers_source = "unknown"

    facts = GgufFacts(
        arch=arch,
        n_layer=n_layer,
        n_embd=n_embd,
        n_vocab=n_vocab,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        value_head_dim=value_head_dim,
        attention_layers=attention_layers,
        attention_layers_source=attention_layers_source,
        sliding_window=sliding_window,
        context_length=context_length,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        has_shared_experts=has_shared_experts,
        bytes_expert_weights=bytes_expert_weights,
        bytes_dense_block_weights=bytes_dense_block_weights,
        bytes_output_head=bytes_output_head,
        bytes_token_embd=bytes_token_embd,
        bytes_lazy_tables=bytes_lazy_tables,
        bytes_global_weights=bytes_global_weights,
        bytes_total=bytes_total,
        recurrent_state_bytes=_recurrent_state_bytes(header, arch, n_layer, attention_layers),
    )
    facts.kv_bytes_per_token_f16 = kv_bytes_per_token(facts, "f16")
    return facts


def _cache_bytes(elements: int, block_elements: int, block_bytes: int) -> int:
    """Bytes one block-quantised cache of ``elements`` values occupies.

    Rounded up. Holding a count that does not divide evenly takes the next whole
    block, so rounding down reports a cache smaller than the one that gets allocated,
    and undersized is the dangerous direction to be wrong in: it tells somebody a
    model fits when it does not. Every cache width in the catalog today is a power of
    two that every block size divides, so the division is exact and the figure does
    not move; that is luck about those shapes, not a property of the arithmetic.
    """
    return (elements + block_elements - 1) // block_elements * block_bytes


def kv_bytes_per_token(facts: GgufFacts, kv_type: str) -> int | None:
    """Bytes both KV caches grow by per token, or ``None`` when the shape is unknown.

    Each cache is sized from its own head dimension and rounded up to a whole number
    of blocks on its own, and the two are then added. No shortcut is taken: doubling
    the key cache would be wrong in proportion for an architecture that declares a
    value length different from its key length, and rounding once over a combined
    element count would model a single tensor that does not exist — llama.cpp
    allocates the key cache and the value cache separately. Every file in the catalog
    today declares the two lengths equal and divides every block size exactly, so this
    is a no-op on today's data and correct on tomorrow's.

    Args:
        facts: The file's facts, which must carry an attention-layer count, a
            key/value head count and a head dimension for a figure to exist at all.
        kv_type: The KV cache quantisation, one of ``f16``, ``q8_0`` or ``q4_0``.

    Returns:
        Bytes the K and V caches together grow by per token, or ``None`` when the
        shape is unknown.
    """
    if facts.attention_layers is None or facts.n_head_kv is None or facts.head_dim is None:
        return None
    value_head_dim = facts.head_dim if facts.value_head_dim is None else facts.value_head_dim
    block_elements, block_bytes = _KV_TYPE_BYTES[kv_type]
    per_cache_heads = facts.attention_layers * facts.n_head_kv
    key_bytes = _cache_bytes(per_cache_heads * facts.head_dim, block_elements, block_bytes)
    value_bytes = _cache_bytes(per_cache_heads * value_head_dim, block_elements, block_bytes)
    return key_bytes + value_bytes


def bits_per_weight(file_bytes: int, lazy_table_bytes: int, total_b: float) -> float:
    """Bits per weight, counting only the bytes of the download that are weights.

    A model's files can hold a large tensor that is a lookup table rather than a
    weight — an n-gram or per-layer embedding table that llama.cpp streams from disk
    rather than holding in memory. Counting it inflates the figure until it no longer
    describes the quantization: Qwen3.8-Flash-Next UD-Q4_K_XL is 111.33 GB of which
    28.80 GB is one such table, which reads as 7.13 bits per weight over the whole
    download and 5.28 over its weights, and only the second is a four-bit quant.

    Args:
        file_bytes: Total bytes of the model's files, every shard included.
        lazy_table_bytes: How many of those bytes are lazy lookup tables, i.e.
            :attr:`~llamafit.models.gguf.GgufFacts.bytes_lazy_tables`. Zero leaves
            this exactly the ratio over the whole download.
        total_b: The model's parameter count, in billions.

    Returns:
        Bits per weight, unrounded.
    """
    return (file_bytes - lazy_table_bytes) * 8 / (total_b * 1e9)
