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


def _recurrent_state_bytes(
    header: GgufHeader, arch: str, n_layer: int | None, attention_layers: int | None
) -> int | None:
    """Bytes of fixed recurrent/SSM state, when the shape and layer split are known."""
    state_size = _int(header, f"{arch}.ssm.state_size")
    inner_size = _int(header, f"{arch}.ssm.inner_size")
    if state_size is None or inner_size is None or n_layer is None or attention_layers is None:
        return None
    return state_size * inner_size * (n_layer - attention_layers)


def derive_facts(header: GgufHeader, *, lazy_tensor_names: Sequence[str] = ()) -> GgufFacts:
    """Derive architecture facts and a byte-size breakdown from a parsed header."""
    arch = str(header.metadata.get("general.architecture", "unknown"))

    n_layer = _int(header, f"{arch}.block_count")
    n_embd = _int(header, f"{arch}.embedding_length")
    n_head = _int(header, f"{arch}.attention.head_count")
    n_head_kv = _int(header, f"{arch}.attention.head_count_kv", f"{arch}.attention.head_count")
    n_expert = _int(header, f"{arch}.expert_count")
    n_expert_used = _int(header, f"{arch}.expert_used_count")

    head_dim = _int(header, f"{arch}.attention.key_length")
    if head_dim is None and n_embd is not None and n_head:
        head_dim = n_embd // n_head

    tokens = header.metadata.get("tokenizer.ggml.tokens")
    n_vocab = len(tokens) if isinstance(tokens, list) else _int(header, f"{arch}.vocab_size")

    bytes_total = 0
    bytes_token_embd = 0
    bytes_output_head = 0
    bytes_lazy_tables = 0
    bytes_expert_weights = 0
    bytes_attention_weights = 0
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
            bytes_attention_weights += tensor.bytes_

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
        attention_layers=attention_layers,
        attention_layers_source=attention_layers_source,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        has_shared_experts=has_shared_experts,
        bytes_expert_weights=bytes_expert_weights,
        bytes_attention_weights=bytes_attention_weights,
        bytes_output_head=bytes_output_head,
        bytes_token_embd=bytes_token_embd,
        bytes_lazy_tables=bytes_lazy_tables,
        bytes_total=bytes_total,
        recurrent_state_bytes=_recurrent_state_bytes(header, arch, n_layer, attention_layers),
    )
    facts.kv_bytes_per_token_f16 = kv_bytes_per_token(facts, "f16")
    return facts


def kv_bytes_per_token(facts: GgufFacts, kv_type: str) -> int | None:
    """Bytes both KV caches grow by per token, or ``None`` when the shape is unknown."""
    if facts.attention_layers is None or facts.n_head_kv is None or facts.head_dim is None:
        return None
    block_elements, block_bytes = _KV_TYPE_BYTES[kv_type]
    elements = 2 * facts.attention_layers * facts.n_head_kv * facts.head_dim
    return elements // block_elements * block_bytes
