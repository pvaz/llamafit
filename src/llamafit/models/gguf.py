"""Data models for a GGUF file's header and the facts derived from it.

``GgufHeader`` mirrors what the GGUF reader parses directly from a file's byte
layout: its version, tensor and metadata counts, the metadata map itself, and
one :class:`TensorInfo` per tensor. ``GgufFacts`` is the smaller,
architecture-aware summary derived from a header — the numbers sizing and
placement decisions actually need.

All three models forbid unknown fields, so a typo in test or catalog data is
caught immediately instead of being silently ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    """Base for models that reject unknown fields and accept field names or aliases."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TensorInfo(_Strict):
    """One tensor's entry in a GGUF file's header.

    Attributes:
        name: The tensor's name, for example ``blk.0.attn_k.weight``.
        dims: Its shape, slowest-varying dimension first.
        type: The GGML tensor type id.
        offset: Byte offset into the file's tensor data section.
        bytes_: Size in bytes, computed from ``dims`` and ``type``. Serialized
            as ``bytes``, its alias, since ``bytes`` is a Python builtin.
    """

    name: str
    dims: list[int]
    type: int
    offset: int
    bytes_: int = Field(default=0, alias="bytes")


class GgufHeader(_Strict):
    """A GGUF file's header, read directly from its bytes.

    Attributes:
        version: The GGUF format version, currently 2 or 3.
        tensor_count: Number of tensors the header declares.
        alignment: Byte alignment of the tensor data section, from the
            ``general.alignment`` metadata key when present.
        metadata: Every metadata key-value pair, keyed by its GGUF key.
        tensors: One entry per tensor, in header order.
        header_bytes: Number of bytes the header itself occupies, i.e. where
            the tensor data section begins before alignment padding.
    """

    version: int
    tensor_count: int
    alignment: int = 32
    metadata: dict[str, object] = Field(default_factory=dict)
    tensors: list[TensorInfo] = Field(default_factory=list)
    header_bytes: int = 0


class GgufFacts(_Strict):
    """Architecture facts and byte-size breakdown derived from a GGUF header.

    Every number here is read from the file, not estimated: layer, head and
    vocabulary counts come from the file's own metadata, and every
    ``bytes_*`` field is a sum of tensor sizes the header already reports.
    ``attention_layers_source`` records how ``attention_layers`` was
    determined: ``"tensors"`` when full-attention blocks were identified by
    the tensors they carry, ``"all-layers"`` when no block could be
    distinguished and every layer was assumed to have full attention, or
    ``"unknown"`` when neither could be established.

    Attributes:
        arch: The model family, from ``general.architecture``.
        n_layer: Number of transformer blocks, or ``None`` if not reported.
        n_embd: Embedding (hidden) dimension.
        n_vocab: Vocabulary size.
        n_head: Number of attention heads.
        n_head_kv: Number of key/value heads (for grouped-query attention).
        head_dim: Dimension of one attention head.
        attention_layers: Number of layers with full (non-linear) attention.
        attention_layers_source: How ``attention_layers`` was determined.
        n_expert: Number of experts, for mixture-of-experts architectures.
        n_expert_used: Number of experts activated per token.
        has_shared_experts: Whether the model has always-on shared experts
            in addition to routed ones.
        bytes_expert_weights: Total bytes of routed expert weight tensors.
        bytes_attention_weights: Total bytes of attention weight tensors.
        bytes_output_head: Bytes of the output (unembedding) tensor.
        bytes_token_embd: Bytes of the token embedding tensor.
        bytes_lazy_tables: Bytes of tensors that can be loaded lazily rather
            than kept resident, such as per-layer embedding tables.
        bytes_total: Total bytes of every tensor in the file.
        kv_bytes_per_token_f16: Bytes of KV cache needed per token at f16
            precision, or ``None`` when it cannot be computed.
        recurrent_state_bytes: Bytes of fixed recurrent/SSM state, for hybrid
            or state-space architectures, or ``None`` when not applicable.
    """

    arch: str
    n_layer: int | None = None
    n_embd: int | None = None
    n_vocab: int | None = None
    n_head: int | None = None
    n_head_kv: int | None = None
    head_dim: int | None = None
    attention_layers: int | None = None
    attention_layers_source: Literal["tensors", "all-layers", "unknown"] = "unknown"
    n_expert: int | None = None
    n_expert_used: int | None = None
    has_shared_experts: bool = False
    bytes_expert_weights: int = 0
    bytes_attention_weights: int = 0
    bytes_output_head: int = 0
    bytes_token_embd: int = 0
    bytes_lazy_tables: int = 0
    bytes_total: int = 0
    kv_bytes_per_token_f16: int | None = None
    recurrent_state_bytes: int | None = None
