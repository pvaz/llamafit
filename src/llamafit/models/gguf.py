"""Data models for a GGUF file's header and the facts derived from it.

``GgufHeader`` mirrors what the GGUF reader parses directly from a file's byte
layout: its version, tensor and metadata counts, the metadata map itself, and
one :class:`TensorInfo` per tensor. ``GgufFacts`` is the smaller,
architecture-aware summary derived from a header — the numbers sizing and
placement decisions actually need.

All three models forbid unknown fields, so a typo in test or catalog data is
caught immediately instead of being silently ignored, and ``GgufFacts`` carries the
bounds real headers actually satisfy: every count and every byte figure is a
non-negative whole number, and the byte buckets add up to the total. Nothing narrower
is asserted, because this module also backs the reader, and a bound argued from
reasoning alone would reject a model that runs.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ByteSize = Annotated[int, Field(ge=0)]
"""A size in bytes. Never negative, whoever supplied it."""

Count = Annotated[int, Field(ge=0)]
"""A count read from a file's metadata: layers, heads, experts, tokens.

Never negative. No upper bound and no positive minimum: a header that reports zero
heads is a badly converted file rather than an impossible one, and refusing to read
it would cost a user a model that llama.cpp may well still load.
"""


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
        head_dim: Dimension of one attention head's key vector, from
            ``{arch}.attention.key_length``, or ``embedding_length`` over
            ``head_count`` when the architecture declares no key length. This is
            the head dimension for everything that legitimately means one; the
            KV cache is the exception and needs ``value_head_dim`` too.
        value_head_dim: Dimension of one attention head's value vector, from
            ``{arch}.attention.value_length``, falling back to ``head_dim`` when
            the architecture declares only a key length. Every model in the
            catalog today declares the two equal, but an architecture is free to
            separate them, and a cache sized from twice the key length would then
            be wrong in proportion.
        attention_layers: Number of layers with full (non-linear) attention.
        attention_layers_source: How ``attention_layers`` was determined.
        sliding_window: Length of the sliding attention window in tokens, from
            ``{arch}.attention.sliding_window``, or ``None`` when the architecture
            declares none. Recorded, not used: a model that slides holds a
            full-length KV cache on only a fraction of its layers, and the header
            does not say which, so the pattern has to come from a per-architecture
            rule or the catalog.
        context_length: The longest context the file itself declares, from
            ``{arch}.context_length``, or ``None`` when the architecture declares
            none. Recorded, not used: it is what the file permits, which is not
            always what the vendor supports.
        n_expert: Number of experts, for mixture-of-experts architectures.
        n_expert_used: Number of experts activated per token.
        has_shared_experts: Whether the model has always-on shared experts
            in addition to routed ones.
        bytes_expert_weights: Total bytes of routed expert weight tensors.
        bytes_dense_block_weights: Total bytes of every non-expert tensor inside
            a block: attention projections, feed-forward weights and norms alike.
            On a dense model that is nearly the whole file, so it is named for
            what it holds rather than for attention alone.
        bytes_output_head: Bytes of the output (unembedding) tensor.
        bytes_token_embd: Bytes of the token embedding tensor.
        bytes_lazy_tables: Bytes of tensors that can be loaded lazily rather
            than kept resident, such as per-layer embedding tables.
        bytes_global_weights: Total bytes of tensors that belong to none of
            the other buckets: not per-block, not the token embedding, the
            output head or a lazy table, for example ``output_norm.weight``.
        bytes_total: Total bytes of every tensor in the file. Always equal to
            the sum of every other ``bytes_*`` field, so the taxonomy above
            accounts for the whole file.
        kv_bytes_per_token_f16: Bytes of KV cache needed per token at f16
            precision, or ``None`` when it cannot be computed.
        recurrent_state_bytes: Bytes of fixed recurrent/SSM state, for hybrid
            or state-space architectures, or ``None`` when not applicable.
    """

    arch: str
    n_layer: Count | None = None
    n_embd: Count | None = None
    n_vocab: Count | None = None
    n_head: Count | None = None
    n_head_kv: Count | None = None
    head_dim: Count | None = None
    value_head_dim: Count | None = None
    attention_layers: Count | None = None
    attention_layers_source: Literal["tensors", "all-layers", "unknown"] = "unknown"
    sliding_window: Count | None = None
    context_length: Count | None = None
    n_expert: Count | None = None
    n_expert_used: Count | None = None
    has_shared_experts: bool = False
    bytes_expert_weights: ByteSize = 0
    bytes_dense_block_weights: ByteSize = 0
    bytes_shared_expert_weights: ByteSize = 0
    """Bytes of shared-expert tensors, which run for every token unlike routed ones.

    Carved out of :attr:`bytes_dense_block_weights` rather than counted beside it, so the
    buckets still partition the file. It has its own bucket because a placement can send
    just these to the other pool with a tensor override, and the reference machine's own
    best measured configuration does exactly that; a budget cannot cost a move it cannot
    measure. Zero for a model with no shared experts, never absent."""
    bytes_output_head: ByteSize = 0
    bytes_token_embd: ByteSize = 0
    bytes_lazy_tables: ByteSize = 0
    bytes_global_weights: ByteSize = 0
    bytes_total: ByteSize = 0
    kv_bytes_per_token_f16: ByteSize | None = None
    recurrent_state_bytes: ByteSize | None = None

    @model_validator(mode="after")
    def _buckets_account_for_the_whole_file(self) -> GgufFacts:
        """Reject facts whose byte buckets do not add up to ``bytes_total``.

        Every tensor in a header lands in exactly one bucket, so the sum is the
        total for any facts the reader derives — checked against five real files
        spanning llama, gemma3, qwen3, qwen3next and qwen4exp, one of them a
        four-shard split model with a streamed lookup table. Only a corrupted or
        hand-edited facts file can break it, and a set of buckets that does not add
        up is precisely what would silently mis-size a graphics card. A refresh
        rewrites the file this rejects.
        """
        buckets = (
            self.bytes_expert_weights
            + self.bytes_dense_block_weights
            + self.bytes_shared_expert_weights
            + self.bytes_output_head
            + self.bytes_token_embd
            + self.bytes_lazy_tables
            + self.bytes_global_weights
        )
        if buckets != self.bytes_total:
            raise ValueError(
                f"the byte buckets sum to {buckets} but bytes_total is {self.bytes_total}"
            )
        return self
