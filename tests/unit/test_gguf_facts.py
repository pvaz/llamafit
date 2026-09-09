import pytest

from llamafit.gguf.facts import _KV_TYPE_BYTES, derive_facts, kv_bytes_per_token
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from llamafit.models.gguf import GgufFacts
from tests.fixtures import gguf_builder as b


def dense_header() -> bytes:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 2),
        b.uint32("llama.embedding_length", 4096),
        b.uint32("llama.attention.head_count", 32),
        b.uint32("llama.attention.head_count_kv", 8),
        b.uint32("llama.attention.key_length", 128),
        b.string_array("tokenizer.ggml.tokens", ["a"] * 32000),
    ]
    tensors = [
        b.tensor("token_embd.weight", [4096, 32000], 8, 0),
        b.tensor("output.weight", [4096, 32000], 8, 1),
        b.tensor("blk.0.attn_k.weight", [4096, 1024], 8, 2),
        b.tensor("blk.0.ffn_down.weight", [11008, 4096], 8, 3),
        b.tensor("blk.1.attn_k.weight", [4096, 1024], 8, 4),
    ]
    return b.build(metadata, tensors)


def hybrid_moe_header() -> bytes:
    metadata = [
        b.string("general.architecture", "qwen3next"),
        b.uint32("qwen3next.block_count", 4),
        b.uint32("qwen3next.embedding_length", 2560),
        b.uint32("qwen3next.attention.head_count", 24),
        b.uint32("qwen3next.attention.head_count_kv", 2),
        b.uint32("qwen3next.attention.key_length", 128),
        b.uint32("qwen3next.expert_count", 512),
        b.uint32("qwen3next.expert_used_count", 10),
        b.string_array("tokenizer.ggml.tokens", ["a"] * 1024),
    ]
    tensors = [
        b.tensor("token_embd.weight", [2560, 1024], 8, 0),
        # One row per layer, as in the real model, so its size cannot coincide with
        # token_embd's by construction.
        b.tensor("per_layer_token_embd.weight", [2560, 1024, 4], 8, 1),
        # three linear-attention layers and one full-attention layer
        b.tensor("blk.0.ssm_out.weight", [2560, 2560], 8, 2),
        b.tensor("blk.0.ffn_down_exps.weight", [512, 2560, 512], 8, 3),
        b.tensor("blk.1.ssm_out.weight", [2560, 2560], 8, 4),
        b.tensor("blk.2.ssm_out.weight", [2560, 2560], 8, 5),
        b.tensor("blk.3.attn_k.weight", [2560, 256], 8, 6),
        b.tensor("blk.3.ffn_down_shexp.weight", [512, 2560], 8, 7),
    ]
    return b.build(metadata, tensors)


def test_dense_model_facts() -> None:
    facts = derive_facts(read_header(FakeSource(dense_header())))
    assert facts.arch == "llama"
    assert (facts.n_layer, facts.n_embd, facts.n_head, facts.n_head_kv) == (2, 4096, 32, 8)
    assert facts.head_dim == 128
    assert facts.n_vocab == 32000
    assert facts.attention_layers == 2
    assert facts.attention_layers_source == "tensors"
    assert facts.n_expert is None and facts.has_shared_experts is False
    assert facts.bytes_expert_weights == 0
    assert facts.bytes_token_embd > 0 and facts.bytes_output_head > 0


def test_hybrid_moe_counts_only_the_full_attention_layers() -> None:
    facts = derive_facts(
        read_header(FakeSource(hybrid_moe_header())), lazy_tensor_names=["per_layer_token_embd"]
    )
    assert facts.n_layer == 4
    assert facts.attention_layers == 1
    assert facts.attention_layers_source == "tensors"
    assert facts.n_expert == 512 and facts.n_expert_used == 10
    assert facts.has_shared_experts is True
    assert facts.bytes_expert_weights > 0
    assert facts.bytes_lazy_tables > 0
    assert facts.bytes_lazy_tables not in (facts.bytes_token_embd,)


def test_the_sliding_window_is_recorded_when_the_architecture_declares_one() -> None:
    # Gemma 3 27B IT really does declare gemma3.attention.sliding_window = 1024.
    metadata = [
        b.string("general.architecture", "gemma3"),
        b.uint32("gemma3.block_count", 62),
        b.uint32("gemma3.embedding_length", 5376),
        b.uint32("gemma3.attention.head_count", 32),
        b.uint32("gemma3.attention.head_count_kv", 16),
        b.uint32("gemma3.attention.key_length", 128),
        b.uint32("gemma3.context_length", 131072),
        b.uint32("gemma3.attention.sliding_window", 1024),
    ]
    facts = derive_facts(read_header(FakeSource(b.build(metadata, []))))
    assert facts.sliding_window == 1024
    assert facts.context_length == 131072


def test_neither_window_nor_context_is_invented_when_the_header_declares_none() -> None:
    facts = derive_facts(read_header(FakeSource(dense_header())))
    assert facts.sliding_window is None
    assert facts.context_length is None


def test_head_dimension_falls_back_to_embedding_over_heads() -> None:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 1),
        b.uint32("llama.embedding_length", 4096),
        b.uint32("llama.attention.head_count", 32),
        b.uint32("llama.attention.head_count_kv", 32),
    ]
    facts = derive_facts(read_header(FakeSource(b.build(metadata, []))))
    assert facts.head_dim == 128


def test_kv_bytes_per_token_by_cache_type() -> None:
    facts = derive_facts(read_header(FakeSource(dense_header())))
    # Each cache is 2 attention layers * 8 kv heads * 128 = 2048 elements per token.
    assert kv_bytes_per_token(facts, "f16") == 2048 * 2 + 2048 * 2
    assert kv_bytes_per_token(facts, "q8_0") == (2048 // 32 * 34) * 2
    assert facts.kv_bytes_per_token_f16 == 2048 * 2 + 2048 * 2


def test_the_value_length_falls_back_to_the_key_length_when_undeclared() -> None:
    facts = derive_facts(read_header(FakeSource(dense_header())))
    assert facts.head_dim == 128
    assert facts.value_head_dim == 128


def test_a_value_length_that_differs_from_the_key_length_is_recorded_and_used() -> None:
    # No file in the catalog separates them today, but nothing stops an architecture
    # from doing it, and a cache sized from twice the key length would be too large.
    metadata = [
        b.string("general.architecture", "acme"),
        b.uint32("acme.block_count", 2),
        b.uint32("acme.embedding_length", 4096),
        b.uint32("acme.attention.head_count", 32),
        b.uint32("acme.attention.head_count_kv", 8),
        b.uint32("acme.attention.key_length", 256),
        b.uint32("acme.attention.value_length", 128),
    ]
    tensors = [
        b.tensor("blk.0.attn_k.weight", [10], 0, 0),
        b.tensor("blk.1.attn_k.weight", [10], 0, 1),
    ]
    facts = derive_facts(read_header(FakeSource(b.build(metadata, tensors))))

    assert (facts.head_dim, facts.value_head_dim) == (256, 128)
    # Key cache 2 * 8 * 256 = 4096 elements, value cache 2 * 8 * 128 = 2048: 6144 in
    # total, not the 8192 that doubling the key length would give.
    assert kv_bytes_per_token(facts, "f16") == 4096 * 2 + 2048 * 2
    assert facts.kv_bytes_per_token_f16 == 6144 * 2


def test_the_head_dimension_is_still_the_key_length_on_its_own() -> None:
    metadata = [
        b.string("general.architecture", "acme"),
        b.uint32("acme.block_count", 1),
        b.uint32("acme.attention.key_length", 256),
        b.uint32("acme.attention.value_length", 128),
    ]
    facts = derive_facts(read_header(FakeSource(b.build(metadata, []))))
    assert facts.head_dim == 256


def test_a_global_tensor_is_bucketed_and_not_lost() -> None:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 1),
    ]
    tensors = [
        b.tensor("token_embd.weight", [10], 0, 0),
        b.tensor("output_norm.weight", [10], 0, 1),
        b.tensor("blk.0.attn_k.weight", [10], 0, 2),
    ]
    facts = derive_facts(read_header(FakeSource(b.build(metadata, tensors))))
    assert facts.bytes_global_weights > 0
    assert facts.bytes_global_weights == facts.bytes_total - (
        facts.bytes_token_embd + facts.bytes_dense_block_weights
    )


def test_the_byte_buckets_sum_exactly_to_the_total() -> None:
    for built in (dense_header(), hybrid_moe_header()):
        facts = derive_facts(read_header(FakeSource(built)))
        buckets = (
            facts.bytes_token_embd
            + facts.bytes_output_head
            + facts.bytes_expert_weights
            + facts.bytes_dense_block_weights
            + facts.bytes_lazy_tables
            + facts.bytes_global_weights
        )
        assert buckets == facts.bytes_total


def test_a_header_without_the_architecture_key_still_returns_facts() -> None:
    facts = derive_facts(read_header(FakeSource(b.build([], []))))
    assert facts.arch == "unknown"
    assert facts.n_layer is None
    assert facts.attention_layers_source == "unknown"


def _one_head_facts(key: int, value: int) -> GgufFacts:
    """Facts for a one-layer, one-KV-head model, so a cache is exactly ``key`` wide."""
    metadata = [
        b.string("general.architecture", "acme"),
        b.uint32("acme.block_count", 1),
        b.uint32("acme.attention.head_count", 1),
        b.uint32("acme.attention.head_count_kv", 1),
        b.uint32("acme.attention.key_length", key),
        b.uint32("acme.attention.value_length", value),
    ]
    tensors = [b.tensor("blk.0.attn_k.weight", [10], 0, 0)]
    return derive_facts(read_header(FakeSource(b.build(metadata, tensors))))


def test_each_cache_is_rounded_up_on_its_own_not_the_pair_together() -> None:
    """llama.cpp allocates two tensors, so each takes its own whole blocks.

    48 elements per cache at q8_0 needs two 32-element blocks, four for the pair.
    Rounding once over the combined 96 elements would claim three, and rounding each
    cache down would claim two.
    """
    facts = _one_head_facts(48, 48)

    assert kv_bytes_per_token(facts, "q8_0") == 2 * (2 * 34)
    assert kv_bytes_per_token(facts, "q8_0") != (48 + 48) // 32 * 34
    assert kv_bytes_per_token(facts, "q8_0") != 2 * (48 // 32 * 34)


def test_a_cache_that_does_not_fill_a_whole_block_still_takes_one() -> None:
    """Rounding down would report a cache smaller than the one that gets allocated.

    Undersized is the dangerous direction: it says a model fits when it does not. 40
    elements at q8_0 take two 32-element blocks, not the one flooring would give.
    """
    facts = _one_head_facts(40, 40)

    assert kv_bytes_per_token(facts, "q8_0") == 2 * (2 * 34)
    assert kv_bytes_per_token(facts, "q8_0") > 2 * (40 // 32 * 34)


# Attention layers, key/value heads, key length and value length as the five seeded
# models really declare them, with the f16 figure each produces. Llama 3.1 declares
# neither length and derives both from embedding length over head count.
_REAL_KV_SHAPES = [
    ("gemma3", 62, 16, 128, 128, 507_904),
    ("llama", 32, 8, 128, 128, 131_072),
    ("qwen3", 28, 8, 128, 128, 114_688),
    ("qwen3next", 12, 2, 256, 256, 24_576),
    ("qwen4exp", 12, 2, 256, 256, 24_576),
]


@pytest.mark.parametrize(
    ("arch", "layers", "heads_kv", "key", "value", "expected_f16"),
    _REAL_KV_SHAPES,
    ids=[shape[0] for shape in _REAL_KV_SHAPES],
)
def test_the_kv_rules_change_nothing_on_the_shapes_real_models_have(
    arch: str, layers: int, heads_kv: int, key: int, value: int, expected_f16: int
) -> None:
    """Per-cache sizing and rounding up are both no-ops on every seeded shape.

    Every real cache width is a power of two that each block size divides, so no
    division leaves a remainder and rounding up cannot differ from rounding down. That
    is luck about these five models, which is why it is asserted rather than assumed.
    """
    facts = GgufFacts(
        arch=arch,
        attention_layers=layers,
        n_head_kv=heads_kv,
        head_dim=key,
        value_head_dim=value,
    )
    assert kv_bytes_per_token(facts, "f16") == expected_f16

    for kv_type in ("f16", "q8_0", "q4_0"):
        block_elements, block_bytes = _KV_TYPE_BYTES[kv_type]
        for name, dimension in (("key", key), ("value", value)):
            elements = layers * heads_kv * dimension
            assert elements % block_elements == 0, (
                f"{arch} at {kv_type}: the {name} cache is {elements} elements, which does "
                f"not fill whole {block_elements}-element blocks, so rounding up is not a "
                "no-op on this shape"
            )
        combined_rounded_down = layers * heads_kv * (key + value) // block_elements * block_bytes
        assert kv_bytes_per_token(facts, kv_type) == combined_rounded_down, (
            f"{arch} at {kv_type}: the per-cache rules must be a no-op on this shape"
        )
