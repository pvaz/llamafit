from llamafit.gguf.facts import derive_facts, kv_bytes_per_token
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
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
    # 2 attention layers * 8 kv heads * (128 key + 128 value) = 4096 elements per token
    assert kv_bytes_per_token(facts, "f16") == 4096 * 2
    assert kv_bytes_per_token(facts, "q8_0") == 4096 * 34 // 32
    assert facts.kv_bytes_per_token_f16 == 4096 * 2


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
    # 2 attention layers * 8 kv heads * (256 key + 128 value) = 6144 elements per token,
    # not the 8192 that doubling the key length would give.
    assert kv_bytes_per_token(facts, "f16") == 6144 * 2
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
