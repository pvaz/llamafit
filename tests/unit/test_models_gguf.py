import pytest
from pydantic import ValidationError

from llamafit.models import GgufFacts, GgufHeader, TensorInfo


def make_header(**overrides: object) -> GgufHeader:
    base: dict[str, object] = {
        "version": 3,
        "tensor_count": 1,
        "alignment": 32,
        "metadata": {"general.architecture": "llama"},
        "tensors": [
            TensorInfo(name="token_embd.weight", dims=[4096, 32000], type=8, offset=0, bytes=512),
        ],
        "header_bytes": 256,
    }
    base.update(overrides)
    return GgufHeader(**base)  # type: ignore[arg-type]


def test_header_round_trips_through_json_with_tensors_intact() -> None:
    header = make_header()
    again = GgufHeader.model_validate_json(header.model_dump_json())
    assert again == header
    assert again.tensors[0].bytes_ == 512


def test_bytes_alias_accepts_field_name_or_alias() -> None:
    by_alias = TensorInfo(name="t", dims=[1], type=0, offset=0, bytes=64)
    by_name = TensorInfo(name="t", dims=[1], type=0, offset=0, bytes_=64)
    assert by_alias.bytes_ == by_name.bytes_ == 64


def test_bytes_alias_is_used_on_serialization() -> None:
    tensor = TensorInfo(name="t", dims=[1], type=0, offset=0, bytes=64)
    assert tensor.model_dump(by_alias=True)["bytes"] == 64
    assert "bytes_" not in tensor.model_dump(by_alias=True)


def test_header_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        make_header(unexpected_field=True)


def test_tensor_info_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        TensorInfo(name="t", dims=[1], type=0, offset=0, bytes=1, extra="nope")  # type: ignore[call-arg]


def test_facts_built_with_only_arch_has_documented_defaults() -> None:
    facts = GgufFacts(arch="llama")
    assert facts.arch == "llama"
    assert facts.n_layer is None
    assert facts.n_embd is None
    assert facts.n_vocab is None
    assert facts.n_head is None
    assert facts.n_head_kv is None
    assert facts.head_dim is None
    assert facts.value_head_dim is None
    assert facts.attention_layers is None
    assert facts.attention_layers_source == "unknown"
    assert facts.sliding_window is None
    assert facts.context_length is None
    assert facts.n_expert is None
    assert facts.n_expert_used is None
    assert facts.has_shared_experts is False
    assert facts.bytes_expert_weights == 0
    assert facts.bytes_dense_block_weights == 0
    assert facts.bytes_output_head == 0
    assert facts.bytes_token_embd == 0
    assert facts.bytes_lazy_tables == 0
    assert facts.bytes_total == 0
    assert facts.kv_bytes_per_token_f16 is None
    assert facts.recurrent_state_bytes is None


def test_attention_layers_source_rejects_value_outside_the_three_allowed() -> None:
    with pytest.raises(ValidationError):
        GgufFacts(arch="llama", attention_layers_source="guessed")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "field",
    [
        "n_layer",
        "n_embd",
        "n_vocab",
        "n_head",
        "n_head_kv",
        "head_dim",
        "value_head_dim",
        "attention_layers",
        "sliding_window",
        "context_length",
        "n_expert",
        "n_expert_used",
        "kv_bytes_per_token_f16",
        "recurrent_state_bytes",
    ],
)
def test_a_negative_count_or_size_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        GgufFacts(arch="llama", **{field: -3})


@pytest.mark.parametrize(
    "field",
    [
        "bytes_expert_weights",
        "bytes_dense_block_weights",
        "bytes_output_head",
        "bytes_token_embd",
        "bytes_lazy_tables",
        "bytes_global_weights",
        "bytes_total",
    ],
)
def test_a_negative_byte_bucket_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        GgufFacts(arch="llama", **{field: -1})


def test_byte_buckets_that_do_not_add_up_to_the_total_are_rejected() -> None:
    with pytest.raises(ValidationError, match="bytes_total"):
        GgufFacts(arch="llama", bytes_token_embd=10, bytes_total=11)


def test_byte_buckets_that_add_up_are_accepted() -> None:
    facts = GgufFacts(
        arch="llama", bytes_token_embd=10, bytes_dense_block_weights=1, bytes_total=11
    )
    assert facts.bytes_total == 11


def test_a_metadata_only_shard_reporting_no_bytes_at_all_is_still_valid() -> None:
    # The first shard of Qwen3.8-Flash-Next really does carry the whole metadata
    # block and no tensors, so zero bytes must stay readable.
    assert GgufFacts(arch="qwen4exp", n_layer=48).bytes_total == 0


def test_a_head_dimension_that_does_not_divide_the_embedding_is_accepted() -> None:
    # Gemma 3 27B: 5,376 embedding, 32 heads, 128 key length. 32 x 128 is 4,096,
    # not 5,376, so nothing here may assume the product.
    facts = GgufFacts(arch="gemma3", n_embd=5376, n_head=32, head_dim=128)
    assert (facts.n_embd, facts.n_head, facts.head_dim) == (5376, 32, 128)
