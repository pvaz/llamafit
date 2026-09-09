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
    assert facts.attention_layers is None
    assert facts.attention_layers_source == "unknown"
    assert facts.sliding_window is None
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
