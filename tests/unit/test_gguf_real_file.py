"""Check the tensor-size table against real files: the arithmetic must close exactly."""

from pathlib import Path

import pytest

from llamafit.gguf.facts import derive_facts
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import LocalSource

CANDIDATES = [
    Path("D:/llama.cpp/models/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf"),
    Path("D:/llama.cpp/models/Qwen3-Coder-Next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"),
]


@pytest.mark.hardware
@pytest.mark.parametrize("path", CANDIDATES, ids=lambda p: p.name)
def test_tensor_sizes_account_for_the_whole_file(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    header = read_header(LocalSource(path))
    data_start = (header.header_bytes + header.alignment - 1) // header.alignment * header.alignment
    total = data_start + sum(t.bytes_ for t in header.tensors)
    actual = path.stat().st_size
    assert total == actual, (
        f"tensor sizes do not account for the file: computed {total}, actual {actual}, "
        f"difference {actual - total}. A row in GGML_TYPES is wrong for one of the types "
        f"used here: {sorted({t.type for t in header.tensors})}"
    )


@pytest.mark.hardware
def test_a_split_model_reports_its_own_shard_only() -> None:
    path = CANDIDATES[1]
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    header = read_header(LocalSource(path))
    assert header.tensor_count == len(header.tensors)
    assert header.metadata.get("general.architecture")


@pytest.mark.hardware
@pytest.mark.parametrize("path", CANDIDATES, ids=lambda p: p.name)
def test_the_byte_buckets_sum_exactly_to_the_total(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    facts = derive_facts(read_header(LocalSource(path)))
    buckets = (
        facts.bytes_token_embd
        + facts.bytes_output_head
        + facts.bytes_expert_weights
        + facts.bytes_attention_weights
        + facts.bytes_lazy_tables
        + facts.bytes_global_weights
    )
    assert buckets == facts.bytes_total


@pytest.mark.hardware
def test_facts_from_the_reference_machines_coder_model() -> None:
    path = CANDIDATES[1]
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    facts = derive_facts(read_header(LocalSource(path)))
    assert facts.n_expert is not None and facts.n_expert > 1
    assert facts.attention_layers is not None
    assert facts.attention_layers < (facts.n_layer or 0), (
        "a hybrid model has fewer attention layers"
    )
    assert facts.bytes_expert_weights > facts.bytes_attention_weights
