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
        + facts.bytes_dense_block_weights
        + facts.bytes_shared_expert_weights
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
    assert facts.bytes_expert_weights > facts.bytes_dense_block_weights


@pytest.mark.hardware
def test_recurrent_state_bytes_matches_the_calibrated_measurement() -> None:
    """Qwen3.8-Flash-Next's recurrent (SSM) state buffer, measured on the reference machine.

    ``docs/calibration/2026-09-09-reference-machine.md`` records llama.cpp reporting a
    "Recurrent" buffer of 113 MiB in every configuration tested for this model
    (constant across context length, as a fixed-size buffer should be); the source
    measurement behind that rounded table entry is
    ``llama_memory_recurrent: CUDA0 RS buffer size = 112.57 MiB``, i.e. 118,036,480
    bytes.

    A single shard's header does not carry every tensor (the first shard here carries
    only metadata, ``tensor_count == 0``), so the full-attention layer count needs
    every shard's tensors merged into one header before deriving facts.
    """
    base = Path("D:/llama.cpp/models/Qwen3.8-Flash-Next")
    shards = sorted(base.glob("Qwen3.8-Flash-Next-UD-Q4_K_XL-*.gguf"))
    if not shards or not all(shard.exists() for shard in shards):
        pytest.skip("Qwen3.8-Flash-Next shards are not on this machine")

    merged = read_header(LocalSource(shards[0]))
    tensors = list(merged.tensors)
    for shard in shards[1:]:
        tensors.extend(read_header(LocalSource(shard)).tensors)
    merged = merged.model_copy(update={"tensors": tensors, "tensor_count": len(tensors)})

    facts = derive_facts(merged)
    measured_bytes = 118_036_480
    assert facts.recurrent_state_bytes is not None
    relative_error = abs(facts.recurrent_state_bytes - measured_bytes) / measured_bytes
    assert relative_error < 0.01, (
        f"derived {facts.recurrent_state_bytes}, measured {measured_bytes}, "
        f"{relative_error:.2%} off"
    )
