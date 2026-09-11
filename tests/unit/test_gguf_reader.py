import pytest

from llamafit.errors import CatalogError
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from llamafit.gguf.types import GGML_TYPES, tensor_bytes
from tests.fixtures import gguf_builder as b


def sample() -> bytes:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 32),
        b.uint32("llama.embedding_length", 4096),
        b.uint32("llama.attention.head_count", 32),
        b.uint32("llama.attention.head_count_kv", 8),
        b.uint32("general.alignment", 32),
        b.string_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
    ]
    tensors = [
        b.tensor("token_embd.weight", [4096, 128256], 12, 0),
        b.tensor("blk.0.attn_q.weight", [4096, 4096], 12, 1024),
        b.tensor("blk.0.ffn_down_exps.weight", [4096, 14336, 8], 12, 2048),
    ]
    return b.build(metadata, tensors)


def test_reads_counts_metadata_and_tensors() -> None:
    header = read_header(FakeSource(sample()))
    assert header.version == 3
    assert header.tensor_count == 3
    assert header.alignment == 32
    assert header.metadata["general.architecture"] == "llama"
    assert header.metadata["llama.block_count"] == 32
    assert header.metadata["tokenizer.ggml.tokens"] == ["a", "b", "c"]
    assert [t.name for t in header.tensors] == [
        "token_embd.weight",
        "blk.0.attn_q.weight",
        "blk.0.ffn_down_exps.weight",
    ]
    assert header.tensors[0].dims == [4096, 128256]


def test_computes_tensor_sizes_from_dimensions_and_type() -> None:
    header = read_header(FakeSource(sample()))
    # Q4_K is type 12: 256 elements per block, 144 bytes per block.
    assert header.tensors[1].bytes_ == (4096 * 4096 // 256) * 144
    assert tensor_bytes([32], 0) == 128  # F32, one byte-per-element block of 4
    assert tensor_bytes([256], 8) == 8 * 34  # Q8_0, 32 per block, 34 bytes


def test_reads_only_what_the_header_needs() -> None:
    source = FakeSource(sample() + b"\x00" * 5_000_000)
    header = read_header(source)
    assert header.header_bytes < 4096
    assert max(offset + length for offset, length in source.reads) < 1 << 20


def test_a_bad_magic_is_a_catalog_error() -> None:
    with pytest.raises(CatalogError, match="not a GGUF file"):
        read_header(FakeSource(b"XXXX" + sample()[4:]))


def test_an_unsupported_version_is_a_catalog_error() -> None:
    with pytest.raises(CatalogError, match="version 99"):
        read_header(FakeSource(b.build([], [], version=99)))


def test_a_truncated_header_is_a_catalog_error_not_a_crash() -> None:
    with pytest.raises(CatalogError, match="truncated"):
        read_header(FakeSource(sample()[:40]))


def test_an_unknown_value_type_is_a_catalog_error() -> None:
    bad = b.build([b._kv("weird", 99, b"\x00" * 4)], [])
    with pytest.raises(CatalogError, match="value type 99"):
        read_header(FakeSource(bad))


def test_an_unknown_tensor_type_refuses_the_file_rather_than_sizing_it_as_zero() -> None:
    # gpt-oss-120b was read as 2.4 GB of weights: every expert tensor is MXFP4, type 39
    # was not in the table, and a size of zero is a number every budget adds up without
    # complaint. The file must fail to read, and the error must name every type at once.
    tensors = [
        b.tensor("blk.0.attn_q.weight", [64], 8, 0),
        b.tensor("blk.0.ffn_down_exps.weight", [64], 4242, 1),
        b.tensor("blk.0.ffn_up_exps.weight", [64], 4242, 2),
        b.tensor("blk.1.ffn_down_exps.weight", [64], 65, 3),
    ]
    with pytest.raises(CatalogError) as info:
        read_header(FakeSource(b.build([], tensors)))
    message = str(info.value)
    assert "3 tensors" in message
    assert "4242 (2 tensors, first blk.0.ffn_down_exps.weight)" in message
    assert "65 (1 tensor, first blk.1.ffn_down_exps.weight)" in message
    assert info.value.hint is not None and "too small" in info.value.hint


@pytest.mark.parametrize(
    ("type_id", "name", "block_elements", "block_bytes"),
    [
        # Each row is the struct in ggml-common.h, added up: the scale bytes first, then
        # the packed weights.
        (39, "MXFP4", 32, 1 + 32 // 2),
        (40, "NVFP4", 64, 64 // 16 + 64 // 2),
        (41, "Q1_0", 128, 2 + 128 // 8),
        (42, "Q2_0", 64, 2 + 64 // 4),
    ],
)
def test_the_newer_ggml_types_size_as_their_structs_say(
    type_id: int, name: str, block_elements: int, block_bytes: int
) -> None:
    assert GGML_TYPES[type_id] == (name, block_elements, block_bytes)
    assert tensor_bytes([block_elements], type_id) == block_bytes
    assert tensor_bytes([block_elements * 3], type_id) == 3 * block_bytes


def test_an_mxfp4_expert_tensor_costs_17_bytes_per_32_weights() -> None:
    # gpt-oss-20b's blk.0.ffn_down_exps.weight, with the shape its own header declares.
    tensors = [b.tensor("blk.0.ffn_down_exps.weight", [2880, 2880, 32], 39, 0)]
    header = read_header(FakeSource(b.build([], tensors)))
    assert header.tensors[0].bytes_ == 2880 * 2880 * 32 // 32 * 17 == 141_004_800
    assert "_misaligned_tensors" not in header.metadata


def test_a_misaligned_tensor_rounds_up_and_is_recorded() -> None:
    # Q8_0: 32 elements per 34-byte block. 33 elements is one full block plus one
    # leftover element, which must round up to two whole blocks, not silently drop it.
    tensors = [b.tensor("blk.0.weird.weight", [33], 8, 0)]
    header = read_header(FakeSource(b.build([], tensors)))
    assert header.tensors[0].bytes_ == 2 * 34
    assert header.metadata["_misaligned_tensors"] == ["blk.0.weird.weight:Q8_0"]
