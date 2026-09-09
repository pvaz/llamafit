import pytest

from llamafit.errors import CatalogError
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from llamafit.gguf.types import tensor_bytes
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
