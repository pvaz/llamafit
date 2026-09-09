"""A split model must read as one model, or refuse to read at all.

The shape here is the real one, measured against ``unsloth/Qwen3.8-Flash-Next-GGUF``
path ``UD-Q4_K_XL/``: the first shard carries the whole metadata block and *no* tensors,
and the remaining shards carry three metadata keys each and all the tensors between
them. Reading only the first shard used to yield complete-looking architecture facts and
zero bytes of weights; it is now refused, because one shard of four is not a single-file
model but the emptiest possible set, and it is the input that produced the bug.
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from llamafit.errors import CatalogError
from llamafit.gguf import bits_per_weight, merge_shard_headers, read_facts
from llamafit.gguf.cache import HeaderCache
from llamafit.gguf.facts import derive_facts
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource, HttpRangeSource
from tests.fixtures import gguf_builder as b

_ARCH = "qwen4exp"
_Q8_0 = 8
_UNKNOWN_TYPE = 4242

# name -> dims, grouped by the shard that carries it; shard 0 carries none, as in the
# real repository. Blocks 0 and 2 own an ``attn_k.weight`` and so are full-attention
# layers, while the model declares four blocks: a fact only the union can establish.
_SHARD_TENSORS: list[list[tuple[str, list[int]]]] = [
    [],
    [("token_embd.weight", [2560, 1024]), ("per_layer_token_embd.weight", [2560, 1024, 4])],
    [
        ("blk.0.attn_k.weight", [2560, 256]),
        ("blk.0.ffn_down_exps.weight", [512, 2560, 512]),
        ("blk.1.ssm_out.weight", [2560, 2560]),
    ],
    [
        ("blk.2.attn_k.weight", [2560, 256]),
        ("output.weight", [2560, 1024]),
        ("output_norm.weight", [2560]),
    ],
]
_TENSOR_TOTAL = sum(len(group) for group in _SHARD_TENSORS)


def _model_metadata() -> list[bytes]:
    """The architecture block only the metadata shard carries."""
    return [
        b.string("general.architecture", _ARCH),
        b.uint32(f"{_ARCH}.block_count", 4),
        b.uint32(f"{_ARCH}.embedding_length", 2560),
        b.uint32(f"{_ARCH}.attention.head_count", 24),
        b.uint32(f"{_ARCH}.attention.head_count_kv", 2),
        b.uint32(f"{_ARCH}.attention.key_length", 128),
        b.uint32(f"{_ARCH}.expert_count", 512),
        b.uint32(f"{_ARCH}.expert_used_count", 10),
        b.string_array("tokenizer.ggml.tokens", ["a"] * 1024),
    ]


def _split_keys(shard_no: int, shard_count: int | None, tensors_total: int | None) -> list[bytes]:
    """The bookkeeping trio; either declared count can be left out to test the guards."""
    keys = [b.uint16("split.no", shard_no)]
    if shard_count is not None:
        keys.append(b.uint16("split.count", shard_count))
    if tensors_total is not None:
        keys.append(b.int32("split.tensors.count", tensors_total))
    return keys


def _encode(group: list[tuple[str, list[int]]], type_id: int = _Q8_0) -> list[bytes]:
    """Encode one shard's tensors, with offsets restarting at 0 as in a real shard."""
    return [b.tensor(name, dims, type_id, index) for index, (name, dims) in enumerate(group)]


def shard(
    shard_no: int,
    *,
    shard_count: int | None = len(_SHARD_TENSORS),
    tensors_total: int | None = _TENSOR_TOTAL,
    tensors: list[bytes] | None = None,
) -> bytes:
    """One shard: the metadata block only when it is shard 0, plus the split keys."""
    metadata = _model_metadata() if shard_no == 0 else []
    metadata += _split_keys(shard_no, shard_count, tensors_total)
    body = _encode(_SHARD_TENSORS[shard_no]) if tensors is None else tensors
    return b.build(metadata, body)


def split_model() -> list[bytes]:
    """Every shard of the split model, in order."""
    return [shard(index) for index in range(len(_SHARD_TENSORS))]


def one_file_model() -> bytes:
    """The same model published as a single file, for comparison."""
    flat = [entry for group in _SHARD_TENSORS for entry in group]
    return b.build(_model_metadata(), _encode(flat))


def write_shards(directory: Path, shards: list[bytes]) -> list[Path]:
    """Write shards to disk under the usual ``-NNNNN-of-MMMMM.gguf`` names."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, data in enumerate(shards, start=1):
        path = directory / f"model-{index:05d}-of-{len(shards):05d}.gguf"
        path.write_bytes(data)
        paths.append(path)
    return paths


def test_the_first_shard_alone_is_refused_rather_than_read_as_a_whole_model(
    tmp_path: Path,
) -> None:
    """The bug's own input. One shard of four is the emptiest set, not a single file.

    File matching returns whatever files it found and never checks the group is
    complete, so a repository mid-upload is enough to hand this in.
    """
    paths = write_shards(tmp_path, split_model())
    with pytest.raises(CatalogError) as caught:
        read_facts(paths[0])
    assert caught.value.message == (
        "incomplete shard set: the model declares 4 shards and 1 arrived (split.no [0])"
    )
    assert caught.value.hint == "Pass every shard of the split model."


def test_a_lone_tail_shard_is_refused_and_says_which_one_it_is(tmp_path: Path) -> None:
    paths = write_shards(tmp_path, split_model())
    with pytest.raises(CatalogError, match=re.escape("1 arrived (split.no [2])")):
        read_facts(paths[2])


def test_the_union_of_the_shards_equals_the_same_model_in_one_file(tmp_path: Path) -> None:
    split = tmp_path / "split"
    single = tmp_path / "model.gguf"
    single.write_bytes(one_file_model())

    from_shards = read_facts(write_shards(split, split_model()))
    from_one_file = read_facts(single)

    assert from_shards == from_one_file
    assert from_shards.bytes_total > 0
    assert from_shards.attention_layers == 2, "only the union shows which blocks attend"
    assert from_shards.attention_layers_source == "tensors"


def test_every_byte_bucket_still_sums_to_the_total(tmp_path: Path) -> None:
    facts = read_facts(write_shards(tmp_path, split_model()))
    buckets = (
        facts.bytes_token_embd
        + facts.bytes_output_head
        + facts.bytes_expert_weights
        + facts.bytes_dense_block_weights
        + facts.bytes_lazy_tables
        + facts.bytes_global_weights
    )
    assert buckets == facts.bytes_total


def test_shards_given_out_of_order_read_the_same(tmp_path: Path) -> None:
    """Order is not rejected: metadata comes from split.no 0 and only sizes are summed."""
    paths = write_shards(tmp_path, split_model())
    shuffled = [paths[2], paths[0], paths[3], paths[1]]
    assert read_facts(shuffled) == read_facts(paths)


def test_a_set_missing_a_shard_raises_instead_of_under_reporting(tmp_path: Path) -> None:
    paths = write_shards(tmp_path, split_model())
    with pytest.raises(CatalogError) as caught:
        read_facts(paths[:-1])
    assert "declares 4 shards" in str(caught.value) and "3 arrived" in str(caught.value)


def test_a_set_without_the_metadata_shard_raises(tmp_path: Path) -> None:
    """No shard declares a count here, so only the absent shard 0 can be complained about."""
    tail = [shard(index, shard_count=None, tensors_total=None) for index in (1, 2, 3)]
    with pytest.raises(CatalogError, match="metadata is missing"):
        read_facts(write_shards(tmp_path, tail))


def test_the_declared_shard_count_is_read_from_a_tail_shard(tmp_path: Path) -> None:
    """A metadata shard that omits the counts must not disarm the guard."""
    shards = split_model()
    shards[0] = shard(0, shard_count=None, tensors_total=None)
    paths = write_shards(tmp_path, shards)
    with pytest.raises(CatalogError, match=re.escape("declares 4 shards and 3 arrived")):
        read_facts(paths[:-1])


def test_the_declared_tensor_total_is_read_from_a_tail_shard(tmp_path: Path) -> None:
    """A truncated tail loses bytes silently if only the metadata shard is consulted."""
    shards = split_model()
    shards[0] = shard(0, shard_count=None, tensors_total=None)
    shards[3] = shard(3, tensors=_encode(_SHARD_TENSORS[3][:1]))
    paths = write_shards(tmp_path, shards)
    with pytest.raises(CatalogError, match=re.escape(f"declares {_TENSOR_TOTAL} tensors")):
        read_facts(paths)


def test_shards_that_disagree_about_the_split_are_refused(tmp_path: Path) -> None:
    mismatched = [shard(0), shard(1), shard(2), shard(3, shard_count=5)]
    with pytest.raises(CatalogError, match=re.escape("disagree about 'split.count'")):
        read_facts(write_shards(tmp_path, mismatched))


def test_a_repeated_shard_raises(tmp_path: Path) -> None:
    paths = write_shards(tmp_path, split_model())
    with pytest.raises(CatalogError, match=re.escape("expected split.no 0 to 3")):
        read_facts([paths[0], paths[1], paths[1], paths[2]])


def test_a_short_union_raises_even_when_the_shard_count_is_right(tmp_path: Path) -> None:
    """A shard that lost tensors keeps the set's length; only the tensor total catches it."""
    shards = split_model()
    shards[3] = shard(3, tensors=_encode(_SHARD_TENSORS[3][:1]))
    paths = write_shards(tmp_path, shards)
    with pytest.raises(CatalogError) as caught:
        read_facts(paths)
    assert f"declares {_TENSOR_TOTAL} tensors" in str(caught.value)
    assert f"hold {_TENSOR_TOTAL - 2} between them" in str(caught.value)


def test_files_that_are_not_shards_of_one_model_are_refused(tmp_path: Path) -> None:
    first = tmp_path / "a.gguf"
    second = tmp_path / "b.gguf"
    first.write_bytes(one_file_model())
    second.write_bytes(one_file_model())
    with pytest.raises(CatalogError, match=re.escape("carry no 'split.no' key")):
        read_facts([first, second])


def test_no_file_at_all_raises() -> None:
    with pytest.raises(CatalogError, match="no GGUF header"):
        read_facts([])


def test_a_single_file_is_returned_unmerged() -> None:
    """A model published as one file declares no split keys, and takes no merge path."""
    header = read_header(FakeSource(one_file_model()))
    assert merge_shard_headers([header]) is header


def test_a_split_of_one_shard_is_a_complete_set_and_reads(tmp_path: Path) -> None:
    """Declaring a split is not the same as being incomplete: one of one is all of them."""
    flat = [entry for group in _SHARD_TENSORS for entry in group]
    whole = tmp_path / "model-00001-of-00001.gguf"
    whole.write_bytes(b.build(_model_metadata() + _split_keys(0, 1, len(flat)), _encode(flat)))
    plain = tmp_path / "plain.gguf"
    plain.write_bytes(one_file_model())
    assert read_facts(whole) == read_facts(plain)


def test_the_single_file_path_is_unchanged(tmp_path: Path) -> None:
    single = tmp_path / "model.gguf"
    single.write_bytes(one_file_model())
    assert read_facts(single) == derive_facts(read_header(FakeSource(one_file_model())))
    assert read_facts(str(single)) == read_facts(single)


def test_a_merged_header_keeps_every_shards_unknown_tensor_types() -> None:
    """A diagnostic raised by a later shard must not be dropped with that shard's metadata."""
    odd = [("blk.3.attn_q.weight", [2560, 256])]
    total = _TENSOR_TOTAL - len(_SHARD_TENSORS[3]) + len(odd)
    shards = [shard(index, tensors_total=total) for index in range(3)]
    shards.append(shard(3, tensors_total=total, tensors=_encode(odd, _UNKNOWN_TYPE)))
    merged = merge_shard_headers([read_header(FakeSource(data)) for data in shards])
    assert merged.metadata["_unknown_tensor_types"] == [
        f"blk.3.attn_q.weight:unknown({_UNKNOWN_TYPE})"
    ]


def test_a_merged_header_counts_the_whole_union() -> None:
    merged = merge_shard_headers([read_header(FakeSource(data)) for data in split_model()])
    assert merged.tensor_count == len(merged.tensors) == _TENSOR_TOTAL
    assert [tensor.name for tensor in merged.tensors] == [
        name for group in _SHARD_TENSORS for name, _ in group
    ]


def _tail_shard(group: list[tuple[str, list[int]]]) -> bytes:
    """The second shard of a two-shard set, carrying ``group`` and no metadata."""
    return b.build(_split_keys(1, 2, len(group)), _encode(group))


def test_two_sets_sharing_their_metadata_shard_do_not_collide_in_the_cache(
    tmp_path: Path,
) -> None:
    """The identity of a set is every member's, so a shared first shard cannot answer."""
    cache = HeaderCache(tmp_path / "cache")
    head = tmp_path / "head.gguf"
    head.write_bytes(shard(0, shard_count=2, tensors_total=2))
    first_tail = tmp_path / "tail-a.gguf"
    first_tail.write_bytes(_tail_shard(_SHARD_TENSORS[1]))
    second_tail = tmp_path / "tail-b.gguf"
    second_tail.write_bytes(_tail_shard(_SHARD_TENSORS[2][:2]))

    first = read_facts([head, first_tail], cache=cache)
    second = read_facts([head, second_tail], cache=cache)

    assert first.bytes_total != second.bytes_total, "the second set must not reuse the first"
    assert read_facts([head, first_tail], cache=cache) == first


def _shard_transport(shards: list[bytes], seen: list[str]) -> httpx.MockTransport:
    """Serve each shard at its own URL, answering HEAD and ranged GET."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(f"{request.method} {request.url.path}")
        index = int(request.url.path.rsplit("-", 1)[-1])
        data = shards[index]
        etag = f'"shard-{index}"'
        if request.method == "HEAD":
            return httpx.Response(200, headers={"content-length": str(len(data)), "etag": etag})
        match = re.match(r"bytes=(\d+)-(\d+)", request.headers["range"])
        assert match is not None
        start, end = int(match.group(1)), min(int(match.group(2)), len(data) - 1)
        return httpx.Response(
            206,
            content=data[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(data)}", "etag": etag},
        )

    return httpx.MockTransport(handler)


def test_a_shard_set_reads_over_http_and_caches_per_shard(tmp_path: Path) -> None:
    shards = split_model()
    urls = [f"https://x/model-{index}" for index in range(len(shards))]
    cache = HeaderCache(tmp_path)

    warm: list[str] = []
    warm_client = httpx.Client(transport=_shard_transport(shards, warm))
    facts = read_facts(urls, cache=cache, client=warm_client)
    assert facts.bytes_total > 0
    assert sum(1 for entry in warm if entry.startswith("GET")) == len(shards)

    hit: list[str] = []
    hit_client = httpx.Client(transport=_shard_transport(shards, hit))
    again = read_facts(urls, cache=cache, client=hit_client)
    assert again == facts
    assert [entry.split(" ")[0] for entry in hit] == ["HEAD"] * len(shards), (
        "a warm shard set must cost one HEAD per shard and no range request"
    )


def test_bits_per_weight_reduces_to_the_old_formula_without_lazy_tables() -> None:
    file_bytes = 111_334_654_784
    total_b = 125.0
    assert bits_per_weight(file_bytes, 0, total_b) == file_bytes * 8 / (total_b * 1e9)
    assert round(bits_per_weight(file_bytes, 0, total_b), 2) == 7.13


def test_bits_per_weight_leaves_out_the_lazy_lookup_table() -> None:
    """The measured figures for Qwen3.8-Flash-Next UD-Q4_K_XL: 111.33 GB, 28.80 GB lazy."""
    assert round(bits_per_weight(111_330_000_000, 28_800_000_000, 125.0), 2) == 5.28


def test_bits_per_weight_is_driven_by_the_facts_a_split_model_reports(tmp_path: Path) -> None:
    """With the lazy table named, the union's bucket is what the figure drops."""
    paths = write_shards(tmp_path, split_model())
    plain = read_facts(paths)
    lazy = read_facts(paths, lazy_tensor_names=["per_layer_token_embd."])
    assert lazy.bytes_lazy_tables > 0 and plain.bytes_lazy_tables == 0
    assert bits_per_weight(lazy.bytes_total, lazy.bytes_lazy_tables, 1.0) < bits_per_weight(
        plain.bytes_total, plain.bytes_lazy_tables, 1.0
    )


_REPO = "https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL"
_REAL_SHARDS = [
    f"{_REPO}/Qwen3.8-Flash-Next-UD-Q4_K_XL-{index:05d}-of-00004.gguf" for index in range(1, 5)
]
_REAL_TENSORS = 1224
_REAL_BYTES = 111.33e9


@pytest.mark.network
def test_the_real_split_model_reads_as_one_model() -> None:
    """Read all four shards of the real repository and check the union against measurement.

    Skipped in the ordinary suite; run it with ``pytest -m network``. Hugging Face
    redirects every file URL to a CDN, so this also exercises redirect following on both
    the ``HEAD`` and the range request.
    """
    with httpx.Client(follow_redirects=True, timeout=60.0) as client:
        headers = [read_header(HttpRangeSource(url, client=client)) for url in _REAL_SHARDS]
        merged = merge_shard_headers(headers)
        assert merged.tensor_count == _REAL_TENSORS
        facts = derive_facts(merged)
        error = abs(facts.bytes_total - _REAL_BYTES) / _REAL_BYTES
        assert error < 0.01, (
            f"summed {facts.bytes_total} bytes, measured {_REAL_BYTES:.0f}, {error:.2%} off"
        )

        # 12 of the 48 blocks are full-attention, per
        # docs/calibration/2026-09-09-reference-machine.md. Only the union shows this:
        # the metadata shard alone carries no tensors, falls back to "every layer
        # attends", and so reports a KV cache four times too large.
        assert (facts.n_layer, facts.attention_layers) == (48, 12)
        assert facts.attention_layers_source == "tensors"

        # The header declares attention.key_length and attention.value_length equal
        # (256 each), so sizing the two caches separately and adding them returns the
        # same figure doubling one of them did. That equality is a property of this
        # file, not of every file, which is why the derivation no longer assumes it.
        assert (facts.head_dim, facts.value_head_dim) == (256, 256)
        assert facts.kv_bytes_per_token_f16 == 12 * 2 * (256 + 256) * 2

        entry_point = read_facts(_REAL_SHARDS, client=client)
        assert entry_point == facts
