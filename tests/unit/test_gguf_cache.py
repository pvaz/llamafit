import os
from pathlib import Path

from llamafit.gguf import read_facts
from llamafit.gguf.cache import HeaderCache, cache_key_for_path, cache_key_for_url
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from tests.unit.test_gguf_facts import dense_header


def test_a_cached_header_round_trips(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    header = read_header(FakeSource(dense_header()))
    assert cache.get("k") is None
    cache.put("k", header)
    again = cache.get("k")
    assert again is not None
    assert again.tensor_count == header.tensor_count
    assert [t.name for t in again.tensors] == [t.name for t in header.tensors]
    assert again.metadata["general.architecture"] == "llama"


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    cache.put("k", read_header(FakeSource(dense_header())))
    next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")
    assert cache.get("k") is None


def test_the_path_key_changes_with_size_and_time(tmp_path: Path) -> None:
    target = tmp_path / "m.gguf"
    target.write_bytes(b"a" * 10)
    first = cache_key_for_path(target)
    target.write_bytes(b"a" * 20)
    assert cache_key_for_path(target) != first


def test_the_url_key_changes_with_the_etag() -> None:
    first = cache_key_for_url("https://x/y.gguf", "abc")
    second = cache_key_for_url("https://x/y.gguf", "def")
    assert first != second


def test_read_facts_from_a_local_file(tmp_path: Path) -> None:
    target = tmp_path / "m.gguf"
    target.write_bytes(dense_header())
    facts = read_facts(target)
    assert facts.arch == "llama" and facts.n_layer == 2


def test_read_facts_uses_the_cache_the_second_time(tmp_path: Path) -> None:
    target = tmp_path / "m.gguf"
    target.write_bytes(dense_header())
    cache = HeaderCache(tmp_path / "cache")
    first = read_facts(target, cache=cache)
    stat = target.stat()
    target.write_bytes(b"XXXX" + dense_header()[4:])  # would fail to parse if re-read
    os.utime(target, (stat.st_atime, stat.st_mtime))
    second = read_facts(target, cache=cache)
    assert second.n_layer == first.n_layer
