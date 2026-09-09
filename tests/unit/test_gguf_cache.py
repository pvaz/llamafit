import os
import re
from pathlib import Path

import httpx

from llamafit.gguf import read_facts
from llamafit.gguf.cache import HeaderCache, cache_key_for_path, cache_key_for_url
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from tests.fixtures import gguf_builder as b
from tests.unit.test_gguf_facts import dense_header


def _handler_for(
    data: bytes,
    etag: str | None,
    requests: list[httpx.Request] | None = None,
    refuse_head: bool = False,
) -> httpx.MockTransport:
    """A mock transport answering both ``HEAD`` and ranged ``GET`` for ``data``."""

    def handler(request: httpx.Request) -> httpx.Response:
        if requests is not None:
            requests.append(request)
        if request.method == "HEAD":
            if refuse_head:
                return httpx.Response(405)
            headers = {"content-length": str(len(data))}
            if etag is not None:
                headers["etag"] = etag
            return httpx.Response(200, headers=headers)
        match = re.match(r"bytes=(\d+)-(\d+)", request.headers["range"])
        assert match is not None
        start, end = int(match.group(1)), min(int(match.group(2)), len(data) - 1)
        headers = {"content-range": f"bytes {start}-{end}/{len(data)}"}
        if etag is not None:
            headers["etag"] = etag
        return httpx.Response(206, content=data[start : end + 1], headers=headers)

    return httpx.MockTransport(handler)


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


def test_read_facts_over_http_does_not_serve_a_stale_header_across_etags(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    url = "https://x/model.gguf"
    second_data = b.build(
        [b.string("general.architecture", "llama"), b.uint32("llama.block_count", 99)], []
    )

    client_a = httpx.Client(transport=_handler_for(dense_header(), "etag-a"))
    first = read_facts(url, cache=cache, client=client_a)
    assert first.n_layer == 2

    client_b = httpx.Client(transport=_handler_for(second_data, "etag-b"))
    second = read_facts(url, cache=cache, client=client_b)
    assert second.n_layer == 99, "a new ETag must not be served the first file's cached header"


def test_read_facts_over_http_still_works_when_the_server_sends_no_etag(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    client = httpx.Client(transport=_handler_for(dense_header(), None))
    facts = read_facts("https://x/model.gguf", cache=cache, client=client)
    assert facts.n_layer == 2


def test_a_cache_hit_costs_one_head_and_no_range_request(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    url = "https://x/model.gguf"

    warm_requests: list[httpx.Request] = []
    warm_client = httpx.Client(transport=_handler_for(dense_header(), "etag-a", warm_requests))
    first = read_facts(url, cache=cache, client=warm_client)
    assert first.n_layer == 2
    assert [r.method for r in warm_requests] == ["HEAD", "GET"]

    hit_requests: list[httpx.Request] = []
    hit_client = httpx.Client(transport=_handler_for(dense_header(), "etag-a", hit_requests))
    second = read_facts(url, cache=cache, client=hit_client)
    assert second.n_layer == first.n_layer
    assert [r.method for r in hit_requests] == ["HEAD"], "a cache hit must not fetch a range"


def test_a_changed_etag_invalidates_the_cache_and_costs_a_head_and_a_range_request(
    tmp_path: Path,
) -> None:
    cache = HeaderCache(tmp_path)
    url = "https://x/model.gguf"
    second_data = b.build(
        [b.string("general.architecture", "llama"), b.uint32("llama.block_count", 99)], []
    )

    first_requests: list[httpx.Request] = []
    first_client = httpx.Client(transport=_handler_for(dense_header(), "etag-a", first_requests))
    first = read_facts(url, cache=cache, client=first_client)
    assert first.n_layer == 2

    second_requests: list[httpx.Request] = []
    second_client = httpx.Client(transport=_handler_for(second_data, "etag-b", second_requests))
    second = read_facts(url, cache=cache, client=second_client)
    assert second.n_layer == 99
    assert [r.method for r in second_requests] == ["HEAD", "GET"], (
        "a changed ETag must miss the cache and fetch the new file"
    )


def test_read_facts_over_http_still_works_when_the_server_refuses_head(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    requests: list[httpx.Request] = []
    client = httpx.Client(
        transport=_handler_for(dense_header(), "etag-a", requests, refuse_head=True)
    )
    facts = read_facts("https://x/model.gguf", cache=cache, client=client)
    assert facts.n_layer == 2
    assert [r.method for r in requests] == ["HEAD", "GET"]
