import re
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from llamafit.errors import CatalogError, NetworkError
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource, HttpRangeSource, LocalSource
from tests.fixtures import gguf_builder as b

URL = "https://example.invalid/model.gguf"


def _parse_range(value: str) -> tuple[int, int]:
    """Parse a ``Range: bytes=start-end`` header value into ``(start, end)``."""
    match = re.match(r"bytes=(\d+)-(\d+)", value)
    assert match is not None
    return int(match.group(1)), int(match.group(2))


def _range_handler(
    data: bytes, requests: list[httpx.Request]
) -> Callable[[httpx.Request], httpx.Response]:
    """A mock transport handler serving byte ranges out of ``data``, recording requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        start, end = _parse_range(request.headers["range"])
        end = min(end, len(data) - 1)
        return httpx.Response(
            206,
            content=data[start : end + 1],
            headers={"content-range": f"bytes {start}-{end}/{len(data)}"},
        )

    return handler


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_read_returns_the_requested_range_and_sends_the_expected_range_header() -> None:
    data = bytes(range(200))
    requests: list[httpx.Request] = []
    source = HttpRangeSource(URL, client=_client(_range_handler(data, requests)), chunk=50)

    result = source.read(10, 5)

    assert result == data[10:15]
    assert requests[0].headers["range"] == "bytes=10-59"


def test_a_second_read_inside_the_fetched_chunk_needs_no_new_request() -> None:
    data = bytes(range(200))
    requests: list[httpx.Request] = []
    source = HttpRangeSource(URL, client=_client(_range_handler(data, requests)), chunk=64)

    first = source.read(0, 4)
    assert first == data[0:4]
    assert len(requests) == 1

    second = source.read(4, 8)
    assert second == data[4:12]
    assert len(requests) == 1


def test_reading_across_a_chunk_boundary_issues_a_second_request() -> None:
    data = bytes(range(200))
    requests: list[httpx.Request] = []
    source = HttpRangeSource(URL, client=_client(_range_handler(data, requests)), chunk=64)

    source.read(0, 4)
    assert len(requests) == 1

    result = source.read(60, 10)

    assert result == data[60:70]
    assert len(requests) == 2


def test_size_reads_the_total_from_the_content_range_header() -> None:
    data = b"x" * 50000
    requests: list[httpx.Request] = []
    source = HttpRangeSource(URL, client=_client(_range_handler(data, requests)), chunk=1024)

    assert source.size() is None
    source.read(0, 10)
    assert source.size() == 50000


def test_size_is_none_when_the_server_sends_no_content_range() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(206, content=b"12345")

    source = HttpRangeSource(URL, client=_client(handler))
    source.read(0, 5)
    assert source.size() is None


@pytest.mark.parametrize("status", [200, 416])
def test_a_non_206_response_is_a_network_error_naming_the_url(status: int) -> None:
    source = HttpRangeSource(URL, client=_client(lambda request: httpx.Response(status)))
    with pytest.raises(NetworkError, match=re.escape(URL)):
        source.read(0, 10)


def test_a_connection_error_is_a_network_error_naming_the_url_not_an_httpx_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    source = HttpRangeSource(URL, client=_client(handler))
    with pytest.raises(NetworkError, match=re.escape(URL)):
        source.read(0, 10)


def test_reading_past_the_end_returns_the_short_tail_without_raising() -> None:
    data = b"hello"
    requests: list[httpx.Request] = []
    source = HttpRangeSource(URL, client=_client(_range_handler(data, requests)), chunk=100)

    result = source.read(0, 100)

    assert result == data


def test_http_range_source_parses_the_same_header_as_a_local_file() -> None:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 2),
        b.uint32("llama.embedding_length", 4096),
    ]
    tensors = [
        b.tensor("token_embd.weight", [4096, 100], 0, 0),
        b.tensor("blk.0.attn_k.weight", [4096, 100], 0, 1),
    ]
    data = b.build(metadata, tensors)
    requests: list[httpx.Request] = []

    remote = read_header(HttpRangeSource(URL, client=_client(_range_handler(data, requests))))
    local = read_header(FakeSource(data))

    assert remote == local
    assert len(requests) <= 2


def test_local_source_a_missing_path_is_a_catalog_error_naming_the_path(tmp_path: Path) -> None:
    missing = tmp_path / "nope.gguf"
    source = LocalSource(missing)
    with pytest.raises(CatalogError, match=re.escape(str(missing))):
        source.read(0, 10)


def test_local_source_a_missing_path_size_is_a_catalog_error_naming_the_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "nope.gguf"
    source = LocalSource(missing)
    with pytest.raises(CatalogError, match=re.escape(str(missing))):
        source.size()


def test_local_source_a_directory_is_a_catalog_error_naming_the_path(tmp_path: Path) -> None:
    source = LocalSource(tmp_path)
    with pytest.raises(CatalogError, match=re.escape(str(tmp_path))):
        source.read(0, 10)
