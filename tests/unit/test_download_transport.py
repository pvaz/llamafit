"""The HTTP surface: the range header it sends, the redirect it follows, what it raises."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from llamafit import __version__
from llamafit.download.transport import HttpRangeReader
from llamafit.errors import NetworkError

URL = "https://huggingface.co/acme/model/resolve/main/model.gguf"
CDN = "https://cdn-lfs.example.invalid/model.gguf"


def _client(handler: object) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_the_range_header_is_inclusive_at_both_ends() -> None:
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["range"])
        return httpx.Response(206, content=b"x" * 10)

    with HttpRangeReader(_client(handle)) as reader, reader.open(URL, 100, 109) as response:
        assert b"".join(response.body) == b"x" * 10
    assert seen == ["bytes=100-109"]


def test_the_request_names_llamafit_and_refuses_a_re_encoded_body() -> None:
    seen: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(206, content=b"x")

    with HttpRangeReader(_client(handle)) as reader, reader.open(URL, 0, 0):
        pass
    assert seen[0]["user-agent"] == f"llamafit/{__version__}"
    assert seen[0]["accept-encoding"] == "identity"


def test_a_redirect_to_the_content_network_is_followed_on_the_request_itself() -> None:
    # The injected client is left at its default, which does not follow redirects. A
    # reader that relied on the client's setting would stop at the 302 here, and no mock
    # transport would ever have told us.
    urls: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        urls.append(str(request.url))
        if str(request.url) == URL:
            return httpx.Response(302, headers={"location": CDN})
        return httpx.Response(206, content=b"weights")

    with HttpRangeReader(_client(handle)) as reader, reader.open(URL, 0, 6) as response:
        assert b"".join(response.body) == b"weights"
    assert urls == [URL, CDN]


def test_a_token_is_sent_when_one_is_given() -> None:
    seen: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(206, content=b"x")

    with HttpRangeReader(_client(handle), token="secret") as reader, reader.open(URL, 0, 0):
        pass
    assert seen[0]["authorization"] == "Bearer secret"


def test_no_token_means_no_authorisation_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    seen: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(206, content=b"x")

    with HttpRangeReader(_client(handle)) as reader, reader.open(URL, 0, 0):
        pass
    assert "authorization" not in seen[0]


def test_the_token_is_read_from_the_environment_the_way_the_catalog_reads_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "from-the-environment")
    seen: list[httpx.Headers] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers)
        return httpx.Response(206, content=b"x")

    with HttpRangeReader(_client(handle)) as reader, reader.open(URL, 0, 0):
        pass
    assert seen[0]["authorization"] == "Bearer from-the-environment"


def test_the_status_and_headers_come_from_the_final_response() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            206, content=b"x", headers={"content-range": "bytes 0-0/4096", "etag": "abc"}
        )

    with HttpRangeReader(_client(handle)) as reader, reader.open(URL, 0, 0) as response:
        assert response.status == 206
        assert response.headers["content-range"] == "bytes 0-0/4096"


def test_a_status_the_engine_will_judge_is_handed_over_rather_than_raised() -> None:
    # Classifying a status is the engine's job: a 429 means something different from a
    # 404, and the transport has no opinion about either.
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    with HttpRangeReader(_client(handle)) as reader, reader.open(URL, 0, 0) as response:
        assert response.status == 429


def test_a_connection_failure_is_a_network_error_naming_the_url() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nothing there")

    with (
        HttpRangeReader(_client(handle)) as reader,
        pytest.raises(NetworkError) as caught,
        reader.open(URL, 0, 0),
    ):
        pass
    assert URL in caught.value.message
    assert caught.value.hint is not None
    assert "resume" in caught.value.hint


def test_a_body_that_fails_mid_stream_is_a_network_error_too() -> None:
    def body() -> Iterator[bytes]:
        yield b"first"
        raise httpx.ReadError("the connection went away")

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(206, content=body())

    with (
        HttpRangeReader(_client(handle)) as reader,
        pytest.raises(NetworkError),
        reader.open(URL, 0, 100) as response,
    ):
        list(response.body)


def test_an_injected_client_is_left_open_for_its_owner_to_close() -> None:
    client = _client(lambda request: httpx.Response(206, content=b"x"))
    reader = HttpRangeReader(client)
    reader.close()
    assert client.is_closed is False
    client.close()


def test_a_client_this_reader_made_is_closed_with_it() -> None:
    reader = HttpRangeReader()
    reader.close()
    assert reader._client.is_closed is True
