"""The downloader itself: resume, verification, and every way a transfer goes wrong.

Nothing here touches the network or writes outside ``tmp_path``. The transport is an
``httpx`` mock; the reader, the engine, the part file and the verifier are the ones that
ship.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from llamafit.download import engine
from llamafit.download.engine import (
    DownloadOptions,
    download_file,
    download_plan,
    probe_size,
)
from llamafit.download.errors import (
    ChecksumError,
    DownloadCancelledError,
    DownloadError,
)
from llamafit.download.plan import DownloadPlan, FileRequest
from llamafit.download.state import open_part, part_path_for, state_path_for
from llamafit.errors import NetworkError
from tests.fixtures.downloads import CANONICAL, FakeHub, RecordingProgress, blob, sha256

CHUNK = 1024
SIZE = CHUNK * 10


def _request(tmp_path: Path, data: bytes, *, checksum: str | None = None) -> FileRequest:
    return FileRequest(
        name="model-Q4_K_M.gguf",
        repo_path="model-Q4_K_M.gguf",
        url=CANONICAL,
        target=tmp_path / "model-Q4_K_M.gguf",
        size=len(data),
        sha256=sha256(data) if checksum is None else checksum,
        role="weights",
    )


def _options(**overrides: object) -> DownloadOptions:
    base: dict[str, object] = {
        "workers": 4,
        "chunk_bytes": CHUNK,
        "max_attempts": 3,
        "backoff_seconds": 0.0,
    }
    base.update(overrides)
    return DownloadOptions(**base)  # type: ignore[arg-type]


# --- the happy path ----------------------------------------------------------


def test_a_file_arrives_whole_and_verified(tmp_path: Path) -> None:
    data = blob(SIZE)
    hub = FakeHub(data)
    with hub.reader() as reader:
        outcome = download_file(_request(tmp_path, data), reader, options=_options())

    assert outcome.path.read_bytes() == data
    assert outcome.verified is True
    assert outcome.sha256 == sha256(data)
    assert outcome.fetched == SIZE
    assert not part_path_for(outcome.path).exists()
    assert not state_path_for(outcome.path).exists()


def test_the_canonical_url_is_followed_to_the_content_network(tmp_path: Path) -> None:
    # A mock transport never redirects on its own, so the reader has to ask on every
    # request. This is the bug the project has already paid for once.
    data = blob(SIZE)
    hub = FakeHub(data, redirect=True)
    with hub.reader() as reader:
        download_file(_request(tmp_path, data), reader, options=_options())
    assert hub.served > 0


def test_every_byte_range_is_asked_for_exactly_once(tmp_path: Path) -> None:
    data = blob(SIZE)
    hub = FakeHub(data)
    with hub.reader() as reader:
        download_file(_request(tmp_path, data), reader, options=_options(workers=1))
    assert sorted(hub.ranges) == [(index * CHUNK, index * CHUNK + CHUNK - 1) for index in range(10)]


def test_a_file_already_here_is_not_fetched_again(tmp_path: Path) -> None:
    data = blob(SIZE)
    target = tmp_path / "model-Q4_K_M.gguf"
    target.write_bytes(data)
    hub = FakeHub(data)
    with hub.reader() as reader:
        outcome = download_file(_request(tmp_path, data), reader, options=_options())
    assert outcome.already_present is True
    assert outcome.fetched == 0
    assert hub.served == 0


# --- resume ------------------------------------------------------------------


def test_an_interrupted_transfer_keeps_the_chunks_that_arrived(tmp_path: Path) -> None:
    data = blob(SIZE)
    stop = threading.Event()
    hub = FakeHub(data, stop=stop, stop_after=3)
    request = _request(tmp_path, data)
    with hub.reader() as reader, pytest.raises(DownloadCancelledError):
        download_file(request, reader, options=_options(workers=1), stop=stop)

    recorded = json.loads(state_path_for(request.target).read_text(encoding="utf-8"))
    assert recorded["size"] == SIZE
    assert recorded["chunk_size"] == CHUNK
    assert recorded["done"] == [0, 1, 2]
    assert part_path_for(request.target).stat().st_size == SIZE
    assert not request.target.exists()


def test_a_resumed_transfer_asks_only_for_what_is_missing(tmp_path: Path) -> None:
    # The proof, rather than the assertion: the second hub raises if it is ever asked for
    # a byte the first run already wrote, and the finished file still hashes correctly.
    data = blob(SIZE)
    stop = threading.Event()
    first = FakeHub(data, stop=stop, stop_after=4)
    request = _request(tmp_path, data)
    with first.reader() as reader, pytest.raises(DownloadCancelledError):
        download_file(request, reader, options=_options(workers=1), stop=stop)
    assert json.loads(state_path_for(request.target).read_text(encoding="utf-8"))["done"] == [
        0,
        1,
        2,
        3,
    ]

    second = FakeHub(data, forbidden_below=4 * CHUNK)
    progress = RecordingProgress()
    with second.reader() as reader:
        outcome = download_file(request, reader, options=_options(workers=1), reporter=progress)

    assert outcome.resumed is True
    assert outcome.fetched == SIZE - 4 * CHUNK
    assert sorted(second.ranges) == [
        (index * CHUNK, index * CHUNK + CHUNK - 1) for index in range(4, 10)
    ]
    assert request.target.read_bytes() == data
    assert sha256(request.target.read_bytes()) == sha256(data)
    assert progress.started == [(request.name, SIZE, 4 * CHUNK)]


def test_a_chunk_cut_off_mid_stream_is_not_recorded_and_is_fetched_again(tmp_path: Path) -> None:
    data = blob(SIZE)
    hub = FakeHub(data, truncate_after=CHUNK // 2, truncate_requests=(0,))
    progress = RecordingProgress()
    with hub.reader() as reader:
        outcome = download_file(
            _request(tmp_path, data), reader, options=_options(workers=1), reporter=progress
        )

    assert outcome.path.read_bytes() == data
    # The half chunk that arrived was counted and then taken back, so the bar never
    # claimed bytes that were not on disk.
    assert progress.at_verify[outcome.name] == SIZE
    assert any(count < 0 for _who, count in progress.deltas)
    assert hub.ranges.count((0, CHUNK - 1)) == 2


def test_a_transfer_that_keeps_being_cut_off_fails_and_keeps_what_it_had(tmp_path: Path) -> None:
    data = blob(SIZE)
    hub = FakeHub(data, truncate_after=CHUNK // 2, truncate_requests=(0, 1, 2, 3, 4, 5))
    request = _request(tmp_path, data)
    with hub.reader() as reader, pytest.raises(NetworkError):
        download_file(request, reader, options=_options(workers=1, max_attempts=3))
    assert part_path_for(request.target).exists()
    assert not request.target.exists()


def test_a_state_file_for_a_different_size_is_not_resumed_into(tmp_path: Path) -> None:
    data = blob(SIZE)
    request = _request(tmp_path, data)
    part_path_for(request.target).write_bytes(b"\0" * SIZE)
    state_path_for(request.target).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "url": CANONICAL,
                "size": SIZE // 2,
                "chunk_size": CHUNK,
                "done": [0, 1, 2],
            }
        ),
        encoding="utf-8",
    )
    hub = FakeHub(data)
    with hub.reader() as reader:
        outcome = download_file(request, reader, options=_options(workers=1))
    assert outcome.fetched == SIZE
    assert outcome.resumed is False


def test_an_unreadable_state_file_costs_bandwidth_not_the_download(tmp_path: Path) -> None:
    data = blob(SIZE)
    request = _request(tmp_path, data)
    part_path_for(request.target).write_bytes(b"\0" * SIZE)
    state_path_for(request.target).write_text("{not json", encoding="utf-8")
    hub = FakeHub(data)
    with hub.reader() as reader:
        outcome = download_file(request, reader, options=_options(workers=1))
    assert outcome.path.read_bytes() == data


# --- verification ------------------------------------------------------------


def test_a_checksum_that_does_not_match_leaves_nothing_behind(tmp_path: Path) -> None:
    data = blob(SIZE)
    hub = FakeHub(data)
    request = _request(tmp_path, data, checksum="0" * 64)
    with hub.reader() as reader, pytest.raises(ChecksumError) as caught:
        download_file(request, reader, options=_options())

    assert request.name in caught.value.message
    assert caught.value.hint is not None
    assert sha256(data) in caught.value.hint
    assert not request.target.exists()
    assert not part_path_for(request.target).exists()
    assert not state_path_for(request.target).exists()


def test_a_file_with_no_checksum_in_the_catalog_is_fetched_but_not_called_verified(
    tmp_path: Path,
) -> None:
    data = blob(SIZE)
    hub = FakeHub(data)
    request = _request(tmp_path, data)
    request = FileRequest(**{**request.__dict__, "sha256": None})
    with hub.reader() as reader:
        outcome = download_file(request, reader, options=_options())
    assert outcome.verified is False
    assert outcome.sha256 is None
    assert outcome.path.read_bytes() == data


def test_recheck_re_reads_a_file_that_is_already_here(tmp_path: Path) -> None:
    data = blob(SIZE)
    target = tmp_path / "model-Q4_K_M.gguf"
    target.write_bytes(data)
    hub = FakeHub(data)
    with hub.reader() as reader:
        outcome = download_file(_request(tmp_path, data), reader, options=_options(recheck=True))
    assert outcome.already_present is True
    assert outcome.verified is True
    assert hub.served == 0


def test_recheck_replaces_a_file_on_disk_that_is_not_the_catalog_s(tmp_path: Path) -> None:
    data = blob(SIZE)
    target = tmp_path / "model-Q4_K_M.gguf"
    target.write_bytes(b"\7" * SIZE)
    hub = FakeHub(data)
    progress = RecordingProgress()
    with hub.reader() as reader:
        outcome = download_file(
            _request(tmp_path, data), reader, options=_options(recheck=True), reporter=progress
        )
    assert outcome.already_present is False
    assert target.read_bytes() == data
    assert any("checksum" in note for note in progress.notes)


# --- servers that misbehave --------------------------------------------------


def test_a_server_that_stops_serving_ranges_halfway_still_delivers_the_file(
    tmp_path: Path,
) -> None:
    data = blob(SIZE)
    hub = FakeHub(data, drop_ranges_after=3)
    progress = RecordingProgress()
    with hub.reader() as reader:
        outcome = download_file(
            _request(tmp_path, data), reader, options=_options(workers=1), reporter=progress
        )

    assert outcome.path.read_bytes() == data
    assert outcome.verified is True
    assert any("one stream" in note for note in progress.notes)
    # The bar was rewound to nothing before the sequential attempt, so it never showed
    # more than the file's size.
    assert progress.at_verify[outcome.name] == SIZE


def test_a_rate_limited_server_is_waited_for_and_given_fewer_connections(
    tmp_path: Path,
) -> None:
    data = blob(SIZE)
    hub = FakeHub(data, rate_limit_requests=(0, 1), retry_after="0")
    progress = RecordingProgress()
    with hub.reader() as reader:
        outcome = download_file(
            _request(tmp_path, data), reader, options=_options(workers=4), reporter=progress
        )
    assert outcome.path.read_bytes() == data
    assert any("fewer connections" in note for note in progress.notes)


def test_a_server_that_only_ever_rate_limits_gives_up_saying_so(tmp_path: Path) -> None:
    data = blob(SIZE)
    hub = FakeHub(data, rate_limit_requests=tuple(range(200)), retry_after="0")
    with hub.reader() as reader, pytest.raises(DownloadError) as caught:
        download_file(_request(tmp_path, data), reader, options=_options(max_attempts=2))
    assert "rate-limiting" in caught.value.message


@pytest.mark.parametrize(("status", "fragment"), [(401, "authorisation"), (404, "404")])
def test_a_refusal_is_not_retried_and_says_what_to_do(
    tmp_path: Path, status: int, fragment: str
) -> None:
    data = blob(SIZE)
    hub = FakeHub(data, fail_status=status)
    with hub.reader() as reader, pytest.raises(DownloadError) as caught:
        download_file(_request(tmp_path, data), reader, options=_options())
    assert fragment in caught.value.message
    assert caught.value.hint


def test_a_file_the_server_says_is_a_different_size_stops_before_anything_is_written(
    tmp_path: Path,
) -> None:
    data = blob(SIZE)
    hub = FakeHub(data)
    request = _request(tmp_path, data)
    request = FileRequest(**{**request.__dict__, "size": SIZE + 4096})
    with hub.reader() as reader, pytest.raises(DownloadError) as caught:
        download_file(request, reader, options=_options())
    assert "re-uploaded" in caught.value.message
    assert not part_path_for(request.target).exists()


def test_the_probe_reads_the_size_out_of_the_content_range(tmp_path: Path) -> None:
    data = blob(SIZE)
    hub = FakeHub(data)
    request = _request(tmp_path, data)
    request = FileRequest(**{**request.__dict__, "size": None})
    with hub.reader() as reader:
        assert probe_size(reader, request) == SIZE


# --- cancellation ------------------------------------------------------------


def test_cancelling_before_anything_starts_writes_nothing(tmp_path: Path) -> None:
    data = blob(SIZE)
    stop = threading.Event()
    stop.set()
    hub = FakeHub(data)
    request = _request(tmp_path, data)
    with hub.reader() as reader, pytest.raises(DownloadCancelledError):
        download_file(request, reader, options=_options(), stop=stop)
    assert not request.target.exists()


# --- a whole plan ------------------------------------------------------------


def test_a_split_model_is_fetched_shard_by_shard_and_each_is_verified(tmp_path: Path) -> None:
    shards = [blob(SIZE, seed=index + 1) for index in range(3)]
    files = tuple(
        FileRequest(
            name=f"model-0000{index + 1}-of-00003.gguf",
            repo_path=f"model-0000{index + 1}-of-00003.gguf",
            url=f"https://cdn-lfs.example.invalid/shard{index}",
            target=tmp_path / f"model-0000{index + 1}-of-00003.gguf",
            size=None,
            sha256=sha256(data),
            role="weights",
        )
        for index, data in enumerate(shards)
    )
    plan = DownloadPlan(
        model_id="acme-model",
        model_name="Acme Model",
        quant="Q4_K_M",
        repo="acme/model-gguf",
        directory=tmp_path,
        files=files,
        total_bytes=SIZE * 3,
    )

    def handle(request: object) -> object:
        import httpx

        assert isinstance(request, httpx.Request)
        index = int(str(request.url)[-1])
        return FakeHub(shards[index], redirect=False).handle(request)

    import httpx

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        from llamafit.download.transport import HttpRangeReader

        outcomes = download_plan(
            plan, HttpRangeReader(client, piece_bytes=256), options=_options(workers=2)
        )

    assert [outcome.verified for outcome in outcomes] == [True, True, True]
    for index, data in enumerate(shards):
        assert (tmp_path / f"model-0000{index + 1}-of-00003.gguf").read_bytes() == data


def test_a_server_that_never_serves_ranges_is_handled_from_the_first_request(
    tmp_path: Path,
) -> None:
    # No Content-Range anywhere, so the size comes from Content-Length and the whole file
    # arrives in one stream. Some mirrors genuinely behave like this.
    data = blob(SIZE)
    hub = FakeHub(data, drop_ranges_after=0)
    progress = RecordingProgress()
    request = _request(tmp_path, data)
    request = FileRequest(**{**request.__dict__, "size": None})
    with hub.reader() as reader:
        outcome = download_file(request, reader, options=_options(workers=2), reporter=progress)
    assert outcome.path.read_bytes() == data
    assert outcome.verified is True


def test_rechecking_a_file_with_no_checksum_leaves_it_alone(tmp_path: Path) -> None:
    data = blob(SIZE)
    target = tmp_path / "model-Q4_K_M.gguf"
    target.write_bytes(data)
    request = _request(tmp_path, data)
    request = FileRequest(**{**request.__dict__, "sha256": None})
    hub = FakeHub(data)
    with hub.reader() as reader:
        outcome = download_file(request, reader, options=_options(recheck=True))
    assert outcome.already_present is True
    assert outcome.verified is False
    assert hub.served == 0


def test_a_part_file_that_is_short_at_the_end_is_refused_as_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The chunks all reported their full length and the file is still short: something
    # outside this process changed it. The size check catches that before the checksum,
    # and says "cut short" rather than reading out two hexadecimal strings.
    data = blob(SIZE)
    hub = FakeHub(data)
    request = _request(tmp_path, data)
    real_check = engine.check_size

    def shrink_then_check(path: Path, expected: int, *, name: str) -> None:
        with path.open("r+b") as handle:
            handle.truncate(expected - 1)
        real_check(path, expected, name=name)

    monkeypatch.setattr(engine, "check_size", shrink_then_check)
    with hub.reader() as reader, pytest.raises(ChecksumError) as caught:
        download_file(request, reader, options=_options(workers=1))
    assert "bytes" in caught.value.message
    assert not part_path_for(request.target).exists()
    assert not state_path_for(request.target).exists()


def test_a_part_file_knows_when_it_is_complete(tmp_path: Path) -> None:
    data = blob(SIZE)
    part = open_part(tmp_path / "x.gguf", CANONICAL, len(data), chunk_size=SIZE)
    part.allocate()
    assert part.is_complete() is False
    part.mark_done(0)
    assert part.is_complete() is True
