"""The part file and the record beside it, which is the whole of what makes resume real."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamafit.download.errors import DownloadError
from llamafit.download.state import (
    STATE_SCHEMA_VERSION,
    PartFile,
    open_part,
    part_path_for,
    state_path_for,
)

URL = "https://example.invalid/model.gguf"


def _part(tmp_path: Path, size: int = 1000, chunk: int = 256) -> PartFile:
    return PartFile(target=tmp_path / "model.gguf", url=URL, size=size, chunk_size=chunk)


def test_the_part_and_state_files_sit_beside_the_target(tmp_path: Path) -> None:
    target = tmp_path / "model.gguf"
    assert part_path_for(target).name == "model.gguf.part"
    assert state_path_for(target).name == "model.gguf.part.state"


def test_chunks_cover_the_file_exactly_once(tmp_path: Path) -> None:
    part = _part(tmp_path, size=1000, chunk=256)
    assert part.chunk_count == 4
    assert [part.bounds(index) for index in range(4)] == [
        (0, 255),
        (256, 511),
        (512, 767),
        (768, 999),
    ]
    assert sum(part.length_of(index) for index in range(4)) == 1000


def test_an_empty_file_is_one_chunk_of_nothing(tmp_path: Path) -> None:
    part = _part(tmp_path, size=0)
    assert part.chunk_count == 1
    assert part.length_of(0) == 0


def test_allocation_creates_the_file_at_its_final_size(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.allocate()
    assert part.path.stat().st_size == 1000


def test_allocation_resizes_a_part_file_that_is_the_wrong_length(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.path.write_bytes(b"\1" * 10)
    part.allocate()
    assert part.path.stat().st_size == 1000


def test_a_volume_with_no_room_is_reported_as_a_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    part = _part(tmp_path)

    def full(*_args: object, **_kwargs: object) -> object:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "open", full)
    with pytest.raises(DownloadError) as caught:
        part.allocate()
    assert "could not create" in caught.value.message
    assert caught.value.hint is not None
    assert "room" in caught.value.hint


def test_marking_a_chunk_writes_the_record(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.mark_done(2)
    recorded = json.loads(part.state_path.read_text(encoding="utf-8"))
    assert recorded == {
        "schema_version": STATE_SCHEMA_VERSION,
        "url": URL,
        "size": 1000,
        "chunk_size": 256,
        "done": [2],
    }
    assert part.bytes_done() == 256
    assert part.pending() == [0, 1, 3]


def test_a_recorded_run_is_resumed(tmp_path: Path) -> None:
    first = _part(tmp_path)
    first.allocate()
    first.mark_done(0)
    first.mark_done(1)

    second = open_part(tmp_path / "model.gguf", URL, 1000, chunk_size=256)
    assert second.done == {0, 1}
    assert second.pending() == [2, 3]
    assert second.bytes_done() == 512


@pytest.mark.parametrize(
    ("field", "value"),
    [("url", "https://elsewhere.invalid/x"), ("size", 999), ("chunk_size", 128)],
)
def test_a_record_that_disagrees_with_the_run_is_thrown_away(
    tmp_path: Path, field: str, value: object
) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.mark_done(0)
    document = json.loads(part.state_path.read_text(encoding="utf-8"))
    document[field] = value
    part.state_path.write_text(json.dumps(document), encoding="utf-8")

    assert open_part(tmp_path / "model.gguf", URL, 1000, chunk_size=256).done == set()


def test_a_record_from_a_future_schema_is_thrown_away(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.mark_done(0)
    document = json.loads(part.state_path.read_text(encoding="utf-8"))
    document["schema_version"] = STATE_SCHEMA_VERSION + 1
    part.state_path.write_text(json.dumps(document), encoding="utf-8")
    assert open_part(tmp_path / "model.gguf", URL, 1000, chunk_size=256).done == set()


def test_a_record_without_its_part_file_is_thrown_away(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.mark_done(0)
    part.path.unlink()
    assert open_part(tmp_path / "model.gguf", URL, 1000, chunk_size=256).done == set()


@pytest.mark.parametrize("text", ["{not json", "[]", '{"schema_version": 1, "done": "all"}'])
def test_a_record_that_will_not_read_is_treated_as_absent(tmp_path: Path, text: str) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.state_path.write_text(text, encoding="utf-8")
    assert open_part(tmp_path / "model.gguf", URL, 1000, chunk_size=256).done == set()


def test_an_index_outside_the_file_is_dropped_from_a_record(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "url": URL,
                "size": 1000,
                "chunk_size": 256,
                "done": [0, 99, "two"],
            }
        ),
        encoding="utf-8",
    )
    assert open_part(tmp_path / "model.gguf", URL, 1000, chunk_size=256).done == {0}


def test_discarding_removes_both_files(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.mark_done(0)
    part.discard()
    assert not part.path.exists()
    assert not part.state_path.exists()
    assert part.pending() == [0, 1, 2, 3]


def test_discarding_what_is_not_there_is_not_a_failure(tmp_path: Path) -> None:
    _part(tmp_path).discard()


def test_finishing_moves_the_file_into_place_and_clears_the_record(tmp_path: Path) -> None:
    part = _part(tmp_path)
    part.allocate()
    part.mark_done(0)
    part.finish()
    assert part.target.stat().st_size == 1000
    assert not part.path.exists()
    assert not part.state_path.exists()


def test_a_move_that_cannot_happen_says_so(tmp_path: Path) -> None:
    part = _part(tmp_path)
    with pytest.raises(DownloadError) as caught:
        part.finish()
    assert "into place" in caught.value.message
