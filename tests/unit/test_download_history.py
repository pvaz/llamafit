"""The downloads history: small, forgiving, and never a reason a download fails."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamafit.download.history import (
    MAX_HISTORY_ENTRIES,
    DownloadRecord,
    append_record,
    history_path,
    read_history,
    recent,
)


def _record(model_id: str = "acme-model", **overrides: object) -> DownloadRecord:
    base: dict[str, object] = {
        "model_id": model_id,
        "quant": "Q4_K_M",
        "repo": "acme/model-gguf",
        "directory": "/models/acme",
        "files": ["model.gguf"],
        "total_bytes": 1000,
        "fetched_bytes": 1000,
        "finished": True,
    }
    base.update(overrides)
    return DownloadRecord(**base)  # type: ignore[arg-type]


def test_the_history_lives_under_the_data_directory(tmp_path: Path) -> None:
    assert history_path(tmp_path).name == "downloads.json"


def test_a_record_survives_a_round_trip(tmp_path: Path) -> None:
    path = history_path(tmp_path)
    append_record(path, _record())
    records = read_history(path)
    assert len(records) == 1
    assert records[0].model_id == "acme-model"
    assert records[0].finished is True
    assert records[0].at.tzinfo is not None


def test_records_accumulate_in_order(tmp_path: Path) -> None:
    path = history_path(tmp_path)
    for index in range(3):
        append_record(path, _record(model_id=f"model-{index}"))
    assert [record.model_id for record in read_history(path)] == ["model-0", "model-1", "model-2"]


def test_the_history_stops_growing(tmp_path: Path) -> None:
    path = history_path(tmp_path)
    for index in range(MAX_HISTORY_ENTRIES + 5):
        append_record(path, _record(model_id=f"model-{index}"))
    records = read_history(path)
    assert len(records) == MAX_HISTORY_ENTRIES
    assert records[0].model_id == "model-5"


def test_the_newest_come_back_first(tmp_path: Path) -> None:
    path = history_path(tmp_path)
    for index in range(5):
        append_record(path, _record(model_id=f"model-{index}"))
    assert [r.model_id for r in recent(read_history(path), 2)] == ["model-4", "model-3"]


def test_a_missing_history_is_an_empty_one(tmp_path: Path) -> None:
    assert read_history(history_path(tmp_path)) == []


@pytest.mark.parametrize("text", ["{not json", '{"a": 1}', "null"])
def test_a_history_that_will_not_read_is_an_empty_one(tmp_path: Path, text: str) -> None:
    path = history_path(tmp_path)
    path.write_text(text, encoding="utf-8")
    assert read_history(path) == []


def test_an_entry_that_is_not_a_record_is_skipped_and_the_rest_are_kept(tmp_path: Path) -> None:
    path = history_path(tmp_path)
    good = _record().model_dump(mode="json")
    path.write_text(json.dumps([{"nonsense": True}, good]), encoding="utf-8")
    assert [record.model_id for record in read_history(path)] == ["acme-model"]


def test_a_history_that_cannot_be_written_is_not_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model is on disk and correct; refusing to say so because a convenience file
    # could not be updated would be the wrong trade.
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", refuse)
    append_record(history_path(tmp_path), _record())
    assert not history_path(tmp_path).exists()
