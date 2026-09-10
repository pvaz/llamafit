"""Installing a model: the order of the steps, the manifest, and the history."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest

from llamafit.download.engine import DownloadOptions
from llamafit.download.errors import (
    ChecksumError,
    DiskSpaceError,
    DownloadCancelledError,
    DownloadError,
)
from llamafit.download.history import history_path, read_history
from llamafit.download.install import install_model, prepare, write_manifest
from llamafit.download.plan import MANIFEST_NAME, DownloadPlan, FileRequest, build_plan
from llamafit.download.transport import HttpRangeReader
from llamafit.errors import NetworkError
from llamafit.models.catalog import CatalogModel, Extra, ModelSource, Quant
from tests.fixtures.downloads import FakeHub, RecordingProgress, blob, sha256
from tests.unit.test_models_catalog import minimal

ROOM = 8 * 1024**3
"""Free space for the disk check: comfortably above the headroom the planner keeps."""

CHUNK = 512
SHARD = CHUNK * 4


def _shards(count: int) -> list[bytes]:
    return [blob(SHARD, seed=index + 1) for index in range(count)]


def _model(shards: list[bytes], extra: bytes | None = None) -> CatalogModel:
    source: dict[str, object] = {
        "repo": "acme/model-gguf",
        "quants": [
            Quant(
                name="Q4_K_M",
                files=[
                    f"model-0000{index + 1}-of-0000{len(shards)}.gguf"
                    for index in range(len(shards))
                ],
                bytes=sum(len(data) for data in shards),
                sha256=[sha256(data) for data in shards],
            )
        ],
    }
    if extra is not None:
        source["extras"] = [
            Extra(role="mmproj", file="mmproj-F16.gguf", bytes=len(extra), sha256=sha256(extra))
        ]
    return minimal(id="acme-model", name="Acme Model", sources=[ModelSource(**source)])  # type: ignore[arg-type]


def _reader(bodies: dict[str, bytes], hubs: dict[str, FakeHub] | None = None) -> HttpRangeReader:
    """A reader serving each file by the last segment of its URL."""
    made = hubs if hubs is not None else {}
    for name, data in bodies.items():
        made.setdefault(name, FakeHub(data, redirect=False))

    def handle(request: httpx.Request) -> httpx.Response:
        name = str(request.url).rsplit("/", 1)[-1]
        return made[name].handle(request)

    return HttpRangeReader(httpx.Client(transport=httpx.MockTransport(handle)), piece_bytes=128)


def _options() -> DownloadOptions:
    return DownloadOptions(workers=2, chunk_bytes=CHUNK, max_attempts=2, backoff_seconds=0.0)


def _bodies(
    model: CatalogModel, shards: list[bytes], extra: bytes | None = None
) -> dict[str, bytes]:
    quant = model.sources[0].quants[0]
    bodies = dict(zip(quant.files, shards, strict=True))
    if extra is not None:
        bodies["mmproj-F16.gguf"] = extra
    return bodies


# --- the whole thing ---------------------------------------------------------


def test_a_split_model_and_its_projector_arrive_as_one_installed_thing(tmp_path: Path) -> None:
    shards, extra = _shards(3), blob(CHUNK, seed=99)
    model = _model(shards, extra)
    plan = build_plan(model, directory=tmp_path)
    with _reader(_bodies(model, shards, extra)) as reader:
        outcome = install_model(
            plan, reader, options=_options(), free_bytes=ROOM, data_dir=tmp_path / "data"
        )

    assert outcome.verified == 4
    assert outcome.fetched_bytes == SHARD * 3 + CHUNK
    for index, data in enumerate(shards):
        assert (tmp_path / f"model-0000{index + 1}-of-00003.gguf").read_bytes() == data
    assert (tmp_path / "mmproj-F16.gguf").read_bytes() == extra
    # Read back from disk: the manifest is what a later run reads to know it is here.
    assert build_plan(model, directory=tmp_path).is_installed() is True

    manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["model"] == "acme-model"
    assert manifest["quant"] == "Q4_K_M"
    assert [entry["name"] for entry in manifest["files"]] == [
        "model-00001-of-00003.gguf",
        "model-00002-of-00003.gguf",
        "model-00003-of-00003.gguf",
        "mmproj-F16.gguf",
    ]
    assert manifest["files"][0]["bytes"] == SHARD
    assert manifest["files"][0]["sha256"] == sha256(shards[0])


def test_a_second_run_over_a_finished_install_fetches_nothing(tmp_path: Path) -> None:
    shards = _shards(2)
    model = _model(shards)
    with _reader(_bodies(model, shards)) as reader:
        install_model(
            build_plan(model, directory=tmp_path), reader, options=_options(), free_bytes=ROOM
        )
    hubs = {name: FakeHub(data, redirect=False) for name, data in _bodies(model, shards).items()}
    with _reader(_bodies(model, shards), hubs) as reader:
        outcome = install_model(
            build_plan(model, directory=tmp_path), reader, options=_options(), free_bytes=ROOM
        )
    assert outcome.fetched_bytes == 0
    assert all(hub.served == 0 for hub in hubs.values())


# --- three of four -----------------------------------------------------------


def test_a_model_that_loses_a_shard_writes_no_manifest_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    shards = _shards(3)
    model = _model(shards)
    bodies = _bodies(model, shards)
    hubs = {
        name: FakeHub(data, redirect=False, fail_status=500 if index == 2 else None)
        for index, (name, data) in enumerate(bodies.items())
    }
    plan = build_plan(model, directory=tmp_path)
    with _reader(bodies, hubs) as reader, pytest.raises(NetworkError):
        install_model(plan, reader, options=_options(), free_bytes=ROOM, data_dir=tmp_path / "data")

    assert (tmp_path / "model-00001-of-00003.gguf").read_bytes() == shards[0]
    assert (tmp_path / "model-00002-of-00003.gguf").read_bytes() == shards[1]
    assert not (tmp_path / "model-00003-of-00003.gguf").exists()
    assert not (tmp_path / MANIFEST_NAME).exists()
    assert build_plan(model, directory=tmp_path).is_installed() is False


def test_the_run_that_lost_a_shard_is_recorded_with_the_sentence_the_user_saw(
    tmp_path: Path,
) -> None:
    shards = _shards(2)
    model = _model(shards)
    bodies = _bodies(model, shards)
    hubs = {name: FakeHub(data, redirect=False, fail_status=500) for name, data in bodies.items()}
    with _reader(bodies, hubs) as reader, pytest.raises(NetworkError):
        install_model(
            build_plan(model, directory=tmp_path),
            reader,
            options=_options(),
            free_bytes=ROOM,
            data_dir=tmp_path / "data",
        )
    records = read_history(history_path(tmp_path / "data"))
    assert len(records) == 1
    assert records[0].finished is False
    assert records[0].error is not None
    assert records[0].model_id == "acme-model"


def test_a_finished_run_is_recorded_as_finished(tmp_path: Path) -> None:
    shards = _shards(1)
    model = _model(shards)
    with _reader(_bodies(model, shards)) as reader:
        install_model(
            build_plan(model, directory=tmp_path),
            reader,
            options=_options(),
            free_bytes=ROOM,
            data_dir=tmp_path / "data",
        )
    records = read_history(history_path(tmp_path / "data"))
    assert [record.finished for record in records] == [True]
    assert records[0].fetched_bytes == SHARD


def test_no_data_directory_means_no_history_and_no_complaint(tmp_path: Path) -> None:
    shards = _shards(1)
    model = _model(shards)
    with _reader(_bodies(model, shards)) as reader:
        install_model(
            build_plan(model, directory=tmp_path), reader, options=_options(), free_bytes=ROOM
        )
    assert not (tmp_path / "downloads.json").exists()


# --- what has to be true first -----------------------------------------------


def test_a_full_disk_stops_the_run_before_a_single_file_is_created(tmp_path: Path) -> None:
    shards = _shards(3)
    model = _model(shards)
    plan = build_plan(model, directory=tmp_path)
    hubs = {name: FakeHub(data, redirect=False) for name, data in _bodies(model, shards).items()}
    with _reader(_bodies(model, shards), hubs) as reader, pytest.raises(DiskSpaceError):
        install_model(plan, reader, options=_options(), free_bytes=SHARD)

    assert all(hub.served == 0 for hub in hubs.values())
    assert list(tmp_path.iterdir()) == []


def test_a_model_with_no_checksums_is_refused_before_anything_is_fetched(
    tmp_path: Path,
) -> None:
    shards = _shards(1)
    model = _model(shards)
    model.sources[0].quants[0].sha256 = []
    plan = build_plan(model, directory=tmp_path)
    hubs = {name: FakeHub(data, redirect=False) for name, data in _bodies(model, shards).items()}
    with _reader(_bodies(model, shards), hubs) as reader, pytest.raises(DownloadError) as caught:
        install_model(plan, reader, options=_options(), free_bytes=ROOM)
    assert "checksum" in caught.value.message
    assert all(hub.served == 0 for hub in hubs.values())


def test_allow_unverified_gets_the_files_but_calls_none_of_them_verified(
    tmp_path: Path,
) -> None:
    shards = _shards(1)
    model = _model(shards)
    model.sources[0].quants[0].sha256 = []
    plan = build_plan(model, directory=tmp_path)
    with _reader(_bodies(model, shards)) as reader:
        outcome = install_model(
            plan, reader, options=_options(), free_bytes=ROOM, allow_unverified=True
        )
    assert outcome.verified == 0
    assert (tmp_path / "model-00001-of-00001.gguf").read_bytes() == shards[0]


def test_prepare_hands_back_the_numbers_it_was_satisfied_by(tmp_path: Path) -> None:
    shards = _shards(2)
    model = _model(shards)
    check = prepare(build_plan(model, directory=tmp_path), free_bytes=ROOM)
    assert check.needed == SHARD * 2
    assert check.free == ROOM
    assert check.ok is True


# --- a bad file --------------------------------------------------------------


def test_a_shard_that_fails_its_checksum_leaves_no_trace_of_itself(tmp_path: Path) -> None:
    shards = _shards(2)
    model = _model(shards)
    # The catalog's checksum for the second shard is wrong, so the bytes that arrive are
    # correct and the entry is not; either way the file must not be kept.
    model.sources[0].quants[0].sha256 = [sha256(shards[0]), "f" * 64]
    plan = build_plan(model, directory=tmp_path)
    with _reader(_bodies(model, shards)) as reader, pytest.raises(ChecksumError):
        install_model(plan, reader, options=_options(), free_bytes=ROOM)

    assert (tmp_path / "model-00001-of-00002.gguf").exists()
    assert not (tmp_path / "model-00002-of-00002.gguf").exists()
    assert not (tmp_path / "model-00002-of-00002.gguf.part").exists()
    assert not (tmp_path / "model-00002-of-00002.gguf.part.state").exists()
    assert not (tmp_path / MANIFEST_NAME).exists()


# --- the manifest ------------------------------------------------------------


def test_a_manifest_that_cannot_be_written_is_not_silently_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = DownloadPlan(
        model_id="m",
        model_name="M",
        quant="Q",
        repo="r",
        directory=tmp_path,
        files=(),
        total_bytes=0,
    )

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "write_text", refuse)
    with pytest.raises(DownloadError) as caught:
        write_manifest(plan, ())
    assert "could not write" in caught.value.message


def test_cancelling_stops_the_run_and_writes_no_manifest(tmp_path: Path) -> None:
    shards = _shards(2)
    model = _model(shards)
    stop = threading.Event()
    bodies = _bodies(model, shards)
    hubs = {
        name: FakeHub(data, redirect=False, stop=stop, stop_after=2)
        for name, data in bodies.items()
    }
    plan = build_plan(model, directory=tmp_path)
    progress = RecordingProgress()
    with _reader(bodies, hubs) as reader, pytest.raises(DownloadCancelledError):
        install_model(
            plan,
            reader,
            options=DownloadOptions(workers=1, chunk_bytes=CHUNK, backoff_seconds=0.0),
            reporter=progress,
            stop=stop,
            free_bytes=ROOM,
        )
    assert not (tmp_path / MANIFEST_NAME).exists()
    assert (tmp_path / "model-00001-of-00002.gguf.part.state").exists()


def test_a_file_request_can_be_built_for_a_file_the_catalog_sized(tmp_path: Path) -> None:
    request = FileRequest(
        name="x.gguf",
        repo_path="x.gguf",
        url="https://example.invalid/x.gguf",
        target=tmp_path / "x.gguf",
        size=10,
        sha256=None,
        role="weights",
    )
    assert request.part.name == "x.gguf.part"
    assert request.state.name == "x.gguf.part.state"
    assert request.already_present() is False
