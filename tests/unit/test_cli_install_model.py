"""``llamafit install model`` and ``llamafit install history`` through Typer's runner.

Every test replaces the catalog and the transport by name, the way ``test_cli_catalog.py``
does. Nothing here reaches the network or writes outside ``tmp_path``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner, Result

from llamafit.cli.app import app
from llamafit.download.errors import ChecksumError, DiskSpaceError, DownloadError
from llamafit.download.plan import MANIFEST_NAME
from llamafit.download.transport import HttpRangeReader
from llamafit.errors import CatalogError
from llamafit.models.catalog import Catalog, CatalogModel, Extra, ModelSource, Quant
from tests.fixtures.downloads import FakeHub, blob, sha256
from tests.unit.test_models_catalog import minimal

runner = CliRunner()

CHUNK = 512
SHARD = CHUNK * 4


def _model(shards: list[bytes], extra: bytes | None = None) -> CatalogModel:
    count = len(shards)
    source: dict[str, object] = {
        "repo": "acme/model-gguf",
        "quants": [
            Quant(
                name="Q4_K_M",
                files=[f"model-0000{index + 1}-of-0000{count}.gguf" for index in range(count)],
                bytes=sum(len(data) for data in shards),
                sha256=[sha256(data) for data in shards],
            ),
            Quant(name="Q8_0", files=["model-Q8_0.gguf"], bytes=SHARD, sha256=["b" * 64]),
        ],
    }
    if extra is not None:
        source["extras"] = [
            Extra(role="mmproj", file="mmproj-F16.gguf", bytes=len(extra), sha256=sha256(extra))
        ]
    return minimal(id="acme-model", name="Acme Model", sources=[ModelSource(**source)])  # type: ignore[arg-type]


@pytest.fixture
def shards() -> list[bytes]:
    return [blob(SHARD, seed=index + 1) for index in range(2)]


@pytest.fixture(autouse=True)
def _isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, shards: list[bytes]
) -> Iterator[dict[str, FakeHub]]:
    """A fixed catalog, a fake hub per file, and every path under ``tmp_path``."""
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    model = _model(shards)
    monkeypatch.setattr("llamafit.cli.common.load_catalog", lambda: (Catalog(models=[model]), []))
    hubs = {
        f"model-0000{index + 1}-of-00002.gguf": FakeHub(data, redirect=False)
        for index, data in enumerate(shards)
    }

    def handle(request: httpx.Request) -> httpx.Response:
        return hubs[str(request.url).rsplit("/", 1)[-1]].handle(request)

    def reader() -> HttpRangeReader:
        return HttpRangeReader(httpx.Client(transport=httpx.MockTransport(handle)), piece_bytes=128)

    monkeypatch.setattr("llamafit.cli.install_cmd.HttpRangeReader", reader)
    monkeypatch.setattr(
        "llamafit.cli.install_cmd.DownloadOptions",
        _small_options(),
    )
    yield hubs


def _small_options() -> object:
    """The real options class with a chunk size a test can afford."""
    from llamafit.download.engine import DownloadOptions

    def make(**kwargs: object) -> DownloadOptions:
        kwargs.setdefault("chunk_bytes", CHUNK)
        kwargs.setdefault("backoff_seconds", 0.0)
        return DownloadOptions(**kwargs)  # type: ignore[arg-type]

    return make


def _install(tmp_path: Path, *extra: str) -> Result:
    return runner.invoke(
        app, ["install", "model", "acme-model", "--dir", str(tmp_path / "out"), *extra]
    )


# --- what a reader sees before the first byte --------------------------------


def test_the_plan_is_printed_before_anything_is_fetched(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["install", "model", "acme-model", "--dir", str(tmp_path / "out"), "--dry-run"],
        # A width, so the screen this test reads is the same screen everywhere. Left to the
        # environment, a narrow one shortens a file name inside its cell and the assertion
        # fails over the terminal rather than over the program.
        env={"COLUMNS": "100"},
    )
    assert result.exit_code == 0, result.output
    # Read with the wrapping taken back out. Rich breaks a line wherever the width runs
    # out, and where that falls depends on how long the temporary directory's path is --
    # which is a property of the machine, not of the program. An earlier version guessed
    # which phrases would survive the wrap and picked wrong: on macOS, whose temporary
    # paths are the longest, the break landed inside "would be left" and the build failed
    # over a line ending. What the test is for is that every part of the plan is printed
    # before a byte is fetched.
    output = " ".join(result.output.split())
    assert "Acme Model" in output
    assert "Q4_K_M" in output
    assert "acme/model-gguf" in output
    assert "model-00001-of-00002.gguf" in output
    assert "model-00002-of-00002.gguf" in output
    assert "2 files" in output
    assert "4.0 KiB" in output  # the total the catalog records
    assert "out, which has" in output
    assert "free" in output
    assert "would be left" in output
    assert not (tmp_path / "out").exists()


def test_dry_run_as_json_carries_the_files_the_sizes_and_the_disk(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "install",
            "model",
            "acme-model",
            "--dir",
            str(tmp_path / "out"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["model"] == "acme-model"
    assert payload["quant"] == "Q4_K_M"
    assert payload["total_bytes"] == SHARD * 2
    assert payload["enough_space"] is True
    assert [file["name"] for file in payload["files"]] == [
        "model-00001-of-00002.gguf",
        "model-00002-of-00002.gguf",
    ]
    assert payload["files"][0]["url"].startswith("https://huggingface.co/acme/model-gguf/")


def test_a_quant_that_does_not_exist_stops_with_a_message(tmp_path: Path) -> None:
    result = _install(tmp_path, "--quant", "Q2_K", "--yes")
    assert result.exit_code == 1
    assert isinstance(result.exception, DownloadError)
    rendered = result.exception.render()
    assert "Q2_K" in rendered
    assert "Q4_K_M" in rendered


def test_a_mistyped_model_id_is_the_catalog_s_own_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["install", "model", "acme-modl", "--yes"])
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "acme-modl" in result.exception.render()


# --- the download ------------------------------------------------------------


def test_a_model_is_fetched_verified_and_recorded(tmp_path: Path, shards: list[bytes]) -> None:
    result = _install(tmp_path, "--yes")
    assert result.exit_code == 0, result.output
    out = tmp_path / "out"
    assert (out / "model-00001-of-00002.gguf").read_bytes() == shards[0]
    assert (out / "model-00002-of-00002.gguf").read_bytes() == shards[1]
    assert (out / MANIFEST_NAME).is_file()
    assert "checked against the catalog" in result.output


def test_json_output_says_what_landed_where(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["--json", "install", "model", "acme-model", "--dir", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["fetched_bytes"] == SHARD * 2
    assert all(file["verified"] for file in payload["files"])
    assert payload["manifest"].endswith(MANIFEST_NAME)


def test_a_second_run_says_there_is_nothing_to_do(tmp_path: Path) -> None:
    assert _install(tmp_path, "--yes").exit_code == 0
    result = _install(tmp_path, "--yes")
    assert result.exit_code == 0, result.output
    assert "already here" in result.output


def test_a_bad_rate_is_refused_by_name(tmp_path: Path) -> None:
    result = _install(tmp_path, "--yes", "--limit-rate", "fast")
    assert result.exit_code == 1
    assert isinstance(result.exception, CatalogError)
    assert "--limit-rate" in result.exception.render()


def test_a_rate_that_parses_is_accepted(tmp_path: Path) -> None:
    result = _install(tmp_path, "--yes", "--limit-rate", "100M")
    assert result.exit_code == 0, result.output


# --- the disk ----------------------------------------------------------------


def test_a_volume_with_no_room_stops_the_command_and_names_the_shortfall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shards: list[bytes]
) -> None:
    import shutil

    monkeypatch.setattr(
        "llamafit.download.plan.shutil.disk_usage",
        lambda _path: shutil._ntuple_diskusage(total=10**9, used=10**9 - 1024, free=1024),
    )
    result = _install(tmp_path, "--yes")
    assert result.exit_code == 1
    assert isinstance(result.exception, DiskSpaceError)
    rendered = result.exception.render()
    assert "free" in rendered
    assert "--dir" in rendered
    assert not (tmp_path / "out" / "model-00001-of-00002.gguf").exists()


# --- checksums ---------------------------------------------------------------


def test_a_model_the_catalog_cannot_vouch_for_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shards: list[bytes]
) -> None:
    model = _model(shards)
    model.sources[0].quants[0].sha256 = []
    monkeypatch.setattr("llamafit.cli.common.load_catalog", lambda: (Catalog(models=[model]), []))
    result = _install(tmp_path, "--yes")
    assert result.exit_code == 1
    assert isinstance(result.exception, DownloadError)
    assert "--allow-unverified" in result.exception.render()
    assert not (tmp_path / "out" / "model-00001-of-00002.gguf").exists()


def test_allow_unverified_goes_ahead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shards: list[bytes]
) -> None:
    model = _model(shards)
    model.sources[0].quants[0].sha256 = []
    monkeypatch.setattr("llamafit.cli.common.load_catalog", lambda: (Catalog(models=[model]), []))
    result = _install(tmp_path, "--yes", "--allow-unverified")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "out" / "model-00001-of-00002.gguf").read_bytes() == shards[0]


def test_a_checksum_that_does_not_match_stops_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shards: list[bytes]
) -> None:
    model = _model(shards)
    model.sources[0].quants[0].sha256 = [sha256(shards[0]), "f" * 64]
    monkeypatch.setattr("llamafit.cli.common.load_catalog", lambda: (Catalog(models=[model]), []))
    result = _install(tmp_path, "--yes")
    assert result.exit_code == 1
    assert isinstance(result.exception, ChecksumError)
    assert "checksum" in result.exception.render()
    out = tmp_path / "out"
    assert not (out / "model-00002-of-00002.gguf").exists()
    assert not (out / "model-00002-of-00002.gguf.part").exists()
    assert not (out / MANIFEST_NAME).exists()


# --- the history -------------------------------------------------------------


def test_the_history_is_empty_until_something_is_downloaded() -> None:
    result = runner.invoke(app, ["install", "history"])
    assert result.exit_code == 0, result.output
    assert "Nothing has been downloaded yet." in result.output


def test_a_finished_download_shows_up_in_the_history(tmp_path: Path) -> None:
    _install(tmp_path, "--yes")
    result = runner.invoke(app, ["install", "history"])
    assert result.exit_code == 0, result.output
    assert "acme-model" in result.output
    assert "Q4_K_M" in result.output
    assert "finished" in result.output


def test_the_history_as_json_is_the_records(tmp_path: Path) -> None:
    _install(tmp_path, "--yes")
    result = runner.invoke(app, ["--json", "install", "history"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["model_id"] == "acme-model"
    assert payload[0]["finished"] is True


# --- Ctrl+C ------------------------------------------------------------------


def test_an_interrupt_asks_the_workers_to_stop_rather_than_raising_into_one() -> None:
    # A KeyboardInterrupt raised into whichever thread happened to be running would leave
    # the pool half torn down and the record possibly a chunk behind the file.
    import signal
    import threading

    from llamafit.cli.install_cmd import _StoppedByInterrupt

    stop = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    with _StoppedByInterrupt(stop) as guard:
        assert signal.getsignal(signal.SIGINT) is not previous
        guard._handle(signal.SIGINT, None)
        assert stop.is_set() is True
    assert signal.getsignal(signal.SIGINT) is previous


def test_a_second_interrupt_hands_the_signal_back_to_python() -> None:
    import signal
    import threading

    from llamafit.cli.install_cmd import _StoppedByInterrupt

    stop = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    try:
        with _StoppedByInterrupt(stop) as guard:
            guard._handle(signal.SIGINT, None)
            with pytest.raises(KeyboardInterrupt):
                guard._handle(signal.SIGINT, None)
    finally:
        signal.signal(signal.SIGINT, previous)  # type: ignore[arg-type]


def test_a_stopped_download_says_it_can_be_carried_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llamafit.download.errors import DownloadCancelledError

    def stopped(*_args: object, **_kwargs: object) -> None:
        raise DownloadCancelledError("Stopped.")

    monkeypatch.setattr("llamafit.cli.install_cmd.install_model", stopped)
    result = _install(tmp_path, "--yes")
    assert result.exit_code == 1
    assert "carry on" in result.output
