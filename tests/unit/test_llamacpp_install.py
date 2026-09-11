"""Installing llama.cpp: whose directory it is, verifying before unpacking, resuming.

Nothing here reaches the network — every download goes through ``FakeFetcher`` — and
nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import shutil
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import TracebackType

import httpx
import pytest

from llamafit.errors import NetworkError
from llamafit.hardware.runner import CommandResult, FakeRunner
from llamafit.llamacpp.detect import parse_version, well_known_dirs
from llamafit.llamacpp.install import (
    MARKER_NAME,
    PROFILE_MARK,
    STAGING_NAME,
    VERSION_FILE,
    FakeFetcher,
    HttpFetcher,
    InstallError,
    InstallMarker,
    apply_path_change,
    bin_dir_of,
    check_space,
    download_asset,
    extract_archive,
    free_bytes,
    inspect_target,
    install_release,
    managed_dir,
    part_path,
    payload_dir,
    plan_install,
    plan_path_change,
    read_marker,
    sha256_of,
    version_text,
)
from llamafit.llamacpp.releases import Release, ReleaseAsset
from llamafit.models.host import Backend, OsName
from tests.fixtures import llamacpp_release as fixture

URL = "https://example.invalid/llama-b6100-bin-win-cuda-12.4-x64.zip"


def _asset(data: bytes, name: str = "llama-b6100-bin-win-cuda-12.4-x64.zip") -> ReleaseAsset:
    """An asset whose checksum really is the checksum of ``data``."""
    return ReleaseAsset(
        name=name,
        size=len(data),
        url=f"https://example.invalid/{name}",
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _release(*assets: ReleaseAsset, commit: str | None = "3f9c1a2b7d5e4c6a8b0d") -> Release:
    return Release(tag="b6100", assets=assets, commit=commit)


def _plan(
    tmp_path: Path,
    *,
    assets: tuple[ReleaseAsset, ...],
    os_name: OsName = "windows",
    backend: Backend = "cuda",
    root: Path | None = None,
    free: int | None = None,
):
    """A plan pointing entirely inside ``tmp_path``."""
    main, *companions = assets
    return plan_install(
        _release(*assets),
        main,
        companions=companions,
        backend=backend,
        os_name=os_name,
        arch="x86_64",
        root=root if root is not None else tmp_path / "install",
        cache_dir=tmp_path / "cache",
        free=(lambda _path: free) if free is not None else free_bytes,
    )


# --------------------------------------------------------------------------------------
# Where it goes
# --------------------------------------------------------------------------------------


def test_the_managed_directory_is_the_first_place_the_detector_looks(tmp_path: Path) -> None:
    """What is installed has to be what is detected, with no configuration in between."""
    root = managed_dir(tmp_path)
    assert bin_dir_of(root) == well_known_dirs("windows", tmp_path)[0]
    assert bin_dir_of(root) == well_known_dirs("linux", tmp_path)[0]


# --------------------------------------------------------------------------------------
# Whose directory is this
# --------------------------------------------------------------------------------------


def test_a_directory_that_does_not_exist_is_ours_to_create(tmp_path: Path) -> None:
    target = inspect_target(tmp_path / "nothing-here")
    assert target.ownership == "absent"
    assert target.replaceable


def test_an_empty_directory_is_ours_to_use(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    assert inspect_target(tmp_path / "empty").ownership == "empty"


def test_a_directory_with_our_marker_is_ours(tmp_path: Path) -> None:
    root = tmp_path / "ours"
    root.mkdir()
    (root / MARKER_NAME).write_text(InstallMarker(tag="b6100").to_json(), encoding="utf-8")
    target = inspect_target(root)
    assert target.ownership == "ours"
    assert target.marker is not None
    assert target.marker.tag == "b6100"
    assert target.replaceable


def test_somebody_elses_build_is_refused_and_the_message_says_what_is_there(
    tmp_path: Path,
) -> None:
    root = tmp_path / "theirs"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "llama-server").write_text("hand-built", encoding="utf-8")
    target = inspect_target(root)
    assert target.ownership == "foreign"
    assert not target.replaceable
    assert target.detail is not None
    assert str(root) in target.detail


def test_a_marker_that_cannot_be_read_does_not_count_as_ours(tmp_path: Path) -> None:
    """Unreadable is not absent: the proof is what licenses deleting what is there."""
    root = tmp_path / "corrupt"
    root.mkdir()
    (root / MARKER_NAME).write_text("{not json", encoding="utf-8")
    (root / "bin").mkdir()
    assert inspect_target(root).ownership == "foreign"
    assert read_marker(root) is None


def test_a_marker_naming_another_tool_does_not_count_as_ours(tmp_path: Path) -> None:
    root = tmp_path / "other-tool"
    root.mkdir()
    (root / MARKER_NAME).write_text(json.dumps({"tool": "somethingelse"}), encoding="utf-8")
    (root / "bin").mkdir()
    assert inspect_target(root).ownership == "foreign"


def test_a_file_where_the_directory_should_be_is_foreign(tmp_path: Path) -> None:
    path = tmp_path / "a-file"
    path.write_text("not a directory", encoding="utf-8")
    target = inspect_target(path)
    assert target.ownership == "foreign"
    assert target.detail is not None


def test_the_marker_round_trips_through_json() -> None:
    marker = InstallMarker(
        installed_at="2026-09-10T09:00:00+00:00",
        tag="b6100",
        build=6100,
        commit="abc123",
        asset="llama-b6100-bin-win-cuda-12.4-x64.zip",
        sha256="f" * 64,
        backend="cuda",
        os_name="windows",
        arch="x86_64",
        backends=("cuda", "cpu"),
    )
    again = InstallMarker.from_json(marker.to_json())
    assert again == marker
    assert again is not None
    assert again.ours


# --------------------------------------------------------------------------------------
# Room on the disk
# --------------------------------------------------------------------------------------


def test_free_space_is_read_from_the_nearest_existing_ancestor(tmp_path: Path) -> None:
    assert free_bytes(tmp_path / "not" / "created" / "yet") is not None


def test_a_volume_with_room_passes_and_one_without_does_not(tmp_path: Path) -> None:
    checks = check_space(
        archive_dir=tmp_path,
        install_dir=tmp_path,
        download_bytes=100,
        unpacked_bytes=300,
        free=lambda _path: 200,
    )
    assert [check.purpose for check in checks] == ["download", "install"]
    assert checks[0].enough
    assert not checks[1].enough


def test_unknown_free_space_counts_as_enough(tmp_path: Path) -> None:
    checks = check_space(
        archive_dir=tmp_path,
        install_dir=tmp_path,
        download_bytes=1,
        unpacked_bytes=1,
        free=lambda _path: None,
    )
    assert all(check.enough for check in checks)


# --------------------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------------------


def test_a_download_that_matches_its_checksum_lands_under_the_real_name(
    tmp_path: Path,
) -> None:
    data = b"a llama.cpp release archive" * 10
    asset = _asset(data)
    fetcher = FakeFetcher(bodies={asset.url: data})
    destination = tmp_path / "archive.zip"
    assert download_asset(asset, destination, fetcher=fetcher) == destination
    assert destination.read_bytes() == data
    assert not part_path(destination).exists()


def test_progress_is_reported_with_the_total_when_the_server_gives_one(
    tmp_path: Path,
) -> None:
    data = b"0123456789" * 4
    asset = _asset(data)
    seen: list[tuple[int, int | None]] = []
    download_asset(
        asset,
        tmp_path / "a.zip",
        fetcher=FakeFetcher(bodies={asset.url: data}, chunk=10),
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (len(data), len(data))
    assert [done for done, _total in seen] == [10, 20, 30, 40]


def test_a_dropped_connection_resumes_from_what_is_already_on_disk(tmp_path: Path) -> None:
    data = bytes(range(200))
    asset = _asset(data)
    fetcher = FakeFetcher(bodies={asset.url: data}, chunk=16, stop_after=48)
    destination = tmp_path / "archive.zip"
    download_asset(asset, destination, fetcher=fetcher)
    assert destination.read_bytes() == data
    # Two requests: the first from zero, the second continuing from where it stopped.
    assert fetcher.calls == [(asset.url, 0), (asset.url, 48)]


def test_a_download_gives_up_after_the_attempts_are_spent(tmp_path: Path) -> None:
    data = bytes(range(100))
    asset = _asset(data)

    class _AlwaysDrops(FakeFetcher):
        def _chunks(self, url: str, chunk_data: bytes):
            self.stop_after = 8
            return super()._chunks(url, chunk_data)

    fetcher = _AlwaysDrops(bodies={asset.url: data}, chunk=8, stop_after=8)
    with pytest.raises(NetworkError, match="stopped early"):
        download_asset(asset, tmp_path / "a.zip", fetcher=fetcher, attempts=2)
    assert len(fetcher.calls) == 2


def test_a_server_that_ignores_the_range_restarts_instead_of_doubling_the_file(
    tmp_path: Path,
) -> None:
    """Appending a fresh whole body to a partial file would build a file that is neither."""
    data = bytes(range(120))
    asset = _asset(data)
    destination = tmp_path / "archive.zip"
    part_path(destination).write_bytes(data[:40])
    fetcher = FakeFetcher(bodies={asset.url: data}, chunk=40, honours_range=False)
    download_asset(asset, destination, fetcher=fetcher)
    assert destination.read_bytes() == data


def test_a_part_file_longer_than_the_asset_is_discarded(tmp_path: Path) -> None:
    data = bytes(range(50))
    asset = _asset(data)
    destination = tmp_path / "archive.zip"
    part_path(destination).write_bytes(b"x" * 500)
    download_asset(asset, destination, fetcher=FakeFetcher(bodies={asset.url: data}))
    assert destination.read_bytes() == data


def test_a_checksum_that_does_not_match_discards_the_file_and_unpacks_nothing(
    tmp_path: Path,
) -> None:
    data = b"the archive that arrived"
    asset = ReleaseAsset(name="a.zip", size=len(data), url=URL, sha256="0" * 64)
    destination = tmp_path / "a.zip"
    with pytest.raises(InstallError, match="did not match its published checksum") as caught:
        download_asset(asset, destination, fetcher=FakeFetcher(bodies={URL: data}))
    assert not destination.exists()
    assert not part_path(destination).exists()
    assert caught.value.hint is not None
    assert "Nothing was unpacked" in caught.value.hint


def test_a_truncated_body_is_reported_rather_than_verified(tmp_path: Path) -> None:
    """A short file would fail its checksum anyway; saying it arrived short is clearer."""
    data = b"0123456789"
    asset = ReleaseAsset(name="a.zip", size=100, url=URL, sha256=hashlib.sha256(data).hexdigest())
    with pytest.raises(NetworkError, match="stopped early"):
        download_asset(asset, tmp_path / "a.zip", fetcher=FakeFetcher(bodies={URL: data}))


def test_an_asset_with_no_published_checksum_is_refused_by_default(tmp_path: Path) -> None:
    asset = ReleaseAsset(name="a.zip", size=4, url=URL, sha256=None)
    with pytest.raises(InstallError, match="no published checksum") as caught:
        download_asset(asset, tmp_path / "a.zip", fetcher=FakeFetcher(bodies={URL: b"data"}))
    assert caught.value.hint is not None
    assert "--allow-unverified" in caught.value.hint


def test_an_unverified_asset_can_be_taken_when_the_caller_says_so_out_loud(
    tmp_path: Path,
) -> None:
    asset = ReleaseAsset(name="a.zip", size=4, url=URL, sha256=None)
    destination = tmp_path / "a.zip"
    download_asset(
        asset,
        destination,
        fetcher=FakeFetcher(bodies={URL: b"data"}),
        require_checksum=False,
    )
    assert destination.read_bytes() == b"data"


def test_an_archive_already_downloaded_and_verified_is_not_downloaded_again(
    tmp_path: Path,
) -> None:
    data = b"already here"
    asset = _asset(data)
    destination = tmp_path / "a.zip"
    destination.write_bytes(data)
    fetcher = FakeFetcher(bodies={asset.url: data})
    download_asset(asset, destination, fetcher=fetcher)
    assert fetcher.calls == []


def test_an_archive_on_disk_that_is_not_the_one_wanted_is_downloaded_again(
    tmp_path: Path,
) -> None:
    data = b"the right bytes"
    asset = _asset(data)
    destination = tmp_path / "a.zip"
    destination.write_bytes(b"the wrong bytes")
    download_asset(asset, destination, fetcher=FakeFetcher(bodies={asset.url: data}))
    assert destination.read_bytes() == data


def test_a_full_disk_is_named_as_a_full_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FullDisk:
        def __enter__(self) -> _FullDisk:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
        ) -> None:
            return None

        def write(self, data: bytes) -> int:
            raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(Path, "open", lambda self, *a, **k: _FullDisk())
    data = b"never lands"
    asset = _asset(data)
    with pytest.raises(InstallError, match="is full") as caught:
        download_asset(asset, tmp_path / "a.zip", fetcher=FakeFetcher(bodies={asset.url: data}))
    assert caught.value.hint is not None
    assert "resumes" in caught.value.hint


def test_a_url_that_is_not_served_is_a_network_error(tmp_path: Path) -> None:
    asset = _asset(b"x")
    with pytest.raises(NetworkError):
        download_asset(asset, tmp_path / "a.zip", fetcher=FakeFetcher(bodies={}), attempts=1)


def test_the_checksum_helper_reads_a_file_in_pieces(tmp_path: Path) -> None:
    path = tmp_path / "big"
    path.write_bytes(b"z" * 5000)
    assert sha256_of(path) == hashlib.sha256(b"z" * 5000).hexdigest()


# --------------------------------------------------------------------------------------
# Unpacking
# --------------------------------------------------------------------------------------


def test_a_flat_archive_unpacks_and_its_payload_is_its_root(tmp_path: Path) -> None:
    archive = fixture.write(tmp_path / "win.zip", fixture.windows_build_zip())
    unpacked = extract_archive(archive, tmp_path / "out")
    assert (unpacked / "llama-server.exe").is_file()
    assert payload_dir(unpacked, "windows") == unpacked


def test_a_nested_archive_unpacks_and_its_payload_is_the_directory_holding_the_server(
    tmp_path: Path,
) -> None:
    archive = fixture.write(tmp_path / "linux.zip", fixture.linux_build_zip())
    unpacked = extract_archive(archive, tmp_path / "out")
    assert payload_dir(unpacked, "linux") == unpacked / "build" / "bin"


@pytest.mark.skipif(os.name == "nt", reason="Windows has no executable bit to restore")
def test_the_executable_bit_survives_the_zip(tmp_path: Path) -> None:
    archive = fixture.write(tmp_path / "linux.zip", fixture.linux_build_zip())
    unpacked = extract_archive(archive, tmp_path / "out")
    server = unpacked / "build" / "bin" / "llama-server"
    assert os.access(server, os.X_OK)


def test_a_member_that_would_escape_the_directory_stops_the_install(tmp_path: Path) -> None:
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escaped.txt", b"gotcha")
    with pytest.raises(InstallError, match="written outside"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_an_archive_that_is_not_an_archive_says_so(tmp_path: Path) -> None:
    archive = tmp_path / "broken.zip"
    archive.write_bytes(b"this is not a zip file at all")
    with pytest.raises(InstallError, match="could not be unpacked"):
        extract_archive(archive, tmp_path / "out")


def test_an_unknown_archive_kind_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "build.rar"
    archive.write_bytes(b"whatever")
    with pytest.raises(InstallError, match="not an archive"):
        extract_archive(archive, tmp_path / "out")


def test_an_archive_without_a_server_is_not_a_llama_cpp_build(tmp_path: Path) -> None:
    archive = fixture.write(tmp_path / "no-server.zip", fixture.build_zip({"README.md": b"hi"}))
    unpacked = extract_archive(archive, tmp_path / "out")
    with pytest.raises(InstallError, match="no llama-server"):
        payload_dir(unpacked, "linux")


# --------------------------------------------------------------------------------------
# VERSION.txt, which the detector has to be able to read
# --------------------------------------------------------------------------------------


def test_the_version_file_is_written_in_the_form_the_detector_parses() -> None:
    text = version_text(_release(commit="3f9c1a2b7d5e4c6a8b0d"))
    assert parse_version(text) == (6100, "3f9c1a2b7d5e")


def test_the_version_file_still_carries_the_build_when_no_commit_is_known() -> None:
    build, commit = parse_version(version_text(_release(commit=None)))
    assert build == 6100
    assert commit is None


# --------------------------------------------------------------------------------------
# The whole install
# --------------------------------------------------------------------------------------


def _windows_install(tmp_path: Path) -> tuple[FakeFetcher, object]:
    build = fixture.windows_build_zip()
    cudart = fixture.cudart_zip()
    main = _asset(build)
    extra = _asset(cudart, name="cudart-llama-bin-win-cuda-12.4-x64.zip")
    fetcher = FakeFetcher(bodies={main.url: build, extra.url: cudart}, chunk=64)
    return fetcher, _plan(tmp_path, assets=(main, extra))


def test_an_install_lands_where_the_detector_will_find_it(tmp_path: Path) -> None:
    fetcher, plan = _windows_install(tmp_path)
    result = install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    assert (result.bin_dir / "llama-server.exe").is_file()
    assert (result.bin_dir / "ggml-cuda.dll").is_file()
    assert (result.bin_dir / VERSION_FILE).is_file()
    assert result.backends == ("cuda", "cpu")
    assert result.warnings == ()


def test_the_cuda_runtime_is_unpacked_beside_the_binaries_not_in_its_own_tree(
    tmp_path: Path,
) -> None:
    """A CUDA build without these DLLs beside it exits before it prints a version."""
    fetcher, plan = _windows_install(tmp_path)
    result = install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    assert (result.bin_dir / "cudart64_12.dll").is_file()
    assert (result.bin_dir / "cublas64_12.dll").is_file()


def test_the_marker_records_what_was_installed(tmp_path: Path) -> None:
    fetcher, plan = _windows_install(tmp_path)
    result = install_release(fetcher=fetcher, plan=plan)  # type: ignore[arg-type]
    marker = read_marker(result.root)
    assert marker is not None
    assert marker.tag == "b6100"
    assert marker.build == 6100
    assert marker.backend == "cuda"
    assert marker.backends == ("cuda", "cpu")
    assert marker.installed_at


def test_the_staging_directory_does_not_survive_a_finished_install(tmp_path: Path) -> None:
    fetcher, plan = _windows_install(tmp_path)
    result = install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    assert not (result.root / STAGING_NAME).exists()


def test_a_second_install_replaces_our_own_and_says_which_one(tmp_path: Path) -> None:
    fetcher, plan = _windows_install(tmp_path)
    install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    fetcher, plan = _windows_install(tmp_path)
    result = install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    assert result.replaced is not None
    assert result.replaced.tag == "b6100"
    assert (result.bin_dir / "llama-server.exe").is_file()


def test_a_directory_llamafit_did_not_create_is_never_overwritten(tmp_path: Path) -> None:
    root = tmp_path / "install"
    (root / "bin").mkdir(parents=True)
    handmade = root / "bin" / "llama-server.exe"
    handmade.write_text("my own build, with my own flags", encoding="utf-8")
    fetcher, plan = _windows_install(tmp_path)
    with pytest.raises(InstallError) as caught:
        install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    assert caught.value.hint is not None
    assert "--force" in caught.value.hint
    assert handmade.read_text(encoding="utf-8") == "my own build, with my own flags"


def test_force_is_what_replaces_somebody_elses_build_and_nothing_less(
    tmp_path: Path,
) -> None:
    root = tmp_path / "install"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "llama-server.exe").write_text("hand-built", encoding="utf-8")
    fetcher, plan = _windows_install(tmp_path)
    result = install_release(plan, fetcher=fetcher, force=True)  # type: ignore[arg-type]
    assert (result.bin_dir / "llama-server.exe").read_bytes() == b"server binary"
    assert read_marker(root) is not None


def test_an_install_that_does_not_fit_is_refused_before_anything_is_downloaded(
    tmp_path: Path,
) -> None:
    build = fixture.windows_build_zip()
    main = _asset(build)
    fetcher = FakeFetcher(bodies={main.url: build})
    plan = _plan(tmp_path, assets=(main,), free=10)
    with pytest.raises(InstallError, match="free"):
        install_release(plan, fetcher=fetcher)
    assert fetcher.calls == []
    assert not (tmp_path / "install").exists()


def test_a_build_missing_its_backend_libraries_is_installed_but_reported(
    tmp_path: Path,
) -> None:
    """What is installed has to be what the detector finds, and this one is not."""
    data = fixture.build_zip({"llama-server.exe": b"server binary"})
    main = _asset(data)
    plan = _plan(tmp_path, assets=(main,))
    result = install_release(plan, fetcher=FakeFetcher(bodies={main.url: data}))
    assert result.backends == ()
    assert any("cuda" in warning for warning in result.warnings)
    assert any("static" in warning for warning in result.warnings)


def test_a_cpu_install_with_backend_libraries_warns_about_nothing(tmp_path: Path) -> None:
    data = fixture.build_zip(
        {"llama-server": b"server binary", "libggml-cpu.so": b"cpu backend"},
        executable=("llama-server",),
    )
    main = _asset(data, name="llama-b6100-bin-ubuntu-x64.zip")
    plan = _plan(tmp_path, assets=(main,), os_name="linux", backend="cpu")
    result = install_release(plan, fetcher=FakeFetcher(bodies={main.url: data}))
    assert result.backends == ("cpu",)
    assert result.warnings == ()


def test_a_failed_swap_puts_the_previous_install_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher, plan = _windows_install(tmp_path)
    first = install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    (first.bin_dir / "keepsake.txt").write_text("still here", encoding="utf-8")

    real_move = shutil.move

    def _fail_on_the_final_swap(src: str, dst: str) -> str:
        # Only the move into the live directory; the staging moves have to still work.
        if Path(dst) == first.bin_dir:
            raise OSError(errno.EACCES, "denied")
        return str(real_move(src, dst))

    monkeypatch.setattr(shutil, "move", _fail_on_the_final_swap)
    fetcher, plan = _windows_install(tmp_path)
    with pytest.raises(InstallError):
        install_release(plan, fetcher=fetcher)  # type: ignore[arg-type]
    assert (first.bin_dir / "keepsake.txt").read_text(encoding="utf-8") == "still here"


def test_the_plan_says_what_is_about_to_happen_without_writing_anything(
    tmp_path: Path,
) -> None:
    fetcher, plan = _windows_install(tmp_path)
    assert plan.release.tag == "b6100"
    assert plan.backend == "cuda"
    assert plan.verified
    assert plan.download_bytes == plan.asset.size + plan.companions[0].size
    assert plan.target.ownership == "absent"
    assert plan.replaces is None
    assert plan.short_space == ()
    assert not (tmp_path / "install").exists()
    assert not (tmp_path / "cache").exists()
    assert fetcher.calls == []


def test_the_plan_counts_a_part_file_as_bytes_already_on_disk(tmp_path: Path) -> None:
    data = fixture.windows_build_zip()
    main = _asset(data)
    plan = _plan(tmp_path, assets=(main,))
    part_path(plan.archive_path).parent.mkdir(parents=True)
    part_path(plan.archive_path).write_bytes(data[:100])
    again = _plan(tmp_path, assets=(main,))
    assert again.resume_bytes == 100
    assert again.download_bytes == main.size - 100


def test_a_plan_for_an_unverifiable_archive_says_so(tmp_path: Path) -> None:
    plan = _plan(tmp_path, assets=(ReleaseAsset("a.zip", 10, URL, sha256=None),))
    assert not plan.verified


# --------------------------------------------------------------------------------------
# The user's PATH
# --------------------------------------------------------------------------------------

_REG_QUERY = "reg query HKCU\\Environment /v Path"
_REG_OUTPUT = (
    "\r\nHKEY_CURRENT_USER\\Environment\r\n    Path    REG_EXPAND_SZ    C:\\Tools;C:\\Bin\r\n"
)


def test_the_windows_change_is_offered_with_its_undo_in_the_same_breath(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(responses={_REG_QUERY: _REG_OUTPUT})
    change = plan_path_change(
        Path("C:/llamafit/bin"), os_name="windows", runner=runner, home=tmp_path, env={}
    )
    assert change.kind == "registry"
    assert change.previous == "C:\\Tools;C:\\Bin"
    assert "C:\\Tools;C:\\Bin" in change.undo
    assert change.needed


def test_a_directory_already_on_the_path_is_left_alone(tmp_path: Path) -> None:
    runner = FakeRunner(responses={_REG_QUERY: _REG_OUTPUT})
    change = plan_path_change(
        Path("C:/Tools"), os_name="windows", runner=runner, home=tmp_path, env={}
    )
    assert change.kind == "present"
    assert not change.needed


def test_a_path_that_cannot_be_read_is_not_guessed_at(tmp_path: Path) -> None:
    runner = FakeRunner(responses={})
    change = plan_path_change(
        Path("C:/llamafit/bin"), os_name="windows", runner=runner, home=tmp_path, env={}
    )
    assert change.kind == "unsupported"
    assert not change.needed


def test_applying_the_windows_change_writes_the_user_scope_registry_value(
    tmp_path: Path,
) -> None:
    runner = FakeRunner(responses={_REG_QUERY: _REG_OUTPUT, "reg": CommandResult([], 0, "", "", 1)})
    added = Path("C:/llamafit/bin")
    change = plan_path_change(added, os_name="windows", runner=runner, home=tmp_path, env={})
    apply_path_change(change, runner=runner)
    written = runner.calls[-1]
    assert written[:3] == ["reg", "add", "HKCU\\Environment"]
    # The directory is appended as the caller spelled it. Hard-coding the separator here
    # asserted that the host builds Windows paths, which is true of Windows and of nothing
    # else; what this test is for is that the existing PATH is kept and the new entry put
    # after it.
    assert f"C:\\Tools;C:\\Bin;{added}" in written


def test_a_registry_write_that_fails_says_so_and_names_the_command(tmp_path: Path) -> None:
    runner = FakeRunner(
        responses={
            _REG_QUERY: _REG_OUTPUT,
            "reg": CommandResult([], 1, "", "Access is denied.", 1),
        }
    )
    change = plan_path_change(
        Path("C:/llamafit/bin"), os_name="windows", runner=runner, home=tmp_path, env={}
    )
    with pytest.raises(InstallError, match="PATH could not be changed") as caught:
        apply_path_change(change, runner=runner)
    assert caught.value.command is not None
    assert caught.value.command.startswith("reg add")


def test_the_posix_change_names_the_profile_the_shell_reads(tmp_path: Path) -> None:
    (tmp_path / ".zshrc").write_text("# my shell\n", encoding="utf-8")
    change = plan_path_change(
        tmp_path / "bin",
        os_name="linux",
        runner=FakeRunner(responses={}),
        home=tmp_path,
        env={"SHELL": "/usr/bin/zsh"},
    )
    assert change.kind == "profile"
    assert change.where == str(tmp_path / ".zshrc")
    assert PROFILE_MARK in change.undo


def test_applying_the_posix_change_appends_a_labelled_block(tmp_path: Path) -> None:
    profile = tmp_path / ".profile"
    profile.write_text("# existing\n", encoding="utf-8")
    change = plan_path_change(
        tmp_path / "bin",
        os_name="linux",
        runner=FakeRunner(responses={}),
        home=tmp_path,
        env={},
    )
    apply_path_change(change, runner=FakeRunner(responses={}))
    text = profile.read_text(encoding="utf-8")
    assert text.startswith("# existing\n")
    assert PROFILE_MARK in text
    assert str(tmp_path / "bin") in text


def test_a_profile_that_already_mentions_the_directory_is_not_appended_to_twice(
    tmp_path: Path,
) -> None:
    profile = tmp_path / ".profile"
    profile.write_text(f'export PATH="{tmp_path / "bin"}:$PATH"\n', encoding="utf-8")
    change = plan_path_change(
        tmp_path / "bin",
        os_name="linux",
        runner=FakeRunner(responses={}),
        home=tmp_path,
        env={},
    )
    assert change.kind == "present"
    apply_path_change(change, runner=FakeRunner(responses={}))
    assert profile.read_text(encoding="utf-8").count("export PATH") == 1


# --------------------------------------------------------------------------------------
# The real HTTP downloader, against a mock transport rather than a network
# --------------------------------------------------------------------------------------


def _http(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serving(data: bytes, seen: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    """A transport that honours ``Range`` the way GitHub's asset host does."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        raw = request.headers.get("range")
        if raw is None:
            return httpx.Response(200, content=data, headers={"content-length": str(len(data))})
        start = int(raw.removeprefix("bytes=").rstrip("-"))
        return httpx.Response(
            206,
            content=data[start:],
            headers={"content-range": f"bytes {start}-{len(data) - 1}/{len(data)}"},
        )

    return handler


def test_the_http_fetcher_reads_a_whole_body_and_reports_its_size() -> None:
    data = b"an archive" * 20
    seen: list[httpx.Request] = []
    with _http(_serving(data, seen)) as client:
        fetcher = HttpFetcher(client=client)
        with fetcher.open_range(URL) as body:
            assert b"".join(body.chunks) == data
            assert body.total == len(data)
            assert not body.resumed
    assert "range" not in seen[0].headers


def test_the_http_fetcher_asks_for_a_range_and_says_when_it_was_honoured() -> None:
    data = bytes(range(120))
    seen: list[httpx.Request] = []
    with _http(_serving(data, seen)) as client:
        fetcher = HttpFetcher(client=client)
        with fetcher.open_range(URL, start=40) as body:
            assert b"".join(body.chunks) == data[40:]
            assert body.resumed
            assert body.total == len(data)
    assert seen[0].headers["range"] == "bytes=40-"


def test_a_real_resume_puts_the_two_halves_back_together(tmp_path: Path) -> None:
    """The resume logic and the real HTTP client, exercised together, off the network."""
    data = bytes(range(200)) * 3
    asset = ReleaseAsset(
        name="a.zip", size=len(data), url=URL, sha256=hashlib.sha256(data).hexdigest()
    )
    destination = tmp_path / "a.zip"
    part_path(destination).write_bytes(data[:150])
    seen: list[httpx.Request] = []
    with _http(_serving(data, seen)) as client:
        download_asset(asset, destination, fetcher=HttpFetcher(client=client))
    assert destination.read_bytes() == data
    assert seen[0].headers["range"] == "bytes=150-"


def test_a_status_the_fetcher_cannot_use_is_a_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"gone")

    with _http(handler) as client:
        opened = HttpFetcher(client=client).open_range(URL)
        with pytest.raises(NetworkError, match="HTTP 404"), opened:
            pass


def test_a_transport_failure_is_a_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with _http(handler) as client:
        opened = HttpFetcher(client=client).open_range(URL)
        with pytest.raises(NetworkError, match="could not download"), opened:
            pass


def test_a_connection_that_dies_mid_body_is_a_network_error_not_an_httpx_one() -> None:
    def dying() -> object:
        yield b"the first part"
        raise httpx.ReadError("connection reset")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=dying())

    with _http(handler) as client:
        opened = HttpFetcher(client=client).open_range(URL)
        with pytest.raises(NetworkError, match="stopped early"), opened as body:
            list(body.chunks)


def test_the_fetcher_closes_only_a_client_it_created() -> None:
    borrowed = _http(lambda request: httpx.Response(200, content=b""))
    HttpFetcher(client=borrowed).close()
    assert not borrowed.is_closed
    borrowed.close()
    own = HttpFetcher()
    own.close()


# --------------------------------------------------------------------------------------
# Tar archives, for a release that ever publishes one
# --------------------------------------------------------------------------------------


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as bundle:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    return path


def test_a_tar_archive_unpacks_like_a_zip(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "build.tar.gz", {"bin/llama-server": b"server binary"})
    unpacked = extract_archive(archive, tmp_path / "out")
    assert payload_dir(unpacked, "linux") == unpacked / "bin"


def test_a_tar_member_that_would_escape_the_directory_stops_the_install(
    tmp_path: Path,
) -> None:
    archive = _tar(tmp_path / "evil.tar.gz", {"../escaped.txt": b"gotcha"})
    with pytest.raises(InstallError, match="written outside"):
        extract_archive(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_a_tar_symlink_is_refused_rather_than_followed(tmp_path: Path) -> None:
    archive = tmp_path / "linked.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        link = tarfile.TarInfo("llama-server")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        bundle.addfile(link)
    with pytest.raises(InstallError, match="written outside"):
        extract_archive(archive, tmp_path / "out")
