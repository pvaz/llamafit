"""``llamafit install llama.cpp`` at the command line.

Every test replaces the GitHub client and the downloader by name, the way
``test_cli_catalog.py`` replaces the catalog loader, so nothing here reaches the network
and nothing is written outside ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from llamafit.cli import install_cmd
from llamafit.cli.app import app
from llamafit.errors import ConfigError, LlamaFitError, NetworkError
from llamafit.llamacpp.install import (
    MARKER_NAME,
    PROFILE_MARK,
    FakeFetcher,
    InstallError,
    read_marker,
)
from llamafit.llamacpp.releases import FakeReleaseClient, Release, ReleaseAsset
from llamafit.models.host import Gpu, OsName, Vendor
from tests.fixtures import llamacpp_release as fixture

runner = CliRunner()

WINDOWS_BUILD = fixture.windows_build_zip()
CUDART = fixture.cudart_zip()
LINUX_BUILD = fixture.linux_build_zip()


def _asset(name: str, data: bytes) -> ReleaseAsset:
    return ReleaseAsset(
        name=name,
        size=len(data),
        url=f"https://example.invalid/{name}",
        sha256=hashlib.sha256(data).hexdigest(),
    )


WIN_ASSET = _asset("llama-b10892-bin-win-cuda-12.4-x64.zip", WINDOWS_BUILD)
WIN_CUDA13_ASSET = _asset("llama-b10892-bin-win-cuda-13.3-x64.zip", WINDOWS_BUILD)
CUDART_ASSET = _asset("cudart-llama-bin-win-cuda-12.4-x64.zip", CUDART)
CUDART13_ASSET = _asset("cudart-llama-bin-win-cuda-13.3-x64.zip", CUDART)
LINUX_ASSET = _asset("llama-b10892-bin-ubuntu-x64.zip", LINUX_BUILD)

RELEASE = Release(
    tag="b10892",
    assets=(WIN_ASSET, WIN_CUDA13_ASSET, CUDART_ASSET, CUDART13_ASSET, LINUX_ASSET),
    url="https://github.com/ggml-org/llama.cpp/releases/tag/b10892",
    commit="3f9c1a2b7d5e4c6a8b0d",
)

BODIES = {
    WIN_ASSET.url: WINDOWS_BUILD,
    WIN_CUDA13_ASSET.url: WINDOWS_BUILD,
    CUDART_ASSET.url: CUDART,
    CUDART13_ASSET.url: CUDART,
    LINUX_ASSET.url: LINUX_BUILD,
}


@pytest.fixture(autouse=True)
def _no_network_and_no_machine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Substitute the release API, the downloader, the machine and the directories."""
    client = FakeReleaseClient(releases={"b10892": RELEASE}, latest_tag="b10892")
    monkeypatch.setattr(install_cmd, "HttpReleaseClient", lambda **_kwargs: client)
    monkeypatch.setattr(
        install_cmd, "HttpFetcher", lambda **_kwargs: FakeFetcher(bodies=BODIES, chunk=64)
    )
    monkeypatch.setattr(install_cmd, "current_os", lambda: "windows")
    monkeypatch.setattr(install_cmd, "current_arch", lambda: "x86_64")
    _set_gpu(monkeypatch, "nvidia")
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "llamafit-home"))


def _set_gpu(
    monkeypatch: pytest.MonkeyPatch, vendor: Vendor | None, driver: str | None = None
) -> None:
    gpus = (
        []
        if vendor is None
        else [
            Gpu(
                index=0,
                vendor=vendor,
                name="Test Card",
                vram_total_bytes=8 * 1024**3,
                driver=driver,
            )
        ]
    )
    monkeypatch.setattr(install_cmd, "detect_gpus", lambda _runner, _os: (gpus, []))


def _rendered(result: Any) -> str:
    """The error a real run would print. CliRunner does not run `main`'s renderer."""
    assert isinstance(result.exception, LlamaFitError), result.exception
    return result.exception.render()


def _invoke(tmp_path: Path, *args: str, **kwargs: Any):
    root = tmp_path / "install"
    return runner.invoke(app, [*args, "--dir", str(root)], **kwargs), root


def test_a_dry_run_prints_the_plan_and_writes_nothing(tmp_path: Path) -> None:
    result, root = _invoke(tmp_path, "install", "llama.cpp", "--dry-run")
    assert result.exit_code == 0, result.output
    assert "b10892" in result.output
    assert "cuda" in result.output
    assert "llama-b10892-bin-win-cuda-12.4-x64.zip" in result.output
    assert "will be created" in result.output
    assert not root.exists()


def test_the_plan_names_the_companion_archive_a_cuda_build_cannot_run_without(
    tmp_path: Path,
) -> None:
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--dry-run")
    assert "cudart-llama-bin-win-cuda-12.4-x64.zip" in result.output


def test_the_plan_says_the_checksum_will_be_checked_before_anything_is_unpacked(
    tmp_path: Path,
) -> None:
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--dry-run")
    assert "before anything is unpacked" in result.output


def test_json_without_yes_is_a_dry_run_because_nobody_is_watching_the_prompt(
    tmp_path: Path,
) -> None:
    result, root = _invoke(tmp_path, "--json", "install", "llama.cpp")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["applied"] is False
    assert payload["tag"] == "b10892"
    assert payload["backend"] == "cuda"
    assert payload["asset"] == "llama-b10892-bin-win-cuda-12.4-x64.zip"
    assert payload["verified"] is True
    assert payload["ownership"] == "absent"
    assert [c["name"] for c in payload["companions"]] == ["cudart-llama-bin-win-cuda-12.4-x64.zip"]
    assert not root.exists()


def test_yes_installs_and_says_where_and_what_was_detected_in_it(tmp_path: Path) -> None:
    result, root = _invoke(tmp_path, "install", "llama.cpp", "--yes")
    assert result.exit_code == 0, result.output
    assert (root / "bin" / "llama-server.exe").is_file()
    assert (root / "bin" / "cudart64_12.dll").is_file()
    assert (root / MARKER_NAME).is_file()
    assert "Installed llama.cpp b10892" in result.output
    assert "cuda" in result.output
    assert "llamafit doctor" in result.output


def test_the_json_result_reports_the_install_and_what_was_detected(tmp_path: Path) -> None:
    result, root = _invoke(tmp_path, "--json", "install", "llama.cpp", "--yes")
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["installed"] is True
    assert payload["backends"] == ["cuda", "cpu"]
    assert payload["bin_dir"] == str(root / "bin")
    assert payload["path_change"] is None


def test_answering_no_writes_nothing(tmp_path: Path) -> None:
    result, root = _invoke(tmp_path, "install", "llama.cpp", input="n\n")
    assert result.exit_code == 1
    assert "Nothing was written" in result.output
    assert not root.exists()


def test_answering_yes_at_the_prompt_installs(tmp_path: Path) -> None:
    result, root = _invoke(tmp_path, "install", "llama.cpp", input="y\n")
    assert result.exit_code == 0, result.output
    assert (root / "bin" / "llama-server.exe").is_file()


def test_a_directory_llamafit_did_not_create_stops_the_command(tmp_path: Path) -> None:
    root = tmp_path / "install"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "llama-server.exe").write_text("my own build", encoding="utf-8")
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--yes")
    assert result.exit_code == 1
    assert isinstance(result.exception, InstallError)
    assert "--force" in _rendered(result)
    assert (root / "bin" / "llama-server.exe").read_text(encoding="utf-8") == "my own build"


def test_the_plan_shows_a_foreign_directory_before_anything_is_downloaded(
    tmp_path: Path,
) -> None:
    root = tmp_path / "install"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "llama-server.exe").write_text("my own build", encoding="utf-8")
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--dry-run")
    assert result.exit_code == 0
    assert "did not install" in result.output


def test_force_replaces_it(tmp_path: Path) -> None:
    root = tmp_path / "install"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "llama-server.exe").write_text("my own build", encoding="utf-8")
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--yes", "--force")
    assert result.exit_code == 0, result.output
    assert (root / "bin" / "llama-server.exe").read_bytes() == b"server binary"


def test_a_second_install_over_our_own_says_what_it_replaced(tmp_path: Path) -> None:
    _invoke(tmp_path, "install", "llama.cpp", "--yes")
    result, root = _invoke(tmp_path, "install", "llama.cpp", "--yes")
    assert result.exit_code == 0, result.output
    assert "Replaced b10892" in result.output
    marker = read_marker(root)
    assert marker is not None
    assert marker.tag == "b10892"


def test_an_unknown_backend_is_refused_by_name_and_the_valid_ones_are_listed(
    tmp_path: Path,
) -> None:
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--backend", "cudah")
    assert result.exit_code == 1
    assert isinstance(result.exception, ConfigError)
    rendered = _rendered(result)
    assert "'cudah'" in rendered
    assert "vulkan" in rendered


def test_a_backend_with_no_published_build_is_reported_not_quietly_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--backend", "sycl", "--yes")
    assert result.exit_code == 1
    rendered = _rendered(result)
    assert "sycl" in rendered
    assert "--backend" in rendered


def test_a_machine_the_release_publishes_nothing_for_is_told_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_cmd, "current_arch", lambda: "other")
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--yes")
    assert result.exit_code == 1
    assert "publishes nothing" in _rendered(result)


def test_a_named_tag_is_the_one_asked_for(tmp_path: Path) -> None:
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--tag", "b9999", "--dry-run")
    assert result.exit_code == 1
    assert isinstance(result.exception, NetworkError)
    assert "no release" in _rendered(result)


def test_a_machine_with_no_gpu_gets_the_cpu_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_cmd, "current_os", lambda: "linux")
    _set_gpu(monkeypatch, None)
    result, _root = _invoke(tmp_path, "--json", "install", "llama.cpp")
    payload = json.loads(result.output)
    assert payload["backend"] == "cpu"
    assert payload["asset"] == "llama-b10892-bin-ubuntu-x64.zip"
    assert payload["companions"] == []


def test_a_driver_new_enough_chooses_the_newer_cuda_archive_and_its_own_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA 13 build will not start below driver 580, so the driver is what decides."""
    _set_gpu(monkeypatch, "nvidia", driver="610.88")
    result, _root = _invoke(tmp_path, "--json", "install", "llama.cpp")
    payload = json.loads(result.output)
    assert payload["asset"] == "llama-b10892-bin-win-cuda-13.3-x64.zip"
    assert [c["name"] for c in payload["companions"]] == ["cudart-llama-bin-win-cuda-13.3-x64.zip"]


def test_an_older_driver_is_given_the_cuda_archive_it_can_actually_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_gpu(monkeypatch, "nvidia", driver="551.23")
    result, _root = _invoke(tmp_path, "--json", "install", "llama.cpp")
    payload = json.loads(result.output)
    assert payload["asset"] == "llama-b10892-bin-win-cuda-12.4-x64.zip"


def test_the_path_offer_shows_the_undo_before_it_asks_and_a_no_leaves_it_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_cmd, "current_os", lambda: "linux")
    _set_gpu(monkeypatch, None)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".profile").write_text("# mine\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--add-to-path", input="y\nn\n")
    assert result.exit_code == 0, result.output
    assert PROFILE_MARK in result.output
    assert "PATH left alone" in result.output
    assert (home / ".profile").read_text(encoding="utf-8") == "# mine\n"


def test_the_path_change_is_applied_when_it_is_agreed_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install_cmd, "current_os", lambda: "linux")
    _set_gpu(monkeypatch, None)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: home))
    result, root = _invoke(tmp_path, "install", "llama.cpp", "--add-to-path", "--yes")
    assert result.exit_code == 0, result.output
    text = (home / ".profile").read_text(encoding="utf-8")
    assert PROFILE_MARK in text
    assert str(root / "bin") in text
    assert "Open a new terminal" in result.output


def test_an_unverified_archive_is_refused_until_the_flag_says_otherwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsigned = ReleaseAsset(name=LINUX_ASSET.name, size=len(LINUX_BUILD), url=LINUX_ASSET.url)
    client = FakeReleaseClient(
        releases={"b10892": Release(tag="b10892", assets=(unsigned,))}, latest_tag="b10892"
    )
    monkeypatch.setattr(install_cmd, "HttpReleaseClient", lambda **_kwargs: client)
    monkeypatch.setattr(install_cmd, "current_os", lambda: "linux")
    _set_gpu(monkeypatch, None)

    refused, root = _invoke(tmp_path, "install", "llama.cpp", "--yes")
    assert refused.exit_code == 1
    assert "no published checksum" in _rendered(refused)
    assert not (root / "bin").exists()

    allowed, root = _invoke(tmp_path, "install", "llama.cpp", "--yes", "--allow-unverified")
    assert allowed.exit_code == 0, allowed.output
    assert (root / "bin" / "llama-server").is_file()


def test_a_corrupt_download_leaves_the_previous_install_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _invoke(tmp_path, "install", "llama.cpp", "--yes")
    root = tmp_path / "install"
    (root / "bin" / "keepsake.txt").write_text("still here", encoding="utf-8")
    monkeypatch.setattr(
        install_cmd,
        "HttpFetcher",
        # The same length as the real archive, so it is the checksum that catches it
        # and not the size.
        lambda **_kwargs: FakeFetcher(bodies={WIN_ASSET.url: bytes(len(WINDOWS_BUILD))}),
    )
    # The verified archive from the first run is still cached, and a cached archive is
    # deliberately not downloaded again; this test is about a fresh, damaged one.
    shutil.rmtree(tmp_path / "llamafit-home" / "cache")
    result, _root = _invoke(tmp_path, "install", "llama.cpp", "--yes")
    assert result.exit_code == 1
    assert "checksum" in _rendered(result)
    assert (root / "bin" / "keepsake.txt").read_text(encoding="utf-8") == "still here"
    assert (root / "bin" / "llama-server.exe").is_file()


def test_the_installed_directory_is_the_managed_one_when_no_dir_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(install_cmd, "managed_dir", lambda: home / ".llamafit" / "llama.cpp")
    result = runner.invoke(app, ["--json", "install", "llama.cpp"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["bin_dir"] == str(home / ".llamafit" / "llama.cpp" / "bin")


@pytest.mark.parametrize("os_name", ["windows", "linux"])
def test_the_command_is_registered_under_install(os_name: OsName) -> None:
    result = runner.invoke(app, ["install", "--help"])
    assert result.exit_code == 0
    assert "llama.cpp" in result.output
