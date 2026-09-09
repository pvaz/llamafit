import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from llamafit.hardware.runner import FakeRunner
from llamafit.llamacpp.detect import (
    detect_backends,
    detect_install,
    find_llamacpp_dir,
    find_local_models,
    parse_version,
    well_known_dirs,
)


def make_install(root: Path, *, windows: bool = True) -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    exe = ".exe" if windows else ""
    for name in ("llama-server", "llama-cli", "llama-bench"):
        (bin_dir / f"{name}{exe}").write_bytes(b"")
    suffix = ".dll" if windows else ".so"
    for lib in ("ggml-base", "ggml-cuda", "ggml-cpu-alderlake", "ggml-rpc"):
        (bin_dir / f"{lib}{suffix}").write_bytes(b"")
    (root / "bin" / "VERSION.txt").write_text("b10867\n")
    return bin_dir


def test_parse_version_variants() -> None:
    assert parse_version("version: 10867 (f3f1a8f27)\nbuilt with clang") == (10867, "f3f1a8f27")
    assert parse_version("build: 10867 (f3f1a8f2) with MSVC") == (10867, "f3f1a8f2")
    assert parse_version("llama-server b10867\n") == (10867, None)
    assert parse_version("nonsense") == (None, None)


def test_find_dir_precedence(tmp_path: Path) -> None:
    env_dir = make_install(tmp_path / "env")
    path_dir = make_install(tmp_path / "path")
    assert (
        find_llamacpp_dir(
            env={"LLAMA_CPP_PATH": str(env_dir)},
            path_dirs=[path_dir],
            well_known=[],
            os_name="windows",
        )
        == env_dir
    )
    assert (
        find_llamacpp_dir(env={}, path_dirs=[path_dir], well_known=[], os_name="windows")
        == path_dir
    )
    assert (
        find_llamacpp_dir(
            env={}, path_dirs=[], well_known=[tmp_path / "missing"], os_name="windows"
        )
        is None
    )


def test_well_known_dirs_include_platform_defaults(tmp_path: Path) -> None:
    dirs = well_known_dirs("windows", tmp_path)
    assert Path("D:/llama.cpp/bin") in dirs or Path("D:\\llama.cpp\\bin") in dirs
    assert tmp_path / ".llamafit" / "llama.cpp" / "bin" in dirs


def test_detect_backends_from_libraries(tmp_path: Path) -> None:
    bin_dir = make_install(tmp_path)
    assert detect_backends(bin_dir) == ["cuda", "rpc", "cpu"]


def test_detect_backends_survives_a_directory_it_cannot_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = make_install(tmp_path)
    assert detect_backends(tmp_path / "does-not-exist") == []

    def denied(self: Path) -> Iterator[Path]:
        raise PermissionError("access is denied")

    monkeypatch.setattr(Path, "iterdir", denied)
    assert detect_backends(bin_dir) == []


def test_find_local_models_skips_unreadable_files_and_missing_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "Good-Q4_K_M.gguf").write_bytes(b"x" * 4)
    unreadable = models / "Bad-Q4_K_M.gguf"
    unreadable.write_bytes(b"x" * 4)
    real_stat = Path.stat

    def flaky_stat(self: Path, **kwargs: object) -> os.stat_result:
        if self == unreadable:
            raise PermissionError("access is denied")
        return real_stat(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", flaky_stat)
    found = find_local_models([models, tmp_path / "does-not-exist"])
    assert [Path(m.path).name for m in found] == ["Good-Q4_K_M.gguf"]


def test_find_local_models_groups_shards(tmp_path: Path) -> None:
    models = tmp_path / "models"
    (models / "a").mkdir(parents=True)
    (models / "a" / "Model-Q4_K_M.gguf").write_bytes(b"x" * 10)
    (models / "a" / "Big-UD-Q4_K_XL-00001-of-00002.gguf").write_bytes(b"x" * 5)
    (models / "a" / "Big-UD-Q4_K_XL-00002-of-00002.gguf").write_bytes(b"x" * 7)
    (models / "a" / "mmproj-F16.gguf").write_bytes(b"x" * 3)
    found = {Path(m.path).name: m.bytes for m in find_local_models([models])}
    assert found == {
        "Model-Q4_K_M.gguf": 10,
        "Big-UD-Q4_K_XL-00001-of-00002.gguf": 12,
        "mmproj-F16.gguf": 3,
    }


def test_find_local_models_skips_split_models_without_their_first_shard(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "Partial-Q4_K_M-00002-of-00003.gguf").write_bytes(b"x" * 7)
    (models / "Whole-Q4_K_M.gguf").write_bytes(b"x" * 4)
    assert [Path(m.path).name for m in find_local_models([models])] == ["Whole-Q4_K_M.gguf"]


def test_detect_install_reads_version_and_backends(tmp_path: Path) -> None:
    bin_dir = make_install(tmp_path)
    server = str(bin_dir / "llama-server.exe")
    runner = FakeRunner({f"{server} --version": "version: 10867 (f3f1a8f27)\n"})
    llamacpp, probes = detect_install(
        runner, "windows", env={"LLAMA_CPP_PATH": str(bin_dir)}, home=tmp_path, path_dirs=[]
    )
    assert llamacpp.installed and llamacpp.path == str(bin_dir)
    assert llamacpp.build == 10867 and llamacpp.commit == "f3f1a8f27"
    assert llamacpp.backends == ["cuda", "rpc", "cpu"]
    assert [p.name for p in probes] == ["llama-server --version"]


def test_detect_install_survives_a_version_file_that_is_not_utf8_text(tmp_path: Path) -> None:
    bin_dir = make_install(tmp_path)
    (bin_dir / "VERSION.txt").write_bytes(b"\xff\xfe b10867")
    llamacpp, _probes = detect_install(
        FakeRunner({}), "windows", env={"LLAMA_CPP_PATH": str(bin_dir)}, home=tmp_path, path_dirs=[]
    )
    assert llamacpp.installed and llamacpp.build is None


def test_detect_install_absent(tmp_path: Path) -> None:
    llamacpp, _probes = detect_install(FakeRunner({}), "linux", env={}, home=tmp_path, path_dirs=[])
    assert not llamacpp.installed
    assert llamacpp.problems and "not found" in llamacpp.problems[0]
