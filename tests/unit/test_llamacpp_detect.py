from pathlib import Path

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


def test_detect_install_absent(tmp_path: Path) -> None:
    llamacpp, _probes = detect_install(FakeRunner({}), "linux", env={}, home=tmp_path, path_dirs=[])
    assert not llamacpp.installed
    assert llamacpp.problems and "not found" in llamacpp.problems[0]
