import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.errors import NotInstalledError
from llamafit.models import Cpu, Disk, Gpu, Host, LlamaCpp, Memory, SystemReport

runner = CliRunner()


def fake_report() -> SystemReport:
    host = Host(
        os="windows",
        os_version="11",
        arch="x86_64",
        cpu=Cpu(
            model="Intel i9-14900KF",
            physical_cores=24,
            logical_cores=32,
            performance_cores=8,
            isa=["avx2", "avx512"],
        ),
        memory=Memory(
            total_bytes=128 * 1024**3,
            available_bytes=100 * 1024**3,
            type="DDR5",
            speed_mts=4200,
            modules=4,
            channels=2,
            bandwidth_gbps=67.2,
            bandwidth_source="estimated",
        ),
        gpus=[
            Gpu(
                index=0,
                vendor="nvidia",
                name="NVIDIA GeForce RTX 4060",
                vram_total_bytes=8188 * 1024**2,
                vram_used_bytes=550 * 1024**2,
                bandwidth_gbps=272,
                compute_tflops_fp16=15,
                backend_hint="cuda",
                driver="610.88",
            )
        ],
        disks=[Disk(path="D:\\", free_bytes=355 * 1024**3, total_bytes=1024**4)],
        scanned_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )
    llamacpp = LlamaCpp(
        installed=True,
        path="D:/llama.cpp/bin",
        build=10867,
        commit="f3f1a8f27",
        backends=["cuda", "rpc", "cpu"],
    )
    return SystemReport(host=host, llamacpp=llamacpp, version="0.1.0a1")


@pytest.fixture(autouse=True)
def patch_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.system_cmd.scan_system", lambda **kwargs: fake_report())
    monkeypatch.setattr("llamafit.cli.doctor_cmd.scan_system", lambda **kwargs: fake_report())


def test_system_table_mentions_gpu_and_memory() -> None:
    result = runner.invoke(app, ["system"])
    assert result.exit_code == 0, result.output
    assert "RTX 4060" in result.output
    assert "DDR5" in result.output
    assert "4 modules" in result.output
    assert "2-channel" in result.output
    assert "67.2" in result.output
    assert "b10867" in result.output
    assert "D:\\" in result.output
    assert "free of" in result.output


def test_system_json_is_the_report() -> None:
    result = runner.invoke(app, ["--json", "system"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["host"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 4060"
    assert data["llamacpp"]["build"] == 10867
    assert data["version"] == "0.1.0a1"


def test_doctor_lists_findings_and_exit_code_zero_when_no_error() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "llama.cpp installed" in result.output


def test_doctor_json_has_findings() -> None:
    result = runner.invoke(app, ["--json", "doctor"])
    data = json.loads(result.output)
    assert isinstance(data["findings"], list)
    assert data["findings"][0]["level"] in {"ok", "warn", "error"}


def test_external_text_with_square_brackets_is_not_parsed_as_markup() -> None:
    from rich.console import Console

    from llamafit.cli.render import render_findings, render_probes
    from llamafit.models import Probe
    from llamafit.services.doctor import Finding

    hostile = "bad path C:/x[/y]/z"
    console = Console(width=200, no_color=True)
    with console.capture() as capture:
        console.print(
            render_probes([Probe(name="wmi-video", ok=False, duration_ms=3, error=hostile)])
        )
        console.print(
            render_findings([Finding(level="warn", title=hostile, detail=hostile, hint=hostile)])
        )
    output = capture.get()
    assert output.count(hostile) == 4


def test_main_prints_an_unexpected_error_containing_markup_verbatim(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import llamafit.cli.app as app_module

    def boom(**kwargs: object) -> None:
        raise ValueError("bad path C:/x[/y]/z")

    monkeypatch.setattr(app_module, "app", boom)
    monkeypatch.setattr(sys, "argv", ["llamafit", "system"])
    with pytest.raises(SystemExit):
        app_module.main()
    assert "bad path C:/x[/y]/z" in capsys.readouterr().err


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llamafit" in result.output


def test_main_maps_known_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import llamafit.cli.app as app_module

    def boom(**kwargs: object) -> None:
        raise NotInstalledError("llama.cpp not found", hint="Install it")

    monkeypatch.setattr(app_module, "app", boom)
    monkeypatch.setattr(sys, "argv", ["llamafit", "system"])
    with pytest.raises(SystemExit) as exit_info:
        app_module.main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "llama.cpp not found" in err and "Install it" in err


def test_main_reports_unexpected_errors_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import llamafit.cli.app as app_module

    def boom(**kwargs: object) -> None:
        raise ValueError("kaboom")

    monkeypatch.setattr(app_module, "app", boom)
    monkeypatch.setattr(sys, "argv", ["llamafit", "system"])
    with pytest.raises(SystemExit) as exit_info:
        app_module.main()
    assert exit_info.value.code == 1
    assert "kaboom" in capsys.readouterr().err


def test_verbose_enables_file_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import llamafit.cli.app as app_module

    def quiet(**kwargs: object) -> None:
        return None

    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path))
    monkeypatch.setattr(app_module, "app", quiet)
    monkeypatch.setattr(sys, "argv", ["llamafit", "--verbose", "system"])
    logger = logging.getLogger("llamafit")
    try:
        app_module.main()
        assert (tmp_path / "log" / "llamafit.log").exists()
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_get_logger_is_a_child_of_the_package_logger() -> None:
    from llamafit.logging import get_logger

    assert get_logger("hardware.gpu").name == "llamafit.hardware.gpu"
