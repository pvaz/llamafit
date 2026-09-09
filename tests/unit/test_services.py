from datetime import datetime, timezone
from pathlib import Path

import pytest

from llamafit.llamacpp.server import FakeHttp
from llamafit.models import (
    Cpu,
    Disk,
    Gpu,
    Host,
    LlamaCpp,
    Memory,
    Probe,
    RunningServer,
    SystemReport,
)
from llamafit.services.doctor import Diagnosis, diagnose
from llamafit.services.scan import scan_system
from tests.fixtures.reference_machine import (
    reference_cores,
    reference_cpuinfo,
    reference_runner,
    reference_vm,
)


def test_scan_system_returns_report_with_version() -> None:
    report = scan_system(
        runner=reference_runner(),
        http=FakeHttp({}),
        os_name="windows",
        env={"PATH": ""},
        measure_bandwidth=False,
        cpuinfo_provider=reference_cpuinfo,
        vm_provider=reference_vm,
        cores_provider=reference_cores,
        well_known=[],
    )
    assert report.version
    assert report.host.primary_gpu is not None
    assert report.llamacpp.installed is False


def test_scan_system_reports_the_disk_llamacpp_lives_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llamafit.hardware.disks as disks_module

    bin_dir = tmp_path / "llama.cpp" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "llama-server.exe").write_bytes(b"")
    monkeypatch.setattr(
        disks_module,
        "_mount_key",
        lambda path: "llama-volume" if str(path).startswith(str(tmp_path)) else "other-volume",
    )
    report = scan_system(
        runner=reference_runner(),
        http=FakeHttp({}),
        os_name="windows",
        env={"PATH": "", "LLAMA_CPP_PATH": str(bin_dir)},
        measure_bandwidth=False,
        cpuinfo_provider=reference_cpuinfo,
        vm_provider=reference_vm,
        cores_provider=reference_cores,
        well_known=[],
    )
    assert report.llamacpp.path == str(bin_dir)
    paths = [d.path for d in report.host.disks]
    assert str(bin_dir) in paths, "the volume llama.cpp lives on must be reported"
    assert len(paths) > 1, "the working directory's volume is still reported too"


def report_with(
    llamacpp: LlamaCpp,
    *,
    gpus: list[Gpu] | None = None,
    probes: list[Probe] | None = None,
    bandwidth_source: str = "measured",
    disks: list[Disk] | None = None,
    unified_memory: bool = False,
) -> SystemReport:
    host = Host(
        os="windows",
        os_version="11",
        arch="x86_64",
        cpu=Cpu(model="i9", physical_cores=24, logical_cores=32, performance_cores=8),
        memory=Memory(
            total_bytes=128 * 1024**3,
            available_bytes=100 * 1024**3,
            bandwidth_gbps=60,
            bandwidth_source=bandwidth_source,  # type: ignore[arg-type]
        ),
        gpus=gpus
        if gpus is not None
        else [
            Gpu(
                index=0,
                vendor="nvidia",
                name="RTX 4060",
                vram_total_bytes=8 * 1024**3,
                vram_used_bytes=0,
                backend_hint="cuda",
            )
        ],
        probes=probes or [],
        disks=disks or [],
        unified_memory=unified_memory,
        scanned_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )
    return SystemReport(host=host, llamacpp=llamacpp, version="0.1.0a1")


def test_diagnose_missing_llamacpp_is_an_error_with_hint() -> None:
    diagnosis = diagnose(
        report_with(LlamaCpp(installed=False, problems=["llama.cpp not found: ..."]))
    )
    errors = [f for f in diagnosis.findings if f.level == "error"]
    assert errors and "llama.cpp" in errors[0].title
    assert errors[0].hint and "install" in errors[0].hint.lower()
    assert diagnosis.worst_level == "error"


def test_diagnose_backend_mismatch_is_a_warning() -> None:
    llamacpp = LlamaCpp(installed=True, path="C:/llama.cpp/bin", build=10867, backends=["cpu"])
    diagnosis = diagnose(report_with(llamacpp))
    titles = [f.title for f in diagnosis.findings if f.level == "warn"]
    assert any("CUDA" in t for t in titles)


def test_diagnose_failed_probe_gets_hint() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=1, backends=["cuda"])
    probes = [Probe(name="nvidia-smi", ok=False, duration_ms=1, error="nvidia-smi: not found")]
    diagnosis = diagnose(report_with(llamacpp, probes=probes))
    finding = next(f for f in diagnosis.findings if f.title.startswith("Probe nvidia-smi"))
    assert finding.level == "warn" and finding.hint and "driver" in finding.hint.lower()


def test_diagnose_all_good() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["cuda", "cpu"])
    diagnosis = diagnose(report_with(llamacpp))
    assert diagnosis.worst_level == "ok"
    assert any(f.level == "ok" and "llama.cpp" in f.title for f in diagnosis.findings)


def test_running_server_is_reported_as_ok() -> None:
    llamacpp = LlamaCpp(
        installed=True,
        path="x",
        build=10867,
        backends=["cuda", "cpu"],
        running_servers=[
            RunningServer(url="http://127.0.0.1:8080", model="qwen3-coder-next", n_ctx=32768)
        ],
    )
    diagnosis = diagnose(report_with(llamacpp))
    finding = next(f for f in diagnosis.findings if "8080" in f.title)
    assert finding.level == "ok" and "qwen3-coder-next" in finding.detail


def test_low_disk_space_is_a_warning() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["cuda", "cpu"])
    disks = [Disk(path="D:", free_bytes=5 * 1024**3, total_bytes=1024**4)]
    diagnosis = diagnose(report_with(llamacpp, disks=disks))
    finding = next(f for f in diagnosis.findings if "Low disk space" in f.title)
    assert finding.level == "warn" and finding.hint


def test_unified_memory_host_does_not_warn_about_unknown_vram() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["metal", "cpu"])
    gpus = [Gpu(index=0, vendor="apple", name="Apple M3 Max", backend_hint="metal")]
    diagnosis = diagnose(report_with(llamacpp, gpus=gpus, unified_memory=True))
    assert not any("VRAM size unknown" in f.title for f in diagnosis.findings)


def test_intel_gpu_warns_about_sycl_and_avoids_the_rocm_hint() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["cpu"])
    gpus = [Gpu(index=0, vendor="intel", name="Intel(R) Arc(TM) A770", backend_hint="vulkan")]
    diagnosis = diagnose(report_with(llamacpp, gpus=gpus))
    assert next(f for f in diagnosis.findings if "SYCL" in f.title).level == "warn"
    vram = next(f for f in diagnosis.findings if "VRAM size unknown" in f.title)
    assert vram.hint is not None and "ROCm" not in vram.hint


def test_worst_level_of_no_findings_is_ok() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["cuda", "cpu"])
    assert Diagnosis(report=report_with(llamacpp), findings=[]).worst_level == "ok"


def test_vendor_probe_failure_is_silent_when_that_vendor_is_absent() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["cuda", "cpu"])
    probes = [Probe(name="rocm-smi", ok=False, duration_ms=1, error="rocm-smi: not found")]
    diagnosis = diagnose(report_with(llamacpp, probes=probes))
    assert not any(f.title.startswith("Probe rocm-smi") for f in diagnosis.findings)


def test_vendor_probe_failure_still_warns_when_no_gpu_was_detected() -> None:
    llamacpp = LlamaCpp(installed=True, path="x", build=10867, backends=["cpu"])
    probes = [Probe(name="nvidia-smi", ok=False, duration_ms=1, error="nvidia-smi: not found")]
    diagnosis = diagnose(report_with(llamacpp, gpus=[], probes=probes))
    assert any(f.title.startswith("Probe nvidia-smi") for f in diagnosis.findings)


def test_unreadable_memory_totals_are_an_error_not_a_missing_detail() -> None:
    # psutil failing leaves the totals at zero, and every budget LlamaFit will compute
    # starts from them, so the machine cannot be sized at all until this is fixed.
    report = report_with(LlamaCpp(installed=True, path="D:/llama.cpp", backends=["cuda"]))
    report.host.memory.total_bytes = 0
    diagnosis = diagnose(report)
    errors = [f for f in diagnosis.findings if f.level == "error"]
    assert errors and "Memory size unknown" in errors[0].title
    assert errors[0].hint and "psutil" in errors[0].hint
    assert diagnosis.worst_level == "error"


def test_a_machine_whose_memory_was_read_says_nothing_about_it() -> None:
    diagnosis = diagnose(
        report_with(LlamaCpp(installed=True, path="D:/llama.cpp", backends=["cuda"]))
    )
    assert not [f for f in diagnosis.findings if "Memory size unknown" in f.title]


def test_the_probe_hints_are_deferred_so_a_language_chosen_later_still_reaches_them() -> None:
    # The table is filled while the module is imported, before any interface has chosen a
    # language. Eager values would be English for the life of the process, silently.
    from llamafit.i18n import LazyString
    from llamafit.services.doctor import PROBE_HINTS

    assert PROBE_HINTS
    for hint in PROBE_HINTS.values():
        assert isinstance(hint, LazyString)
