from datetime import datetime, timezone

from llamafit.llamacpp.server import FakeHttp
from llamafit.models import Cpu, Gpu, Host, LlamaCpp, Memory, Probe, SystemReport
from llamafit.services.doctor import diagnose
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


def report_with(
    llamacpp: LlamaCpp,
    *,
    gpus: list[Gpu] | None = None,
    probes: list[Probe] | None = None,
    bandwidth_source: str = "measured",
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
