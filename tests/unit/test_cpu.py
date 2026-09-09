from llamafit.hardware.cpu import detect_cpu, isa_from_flags, performance_cores_for
from llamafit.hardware.runner import FakeRunner


def reference_cores() -> tuple[int, int]:
    return 24, 32


def reference_cpuinfo() -> dict[str, object]:
    return {"brand_raw": "Intel(R) Core(TM) i9-14900KF", "flags": ["avx2"]}


def test_isa_from_flags_maps_known_flags() -> None:
    flags = ["sse4_2", "avx2", "avx512f", "avx512_vnni", "amx_tile", "fma"]
    assert isa_from_flags(flags) == ["avx2", "avx512", "avx512_vnni", "amx"]


def test_isa_from_flags_arm() -> None:
    assert isa_from_flags(["asimd", "sve"]) == ["neon", "sve"]


def test_performance_cores_table() -> None:
    assert performance_cores_for("Intel(R) Core(TM) i9-14900KF", 24) == 8
    assert performance_cores_for("13th Gen Intel(R) Core(TM) i5-13600K", 14) == 6
    assert performance_cores_for("Intel(R) Core(TM) Ultra 9 285K", 24) == 8
    assert performance_cores_for("AMD Ryzen 9 7950X 16-Core Processor", 16) == 16
    assert performance_cores_for("Intel(R) Core(TM) i7-9700K", 8) == 8


def test_detect_cpu_uses_providers() -> None:
    def provider() -> dict[str, object]:
        return {"brand_raw": "Intel(R) Core(TM) i9-14900KF", "flags": ["avx2", "avx512f"]}

    cpu, probes = detect_cpu(
        FakeRunner({}), "windows", cpuinfo_provider=provider, cores_provider=reference_cores
    )
    assert cpu.model == "Intel(R) Core(TM) i9-14900KF"
    assert cpu.isa == ["avx2", "avx512"]
    assert cpu.physical_cores == 24 and cpu.logical_cores == 32
    assert cpu.performance_cores == 8
    assert [p.name for p in probes] == ["cpuinfo", "cpu-cores"]
    assert probes[0].ok


def test_detect_cpu_apple_perf_cores_from_sysctl() -> None:
    def provider() -> dict[str, object]:
        return {"brand_raw": "Apple M3 Max", "flags": ["asimd"]}

    runner = FakeRunner({"sysctl -n hw.perflevel0.physicalcpu": "12\n"})
    cpu, probes = detect_cpu(
        runner, "macos", cpuinfo_provider=provider, cores_provider=reference_cores
    )
    assert cpu.performance_cores == 12
    assert [p.name for p in probes] == ["cpuinfo", "cpu-cores", "sysctl-perflevel"]


def test_detect_cpu_survives_provider_failure() -> None:
    def broken() -> dict[str, object]:
        raise RuntimeError("no cpuinfo")

    cpu, probes = detect_cpu(FakeRunner({}), "linux", cpuinfo_provider=broken)
    assert cpu.model == "unknown"
    assert not probes[0].ok and "no cpuinfo" in (probes[0].error or "")


def test_detect_cpu_survives_cores_failure() -> None:
    def broken() -> tuple[int, int]:
        raise RuntimeError("no psutil")

    cpu, probes = detect_cpu(
        FakeRunner({}), "linux", cpuinfo_provider=reference_cpuinfo, cores_provider=broken
    )
    assert (cpu.physical_cores, cpu.logical_cores) == (1, 1)
    assert not next(p for p in probes if p.name == "cpu-cores").ok
