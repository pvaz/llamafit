from llamafit.hardware.gputable import enrich_gpu, lookup_gpu
from llamafit.models import Gpu


def test_lookup_prefers_the_longest_matching_pattern() -> None:
    spec = lookup_gpu("NVIDIA GeForce RTX 4060 Ti")
    assert spec is not None and spec.pattern == "rtx 4060 ti"
    spec = lookup_gpu("NVIDIA GeForce RTX 4060")
    assert spec is not None and spec.pattern == "rtx 4060"
    assert spec.bandwidth_gbps == 272


def test_lookup_unknown_returns_none() -> None:
    assert lookup_gpu("Matrox G200") is None


def test_enrich_fills_only_unknown_fields() -> None:
    gpu = Gpu(index=0, vendor="nvidia", name="NVIDIA GeForce RTX 4060", bandwidth_gbps=999)
    enriched = enrich_gpu(gpu)
    assert enriched.bandwidth_gbps == 999
    assert enriched.compute_tflops_fp16 is not None


def test_apple_entry_is_unified() -> None:
    spec = lookup_gpu("Apple M3 Max")
    assert spec is not None and spec.unified and spec.bandwidth_gbps >= 300
