"""Builders for speed-estimator tests: the reference machine and placements on it."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llamafit.catalog.loader import load_catalog
from llamafit.models.catalog import CatalogModel
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Cpu, Gpu, Host, Memory
from llamafit.models.plan import Budget, Placement, Pool, RunMode, Verdict

REFERENCE_RAM_GBPS = 57.0
"""What the NumPy read benchmark reports on the reference machine (bandwidth.py)."""


def reference_host(
    *,
    ram_gbps: float | None = REFERENCE_RAM_GBPS,
    bandwidth_source: str = "measured",
    with_gpu: bool = True,
) -> Host:
    """The machine in docs/calibration/2026-09-09-reference-machine.md."""
    gpus = []
    if with_gpu:
        gpus.append(
            Gpu(
                index=0,
                vendor="nvidia",
                name="NVIDIA GeForce RTX 4060",
                vram_total_bytes=8188 * 1024**2,
                vram_used_bytes=550 * 1024**2,
                vram_source="measured",
                bandwidth_gbps=272.0,
                compute_tflops_fp16=60.4,
                backend_hint="cuda",
                driver="610.88",
            )
        )
    return Host(
        os="windows",
        os_version="11 (26200)",
        arch="x86_64",
        cpu=Cpu(
            model="Intel(R) Core(TM) i9-14900KF",
            physical_cores=24,
            logical_cores=32,
            performance_cores=8,
            isa=["avx2"],
        ),
        memory=Memory(
            total_bytes=128 * 1024**3,
            available_bytes=100 * 1024**3,
            type="DDR5",
            speed_mts=4200,
            modules=2,
            channels=2,
            bandwidth_gbps=ram_gbps,
            bandwidth_source=bandwidth_source,  # type: ignore[arg-type]
        ),
        gpus=gpus,
        disks=[],
        probes=[],
        scanned_at=datetime(2026, 9, 9, 20, 0, tzinfo=timezone.utc),
    )


def budget(verdict: Verdict = "fits") -> Budget:
    """A budget with the verdict a test cares about and nothing else of interest."""
    return Budget(
        lines=(),
        vram_required=7 * 1024**3,
        ram_required=60 * 1024**3,
        vram_available=8 * 1024**3,
        ram_available=100 * 1024**3,
        vram_utilisation=0.87,
        ram_utilisation=0.6,
        verdict=verdict,
    )


def placement(
    *,
    mode: RunMode = "moe-offload",
    context: int = 32768,
    micro_batch: int = 1024,
    kv_type: str = "f16",
    gpu_layers: int = 99,
    cpu_moe_layers: int | None = 48,
    shared_experts_pool: Pool | None = None,
    projector_pool: Pool | None = None,
    verdict: Verdict = "fits",
) -> Placement:
    """A placement, defaulting to the reference machine's expert-offload configuration."""
    return Placement(
        mode=mode,
        context=context,
        micro_batch=micro_batch,
        batch=max(2 * micro_batch, 2048),
        kv_type=kv_type,
        gpu_layers=gpu_layers,
        cpu_moe_layers=cpu_moe_layers,
        shared_experts_pool=shared_experts_pool,
        projector_pool=projector_pool,
        threads=8,
        budget=budget(verdict),
        max_context_fit=context,
    )


def catalog_model(model_id: str) -> CatalogModel:
    """One bundled catalog entry, with the facts read from its GGUF headers merged in."""
    catalog, _problems = load_catalog(custom_path=Path("no-such-custom-models.yaml"))
    for model in catalog.models:
        if model.id == model_id:
            return model
    raise AssertionError(f"{model_id} is not in the bundled catalog")


def catalog_facts(model_id: str, quant_name: str) -> GgufFacts:
    """The GGUF facts the bundled catalog records for one quant of one model."""
    for source in catalog_model(model_id).sources:
        for quant in source.quants:
            if quant.name == quant_name and quant.gguf_facts is not None:
                return quant.gguf_facts
    raise AssertionError(f"{model_id} {quant_name} has no recorded facts")
