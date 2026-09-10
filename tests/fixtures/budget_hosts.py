"""Hosts the budget tests size models against: the reference machine, and made-up ones."""

from datetime import datetime, timezone

from llamafit.hardware import scan
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.host import Cpu, Gpu, Host, Memory
from tests.fixtures.reference_machine import (
    reference_cores,
    reference_cpuinfo,
    reference_runner,
    reference_vm,
)

GIB = 1024**3
MIB = 1024**2


def reference_host() -> Host:
    """The 2026-09-09 reference machine, scanned from its recorded probe output.

    An RTX 4060 with 8,188 MiB of which 550 were in use, 128 GB of DDR5 with 100 free.
    """
    return scan(
        reference_runner(),
        os_name="windows",
        measure_bandwidth=False,
        cpuinfo_provider=reference_cpuinfo,
        vm_provider=reference_vm,
        cores_provider=reference_cores,
    )


def machine(
    *,
    vram_total: int | None = 8 * GIB,
    vram_used: int = 0,
    ram_total: int = 64 * GIB,
    ram_available: int = 48 * GIB,
    unified: bool = False,
) -> Host:
    """A machine with exactly the memory a test wants, and nothing else of interest.

    ``vram_total`` of ``None`` means no graphics card at all.
    """
    gpus = []
    if vram_total is not None:
        gpus.append(
            Gpu(
                index=0,
                vendor="nvidia",
                name="Test GPU",
                vram_total_bytes=vram_total,
                vram_used_bytes=vram_used,
                backend_hint="cuda",
            )
        )
    return Host(
        os="windows",
        os_version="11",
        arch="x86_64",
        cpu=Cpu(model="Test CPU", physical_cores=8, logical_cores=16, performance_cores=8),
        memory=Memory(total_bytes=ram_total, available_bytes=ram_available),
        gpus=gpus,
        unified_memory=unified,
        scanned_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )


def card_with_free(free_bytes: int, *, ram_available: int = 100 * GIB) -> Host:
    """A machine whose card has exactly ``free_bytes`` unused, before LlamaFit's reserve."""
    total = free_bytes + 4 * GIB
    return machine(
        vram_total=total,
        vram_used=total - free_bytes,
        ram_total=ram_available + 28 * GIB,
        ram_available=ram_available,
    )


def first_quant(model: CatalogModel) -> Quant:
    """The first quantisation of a catalog model, which is the only one the seed carries."""
    return model.sources[0].quants[0]
