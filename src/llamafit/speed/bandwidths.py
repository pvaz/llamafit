# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Turning a scanned host into the bandwidths the speed formula divides by.

Three numbers come out of here, and each one is a raw bandwidth multiplied by the fraction
of it the corresponding access pattern actually reaches: the graphics card's, system
memory's for a contiguous read, and system memory's for a scattered one. The last is
:data:`~llamafit.constants.EFF_RAM_SCATTERED`, and it is meaningfully below the other,
which is the whole reason this module reports three figures instead of two.

When a pool's bandwidth could not be established, section 10.1's per-backend fallback
stands in for the raw figure -- not for the effective one. A fallback is a conservative
guess at what the hardware can do, so the same efficiency that applies to a measured
bandwidth applies to it, and the estimate built on it can never be labelled better than
``estimated``.

**Prompt processing needs two compute rates, not one, because it runs in two places.** A
layer on the card multiplies its matrices on the card and a layer in system memory
multiplies them on the CPU, so section 10.2's compute term is charged to both in the
proportion the placement splits the layers -- which :mod:`llamafit.speed.estimate` does,
with what comes out of here. ``compute_flops`` is the card's published peak and is scaled
by an efficiency where it is spent; ``cpu_compute_flops`` is what the CPU was measured
reaching and is not scaled again. A single rate hid a category error for as long as it
existed: every prompt figure on this project's own reference machine, including the ones
for models with six layers out of sixty-two on the card, was charged to the card.
"""

from __future__ import annotations

from dataclasses import dataclass

from llamafit.constants import (
    ASSUMED_PCIE_GBPS,
    BACKEND_FALLBACK_GBPS,
    CPU_FALLBACK_DEFAULT_GBPS,
    CPU_FALLBACK_GBPS,
    CPU_PP_TFLOPS_PER_CORE,
    EFF_PCIE,
    EFF_RAM_SCATTERED,
    EFF_RAM_SEQUENTIAL,
    EFF_VRAM,
)
from llamafit.hardware.gputable import lookup_gpu
from llamafit.i18n import _
from llamafit.models.host import Host


@dataclass(frozen=True)
class EffectiveBandwidths:
    """What each access pattern actually achieves on this host, in bytes per second.

    Attributes:
        device: The graphics card's read bandwidth for generation, or ``None`` when the
            host has no usable card.
        sequential: System memory's read bandwidth for a contiguous weight read.
        scattered: System memory's read bandwidth for a routed expert read, which is about
            half the contiguous figure and is why the two are kept apart.
        pcie: What an expert set streams at across the link, for prompt processing.
        compute_flops: The card's *peak* dense fp16 matrix throughput, which section
            10.2's compute term then multiplies by
            :data:`~llamafit.constants.EFF_PP`. Zero when the host has no card figure at
            all, which is how a caller knows there is no card side to charge anything to.
        cpu_compute_flops: What the CPU *reaches* on the same arithmetic, already
            effective. The two are not the same kind of number and the docstring of
            :data:`~llamafit.constants.CPU_PP_TFLOPS_PER_CORE` says why.
        ram_gbps: The raw system-memory figure the two RAM numbers were derived from.
        device_gbps: The raw graphics-card figure, or ``None``.
        assumed: True when any figure above came from a fallback rather than from the
            host. An estimate built on one is never better than ``estimated``.
        notes: What was substituted and why, when anything was.
    """

    device: float | None
    sequential: float
    scattered: float
    pcie: float
    compute_flops: float
    cpu_compute_flops: float
    ram_gbps: float
    device_gbps: float | None
    assumed: bool
    notes: tuple[str, ...]


def _ram_bandwidth(host: Host) -> tuple[float, bool]:
    """System memory's raw read bandwidth in GB/s, and whether it had to be assumed."""
    measured = host.memory.bandwidth_gbps
    if measured and host.memory.bandwidth_source in ("measured", "estimated"):
        return measured, host.memory.bandwidth_source != "measured"
    if measured:
        return measured, True
    return CPU_FALLBACK_GBPS.get(host.arch, CPU_FALLBACK_DEFAULT_GBPS), True


def resolve_bandwidths(host: Host) -> EffectiveBandwidths:
    """Work out what this host's pools achieve, substituting where it could not be read.

    Args:
        host: The scanned machine.

    Returns:
        The effective bandwidths, in bytes per second, with a note for every substitution.
    """
    notes: list[str] = []
    ram_gbps, ram_assumed = _ram_bandwidth(host)
    if ram_assumed:
        notes.append(
            _("System memory bandwidth was not measured; %(gbps).0f GB/s assumed for %(arch)s.")
            % {"gbps": ram_gbps, "arch": host.arch}
        )

    gpu = host.primary_gpu
    device_gbps: float | None = None
    device_assumed = False
    compute_tflops = 0.0
    pcie_gbps = ASSUMED_PCIE_GBPS
    if gpu is not None:
        spec = lookup_gpu(gpu.name)
        device_gbps = gpu.bandwidth_gbps or (spec.bandwidth_gbps if spec else None)
        compute_tflops = gpu.compute_tflops_fp16 or (spec.compute_tflops_fp16 if spec else 0.0)
        if spec is not None and spec.pcie_gbps:
            pcie_gbps = spec.pcie_gbps
        if device_gbps is None:
            device_gbps = BACKEND_FALLBACK_GBPS.get(gpu.backend_hint, CPU_FALLBACK_DEFAULT_GBPS)
            device_assumed = True
            notes.append(
                _(
                    "%(gpu)s bandwidth is unknown; the %(backend)s fallback of %(gbps).0f GB/s"
                    " stands in."
                )
                % {"gpu": gpu.name, "backend": gpu.backend_hint, "gbps": device_gbps}
            )

    cores = host.cpu.performance_cores or host.cpu.physical_cores
    cpu_compute_flops = max(cores, 1) * CPU_PP_TFLOPS_PER_CORE * 1e12
    compute_assumed = compute_tflops <= 0
    if compute_assumed:
        notes.append(
            _("No graphics card compute figure: prompt processing is estimated on the CPU.")
        )

    return EffectiveBandwidths(
        device=None if device_gbps is None else device_gbps * EFF_VRAM * 1e9,
        sequential=ram_gbps * EFF_RAM_SEQUENTIAL * 1e9,
        scattered=ram_gbps * EFF_RAM_SCATTERED * 1e9,
        pcie=pcie_gbps * EFF_PCIE * 1e9,
        compute_flops=max(compute_tflops, 0.0) * 1e12,
        cpu_compute_flops=cpu_compute_flops,
        ram_gbps=ram_gbps,
        device_gbps=device_gbps,
        assumed=ram_assumed or device_assumed or compute_assumed,
        notes=tuple(notes),
    )
