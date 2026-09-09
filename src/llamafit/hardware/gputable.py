"""Bundled GPU specifications: bandwidth and compute the probes cannot read."""

from __future__ import annotations

import json
from functools import lru_cache

from pydantic import BaseModel

from llamafit.data import packaged_text
from llamafit.models.host import Gpu


class GpuSpec(BaseModel):
    """One row of ``data/gpus.json``."""

    pattern: str
    bandwidth_gbps: float
    compute_tflops_fp16: float
    pcie_gbps: float | None = None
    unified: bool = False


@lru_cache(maxsize=1)
def _table() -> list[GpuSpec]:
    """Read the packaged table once.

    Raises:
        PackagedDataError: The file did not ship in this installation.
    """
    text = packaged_text("llamafit.data", "gpus.json", what="its GPU specification table")
    return [GpuSpec(**row) for row in json.loads(text)["gpus"]]


def lookup_gpu(name: str) -> GpuSpec | None:
    """Find the row whose pattern is the longest substring of ``name`` (case-insensitive)."""
    lowered = name.lower()
    matches = [spec for spec in _table() if spec.pattern in lowered]
    if not matches:
        return None
    return max(matches, key=lambda spec: len(spec.pattern))


def enrich_gpu(gpu: Gpu) -> Gpu:
    """Fill bandwidth and compute from the table when the probe left them unknown."""
    spec = lookup_gpu(gpu.name)
    if spec is None:
        return gpu
    if gpu.bandwidth_gbps is None:
        gpu.bandwidth_gbps = spec.bandwidth_gbps
    if gpu.compute_tflops_fp16 is None:
        gpu.compute_tflops_fp16 = spec.compute_tflops_fp16
    return gpu
