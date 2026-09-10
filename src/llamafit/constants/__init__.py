# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Constants the budget, the placement planner and the estimator are built from.

Nothing here is a round number chosen because it looked safe. Every value is either
written down in ``docs/specs/2026-09-09-llamafit-design.md`` or read out of
``docs/calibration/2026-09-09-reference-machine.md``, and the docstring under it says
which and cites the section or the run, because a constant whose provenance has been lost
cannot be argued with when a later measurement disagrees with it.

The package is one module per area that built it -- :mod:`~llamafit.constants.budget` for
the memory budget (design section 8), :mod:`~llamafit.constants.placement` for the
placement planner (section 9) and :mod:`~llamafit.constants.speed` for the speed
estimator (section 10) -- because a constant belongs to whichever of the three measured
it, and three files a merge can touch independently are worth more than one file long
enough that every branch collided in it. Everything is re-exported here, so an existing
``from llamafit.constants import ...`` keeps working unchanged.
"""

from llamafit.constants.budget import (
    COMPUTE_BUFFER_FIT,
    COMPUTE_BUFFER_KNEE_TOKENS,
    CUDA_CONTEXT_BYTES,
    GIB,
    LOGITS_BYTES_PER_TOKEN,
    MEASURED_RUNTIME_OVERHEAD_BYTES,
    MIB,
    MIN_BATCH_TOKENS,
    PROCESS_OVERHEAD_BYTES,
    PROJECTOR_COMPUTE_BYTES,
    PROJECTOR_MAIN_COMPUTE_FLOOR_BYTES,
    UNACCOUNTED_KV_CACHE_BYTES_PER_1K,
    UTILISATION_COMFORTABLE,
    UTILISATION_FITS,
    UTILISATION_TIGHT,
    VRAM_RESERVE_BYTES,
    ComputeBufferFit,
    MeasuredRuntimeOverhead,
)
from llamafit.constants.placement import (
    ALL_GPU_LAYERS,
    CONTEXT_TIERS,
    DEFAULT_MICRO_BATCH,
    DEFAULT_PARALLEL_SLOTS,
    DEFAULT_REQUESTED_CONTEXT,
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    KV_TYPE_DEFAULT,
    KV_TYPE_QUANTISED,
    MAX_CONTEXT_SEARCH_STEP,
    MICRO_BATCH_LADDER,
    MIN_CONTEXT_TOKENS,
    SHARED_EXPERT_OVERRIDE,
)
from llamafit.constants.speed import (
    ASSUMED_BITS_PER_WEIGHT,
    ASSUMED_PCIE_GBPS,
    BACKEND_FALLBACK_GBPS,
    CPU_FALLBACK_DEFAULT_GBPS,
    CPU_FALLBACK_GBPS,
    CPU_FP16_TFLOPS_PER_CORE,
    DEFAULT_WORKING_CONTEXT,
    EFF_PCIE,
    EFF_PP,
    EFF_RAM_SCATTERED,
    EFF_RAM_SEQUENTIAL,
    EFF_VRAM,
    FIXED_OVERHEAD_S,
    KV_TYPE_BITS,
)

__all__ = [
    "ALL_GPU_LAYERS",
    "ASSUMED_BITS_PER_WEIGHT",
    "ASSUMED_PCIE_GBPS",
    "BACKEND_FALLBACK_GBPS",
    "COMPUTE_BUFFER_FIT",
    "COMPUTE_BUFFER_KNEE_TOKENS",
    "CONTEXT_TIERS",
    "CPU_FALLBACK_DEFAULT_GBPS",
    "CPU_FALLBACK_GBPS",
    "CPU_FP16_TFLOPS_PER_CORE",
    "CUDA_CONTEXT_BYTES",
    "DEFAULT_MICRO_BATCH",
    "DEFAULT_PARALLEL_SLOTS",
    "DEFAULT_REQUESTED_CONTEXT",
    "DEFAULT_SERVER_HOST",
    "DEFAULT_SERVER_PORT",
    "DEFAULT_WORKING_CONTEXT",
    "EFF_PCIE",
    "EFF_PP",
    "EFF_RAM_SCATTERED",
    "EFF_RAM_SEQUENTIAL",
    "EFF_VRAM",
    "FIXED_OVERHEAD_S",
    "GIB",
    "KV_TYPE_BITS",
    "KV_TYPE_DEFAULT",
    "KV_TYPE_QUANTISED",
    "LOGITS_BYTES_PER_TOKEN",
    "MAX_CONTEXT_SEARCH_STEP",
    "MEASURED_RUNTIME_OVERHEAD_BYTES",
    "MIB",
    "MICRO_BATCH_LADDER",
    "MIN_BATCH_TOKENS",
    "MIN_CONTEXT_TOKENS",
    "PROCESS_OVERHEAD_BYTES",
    "PROJECTOR_COMPUTE_BYTES",
    "PROJECTOR_MAIN_COMPUTE_FLOOR_BYTES",
    "SHARED_EXPERT_OVERRIDE",
    "UNACCOUNTED_KV_CACHE_BYTES_PER_1K",
    "UTILISATION_COMFORTABLE",
    "UTILISATION_FITS",
    "UTILISATION_TIGHT",
    "VRAM_RESERVE_BYTES",
    "ComputeBufferFit",
    "MeasuredRuntimeOverhead",
]
