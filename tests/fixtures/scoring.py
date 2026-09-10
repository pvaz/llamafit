"""Candidates for the reference machine, built from the seeded catalog's own facts.

The five seeded models are real entries with real quant sizes, and two of them —
``qwen3-coder-next`` and ``qwen3.8-flash-next`` — were measured on the reference machine
on 2026-09-09 and carry those figures in the catalog. The placements below reproduce what
the planner is expected to find for each of them on that machine (RTX 4060 8 GB, 128 GB
DDR5 with 100 GB free, Intel i9-14900KF), and the speeds come from the measurements where
measurements exist.

The three models with no measurement carry ``estimated`` speeds, which is exactly what the
estimator would label them, and their placements follow from their file sizes: a 0.6B Q8_0
sits entirely on the card, a 4.9 GB Llama 3.1 8B does too, and a 16.5 GB Gemma 3 27B does
not and runs hybrid.
"""

from __future__ import annotations

from datetime import date

from llamafit.models.plan import (
    Budget,
    BudgetLine,
    Confidence,
    Placement,
    RunMode,
    SpeedEstimate,
    Verdict,
)

MIB = 1024**2
GB = 1000**3

CARD_TOTAL = 8188 * MIB
"""What nvidia-smi reports for the reference machine's RTX 4060."""

DESKTOP_RESERVE = 512 * MIB
"""What the budget leaves the desktop compositor, so the card is not filled to the brim."""

VRAM_AVAILABLE = CARD_TOTAL - DESKTOP_RESERVE
RAM_AVAILABLE = 100 * 1024**3


def budget(
    *,
    vram_required: int,
    ram_required: int,
    vram_available: int = VRAM_AVAILABLE,
    ram_available: int = RAM_AVAILABLE,
    verdict: Verdict = "fits",
) -> Budget:
    """A budget with one line per pool and the utilisations that follow from it."""
    lines = (
        BudgetLine(component="weights", pool="vram", bytes=vram_required, exact=True),
        BudgetLine(component="weights", pool="ram", bytes=ram_required, exact=True),
    )
    return Budget(
        lines=lines,
        vram_required=vram_required,
        ram_required=ram_required,
        vram_available=vram_available,
        ram_available=ram_available,
        vram_utilisation=None if vram_available == 0 else vram_required / vram_available,
        ram_utilisation=ram_required / ram_available,
        verdict=verdict,
    )


def placement(
    *,
    mode: RunMode = "gpu",
    context: int = 32768,
    max_context_fit: int = 32768,
    vram_required: int,
    ram_required: int = 0,
    kv_type: str = "f16",
    gpu_layers: int = 99,
    cpu_moe_layers: int | None = None,
    threads: int = 16,
    verdict: Verdict = "fits",
) -> Placement:
    """A placement on the reference machine, with the budget its pools imply."""
    return Placement(
        mode=mode,
        context=context,
        micro_batch=1024,
        batch=4096,
        kv_type=kv_type,
        gpu_layers=gpu_layers,
        cpu_moe_layers=cpu_moe_layers,
        threads=threads,
        budget=budget(vram_required=vram_required, ram_required=ram_required, verdict=verdict),
        max_context_fit=max_context_fit,
    )


def speed(
    gen_tps: float,
    pp_tps: float,
    *,
    confidence: Confidence = "estimated",
    measured_on: date | None = None,
) -> SpeedEstimate:
    """A speed estimate, measured or otherwise."""
    return SpeedEstimate(
        gen_tps=gen_tps,
        pp_tps=pp_tps,
        confidence=confidence,
        measured_on=measured_on,
    )


MEASURED_ON = date(2026, 9, 9)

QWEN3_CODER_NEXT = (
    # 49.6 GB of weights: the shared parts and the KV cache on the card, the 512 routed
    # experts in system memory (--n-cpu-moe 48). Peak VRAM measured at 7.8 GB.
    placement(
        mode="moe-offload",
        context=262144,
        max_context_fit=262144,
        vram_required=int(7.8 * GB),
        ram_required=int(44 * GB),
        kv_type="q8_0",
        cpu_moe_layers=48,
        verdict="tight",
    ),
    # llama-bench tg128 24.7 t/s, pp2048 323 t/s, build 10867, 2026-09-09.
    speed(24.7, 323.0, confidence="measured", measured_on=MEASURED_ON),
)

QWEN3_8_FLASH_NEXT = (
    # 111.3 GB downloaded, 28.8 GB of it an n-gram table streamed from disk. Shared
    # experts and vision on the CPU; peak VRAM measured at 7.3 GB.
    placement(
        mode="moe-offload",
        context=40960,
        max_context_fit=262144,
        vram_required=int(7.3 * GB),
        ram_required=int(75 * GB),
        cpu_moe_layers=48,
        verdict="tight",
    ),
    # The winning configuration on the reference machine: 13.9 t/s generation, 49.4 t/s
    # prompt, build 10867, 2026-09-09.
    speed(13.9, 49.4, confidence="measured", measured_on=MEASURED_ON),
)

QWEN3_0_6B = (
    # 639 MB of Q8_0 weights plus a small cache: barely touches the card.
    placement(context=40960, max_context_fit=40960, vram_required=int(1.2 * GB)),
    speed(120.0, 3000.0),
)

LLAMA_3_1_8B = (
    # 4.9 GB of Q4_K_M weights and a 32K f16 cache, entirely on the card.
    placement(vram_required=int(6.6 * GB)),
    speed(45.0, 1400.0),
)

GEMMA_3_27B = (
    # 16.5 GB of Q4_K_M weights: twice the card, so half the layers run on the CPU.
    placement(
        mode="hybrid",
        context=16384,
        max_context_fit=16384,
        vram_required=int(7.6 * GB),
        ram_required=int(11 * GB),
        gpu_layers=28,
        verdict="tight",
    ),
    speed(4.2, 180.0),
)

BOARD = {
    "qwen3-coder-next": QWEN3_CODER_NEXT,
    "qwen3.8-flash-next": QWEN3_8_FLASH_NEXT,
    "qwen3-0.6b": QWEN3_0_6B,
    "llama-3.1-8b-instruct": LLAMA_3_1_8B,
    "gemma-3-27b-it": GEMMA_3_27B,
}
"""Every seeded model, placed and estimated on the reference machine."""
