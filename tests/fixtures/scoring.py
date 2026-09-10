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

Every placement says how much of each pool is the model's own weights and how much is the
cache and buffers around them, because section 11.3 reads those two apart. The figures are
the ones the real budget produces for these five models against this machine's profile.
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
    weight_vram: int | None = None,
    weight_ram: int | None = None,
    vram_available: int = VRAM_AVAILABLE,
    ram_available: int = RAM_AVAILABLE,
    verdict: Verdict = "fits",
) -> Budget:
    """A budget with the model's own weights split out from the cost of running it.

    Section 11.3 reads two different quantities off a budget — the whole of it, for how
    crowded the tightest pool is, and the weight lines alone, for whether the model is
    worth the machine — so a fixture that charged every byte to one anonymous component
    could only exercise half of the score. ``weight_vram`` and ``weight_ram`` say how much
    of each pool is the model itself; whatever is left over becomes the cache and buffer
    lines that a real budget carries alongside it. Left out, a pool is all weights.
    """
    weight_vram = vram_required if weight_vram is None else weight_vram
    weight_ram = ram_required if weight_ram is None else weight_ram
    assert weight_vram <= vram_required, "more weights on the card than the card holds"
    assert weight_ram <= ram_required, "more weights in memory than the placement needs"
    candidates = (
        ("dense-weights", "vram", weight_vram),
        ("dense-weights", "ram", weight_ram),
        ("kv-cache", "vram", vram_required - weight_vram),
        ("output-buffer", "ram", ram_required - weight_ram),
    )
    lines = tuple(
        BudgetLine(component=component, pool=pool, bytes=size, exact=True)
        for component, pool, size in candidates
        if size > 0
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
    weight_vram: int | None = None,
    weight_ram: int | None = None,
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
        budget=budget(
            vram_required=vram_required,
            ram_required=ram_required,
            weight_vram=weight_vram,
            weight_ram=weight_ram,
            verdict=verdict,
        ),
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
    # 49.6 GB of weights: 3.3 GB of shared parts and the KV cache on the card, the 512
    # routed experts in system memory (--n-cpu-moe 48). Peak VRAM measured at 7.8 GB, and
    # the 49.8 GB of system memory is those experts plus the output buffer and the
    # process overhead.
    placement(
        mode="moe-offload",
        context=262144,
        max_context_fit=262144,
        vram_required=int(7.8 * GB),
        ram_required=int(49.8 * GB),
        weight_vram=int(3.3 * GB),
        weight_ram=int(46.3 * GB),
        kv_type="q8_0",
        cpu_moe_layers=48,
        verdict="tight",
    ),
    # llama-bench tg128 24.7 t/s, pp2048 323 t/s, build 10867, 2026-09-09.
    speed(24.7, 323.0, confidence="measured", measured_on=MEASURED_ON),
)

QWEN3_8_FLASH_NEXT = (
    # 111.3 GB downloaded, 28.8 GB of it an n-gram table streamed from disk, so 82.5 GB
    # of it is resident: 4.6 GB on the card and 77.9 GB in system memory. Shared experts
    # and vision on the CPU; peak VRAM measured at 7.3 GB.
    placement(
        mode="moe-offload",
        context=40960,
        max_context_fit=262144,
        vram_required=int(7.3 * GB),
        ram_required=int(81 * GB),
        weight_vram=int(4.6 * GB),
        weight_ram=int(77.9 * GB),
        cpu_moe_layers=48,
        verdict="tight",
    ),
    # The winning configuration on the reference machine: 13.9 t/s generation, 49.4 t/s
    # prompt, build 10867, 2026-09-09.
    speed(13.9, 49.4, confidence="measured", measured_on=MEASURED_ON),
)

QWEN3_0_6B = (
    # 639 MB of Q8_0 weights plus a cache and buffers many times their size: the card
    # looks half taken and the machine is barely touched, which is the whole reason
    # section 11.3 reads the weights rather than the pool.
    placement(
        context=40960,
        max_context_fit=40960,
        vram_required=int(1.2 * GB),
        weight_vram=int(0.639 * GB),
    ),
    speed(120.0, 3000.0),
)

LLAMA_3_1_8B = (
    # 4.9 GB of Q4_K_M weights and a 32K f16 cache, entirely on the card.
    placement(vram_required=int(6.6 * GB), weight_vram=int(4.9 * GB)),
    speed(45.0, 1400.0),
)

GEMMA_3_27B = (
    # 16.5 GB of Q4_K_M weights: twice the card, so half the layers run on the CPU. 2.2 GB
    # of them stay on it and 14.3 GB go to system memory, with the cache split behind them.
    placement(
        mode="hybrid",
        context=16384,
        max_context_fit=16384,
        vram_required=int(7.6 * GB),
        ram_required=int(17 * GB),
        weight_vram=int(2.2 * GB),
        weight_ram=int(14.3 * GB),
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
