# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Constants for the speed estimator (design section 10).

Nothing measured here is meant to be the last word. Phase 3 replaces the measured
constants per host by calibration, and an estimate built on a calibrated figure is
relabelled accordingly (section 10.3), so a constant here is the starting point rather
than the answer.
"""

from __future__ import annotations

from typing import Final

# The three efficiencies below are not independent guesses. They were identified together
# from four measurements on the reference machine, two of which exist only to isolate a
# pool: the same small dense model run entirely in system memory and entirely on the card,
# where the traffic is exactly its block weights and nothing else competes. Changing one of
# them without re-deriving the others against all four runs will make the estimator worse.

# --- Generation: how much of a pool's bandwidth the decode loop actually reaches -----
#
# The four runs, all on the reference machine (RTX 4060 rated 272 GB/s, DDR5 measuring 57
# GB/s sequential read). The traffic column excludes the token embedding table, of which a
# token reads one row:
#
#   Qwen3-0.6B Q8_0, -ngl 0        0.47 GB, system memory, contiguous   78.0 tok/s
#   Qwen3-0.6B Q8_0, -ngl 99       0.47 GB, card                       279.5 tok/s
#   Qwen3-Coder-Next UD-Q4_K_XL    2.36 GB card, 0.916 GB scattered     23.0 tok/s
#   Qwen3.8-Flash-Next UD-Q4_K_XL  4.83 GB card, 1.504 GB scattered     13.9 tok/s

EFF_VRAM: Final = 0.67
"""Fraction of a card's peak bandwidth a generation kernel reaches (section 10.1).

From the second run above: 0.47 GB at 279.5 tokens per second is 3.58 ms a token, and less
the 1 ms fixed overhead that is 182 GB/s of a 272 GB/s card. A small dense model held
entirely on the card is the only run in which the card carries all of the traffic, which is
what makes it the run that settles this figure.
"""

EFF_RAM_SEQUENTIAL: Final = 0.70
"""Fraction of measured read bandwidth a *contiguous* weight read reaches (section 10.1).

From the first run above: 0.47 GB at 78.0 tokens per second is 12.8 ms a token, and less
the 1 ms overhead that is 40 GB/s of a measured 57. A dense model's weights are laid out in
the order they are used, so a layer's read is one long run and the memory system gets to
stream. This is the efficiency for every read from system memory that is not a routed
expert: block weights on the CPU, an output head left in RAM, a KV cache the card does not
hold.
"""

EFF_RAM_SCATTERED: Final = 0.57
"""Fraction of measured read bandwidth a *routed expert* read reaches (section 10.1).

**System memory has two effective bandwidths, not one.** Each token selects a different
handful of experts, so a layer's read is a scatter of small blocks across tens of gigabytes
and the memory system never gets to stream. A contiguous read of the same bytes reaches
0.70; charging a routed expert set the contiguous rate overstates a mixture-of-experts
model by about a seventh of its whole token, which is not a rounding error but is not the
factor of two an earlier revision claimed either.

With :data:`EFF_VRAM` and :data:`FIXED_OVERHEAD_S` already settled by the two dense runs,
the two expert models give this figure independently, out of the time left once their card
traffic and overhead are accounted for:

===============================  =================  =========  =========
Model                            Active expert/tok  Remainder  Effective
===============================  =================  =========  =========
Qwen3-Coder-Next UD-Q4_K_XL      0.916 GB           29.5 ms    0.54
Qwen3.8-Flash-Next UD-Q4_K_XL    1.504 GB           44.3 ms    0.59
===============================  =================  =========  =========

Nine percent apart while carrying 64 percent different traffic, which is what makes it a
property of the access pattern rather than a fit to one model.

**An earlier revision of section 10.1 gave 0.36 and was wrong.** That figure was the expert
traffic divided by the *whole* token time, which already contains the card traffic and the
overhead the formula then adds again: used as a coefficient inside the sum it alone
exceeded the measured token, and the estimator built on it came out 42 percent low. An
apparent end-to-end rate and a coefficient inside a sum of terms are not the same quantity.
"""

FIXED_OVERHEAD_S: Final = 0.001
"""Everything a token costs that is not a byte moved, in seconds (section 10.1).

One millisecond, the intercept the two dense runs share: sampling, the graph launch, and
the small tensors the traffic model does not itemise.

**It is not per layer.** The specification carried 0.20 ms per transformer block, which for
a 28-layer model is 5.6 ms against a whole measured token of 3.58 ms -- not merely too
large but impossible, and it imposed a ceiling near 105 tokens per second on any model of
that depth, however small.
"""

DEFAULT_WORKING_CONTEXT: Final = 8192
"""Context the KV-cache read is sized for when nobody says otherwise (section 10.1).

The board estimates every candidate at 8K so the column compares like with like; ``plan``
passes the context the user actually asked for.
"""

# --- Generation: what to use when a pool's bandwidth could not be measured -----------

BACKEND_FALLBACK_GBPS: Final[dict[str, float]] = {
    "cuda": 250.0,
    "hip": 200.0,
    "metal": 150.0,
    "vulkan": 120.0,
    "sycl": 100.0,
}
"""Stand-in device bandwidth per backend, in GB/s (section 10.1).

Used only when the graphics card's bandwidth is unknown and the bundled table has no row
for it. Deliberately conservative, and an estimate built on one is never labelled better
than ``estimated``.
"""

CPU_FALLBACK_GBPS: Final[dict[str, float]] = {"arm64": 80.0, "x86_64": 60.0}
"""Stand-in system-memory bandwidth per CPU architecture, in GB/s (section 10.1)."""

CPU_FALLBACK_DEFAULT_GBPS: Final = 60.0
"""Stand-in system-memory bandwidth for an architecture not in :data:`CPU_FALLBACK_GBPS`."""

# --- KV cache -----------------------------------------------------------------------

KV_TYPE_BITS: Final[dict[str, float]] = {
    "f32": 32.0,
    "f16": 16.0,
    "bf16": 16.0,
    "q8_0": 8.5,
    "q5_1": 6.0,
    "q5_0": 5.5,
    "q4_1": 5.0,
    "q4_0": 4.5,
    "iq4_nl": 4.5,
}
"""Bits per cached element for each KV cache type, from ggml's block layouts.

A ggml block holds 32 weights plus its scales: ``q8_0`` is 32 bytes of data and one fp16
scale in 34 bytes, so 8.5 bits; ``q4_0`` is 16 bytes and one scale in 18, so 4.5; ``q5_0``
adds four bytes of high bits, ``q4_1`` and ``q5_1`` a second fp16 for the minimum. A GGUF
file reports its cache size at f16, and these ratios scale it.
"""

# --- Prompt processing --------------------------------------------------------------

EFF_PP: Final = 0.30
"""Fraction of a card's peak fp16 throughput prompt processing reaches (section 10.2).

From the slope of the reference machine's ``llama-bench`` micro-batch sweep for
Qwen3-Coder-Next: 118 tokens per second at ``-ub 512``, 194 at 1024 and 323 at 2048, which
is 1.30 ms of extra time per extra prompt token. The slope isolates the compute term,
because the per-micro-batch streaming cost is the same at every micro-batch size. Against
2 x 3e9 active parameters on a 15 TFLOP/s card that is 0.307.
"""

EFF_PCIE: Final = 0.33
"""Fraction of the link's bandwidth an expert set reaches while streaming (section 10.2).

Section 10.2 fixes the reference machine at 4 GB/s: a 78 GB expert set at 52 tokens per
second with ``-ub 1024``, and 25 at ``-ub 512``, both give the same figure. The bundled
table rates that machine's RTX 4060 link at 12 GB/s, so 0.33.

The figure comes from Qwen3.8-Flash-Next, whose 111 GB of weights plus a 29 GB streamed
lookup table do not fit in the machine's 128 GB of RAM, so part of every micro-batch's
expert read comes off the disk. Qwen3-Coder-Next, whose 50 GB expert set stays in the page
cache, streams at about 12.8 GB/s on the same machine and is under-predicted about
twofold as a result. Residency, not the link, is the real variable; modelling it needs a
disk-bandwidth probe this phase does not have.
"""

ASSUMED_PCIE_GBPS: Final = 12.0
"""Link bandwidth assumed when the bundled GPU table has no figure, in GB/s.

A PCIe 4.0 x8 link, which is what the reference machine has and roughly the middle of what
a consumer card is given.
"""

CPU_FP16_TFLOPS_PER_CORE: Final = 0.05
"""Prompt-processing throughput of one performance core, in TFLOP/s.

An AVX2 core issuing two eight-wide fused multiply-adds per cycle at 4 GHz has about 256
GFLOP/s of fp32 peak, and llama.cpp's quantised matrix multiply reaches roughly a fifth of
it. Not measured: it exists so a CPU-only placement gets a prompt figure at all, and any
estimate that depends on it is labelled ``estimated``.
"""

ASSUMED_BITS_PER_WEIGHT: Final = 4.5
"""Bits per weight assumed when a caller gives no active parameter count.

The average of the Q4_K family, which is what nearly every recommended quantisation is.
Only ever used to turn bytes back into a parameter count for the prompt-processing compute
term, and the estimate says so in its notes.
"""
