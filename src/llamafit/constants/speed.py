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
#
# Prompt arithmetic runs where the layer is: a layer on the card multiplies on the card, a
# layer in system memory multiplies on the CPU, and section 10.2's compute term is split
# between them by that share. So there are two rates here rather than one, and each is
# fitted to the one reference run that isolates it -- the same small dense model, held
# first entirely on the card and then entirely in system memory:
#
#   Qwen3-0.6B Q8_0, -ngl 99   everything on the card    21,734 prompt tok/s   EFF_PP
#   Qwen3-0.6B Q8_0, -ngl 0    everything in RAM          2,924 prompt tok/s   CPU_PP_...
#
# Two runs, two constants, and neither has a second machine behind it.

EFF_PP: Final = 0.43
"""Fraction of a card's dense fp16 matrix throughput prompt processing reaches (10.2).

**One measurement, and it wants a second machine.** Qwen3-0.6B Q8_0 with every layer on
the card is the only run this project has in which the compute term is the whole of the
prompt formula: nothing streams and nothing is read from system memory, so the measurement
reads the product ``tflops_fp16 x eff_pp`` off directly. It managed 21,734 prompt tokens
per second, and at the ``2 x active_params x ub`` the formula charges, that is
2 x 0.6e9 x 21,734 = **26.1 TFLOP/s achieved**. The bundled table now rates that machine's
RTX 4060 at 60.4 TFLOPS of dense fp16 matrix throughput, so 26.1 / 60.4 = 0.43.

It has to be fitted the way the formula spends it, which means against the catalog's
``active_b`` and not against some tidier parameter count: Qwen3-0.6B's 0.6 billion includes
a tied embedding table that prompt processing barely multiplies by, so the real arithmetic
is less than 26.1 TFLOP/s and the real silicon efficiency lower than 0.43. An apparent
end-to-end rate and a coefficient inside a sum of terms are not the same quantity -- the
mistake section 10.1 already made once with ``eff_ram_scattered`` -- and this is the
coefficient.

**An earlier revision gave 0.30 against a table that held 15 TFLOPS, and the pair was
impossible.** 15 was the RTX 4060's *fp32 shader* rate; llama.cpp multiplies matrices on
the tensor cores, and no efficiency at or below one reaches 26.1 from 15. The two errors
cancelled wherever the streaming term was large enough to hide them, and on a dense model
held on the card, where nothing hides them, the prompt figure came out roughly fourfold
low. That 0.30 came from the slope of a ``llama-bench`` micro-batch sweep of
Qwen3-Coder-Next -- 118 tokens per second at ``-ub 512``, 194 at 1024, 323 at 2048 -- which
is not a clean read of this constant: that model's experts stream across the link, its
three points do not lie on a line to better than 13 percent, and the slope through them
implies about 5 TFLOP/s where the dense run says 26. The dense run is the one that isolates
the term, so the dense run is the one this is fitted to. The disagreement is real, it is
not resolved here, and it is the reason a second machine is wanted.

Two things this figure quietly absorbs, neither of which it should carry for ever:
llama.cpp runs a *quantised* matrix multiply on the integer tensor cores, whose dense rate
is twice the fp16 one on the card measured here, and the formula counts no attention
arithmetic at all.
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

CPU_PP_TFLOPS_PER_CORE: Final = 0.44
"""Prompt-processing throughput one performance core actually reaches, in TFLOP/s.

**Effective, not peak, and it is not multiplied by :data:`EFF_PP`.** There is no table of
CPU peaks to scale, and the two halves of the prompt formula are asked for different
things: the card side gets a published ceiling and an efficiency against it, this side gets
the rate itself.

From the reference machine's other dense run: Qwen3-0.6B Q8_0 at ``-ngl 0 -t 8``, 2,924
prompt tokens per second, which at ``2 x 0.6e9`` per token is 3.51 TFLOP/s across eight
performance cores, so 0.44 each.

**The name it replaces said fp16 and meant nothing measurable.** ``CPU_FP16_TFLOPS_PER_CORE``
was 0.05, a fifth of an AVX2 core's fp32 peak, arrived at by argument rather than by
measurement -- and then multiplied by ``eff_pp`` on the way out, so the rate the formula
actually spent was a fiftieth of that core's peak and about twentyfold below what the
machine was measured doing. Nothing caught it because the term only ever fired on a host
with no graphics card at all.

**One machine, one thread count, one quantisation.** The run used eight threads on eight
performance cores, which is what makes "per performance core" meaningful here and is not
what section 9.4 asks llama.cpp for; a Q8_0 matrix multiply on a Raptor Lake core runs on
integer dot-product instructions rather than on fp32 fused multiply-adds, which is why a
figure above that core's fp32 peak is not the contradiction it looks like; and every other
architecture inherits this number by division. It wants a second machine at least as badly
as :data:`EFF_PP` does.
"""

ASSUMED_BITS_PER_WEIGHT: Final = 4.5
"""Bits per weight assumed when a caller gives no active parameter count.

The average of the Q4_K family, which is what nearly every recommended quantisation is.
Only ever used to turn bytes back into a parameter count for the prompt-processing compute
term, and the estimate says so in its notes.
"""
