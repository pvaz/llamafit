"""Constants the sizing and speed models are built from, each with its provenance.

A constant with no source is a guess that has learned to look like a measurement, so every
number here says where it came from: a section of the design specification, a published
figure, or a run on the reference machine recorded in
``docs/calibration/2026-09-09-reference-machine.md``. Nothing here is meant to be the last
word. A measurement taken on a real host supersedes the constant for that host, and the
estimate built from it is relabelled accordingly (section 10.3).

The one constant that decides whether a recommendation is right or twice as fast as the
truth is :data:`EFF_RAM_SCATTERED`. Read its docstring before changing anything in this
file.
"""

from __future__ import annotations

from typing import Final

# --- Generation: how much of a pool's bandwidth the decode loop actually reaches -----

EFF_VRAM: Final = 0.60
"""Fraction of a card's peak bandwidth a generation kernel reaches.

Section 10.1: the fraction decode kernels reach on consumer GPUs in published
llama-bench results. Not measured on the reference machine, where the graphics card
carries the smaller half of the traffic and the figure could not be isolated.
"""

EFF_RAM_SEQUENTIAL: Final = 0.70
"""Fraction of measured read bandwidth a *contiguous* weight read reaches (section 10.1).

A dense model's weights are laid out in the order they are used, so a layer's read is one
long run and the memory system gets to stream. This is the efficiency for every read from
system memory that is not a routed expert: dense block weights on the CPU, an output head
left in RAM, a KV cache the graphics card does not hold.
"""

EFF_RAM_SCATTERED: Final = 0.36
"""Fraction of measured read bandwidth a *routed expert* read reaches (section 10.1).

**System memory has two effective bandwidths, not one, and the difference is a factor of
two.** Each token selects a different handful of experts, so a layer's read is a scatter
of small blocks across tens of gigabytes and the memory system never gets to stream.
Treating a routed expert set like a dense weight matrix makes every mixture-of-experts
model look about twice as fast as it runs.

Derived on the reference machine from two models whose per-token expert traffic differs by
64 percent:

===================================  =================  ========  =========
Model                                Active expert/tok  Measured  Effective
===================================  =================  ========  =========
Qwen3-Coder-Next UD-Q4_K_XL          0.916 GB           23.0 t/s  21.1 GB/s
Qwen3.8-Flash-Next UD-Q4_K_XL        1.504 GB           13.9 t/s  20.9 GB/s
===================================  =================  ========  =========

Against a measured sequential read of 55 to 60 GB/s on the same machine, that is 0.36. The
two agree to within one percent while the traffic they carry differs by more than half,
which is what makes it a constant of the access pattern rather than a fit to one model.

**Known caveat, recorded rather than silently corrected.** The two effective figures above
were obtained by dividing the expert traffic by the *whole* token time, so they already
contain the graphics-card term and the per-layer overheads that
:func:`llamafit.speed.estimate_speed` then adds separately. Using 0.36 inside the full
four-term formula therefore counts a token's time roughly 1.6 times over and lands about
37 percent below both measurements; see
``.superpowers/sdd/2026-09-09-phase1b-catalog-and-gguf-facts/speed-report.md`` for the
arithmetic. The constant is kept exactly as section 10.1 states it, because it is the
specification's number and because the relative ordering it produces is right to within a
few percent; the absolute level is what a phase 3 host measurement is for.
"""

LAYER_OVERHEAD_S: Final = 0.000_20
"""Fixed cost per transformer block per token, in seconds (section 10.1).

0.20 ms, from the residual between a bandwidth-only prediction and measured generation on
the reference machine. It stands for everything that is not a byte moved: kernel launches,
the synchronisation between the graphics card and the CPU threads computing the experts,
and the small per-block tensors the traffic model does not itemise.
"""

SAMPLING_OVERHEAD_S: Final = 0.001
"""Fixed cost per token for sampling, in seconds (section 10.1)."""

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
