# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Constants for the memory budget (design section 8).

Nothing here is a round number chosen because it looked safe. Every value is either
written down in ``docs/specs/2026-09-09-llamafit-design.md`` or read out of
``docs/calibration/2026-09-09-reference-machine.md``, and the docstring under it says
which and cites the section or the run, because a constant whose provenance has been lost
cannot be argued with when a later measurement disagrees with it.

The distinction that matters most here is between the constants that describe a rule and
the constants that were fitted to data. A reserve of 256 MiB is a policy: it is exactly as
right as the people who chose it. The compute-buffer table is a *model*, fitted to ten
measurements on one graphics card with one llama.cpp build, and it will be wrong on
another machine in ways nobody has measured yet. Every budget line built from the second
kind is marked ``exact=False`` so a reader can tell them apart.
"""

from __future__ import annotations

from typing import NamedTuple

MIB = 1024**2
"""One mebibyte. Buffer sizes are reported in these by llama.cpp, so budgets speak them."""

GIB = 1024**3
"""One gibibyte."""

VRAM_RESERVE_BYTES = 256 * MIB
"""What is held back on the graphics card for whatever else the desktop is doing.

Section 8.1 of the specification, which gives 256 MiB as the default. The calibration
record measured the desktop itself using between 530 and 1,340 MiB on the reference
machine depending on what was open, so this reserve does not cover a desktop; it covers
the drift between the scan and the moment the server actually allocates.
"""

CUDA_CONTEXT_BYTES = 300 * MIB
"""VRAM a GPU backend costs before llama.cpp allocates a single buffer, on a card nobody
has measured.

Section 8.1's default, and the fallback under
:data:`MEASURED_RUNTIME_OVERHEAD_BYTES`. It is the conservative figure on purpose: being
too high withholds a configuration, being too low recommends one that pages silently, and
of those two the first is the one a user can argue with.
"""


class MeasuredRuntimeOverhead(NamedTuple):
    """What one card was measured to cost, and what it was measured on.

    Attributes:
        bytes_: The overhead measured on that card.
        driver: The graphics driver version it was measured under.
        build: The llama.cpp build it was measured with.
    """

    bytes_: int
    driver: str
    build: str


MEASURED_RUNTIME_OVERHEAD_BYTES: dict[str, MeasuredRuntimeOverhead] = {
    "nvidia geforce rtx 4060": MeasuredRuntimeOverhead(120 * MIB, "610.88", "b10867"),
}
"""What the GPU backend actually cost, per card that somebody has measured.

Keyed by the card's name as the scan reports it, casefolded with runs of whitespace
collapsed. A card that is not here gets :data:`CUDA_CONTEXT_BYTES`, so the safe figure is
what every unmeasured machine still plans with; this table is the project's confidence
ladder applied to a constant rather than to a speed.

The one row is one measurement: VRAM in use minus the sum of the buffers ``llama-server
-v`` printed, on the reference machine, under the driver and build recorded beside it. It
is worth having because on that machine the 180 MiB between the measured figure and the
conservative one is two rungs of the context ladder -- it is the difference between the
planner offering the best configuration anybody has measured there and withholding it.

What would generalise it is a second card. The overhead is a property of the driver's
context and the backend's own allocations rather than of the model, so a handful of rows
across vendors, driver generations and llama.cpp builds would say whether 120 MiB is this
card's number, this driver's, or roughly everyone's -- and until somebody takes them, a
row here is a claim about one machine and the budget line built from it says so.
"""

PROCESS_OVERHEAD_BYTES = 1 * GIB
"""System memory the server process costs before any model buffer.

Section 8.1, which gives 1 GiB as the default.
"""

PROJECTOR_COMPUTE_BYTES = 248 * MIB
"""The vision projector's own compute space on the card, on top of its file bytes.

Measured on the reference machine: every configuration in the calibration record with the
projector offloaded reports a second compute buffer of exactly 248 MiB beside the main
one, and every configuration without it reports none.
"""

PROJECTOR_MAIN_COMPUTE_FLOOR_BYTES = 1922 * MIB
"""How large the *main* compute buffer becomes when the projector is on the card.

The compute buffer is one arena sized by the largest graph that will run in it, so an
offloaded projector does not add to it: it sets a floor under it. The calibration record
shows both halves of that. At 65,536 tokens and ``-ub 512`` the main buffer is 1,142 MiB
with vision off and 1,922 MiB with the projector offloaded, a rise of 780 MiB; at
``-ub 2048`` the same two configurations report 2,586 and 2,592 MiB, so the text graph has
overtaken the vision graph and offloading the projector costs nothing there at all.

Taking the maximum of the fitted text formula and this floor reproduces all four of those
rows. It is a floor observed at one micro-batch, so it is a model and not a measurement of
every case: a budget line built on it is never marked exact.
"""


class ComputeBufferFit(NamedTuple):
    """The fitted compute-buffer model for one micro-batch.

    Attributes:
        base_bytes: What the buffer costs at no context at all.
        per_1k_context_bytes: What each 1,024 tokens of context adds, up to
            :data:`COMPUTE_BUFFER_KNEE_TOKENS`.
        per_1k_context_bytes_beyond_knee: What each 1,024 tokens adds past that knee,
            which is the same figure again for every micro-batch that has never been
            measured past it.
    """

    base_bytes: int
    per_1k_context_bytes: int
    per_1k_context_bytes_beyond_knee: int


COMPUTE_BUFFER_KNEE_TOKENS = 64 * 1024
"""The context past which the compute buffer grows faster, from section 8.2's table."""

COMPUTE_BUFFER_FIT: dict[int, ComputeBufferFit] = {
    512: ComputeBufferFit(1090 * MIB, 3 * MIB // 2, 3 * MIB // 2),
    1024: ComputeBufferFit(1240 * MIB, 3 * MIB, 3 * MIB),
    2048: ComputeBufferFit(2080 * MIB, 8 * MIB, 25 * MIB),
}
"""The compute-buffer model of section 8.2, keyed by micro-batch.

The bases and the slopes are section 8.2's own table: 1,090, 1,240 and 2,080 MiB, and 1.5,
3.0 and 8.0 MiB per 1,024 tokens of context, the last rising to 25 beyond 64K.

Each slope belongs to the micro-batch it is filed under and is applied once, per 1,024
tokens of context: at ``-ub 1024`` the buffer grows about 3 MiB per 1K of context, at
``-ub 2048`` about 8 MiB per 1K up to 64K and about 25 MiB per 1K beyond. That is how the
calibration record states the fit these constants came from, and it reproduces every row of
that record at ``-ub`` 1024 and 2048 to better than one percent -- 1,336 MiB predicted
against 1,337 measured at 32K, 2,336 against 2,330, 4,192 against 4,168.
"""

UNACCOUNTED_KV_CACHE_BYTES_PER_1K: dict[str, int] = {
    "qwen4exp": 9 * MIB,
}
"""Cache an architecture allocates per 1,024 tokens that its GGUF header does not describe.

Section 7.2 derives the key and value caches from the shapes the file declares, and for
almost every architecture that is the whole cache. Qwen4exp is not one of those. The
calibration record measured 33 MiB per 1,024 tokens on Qwen3.8-Flash-Next -- "two caches,
24 and 9 MiB per 1K" -- where the declared shape (12 attention layers, 2 key/value heads,
256 for each of the key and value lengths) accounts for 24. The remaining 9 MiB per 1,024
tokens is real memory llama.cpp allocates, most likely the index Qwen Sparse Attention
keeps over past keys, and nothing in the header predicts it.

It is here rather than absent because the direction of the error decides what it costs
somebody. At 32K it is 288 MiB, 27 percent of that model's cache, and it grows with the
context: at 128K it is 1.1 GB. A budget that left it out would report a configuration
fitting a card it would page off, which is the one way of being wrong that section 8 exists
to prevent. It is carried on a line of its own so that nobody mistakes a figure measured on
one machine for one derived from the file.

Measured at ``f16``, the only cache type this model's catalog entry allows, and applied
whatever the cache type is asked for, since nothing says how it would quantise. An
architecture added here whose cache can be quantised needs that question answered first.
"""

MIN_BATCH_TOKENS = 2048
"""The floor under the logical batch, from section 8.1's ``b = max(2 x ub, 2048)``."""

LOGITS_BYTES_PER_TOKEN = 4
"""Bytes one vocabulary entry of the output buffer occupies: llama.cpp keeps logits in F32."""

UTILISATION_COMFORTABLE = 0.65
"""At or below this share of a pool, there is room for a bigger context or a second model.

Section 8.3's table.
"""

UTILISATION_FITS = 0.85
"""At or below this share of a pool, the intended configuration runs as planned.

Section 8.3's table.
"""

UTILISATION_TIGHT = 0.95
"""At or below this share of a pool it still runs, but a browser can push it over.

Section 8.3's table. Above it, section 8.4 applies: on the graphics card the driver pages
rather than refusing, so the configuration is named ``too-tight`` rather than rejected.
"""
