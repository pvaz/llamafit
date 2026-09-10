<<<<<<< HEAD
"""Constants the memory budget is built from, each with the measurement it came from.

Nothing here is a round number chosen because it looked safe. Every value is either
written down in section 8 of ``docs/specs/2026-09-09-llamafit-design.md`` or read out of
``docs/calibration/2026-09-09-reference-machine.md``, and the docstring under it says
which, because a constant whose provenance has been lost cannot be argued with when a
later measurement disagrees with it.

The distinction that matters most here is between the constants that describe a rule and
the constants that were fitted to data. A reserve of 256 MiB is a policy: it is exactly as
right as the people who chose it. The compute-buffer table is a *model*, fitted to ten
measurements on one graphics card with one llama.cpp build, and it will be wrong on
another machine in ways nobody has measured yet. Every budget line built from the second
kind is marked ``exact=False`` so a reader can tell them apart.
=======
"""Constants the estimator and the placement planner are built from, with their provenance.

Every number here is written down once, with a comment saying where it came from, because
a constant with no provenance is indistinguishable from a guess and nobody dares change
it. Values measured on the reference machine cite
``docs/calibration/2026-09-09-reference-machine.md``; values the design fixed cite the
section of ``docs/specs/2026-09-09-llamafit-design.md`` that fixed them.

Phase 3 replaces the measured ones per host by calibration, so a constant here is the
starting point rather than the answer.
>>>>>>> feat/placement
"""

from __future__ import annotations

<<<<<<< HEAD
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
"""VRAM a CUDA context costs before llama.cpp allocates a single buffer.

Section 8.1 gives 300 MiB as the default. The calibration record measured 120 MiB on the
reference machine (VRAM in use minus the sum of the buffers llama.cpp reported) and
records 300 as the conservative figure to plan with, since it is the direction that
refuses a configuration rather than the direction that silently pages.
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
=======
MIB = 1024**2
"""One mebibyte, so a size below can be written the way llama.cpp reports it."""

# --------------------------------------------------------------------------------------
# Placement planner (design section 9)
# --------------------------------------------------------------------------------------

CONTEXT_TIERS: tuple[int, ...] = (
    16384,
    24576,
    32768,
    40960,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
)
"""The context ladder a launch script chooses from at start time (section 9.3).

Sixteen thousand to a quarter of a million tokens, in the steps the design names. The
lower rungs are close together because that is where an eight-gigabyte card lives: the
reference machine's own winning configurations sit at 32,768 and 40,960, four hundred
megabytes apart.
"""

MICRO_BATCH_LADDER: tuple[int, ...] = (2048, 1024, 512, 256)
"""Micro-batch sizes to try, largest first (section 9.2).

A larger micro-batch makes prompt processing faster and the compute buffer bigger: the
reference machine measured 25, 50 and 77 prompt tokens per second at 512, 1024 and 2048
for the same model, and 4096 was slower than 2048, which is why the ladder stops there.
"""

MIN_BATCH_TOKENS = 2048
"""The floor under the logical batch: ``b = max(2 x ub, 2048)`` (section 8.1)."""

ALL_GPU_LAYERS = 99
"""What ``-ngl`` is when every layer goes to the card (section 9.1).

llama.cpp reads any number at or above the layer count as "all of them", and 99 is what
the reference machine's recorded configurations actually passed, so a rendered command
line matches the one that was measured.
"""

KV_TYPE_DEFAULT = "f16"
"""llama.cpp's own KV cache type, and the one every architecture accepts."""

KV_TYPE_QUANTISED = "q8_0"
"""The quantised KV cache the search falls back to when it buys the requested context.

Section 9.2 names this one type. It halves the cache and costs little quality, but some
architectures assert on it, so the catalog's ``kv_types_allowed`` has the last word.
"""

MIN_CONTEXT_TOKENS = CONTEXT_TIERS[0]
"""The floor the context ladder halves down to when the user names no minimum.

Section 9.2 halves "down to the user's minimum", and :class:`~llamafit.models.plan.Needs`
lets that minimum be zero, which would halve forever. The bottom rung of section 9.3's
own ladder is the honest floor: the design offers a launch script no context below 16,384
tokens, so nothing below it is a context this project is willing to recommend either.

It carries weight beyond ending a loop. The planner prefers a faster mode to a longer
context (section 9.2's mode order), so the floor is what stops that preference running
away with itself: a placement has to reach a usable context before its speed counts for
anything. A user who genuinely wants less says so with an explicit shorter request.
"""

DEFAULT_REQUESTED_CONTEXT = 32768
"""What to size for when the caller states no context.

Section 12.1's own ``Needs`` example asks for 32,768, it is a rung of
:data:`CONTEXT_TIERS`, and it is the context the reference machine's winning
configuration was measured at. Only a default: ``plan --context`` overrides it.
"""

MAX_CONTEXT_SEARCH_STEP = 1024
"""Granularity of the search for the largest context a placement holds (section 9.3).

Reporting ``max_context_fit`` to the nearest token would be false precision: the compute
buffer is a fitted formula, and the KV cache of the reference model costs 33 MiB per
thousand tokens, so a thousand tokens is already finer than the model behind the number.
"""

# --------------------------------------------------------------------------------------
# llama-server command line (design section 9.5)
# --------------------------------------------------------------------------------------

DEFAULT_SERVER_HOST = "127.0.0.1"
"""Loopback, because a rendered command line must not open a port to the network.

llama-server has no authentication of any kind; section 13.3 makes the same choice for
LlamaFit's own dashboard and warns whenever a user asks for anything else.
"""

DEFAULT_SERVER_PORT = 8080
"""llama.cpp's own default port, and the first one section 5.1's discovery probes.

Deliberately not LlamaFit's 8765 (section 21): that port belongs to ``llamafit serve``,
and handing it to a model server would make the two collide on the same machine.
"""

DEFAULT_PARALLEL_SLOTS = 1
"""``-np``: one slot (section 9.5).

Every budget in section 8 is computed for one sequence. A second slot divides the context
between them and changes the KV cache, so the number that is planned and the number that
is launched have to be the same one.
>>>>>>> feat/placement
"""
