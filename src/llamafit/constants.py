"""Constants the estimator and the placement planner are built from, with their provenance.

Every number here is written down once, with a comment saying where it came from, because
a constant with no provenance is indistinguishable from a guess and nobody dares change
it. Values measured on the reference machine cite
``docs/calibration/2026-09-09-reference-machine.md``; values the design fixed cite the
section of ``docs/specs/2026-09-09-llamafit-design.md`` that fixed them.

Phase 3 replaces the measured ones per host by calibration, so a constant here is the
starting point rather than the answer.
"""

from __future__ import annotations

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
"""
