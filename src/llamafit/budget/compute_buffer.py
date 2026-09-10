"""The compute buffer and the output buffer: the two lines that are models, not sums.

The compute buffer is the least predictable component in the whole budget. It is the
scratch arena llama.cpp allocates for the graph it is about to run, it grows with both the
micro-batch and the context, and no header says how big it will be. Section 8.2's model is
piecewise linear, fitted to ten measurements on one graphics card with one build, and this
module implements it and nothing more. Every line built here is marked ``exact=False``,
because the honest thing to say about a fitted formula is that it is a fitted formula.

Two things about the fit are worth knowing before trusting a number out of it. It was
fitted at micro-batches of 512, 1024 and 2048, and anything else is served by the smallest
of those at or above the request, which overstates rather than understates. And it was
fitted on a single architecture -- a mixture-of-experts model with most of its weights in
system memory -- so a dense model that keeps everything on the card is extrapolation.

The file is named ``compute_buffer`` rather than ``compute`` so that
``llamafit.budget.compute``, the function this package exists to provide, can never be
shadowed by a submodule of the same name.
"""

from __future__ import annotations

from llamafit.constants import (
    COMPUTE_BUFFER_FIT,
    COMPUTE_BUFFER_KNEE_TOKENS,
    LOGITS_BYTES_PER_TOKEN,
    MIN_BATCH_TOKENS,
    PROJECTOR_MAIN_COMPUTE_FLOOR_BYTES,
)
from llamafit.i18n import _
from llamafit.models.plan import BudgetLine, Pool


def batch_for(micro_batch: int) -> int:
    """The logical batch that goes with a micro-batch, from section 8.1's ``max(2 x ub, 2048)``."""
    return max(2 * micro_batch, MIN_BATCH_TOKENS)


def fit_for(micro_batch: int) -> tuple[int, int, int]:
    """The fitted compute-buffer model to use for a micro-batch that may not be tabulated.

    The smallest tabulated micro-batch at or above the one asked for is used, so a request
    the fit has never seen is answered with the next larger buffer rather than a smaller
    one; a micro-batch above the largest tabulated value falls back to that largest row,
    which is the only direction where this understates.

    Args:
        micro_batch: The micro-batch, i.e. what ``-ub`` would be.

    Returns:
        The base bytes, the bytes per 1,024 tokens of context, and the bytes per 1,024
        tokens past the knee.
    """
    tabulated = sorted(COMPUTE_BUFFER_FIT)
    key = next((ub for ub in tabulated if ub >= micro_batch), tabulated[-1])
    return COMPUTE_BUFFER_FIT[key]


def compute_buffer_bytes(*, micro_batch: int, context: int, projector_on_gpu: bool = False) -> int:
    """What the main compute buffer costs on the card.

    Args:
        micro_batch: What ``-ub`` would be.
        context: The context length in tokens.
        projector_on_gpu: Whether the vision projector is offloaded, which puts a floor
            under the buffer rather than adding to it, since one arena is sized by the
            largest graph that will run in it.

    Returns:
        Bytes of VRAM the buffer occupies.
    """
    base, per_1k, per_1k_beyond = fit_for(micro_batch)
    tokens = max(context, 0)
    below = min(tokens, COMPUTE_BUFFER_KNEE_TOKENS)
    beyond = max(tokens - COMPUTE_BUFFER_KNEE_TOKENS, 0)
    total = base + per_1k * below // 1024 + per_1k_beyond * beyond // 1024
    if projector_on_gpu:
        total = max(total, PROJECTOR_MAIN_COMPUTE_FLOOR_BYTES)
    return total


def output_buffer_bytes(*, n_vocab: int, batch: int) -> int:
    """What the logits buffer costs in system memory, from section 8.1's ``n_vocab x 4 x b``."""
    return max(n_vocab, 0) * LOGITS_BYTES_PER_TOKEN * max(batch, 0)


def buffer_lines(
    *,
    micro_batch: int,
    context: int,
    n_vocab: int,
    batch: int | None = None,
    projector_on_gpu: bool = False,
    pool: Pool = "vram",
) -> tuple[BudgetLine, ...]:
    """The compute buffer and the output buffer in system memory.

    Args:
        micro_batch: What ``-ub`` would be.
        context: The context length in tokens.
        n_vocab: The model's vocabulary size, which is what the output buffer scales with.
        batch: What ``-b`` would be, defaulting to :func:`batch_for` of the micro-batch.
        projector_on_gpu: Whether the vision projector is offloaded.
        pool: Where the compute buffer lives, which is with the layers whose graph runs in
            it: the card for any placement that has a layer there, and system memory for
            one that has none. Charging the arena to a card that is doing nothing would
            report a processor-only run needing VRAM it never touches.

    Returns:
        The two lines, both modelled.
    """
    lines = [
        BudgetLine(
            component="compute-buffer",
            pool=pool,
            bytes=compute_buffer_bytes(
                micro_batch=micro_batch, context=context, projector_on_gpu=projector_on_gpu
            ),
            exact=False,
            note=_("fitted to one machine's measurements; the least certain line here"),
        )
    ]
    logits = output_buffer_bytes(n_vocab=n_vocab, batch=batch or batch_for(micro_batch))
    if logits:
        lines.append(
            BudgetLine(
                component="output-buffer",
                pool="ram",
                bytes=logits,
                exact=False,
                note=_("room for one logit per vocabulary entry per token of the batch"),
            )
        )
    return tuple(lines)
