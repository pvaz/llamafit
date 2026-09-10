# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Turn four scores into one ordered list, and keep the candidates that did not make it.

This is the step that produces the answer a person actually came for. Everything before it
is arithmetic about one model at a time; this is where models are compared.

Two rules shape the module.

**Nothing disappears.** A candidate that fails a filter comes back with
``excluded_because`` set and its placement and speed still attached, and it is ranked last
rather than dropped. A shorter list tells a person nothing. "Llama 3.1 8B was excluded
because its entry lists general, chat and reasoning, not coding" tells them why a model
they expected to see is missing, and what to change — the request, or the catalog entry —
which is the difference between a tool that answers and one that merely responds.

That is also what makes an exclusion the right home for a fact a score cannot carry. A
score of zero says "this is bad at the job"; an exclusion says "this is not that kind of
tool, and here is the number that decides it". The second is the more useful sentence and
the more falsifiable one, and it costs the reader nothing, because the candidate is still
on the page.

**Nothing is a bare number.** Every candidate that is scored carries the four parts, the
weights that combined them and the total, because a ranking whose reason is invisible asks
a reader to take it on faith. This project has already fixed that defect once, in a table
that sorted by a number it did not show.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from llamafit.i18n import _
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.plan import (
    Candidate,
    Needs,
    Placement,
    ScoreBreakdown,
    SpeedEstimate,
)
from llamafit.quality import (
    declares_use_case,
    missing_capabilities,
    penalty_for,
    score_quality,
)
from llamafit.scoring.context_score import context_score, requested_context
from llamafit.scoring.fit_score import fit_score_for
from llamafit.scoring.speed_score import (
    floor_tps,
    keeps_up_with_reader,
    observed_tps,
    speed_score,
)
from llamafit.scoring.weights import weights_for
from llamafit.units import format_bytes, format_grouped, localise_number


def evaluate(
    model: CatalogModel,
    quant: Quant,
    needs: Needs,
    placement: Placement | None,
    speed: SpeedEstimate | None,
    *,
    weights: Mapping[str, float] | None = None,
) -> Candidate:
    """Score one model at one quantisation against one request, or say why it cannot be.

    Args:
        model: The catalog entry.
        quant: The quantisation being considered, with its download size when the
            catalog has been refreshed.
        needs: What the user asked for.
        placement: Where this configuration's bytes would go, or ``None`` when the
            planner found nowhere to put them.
        speed: How fast it is expected to run, or ``None`` when no estimate could be
            made.
        weights: The weights to combine the four parts with, already resolved. Left out,
            the use case's defaults are used; pass
            :func:`~llamafit.scoring.weights.weights_for` once for a whole board rather
            than resolving the config's overrides per candidate.

    Returns:
        A scored candidate, or an excluded one carrying the reason and whatever was
        learned about it before it was excluded.
    """
    reason = _exclusion(model, quant, needs, placement, speed)
    if reason is None and placement is not None and speed is not None:
        return _scored(model, quant, needs, placement, speed, weights)
    # Unreachable with a reason of None: every branch that leaves placement or speed
    # unset above returns one. The fallback is written this way so the type checker can
    # see the narrowing rather than be told to trust it.
    return Candidate(
        model_id=model.id,
        quant=quant.name,
        placement=placement,
        speed=speed,
        excluded_because=reason,
    )


def rank(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Order candidates best first, with the excluded ones kept at the end.

    Ties are broken by quality, as section 12.2 requires: two candidates that score the
    same overall are separated by the better model, not by whichever the catalog happened
    to list first. Below that, the faster one wins, and below that the ordering falls back
    to the model id and quant name so that the same board is always in the same order.

    Excluded candidates keep the order they arrived in. They are not ranked against each
    other — a list of things that did not qualify has no best — but they are shown.

    Args:
        candidates: What :func:`evaluate` produced, in any order.

    Returns:
        A new list: scored candidates best first, then the excluded ones.
    """
    # Read once into a list: candidates arrives as a generator from evaluate_and_rank,
    # and two passes over one of those would leave the second empty — which is exactly
    # how the excluded candidates would silently vanish from the board.
    everything = list(candidates)
    scored = [c for c in everything if c.score is not None]
    excluded = [c for c in everything if c.score is None]
    scored.sort(key=_ordering)
    return scored + excluded


def evaluate_and_rank(
    entries: Sequence[tuple[CatalogModel, Quant, Placement | None, SpeedEstimate | None]],
    needs: Needs,
    *,
    weight_overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> list[Candidate]:
    """Score a whole board and order it, resolving the weights once.

    Args:
        entries: One tuple per candidate: the catalog entry, the quantisation, the
            placement the planner found and the estimator's speed for it.
        needs: What the user asked for.
        weight_overrides: The config file's weights, keyed by use case and then by part.

    Returns:
        The ranked board, excluded candidates last.

    Raises:
        ConfigError: If the use case is unknown or an override is malformed.
    """
    weights = weights_for(needs.use_case, weight_overrides)
    return rank(
        evaluate(model, quant, needs, placement, speed, weights=weights)
        for model, quant, placement, speed in entries
    )


def _ordering(candidate: Candidate) -> tuple[float, float, float, str, str]:
    """Sort key: total, then quality, then speed, then identity, all best first."""
    score = candidate.score
    total = 0.0 if score is None else score.total
    quality = 0.0 if score is None else score.quality
    gen_tps = 0.0 if candidate.speed is None else candidate.speed.gen_tps
    return (-total, -quality, -gen_tps, candidate.model_id, candidate.quant)


def _scored(
    model: CatalogModel,
    quant: Quant,
    needs: Needs,
    placement: Placement,
    speed: SpeedEstimate,
    weights: Mapping[str, float] | None,
) -> Candidate:
    """Build a scored candidate, keeping every part that went into its total."""
    applied = dict(weights) if weights is not None else weights_for(needs.use_case)
    quality = score_quality(model, quant.name, needs)
    parts = {
        "quality": quality.quality,
        "speed": speed_score(speed, needs.use_case, needs.min_tps),
        "fit": fit_score_for(placement.budget),
        "context": context_score(placement.max_context_fit, requested_context(needs)),
    }
    total = sum(applied[part] * parts[part] for part in applied)
    return Candidate(
        model_id=model.id,
        quant=quant.name,
        placement=placement,
        speed=speed,
        quality=quality,
        score=ScoreBreakdown(
            quality=parts["quality"],
            speed=parts["speed"],
            fit=parts["fit"],
            context=parts["context"],
            weights=applied,
            # The weights sum to one, so the total is already on the scale; the clamp is
            # against the last bit of floating-point error, not against the arithmetic.
            total=max(0.0, min(100.0, total)),
        ),
    )


def _exclusion(
    model: CatalogModel,
    quant: Quant,
    needs: Needs,
    placement: Placement | None,
    speed: SpeedEstimate | None,
) -> str | None:
    """Return why this candidate cannot be ranked, or ``None`` when it can.

    The order of the checks is the order of what a person can do about them. What the
    request asked for comes first, because that is the part the reader controls outright:
    the job itself, then a required capability, then the download ceiling they set. The
    machine comes second, because "it does not fit" is only worth saying once it is clear
    the model was wanted at all. Each reason names the thing to change, and only the first
    is reported — a list of every way a candidate failed is a worse answer than the one
    that comes first.

    The job comes before the capability because it is the broader statement about the
    same thing: a model whose entry does not offer itself for this work at all should not
    be explained away by whichever capability it also happens to lack.

    **The reading floor is the last check, and it is a check and not a penalty.** A model
    that generates more slowly than its reader reads cannot do an interactive job at any
    quality, and section 11.4's floor already says so — it scores such a candidate zero.
    Zero was not enough on its own: the other three parts carry three quarters of every
    use case's weight, so a slow, large, well-fitting model finishes above a fast one that
    can actually be used, which is what Gemma 3 27B at 2.1 tokens per second did to Llama
    3.1 8B at 10.2 on the reference machine. Reweighting cannot answer it, because the
    complaint is not that the candidate was ranked too high. It is that a batch tool was
    entered in a race about waiting.

    It comes last because it is the most machine-ish of the machine's reasons: it needs a
    placement, and then a speed estimated from that placement, so everything that could
    have been said about the request or about the memory has already been said. And the
    exclusion changes nothing about what a reader sees except the words: an excluded
    candidate keeps its placement and its speed and is shown, so the row that used to read
    "ranked third, 2.1 tokens per second" now reads "2.1 tokens per second, below the six a
    person reads at" — which is the same number with the consequence attached.

    Six is the default and not the only answer. ``--min-tps`` puts the request's own figure
    in its place, and ``--min-tps 0`` says nobody is waiting, which is the batch case and
    excludes nothing here. The reason then names the figure it actually used and says which
    flag set it, because a reader who moved a boundary should meet the boundary they moved
    and not the one the specification argued for.
    """
    if not declares_use_case(model, needs.use_case):
        return _(
            "not a %(use_case)s model; its entry lists %(use_cases)s, so ask for one of those"
        ) % {"use_case": needs.use_case, "use_cases": ", ".join(model.use_cases)}
    missing = missing_capabilities(model, needs)
    if missing:
        return _("no %(capability)s capability; drop it from the request to see this model") % {
            "capability": missing[0]
        }
    if (
        needs.max_download_bytes is not None
        and quant.bytes_ is not None
        and quant.bytes_ > needs.max_download_bytes
    ):
        return _("%(size)s to download, over the %(limit)s this request allows") % {
            "size": format_bytes(quant.bytes_),
            "limit": format_bytes(needs.max_download_bytes),
        }
    if penalty_for(quant.name) is None:
        return _("unknown quantisation %(quant)s, so its cost in quality cannot be told") % {
            "quant": quant.name
        }
    if placement is None:
        return _("nowhere to put it: no placement fits this machine at any context")
    if placement.mode == "unsupported":
        return _("no run mode supports this model on this machine")
    if placement.budget.verdict == "does-not-fit":
        return _("needs more memory than this machine has, even at its smallest context")
    if placement.max_context_fit < needs.min_context:
        return _(
            "holds %(fit)s tokens at most, under the %(minimum)s asked for; "
            "lower the minimum context or free memory to see it ranked"
        ) % {
            "fit": format_grouped(placement.max_context_fit),
            "minimum": format_grouped(needs.min_context),
        }
    if speed is None:
        return _("no speed estimate, so it cannot be ranked against models that have one")
    if not keeps_up_with_reader(speed, needs.use_case, needs.min_tps):
        observed = _tps(observed_tps(speed, needs.use_case))
        floor = _tps(floor_tps(needs.use_case, needs.min_tps))
        if needs.min_tps is not None:
            return _(
                "generates %(tps)s tokens per second, below the %(floor)s this request asks "
                "for; lower --min-tps or choose a smaller model or quantisation"
            ) % {"tps": observed, "floor": floor}
        return _(
            "generates %(tps)s tokens per second, below the %(floor)s a person reads at: "
            "a batch tool on this machine and not one to sit in front of; "
            "a smaller model or quantisation would keep up"
        ) % {"tps": observed, "floor": floor}
    return None


def _tps(value: float) -> str:
    """A tokens-per-second figure for a sentence, in this language's punctuation.

    One decimal, and none at all when the figure is a whole number: the floor is six and
    reads better in a sentence as six than as 6.0, while an estimate of 2.65 has to keep
    its fraction or the reason would round a model up to three tokens a second.
    """
    whole = value == int(value)
    return localise_number(f"{value:,.0f}" if whole else f"{value:,.1f}")
