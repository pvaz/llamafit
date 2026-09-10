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
from llamafit.scoring.speed_score import speed_score
from llamafit.scoring.weights import weights_for
from llamafit.units import format_bytes, format_grouped


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
        "speed": speed_score(speed, needs.use_case),
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
    return None
