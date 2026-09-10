# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Data models for sizing a model, placing it, estimating its speed and ranking it.

These are the types phase 1C's four modules hand to one another, and they are written
down together because the seam between them is where a mistake would be expensive: the
budget decides what fits, the planner decides where the bytes go, the estimator turns
that placement into a speed, and the scorer turns the speed into a ranking. Each step
consumes the one before it.

Two habits run through all of them. Every number a user will see carries a
:class:`Confidence`, because a figure derived from a formula and a figure measured on
this machine are not the same claim and must never look alike. And every composite
number keeps the parts it was made from, so a recommendation can always be expanded
into the inputs that produced it rather than asked to be taken on faith.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ByteSize = Annotated[int, Field(ge=0)]
"""A size in bytes. Never negative, whoever supplied it."""

Score = Annotated[float, Field(ge=0, le=100)]
"""A score on the project's common 0 to 100 scale."""

Pool = Literal["vram", "ram", "disk"]
"""Where a component's bytes live. ``disk`` means streamed rather than held."""

RunMode = Literal["gpu", "moe-offload", "hybrid", "cpu", "unsupported"]
"""How the weights are divided between the graphics card and system memory."""

Verdict = Literal["comfortable", "fits", "tight", "too-tight", "does-not-fit"]
"""How well a configuration fits, decided on whichever pool is worst.

``too-tight`` is separate from ``does-not-fit`` on purpose. An NVIDIA driver does not
refuse an allocation larger than the card: it pages to system memory and the speed
collapses, silently, so the server starts and looks healthy while running at a fraction
of its speed. A configuration that would do that has to be named, not merely rejected,
because the user would otherwise never learn why their machine felt slow.
"""

Confidence = Literal["measured", "calibrated", "estimated", "unsupported"]
"""How much weight a number deserves, in decreasing order.

``measured`` is a benchmark taken on this machine for this model, quant and flags.
``calibrated`` is the formula corrected by measurements from this machine. ``estimated``
is the formula with the project's default constants. ``unsupported`` means no number
could be produced at all. The label travels with the number into every interface, and a
measured number always carries the date it was taken.
"""


class BudgetLine(BaseModel):
    """One component's contribution to what a configuration needs.

    Attributes:
        component: What this is, for example ``kv-cache`` or ``compute-buffer``.
        pool: Where the bytes live.
        bytes_: How many bytes it needs.
        exact: True when the figure comes from the file's own tensor table, false when
            it comes from a formula. A budget mixes both, and a reader deserves to know
            which lines are arithmetic on measured sizes and which are models of
            something that has to be predicted.
        note: Why this line is what it is, when that is not obvious.
    """

    model_config = ConfigDict(extra="forbid")

    component: str
    pool: Pool
    bytes_: ByteSize = Field(alias="bytes")
    exact: bool
    note: str | None = None


class Budget(BaseModel):
    """What one configuration of one model needs, component by component.

    Attributes:
        lines: Every component, in the order a reader should meet them.
        vram_required: Sum of the lines that live on the graphics card.
        ram_required: Sum of the lines that live in system memory.
        vram_available: What the card has free, less the reserve left for the desktop.
        ram_available: What the machine has free.
        vram_utilisation: Required over available, or ``None`` with no card.
        ram_utilisation: Required over available.
        verdict: Decided on whichever pool is worse.
    """

    model_config = ConfigDict(extra="forbid")

    lines: tuple[BudgetLine, ...]
    vram_required: ByteSize
    ram_required: ByteSize
    vram_available: ByteSize
    ram_available: ByteSize
    vram_utilisation: float | None
    ram_utilisation: float
    verdict: Verdict


class ContextTier(BaseModel):
    """One rung of the context ladder, and what it costs to stand on it.

    A launch script reads this at start time and picks the largest tier the free VRAM
    allows, which is how the same script keeps working when a browser is open.

    Attributes:
        tokens: The context length this tier offers.
        vram_required: What that context needs on the card.
        fits: Whether it fits the machine as scanned.
        verdict: How it fits, or how it fails. ``fits`` cannot tell the two failures apart
            and they are not the same thing to a reader: a rung that overflows the card
            still starts and then pages, silently, while a rung that overflows system
            memory does not start at all. Carrying the verdict is what lets a ladder show
            the paging band rather than one flat "no", and it is already computed —
            costing a rung means computing a whole budget, and the verdict was being
            thrown away.
    """

    model_config = ConfigDict(extra="forbid")

    tokens: int = Field(gt=0)
    vram_required: ByteSize
    fits: bool
    verdict: Verdict = "fits"


class Placement(BaseModel):
    """Where a model's bytes go, and the settings that put them there.

    Attributes:
        mode: How the weights are divided.
        context: The context length this placement is for.
        micro_batch: The micro-batch size, which drives the compute buffer.
        batch: The batch size.
        kv_type: The cache type, for example ``f16`` or ``q8_0``.
        gpu_layers: What ``-ngl`` would be.
        cpu_moe_layers: What ``--n-cpu-moe`` would be, or ``None``.
        projector_pool: Where the vision projector goes, or ``None`` when the model has
            none or it was left out.
        shared_experts_pool: Where the always-on shared experts go, or ``None`` when they
            stay with their layers. ``ram`` is what ``-ot ffn_.*_shexp=CPU`` does: they
            run for every token, so moving them costs speed, but they are the one part of
            a mixture-of-experts model that can leave the card without the attention path
            following, and on the reference machine it is what the best measured
            configuration does.
        threads: How many threads to run, chosen from performance cores.
        budget: What this placement needs.
        max_context_fit: The largest context this mode fits.
        tiers: The context ladder, for a launch script to choose from at start time.
        notes: Anything a reader should know, for example that vision was turned off to
            make the rest fit.
    """

    model_config = ConfigDict(extra="forbid")

    mode: RunMode
    context: int = Field(gt=0)
    micro_batch: int = Field(gt=0)
    batch: int = Field(gt=0)
    kv_type: str
    gpu_layers: int = Field(ge=0)
    cpu_moe_layers: int | None = None
    projector_pool: Pool | None = None
    shared_experts_pool: Pool | None = None
    threads: int = Field(gt=0)
    budget: Budget
    max_context_fit: int = Field(ge=0)
    tiers: tuple[ContextTier, ...] = ()
    notes: tuple[str, ...] = ()


class SpeedEstimate(BaseModel):
    """How fast a placement is expected to run, and how much to trust that.

    Both figures carry their terms, and each trio adds up to one over the figure it
    belongs to. Generation's three are section 10.1's; prompt processing's three are
    section 10.2's, per prompt token rather than per micro-batch so that the two read the
    same way. Three more fields rather than a second shape, because a reader who has
    learnt to check one figure has then learnt to check the other.

    Attributes:
        gen_tps: Generated tokens per second.
        pp_tps: Prompt tokens per second.
        confidence: How the two figures were arrived at.
        measured_on: The date a measured figure was taken, and ``None`` otherwise.
        vram_seconds_per_token: The share of a token's time spent reading the card.
        ram_seconds_per_token: The share spent reading system memory.
        overhead_seconds_per_token: Per-layer and sampling overheads.
        prompt_compute_seconds_per_token: The share of a prompt token's time spent on
            arithmetic.
        prompt_link_seconds_per_token: The share spent streaming the expert set across
            the link to the card. Its own term because its constant is the one this
            project knows least about: it was fitted on the single model whose expert set
            does not fit in system memory, and a model whose set stays in the page cache
            streams about three times faster. Inside a total that gap is invisible;
            standing alone the term says how much of the answer rests on it.
        prompt_ram_seconds_per_token: The share spent reading the expert set out of
            system memory, which happens instead of the term above when there is no card.
        notes: Why the number is what it is, for example that the expert set is read in
            a scattered pattern and reaches a third of the machine's sequential
            bandwidth rather than most of it.
    """

    model_config = ConfigDict(extra="forbid")

    gen_tps: float = Field(ge=0)
    pp_tps: float = Field(ge=0)
    confidence: Confidence
    measured_on: date | None = None
    vram_seconds_per_token: float = Field(default=0.0, ge=0)
    ram_seconds_per_token: float = Field(default=0.0, ge=0)
    overhead_seconds_per_token: float = Field(default=0.0, ge=0)
    prompt_compute_seconds_per_token: float = Field(default=0.0, ge=0)
    prompt_link_seconds_per_token: float = Field(default=0.0, ge=0)
    prompt_ram_seconds_per_token: float = Field(default=0.0, ge=0)
    notes: tuple[str, ...] = ()


class QualityBreakdown(BaseModel):
    """A model's quality for one request, and the three parts it is made of.

    Attributes:
        baseline: The curator's editorial score, before anything is taken off.
        quant_penalty: What this quantisation costs.
        alignment_bonus: What matching the request is worth.
        quality: The three combined and clamped to the scale.
    """

    model_config = ConfigDict(extra="forbid")

    baseline: Score
    quant_penalty: float = Field(ge=0)
    alignment_bonus: float = Field(ge=0)
    quality: Score


class ScoreBreakdown(BaseModel):
    """A composite score and every part that went into it.

    Kept whole rather than reduced to one number because a ranking whose reason is
    invisible asks a reader to take it on faith, and this project's standing rule is
    that any number can be expanded into its inputs.

    Attributes:
        quality: Quality after the quantisation penalty and the alignment bonus.
        speed: How the estimated speed compares with what this use case needs.
        fit: How well the model uses the machine, penalising both ends.
        context: How much of the requested context it can actually hold.
        weights: What each part counted for, which varies by use case.
        total: The weighted sum.
    """

    model_config = ConfigDict(extra="forbid")

    quality: Score
    speed: Score
    fit: Score
    context: Score
    weights: dict[str, float]
    total: Score


class Candidate(BaseModel):
    """One model, one quantisation, placed, estimated and scored for one request.

    Attributes:
        model_id: The catalog id.
        quant: The quantisation name.
        placement: Where its bytes would go.
        speed: How fast it is expected to run there.
        quality: Its quality for this request.
        score: The composite, with its parts.
        excluded_because: Why it is not a candidate at all, when it is not. A candidate
            that fails is kept rather than dropped, because "this one was excluded and
            here is why" is far more use than a silently shorter list.
        excluded_tag: The same answer in one or two words -- "no room", "too slow", "no
            vision" -- for a surface that has a cell rather than a paragraph. A list that
            gives one row a sentence and its neighbours a number stops being a list, and
            fifty sentences are read by nobody; the tag keeps the grid a grid and the
            sentence keeps its place where the row is opened. Both are decided in the same
            branch of the same function, so they cannot come to disagree.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    quant: str
    placement: Placement | None = None
    speed: SpeedEstimate | None = None
    quality: QualityBreakdown | None = None
    score: ScoreBreakdown | None = None
    excluded_because: str | None = None
    excluded_tag: str | None = None


class Needs(BaseModel):
    """What a person is asking for, which is what everything above is judged against.

    Attributes:
        use_case: The primary job, which sets the score weights and the speed target.
        capabilities: What the model must be able to do. A model missing one of these
            is excluded rather than penalised.
        min_context: The smallest context worth having.
        min_tps: The slowest generation worth having, in tokens per second, or ``None``
            to use section 11.4's reading floor for the use case. Zero says nobody is
            waiting on the tokens, which is the batch case: no candidate is then
            excluded for being slow. See :func:`~llamafit.scoring.speed_score.floor_tps`.
        requested_context: What to size for, defaulting by use case.
        max_context: A ceiling on every context this request sizes for, reports or
            scores against, or ``None`` for the model's own native length. It stands
            beside ``min_context`` and not beside the simulation options: a substituted
            machine changes what the arithmetic runs on, and this changes the question
            being asked of it.
        allow_kv_quant: Whether the cache may be quantised to buy context.
        max_download_bytes: A ceiling on what the user is willing to fetch.

    ``min_tps`` is ``None`` rather than zero when it is unset, and the two mean opposite
    things: unset is "use the figure the specification derived from reading rates", zero
    is "there is no such figure for this request". A single number could not carry both,
    and a default of zero would have quietly turned the exclusion off for everybody.

    A ``max_context`` below ``min_context`` is refused here rather than at each
    interface. No machine and no model could satisfy it, and the planner would otherwise
    answer it with "this model holds fewer tokens than you asked for", which names the
    wrong culprit.
    """

    model_config = ConfigDict(extra="forbid")

    use_case: str = "general"
    capabilities: tuple[str, ...] = ()
    min_context: int = Field(default=0, ge=0)
    min_tps: float | None = Field(default=None, ge=0)
    requested_context: int | None = None
    max_context: int | None = Field(default=None, ge=1)
    allow_kv_quant: bool = True
    max_download_bytes: ByteSize | None = None

    @model_validator(mode="after")
    def _ceiling_above_floor(self) -> Needs:
        """Refuse a context ceiling below the context floor."""
        if self.max_context is not None and self.max_context < self.min_context:
            raise ValueError(
                f"max_context {self.max_context} is below min_context {self.min_context}"
            )
        return self
