# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The join between section 8's budget and section 9's search, and the plan built on it.

The planner takes what costs a configuration as an injected callable so the two can be
read, tested and changed apart, and section 8 knows nothing about placements. Something
has to hold them together, and it is this: the one function that turns a
:class:`~llamafit.placement.modes.PlacementSettings` into a
:class:`~llamafit.models.plan.Budget`, and the one that hands it to the planner.

It is a service rather than a member of either package because either home would be
wrong. In ``budget`` it would make section 8 depend on section 9, which is backwards; in
``placement`` it would close the seam the planner deliberately leaves open.

Small as it is, the absence of it was a real gap. Both ends existed for a while and
nothing joined them, so the budget could cost a configuration the planner could not
produce, and everything looked finished from either side.

On top of that join sits :func:`plan_report`, which is what ``llamafit plan`` prints: one
model, one quantisation, the placement, the budget behind it, the context ladder a launch
script chooses from, the speed and the command line a person pastes. Two of its fields
exist because a plan that only showed what fits would hide the thing section 8.4 exists to
name. ``requested_budget`` is what the context the user actually asked for would cost, kept
whenever the planner had to size down, so the reader sees the overflow rather than only the
retreat from it. And ``measurements`` carries the catalog's own recorded runs beside the
estimate, never in place of it, so the two can be compared.

Two more things sit on the same join, and both answer the question a person asks straight
after "what do I get": what would I have to change. :func:`size_quants` costs every
quantisation a model publishes against one machine, which is section 13.1's budget per
quant for ``info`` — the breadth ``plan`` does not have, stopped one step short of the
command line so the two commands do not print the same screen. :func:`counterfactuals`
answers section 12.3 for one candidate: the next rung of the context ladder this machine
cannot take and what freeing the card would buy, and the next quantisation down.

The rule both obey is the one that makes them worth printing. **Nothing here is arithmetic
on a placement somebody else computed.** The context rung carries the budget computed for
that rung; the quantisation carries a placement searched for that quantisation. A file four
fifths the size does not make a budget four fifths the size, and a sentence promising that a
smaller quantisation would fit when it would not is worse than printing no sentence at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from pathlib import PurePath

from pydantic import BaseModel, ConfigDict, Field, computed_field

from llamafit.budget import compute
from llamafit.constants import DEFAULT_REQUESTED_CONTEXT, UTILISATION_TIGHT
from llamafit.errors import CatalogError, LlamaFitError
from llamafit.i18n import _
from llamafit.models.catalog import CatalogModel, Extra, Measured, Quant
from llamafit.models.host import Host, Simulation
from llamafit.models.plan import (
    Budget,
    ByteSize,
    ContextTier,
    Needs,
    Placement,
    RunMode,
    SpeedEstimate,
    Verdict,
)
from llamafit.paths import get_paths
from llamafit.placement import (
    LaunchOptions,
    PlacementSettings,
    command_line,
    context_tiers,
    max_context_fit,
    placement_notes,
    plan_placement,
    render_flags,
)
from llamafit.placement.modes import context_ceiling, projector_of, thread_count
from llamafit.services.catalog import ModelDetail, QuantDetail
from llamafit.speed import estimate_speed


def budget_for(
    model: CatalogModel, quant: Quant, host: Host, settings: PlacementSettings, /
) -> Budget:
    """Cost one configuration: section 9's knobs in, section 8's budget out.

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        host: The scanned machine.
        settings: The knobs the search is currently holding.

    Returns:
        What that configuration needs, component by component.

    The vision projector is looked up here rather than carried in the settings, because
    the settings say *where* it goes and the catalog says *how large it is*, and only one
    of those two belongs to the search.
    """
    return compute(
        model,
        quant,
        host,
        context=settings.context,
        mode=settings.mode,
        micro_batch=settings.micro_batch,
        batch=settings.batch,
        kv_type=settings.kv_type,
        gpu_layers=settings.gpu_layers,
        cpu_moe_layers=settings.cpu_moe_layers,
        projector=projector_of(model, quant),
        projector_pool=settings.projector_pool,
        shared_experts_pool=settings.shared_experts_pool,
    )


def plan_model(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    *,
    needs: Needs | None = None,
    vision: bool = True,
) -> Placement:
    """The best placement for one model and quantisation on one machine.

    Args:
        model: The catalog entry.
        quant: The quantisation to place.
        host: The scanned machine.
        needs: What the user asked for; the project's defaults when omitted.
        vision: Whether to try to keep the vision projector at all.

    Returns:
        The placement, with the budget that decided it and the flags it implies.
    """
    return plan_placement(model, quant, host, budget_for=budget_for, needs=needs, vision=vision)


class TargetCheck(BaseModel):
    """Whether a plan reaches a speed the user named, and what would.

    ``--target-tps`` asks a question the rest of the plan cannot answer on its own: not
    "how fast is this" but "what would I have to give up to make it fast enough". The
    answer is a context, because context is the one thing a reader can trade for speed
    without changing the model, the quantisation or the machine.

    Attributes:
        target_tps: The generation speed asked for.
        reached: Whether the planned configuration already reaches it.
        planned_tps: What the planned configuration is estimated to generate.
        best_context: The largest context of section 9.3's ladder whose estimate reaches
            the target, or ``None`` when no rung does.
        best_tps: What the fastest rung tried is estimated to generate, which is what a
            reader is owed when the answer is that nothing reaches the target.
    """

    model_config = ConfigDict(extra="forbid")

    target_tps: float = Field(gt=0)
    reached: bool
    planned_tps: float = Field(ge=0)
    best_context: int | None = None
    best_tps: float = Field(default=0.0, ge=0)


class PlanReport(BaseModel):
    """One model, one quantisation, placed on this machine and ready to launch.

    Attributes:
        model_id: The catalog id.
        name: The model's display name.
        quant: The quantisation planned.
        placement: Where the bytes go, with its budget, context ladder and notes.
        speed: How fast it is expected to run, at the context planned for.
        requested_context: The context the request asked to be sized for.
        requested_budget: What that context would have cost, kept only when the planner
            had to size below it. This is where a reader meets section 8.4: a
            configuration over the card does not fail, it pages, and a plan that showed
            only the smaller configuration it retreated to would never say so.
        flags: The ``llama-server`` arguments, in section 9.5's order.
        command: The same list with the program name in front, ready to paste.
        model_path: The GGUF file the command line names.
        model_present: Whether that file is already on this machine.
        projector_path: The vision projector the command line names, when there is one.
        download_bytes: What fetching this quantisation would cost, when known.
        measurements: The benchmarks the catalog records for this model and quantisation,
            to be shown beside the estimate rather than in place of it.
        target: The ``--target-tps`` answer, when one was asked for.
        simulation: What was substituted for the machine this plan was computed on, or
            ``None`` when it was computed on the machine the reader is sitting at.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    name: str
    quant: str
    placement: Placement
    speed: SpeedEstimate | None = None
    requested_context: int
    requested_budget: Budget | None = None
    flags: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    model_path: str
    model_present: bool = False
    projector_path: str | None = None
    download_bytes: int | None = None
    measurements: list[Measured] = Field(default_factory=list)
    target: TargetCheck | None = None
    simulation: Simulation | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def simulated(self) -> bool:
        """Whether this plan was computed for a machine other than this one.

        The pair :class:`~llamafit.models.host.Host` carries, for the same reason: a plan
        carries no host, so this is the only thing in ``plan --json`` that distinguishes
        a command line sized for the reader's card from one sized for somebody else's.
        """
        return self.simulation is not None


def settings_of(placement: Placement) -> PlacementSettings:
    """The knobs a placement was built from, for costing a variation of it.

    A :class:`~llamafit.models.plan.Placement` is settings plus results, and everything
    that recomputes one -- a forced micro-batch, the budget at the context the user asked
    for -- needs the settings back out. Written once here so no caller reconstructs them
    by hand and quietly leaves one field behind.
    """
    return PlacementSettings(
        mode=placement.mode,
        context=placement.context,
        micro_batch=placement.micro_batch,
        batch=placement.batch,
        kv_type=placement.kv_type,
        gpu_layers=placement.gpu_layers,
        cpu_moe_layers=placement.cpu_moe_layers,
        projector_pool=placement.projector_pool,
        shared_experts_pool=placement.shared_experts_pool,
    )


def replan(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    settings: PlacementSettings,
    *,
    requested_context: int,
    vision: bool,
    ceiling: int | None = None,
) -> Placement:
    """Rebuild a placement around settings a user chose by hand, keeping everything true.

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        host: The scanned machine.
        settings: The knobs to cost.
        requested_context: What the user asked to be sized for, for the notes.
        vision: Whether vision was wanted at all, for the notes.
        ceiling: The longest context to offer, from
            :func:`~llamafit.placement.modes.context_ceiling`; the model's native length
            when the caller states none.

    Returns:
        A placement for ``settings``, with the budget, the context ladder, the largest
        context it holds and the notes all recomputed.

    A hand-set micro-batch is not a cosmetic change: it moves the compute buffer, which
    moves the verdict, the largest context that fits and every rung of the ladder. Reusing
    any of the planner's answers would print a budget for one configuration under the
    flags of another, which is the failure ``--fit off`` exists to prevent, arrived at
    from inside.
    """
    budget = budget_for(model, quant, host, settings)
    threads = thread_count(host.cpu)
    return settings.to_placement(
        budget,
        threads=threads,
        max_context_fit=max_context_fit(
            model, quant, host, settings, budget_for=budget_for, ceiling=ceiling
        ),
        tiers=context_tiers(model, quant, host, settings, budget_for=budget_for, ceiling=ceiling),
        notes=placement_notes(
            model,
            quant,
            host,
            settings,
            budget,
            requested_context=requested_context,
            threads=threads,
            vision=vision,
        ),
    )


def launch_paths(
    model: CatalogModel,
    quant: Quant,
    projector: Extra | None,
    local_files: Sequence[str],
) -> tuple[str, bool, str | None]:
    """Where the files are, or where an installer would put them.

    Args:
        model: The catalog entry.
        quant: The quantisation being planned.
        projector: The vision projector to name, when the plan keeps one.
        local_files: Paths of the GGUF files llama.cpp already has.

    Returns:
        The model path, whether it is already there, and the projector path when the
        model publishes one.

    A command line has to name a file whether or not the file exists yet, and the honest
    stand-in is the directory ``llamafit install model`` will write to. The flag that says
    which of the two a reader is looking at is ``model_present``, so nobody pastes a path
    believing a download has happened.
    """
    by_name = {PurePath(path).name: path for path in local_files}
    downloads = get_paths().downloads_dir / model.id
    first = quant.files[0] if quant.files else f"{model.id}-{quant.name}.gguf"
    # A catalog file name is a path inside the publishing repository and often carries a
    # directory; a local file is wherever its owner put it. Only the bare names can be
    # compared, or every sharded model reads as missing from a machine that has it.
    found = by_name.get(PurePath(first).name)
    model_path = found if found is not None else str(downloads / first)
    projector_path: str | None = None
    if projector is not None:
        projector_path = by_name.get(PurePath(projector.file).name) or str(
            downloads / projector.file
        )
    return model_path, found is not None, projector_path


def _speed_at(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    placement: Placement,
    context: int,
) -> SpeedEstimate | None:
    """Estimate one placement at one context, or ``None`` without the file's facts.

    No measurement is passed in, so nothing here can come back labelled ``measured``:
    section 10.3 reserves that for a benchmark taken on this machine, and phase 3 is what
    will store one. The catalog's own records travel beside the estimate instead, on
    :attr:`PlanReport.measurements`.
    """
    facts = quant.gguf_facts
    if facts is None:
        return None
    return estimate_speed(
        placement,
        facts,
        host,
        working_context=context,
        active_params=model.params.active_b * 1e9,
        quant=quant.name,
    )


def _target_check(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    placement: Placement,
    tiers: Sequence[ContextTier],
    *,
    target_tps: float,
    planned_tps: float,
) -> TargetCheck:
    """Answer ``--target-tps``: does this reach the speed, and what context would.

    Every rung of the ladder that fits is estimated at its own context, because the KV
    cache a token reads grows with the context and is the only term of section 10.1 that
    does. The largest rung that reaches the target is the answer; when none does, the
    fastest figure found is reported instead, so that "no" comes with a number rather than
    alone.
    """
    best_context: int | None = None
    best_tps = planned_tps
    for tier in tiers:
        if not tier.fits:
            continue
        estimate = _speed_at(
            model, quant, host, placement.model_copy(update={"context": tier.tokens}), tier.tokens
        )
        if estimate is None:
            continue
        best_tps = max(best_tps, estimate.gen_tps)
        if estimate.gen_tps >= target_tps:
            best_context = tier.tokens if best_context is None else max(best_context, tier.tokens)
    return TargetCheck(
        target_tps=target_tps,
        reached=planned_tps >= target_tps,
        planned_tps=planned_tps,
        best_context=best_context,
        best_tps=best_tps,
    )


def plan_report(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    *,
    needs: Needs | None = None,
    vision: bool = True,
    micro_batch: int | None = None,
    target_tps: float | None = None,
    local_files: Sequence[str] = (),
    alias: str | None = None,
) -> PlanReport:
    """Everything ``llamafit plan`` shows for one model and one quantisation.

    Args:
        model: The catalog entry.
        quant: The quantisation to plan.
        host: The scanned machine.
        needs: What the user asked for; the project's defaults when omitted.
        vision: Whether to try to keep the vision projector.
        micro_batch: A micro-batch chosen by hand, which the whole budget is rebuilt
            around rather than merely printed.
        target_tps: A generation speed to answer against.
        local_files: Paths of the GGUF files llama.cpp already has, so the command line
            names a real file when there is one.
        alias: What the server should call the model; its catalog id when unset.

    Returns:
        The plan, ready to render or to serialise.

    Raises:
        BudgetError: If nobody has read this quantisation's header, so it cannot be sized.
    """
    needs = needs or Needs()
    ceiling = context_ceiling(model, needs)
    placement = plan_model(model, quant, host, needs=needs, vision=vision)
    # The context asked for, but never above the ceiling: a report that costed a context
    # the request itself forbade would print a budget for a configuration nobody wanted.
    wanted = min(max(needs.requested_context or placement.context, needs.min_context), ceiling)

    if micro_batch is not None and micro_batch != placement.micro_batch:
        placement = replan(
            model,
            quant,
            host,
            settings_of(placement).with_micro_batch(micro_batch),
            requested_context=wanted,
            vision=vision,
            ceiling=ceiling,
        )

    requested_budget: Budget | None = None
    if placement.mode != "unsupported" and placement.context < wanted:
        requested_budget = budget_for(
            model, quant, host, settings_of(placement).with_context(wanted)
        )

    speed = _speed_at(model, quant, host, placement, placement.context)
    projector = projector_of(model, quant) if vision else None
    model_path, present, projector_path = launch_paths(model, quant, projector, local_files)
    launch = LaunchOptions(model_path=model_path, projector_path=projector_path, alias=alias)

    target = (
        None
        if target_tps is None or speed is None
        else _target_check(
            model,
            quant,
            host,
            placement,
            placement.tiers,
            target_tps=target_tps,
            planned_tps=speed.gen_tps,
        )
    )

    return PlanReport(
        model_id=model.id,
        name=model.name,
        quant=quant.name,
        placement=placement,
        speed=speed,
        requested_context=wanted,
        requested_budget=requested_budget,
        flags=render_flags(placement, model, launch),
        command=command_line(placement, model, launch),
        model_path=model_path,
        model_present=present,
        projector_path=projector_path,
        download_bytes=quant.bytes_,
        measurements=[m for m in model.measured if m.quant == quant.name],
        target=target,
        simulation=host.simulation,
    )


def quant_named(model_id: str, quants: Sequence[Quant], name: str) -> Quant:
    """The quantisation a user named, matched the way a model id is matched.

    Args:
        model_id: The catalog id, for the messages; a reader who mistyped a quant name
            has to be told which model was being asked about.
        quants: What the model publishes, in catalog order.
        name: What ``--quant`` said, with any surrounding whitespace and whatever case
            the user's shell history happened to hold.

    Returns:
        The quantisation with that name.

    Raises:
        CatalogError: If the model publishes none at all, or none by that name; the
            second message lists the names it does publish, because a reader who
            mistyped ``UD-Q4_K_XL`` cannot be expected to guess the underscores.

    One matcher rather than one per command. ``plan --quant`` and ``info --quant`` ask the
    same question of the same catalog, and two spellings of "case-insensitively" would
    eventually disagree about a name with a capital in it, which is every name here.
    """
    if not quants:
        raise CatalogError(
            _("%(model)s publishes no quantisations") % {"model": model_id},
            hint=_("Run `llamafit catalog refresh --model %(model)s`.") % {"model": model_id},
        )
    wanted = name.strip().casefold()
    for quant in quants:
        if quant.name.casefold() == wanted:
            return quant
    raise CatalogError(
        _("%(model)s has no %(quant)s quantisation") % {"model": model_id, "quant": repr(name)},
        hint=_("It publishes: %(names)s") % {"names": ", ".join(q.name for q in quants)},
    )


def _sized(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    *,
    needs: Needs,
    vision: bool,
    working_context: int | None = None,
) -> tuple[Placement | None, SpeedEstimate | None, str | None]:
    """Plan and estimate one quantisation, turning a refusal to size it into a sentence.

    The budget raises rather than guesses when nobody has read a quant's header, which is
    the right answer -- a size nobody has read is not a size -- and it reaches a table as
    one row saying so rather than as a traceback that loses the other two.
    """
    try:
        placement = plan_model(model, quant, host, needs=needs, vision=vision)
    except LlamaFitError as exc:
        return None, None, exc.message
    context = placement.context if working_context is None else working_context
    return placement, _speed_at(model, quant, host, placement, context), None


def sizing_context(model: CatalogModel, needs: Needs | None = None) -> int:
    """The context a sizing run will actually use, which is the only one a caption may name.

    Args:
        model: The catalog entry, whose native length is a ceiling nothing raises.
        needs: What the user asked for; the project's defaults when omitted.

    Returns:
        What the request asks to be sized for, never above what the model supports or what
        ``--max-context`` allows.

    ``--context 999999`` on a model that holds 131,072 is sized for 131,072, and a caption
    that named the figure the user typed would be labelling one configuration with another
    one's number. :func:`plan_report` caps the same request the same way, through the same
    :func:`~llamafit.placement.modes.context_ceiling`, so the two commands cannot disagree
    about what a request asked for.
    """
    needs = needs or Needs()
    wanted = max(needs.requested_context or DEFAULT_REQUESTED_CONTEXT, needs.min_context)
    return min(wanted, context_ceiling(model, needs))


def size_quants(
    detail: ModelDetail,
    host: Host,
    *,
    needs: Needs | None = None,
    vision: bool = True,
) -> ModelDetail:
    """Cost every quantisation a model publishes against one machine.

    Args:
        detail: The catalog's own description of the model, from
            :func:`~llamafit.services.catalog.describe`.
        host: The machine to size against.
        needs: What the user asked for; the project's defaults when omitted.
        vision: Whether to try to keep the vision projector.

    Returns:
        The same detail with :attr:`~llamafit.services.catalog.QuantDetail.placement`,
        ``speed`` and ``unplaceable_because`` filled in on every quantisation.

    This is the breadth ``plan`` does not have, and it stops exactly where ``plan``
    begins. ``plan`` takes one quantisation as far as a command line: the budget
    component by component, the ladder, where a token's time goes. This takes all of them
    one step, far enough to answer "which of these can this machine run, and how fast",
    and no further. A reader who has picked a row out of it goes to ``plan`` for the
    working behind that row, and neither command prints the other's screen.
    """
    needs = needs or Needs()
    by_name: dict[str, Quant] = {}
    for source in detail.model.sources:
        for quant in source.quants:
            by_name.setdefault(quant.name, quant)

    sized: list[QuantDetail] = []
    for entry in detail.quants:
        published = by_name.get(entry.name)
        if published is None:  # pragma: no cover - describe() reads these same sources
            sized.append(entry)
            continue
        placement, speed, problem = _sized(
            detail.model, published, host, needs=needs, vision=vision
        )
        sized.append(
            entry.model_copy(
                update={"placement": placement, "speed": speed, "unplaceable_because": problem}
            )
        )
    # The simulation travels with the figures, not with the command that asked for them:
    # a document that says which machine it is about is the only thing that can still say
    # so once it has been piped somewhere else.
    return detail.model_copy(update={"quants": sized, "simulation": host.simulation})


MEANINGFUL_SPEEDUP = 1.05
"""How much faster another quantisation has to be before the difference is worth naming.

Five per cent, because the estimator is a formula with fitted constants and this project
does not claim it to the last token per second. A counterfactual offering a reader a two
per cent gain would be offering them the width of its own error bar as a reason to fetch
another forty gigabytes.
"""

_SEVERITY: dict[Verdict, int] = {
    "comfortable": 0,
    "fits": 1,
    "tight": 2,
    "too-tight": 3,
    "does-not-fit": 4,
}
"""How bad each verdict is, for saying whether another quantisation fits better.

The same order :mod:`llamafit.budget.budget` compares pools by, written again rather than
imported because that one is private to the module that *decides* a verdict and this one
only reads verdicts somebody else decided.
"""


class ContextReach(BaseModel):
    """The next rung of the context ladder this machine cannot take, and what it needs.

    Section 12.3's "free 1.2 GB of VRAM to reach 64K context", with every figure coming
    from a budget computed for that rung rather than from arithmetic on the rung below it.

    Attributes:
        tokens: The rung.
        vram_required: What that rung's own budget needs on the card.
        vram_available: What the card had free when it was costed, after the reserve left
            for the desktop.
        vram_to_free: How much more free card memory would bring that rung inside the band
            this program is willing to recommend, or zero when freeing the card is not
            what stands in the way.
        verdict: How the rung fails. ``too-tight`` is section 8.4's paging band and is the
            case ``vram_to_free`` answers; ``does-not-fit`` means nothing would absorb it
            and no amount of closing a browser reaches it.
    """

    model_config = ConfigDict(extra="forbid")

    tokens: int = Field(gt=0)
    vram_required: ByteSize
    vram_available: ByteSize
    vram_to_free: ByteSize
    verdict: Verdict


class QuantSwap(BaseModel):
    """The next quantisation down, planned on this machine rather than guessed at.

    Every field here is read off a placement this program computed for that quantisation.
    Nothing is scaled from the placement it is compared with: a file four fifths the size
    does not make a budget four fifths the size -- the cache, the compute buffer and the
    overheads do not shrink with the weights -- and a sentence saying a quantisation would
    fit when it would not is worse than no sentence at all.

    Attributes:
        quant: Its name.
        download_bytes: What fetching it would cost, when known.
        mode: How its weights would be divided, or ``None`` when it could not be sized.
        verdict: How it fits, or ``None`` when it could not be sized.
        max_context_fit: The largest context it holds in that mode.
        gen_tps: What it is estimated to generate, at the same context the placement it is
            compared with was estimated at, or ``None`` when it cannot be estimated.
        unplaceable_because: Why it could not be sized at all, when it could not.
        entirely_on_card: Whether it runs with everything on the graphics card.
        better_verdict: Whether it fits better than the placement it is compared with.
        more_context: Whether it holds more tokens.
        faster: Whether it generates at least :data:`MEANINGFUL_SPEEDUP` times as fast.
    """

    model_config = ConfigDict(extra="forbid")

    quant: str
    download_bytes: int | None = None
    mode: RunMode | None = None
    verdict: Verdict | None = None
    max_context_fit: int = 0
    gen_tps: float | None = None
    unplaceable_because: str | None = None
    entirely_on_card: bool = False
    better_verdict: bool = False
    more_context: bool = False
    faster: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def improves(self) -> bool:
        """Whether dropping to it would change anything a reader asked about.

        Computed rather than stored, so it cannot come to disagree with the three
        comparisons it is made of, and serialised, so a program reading ``--json`` does
        not have to re-derive the rule from them.
        """
        return self.better_verdict or self.more_context or self.faster


class Counterfactuals(BaseModel):
    """What would move one candidate, each answer from a placement actually computed.

    Section 12.3 asks an explanation to say what would move a candidate up a tier. There
    are exactly two things a reader can change without changing the machine or the model:
    how much of the card is free, and which quantisation they fetch. Both are answered
    here and neither is answered by arithmetic -- the context rung carries its own budget,
    and the quantisation carries its own placement.

    Attributes:
        context: The next rung this machine cannot take, or ``None`` when there is none
            above the one planned.
        every_context_fits: Whether every rung this model offers already fits, which is
            one of the two reasons ``context`` can be empty. The other is that the
            placement is already at the top of the ladder. They are not the same answer
            and a reader is owed the difference.
        quant: The next quantisation down, or ``None`` when the model publishes nothing
            smaller, or nothing whose size can be compared with this one's.
    """

    model_config = ConfigDict(extra="forbid")

    context: ContextReach | None = None
    every_context_fits: bool = False
    quant: QuantSwap | None = None


def _vram_to_free(tier: ContextTier, budget: Budget) -> int:
    """How much more free card memory would bring one rung inside the recommendable band.

    Section 8.3's thresholds are a share of what is free, so the question inverts: at what
    ``vram_available`` does ``vram_required / vram_available`` fall to
    :data:`~llamafit.constants.UTILISATION_TIGHT`? The requirement is the rung's own, out
    of the rung's own budget; only the denominator is solved for, and closing a browser is
    exactly a change to that denominator.

    Zero for ``does-not-fit``, which is not a card problem: either system memory is over on
    its own, or the overflow off the card is larger than what is left to page into, and in
    neither case does freeing the card reach the rung.
    """
    if tier.verdict != "too-tight":
        return 0
    # Rounded up, and then a byte more. The division is in floating point, and the exact
    # quotient can come back a hair above the threshold it was derived from -- 0.95 lands
    # as 0.9500000000603609 on the reference machine -- which would hand a reader a figure
    # one byte short of the thing it promises them.
    needed = ceil(tier.vram_required / UTILISATION_TIGHT) + 1
    return max(0, needed - budget.vram_available)


def _next_rung(placement: Placement) -> tuple[ContextReach | None, bool]:
    """The first rung above the planned context that does not fit, and whether any does not.

    Returns:
        The rung and ``False``, or ``None`` and whether every rung of the ladder fits.

    The ladder stops at the model's native length or at the request's ceiling, so "every
    rung fits" is a real answer rather than the absence of one: it says this machine is not
    what limits the context here.

    An empty ladder is neither answer. A placement that fits nowhere carries no rungs at
    all, and reading that as "every rung fits" would turn the worst answer this program can
    give into the best sentence on the page.
    """
    above = [tier for tier in placement.tiers if tier.tokens > placement.context]
    unreachable = [tier for tier in above if not tier.fits]
    if not unreachable:
        return None, bool(placement.tiers) and all(tier.fits for tier in placement.tiers)
    tier = unreachable[0]
    return (
        ContextReach(
            tokens=tier.tokens,
            vram_required=tier.vram_required,
            vram_available=placement.budget.vram_available,
            vram_to_free=_vram_to_free(tier, placement.budget),
            verdict=tier.verdict,
        ),
        False,
    )


def _comparable_sizes(quant: Quant, other: Quant) -> tuple[float, float] | None:
    """Two quantisations' sizes on one scale, or ``None`` when they share no scale.

    Bytes first, because bytes are what has to fit; bits per weight when a refresh has not
    filled the sizes in but both widths are known. Mixing the two would put a size beside a
    width and call one of them smaller.
    """
    if quant.bytes_ is not None and other.bytes_ is not None:
        return float(quant.bytes_), float(other.bytes_)
    if quant.bpw is not None and other.bpw is not None:
        return quant.bpw, other.bpw
    return None


def next_quant_down(quants: Sequence[Quant], current: Quant) -> Quant | None:
    """The largest quantisation this model publishes that is smaller than ``current``.

    Args:
        quants: Everything the model publishes.
        current: The one being explained.

    Returns:
        The next one down, or ``None`` when there is nothing smaller to compare with.

    One step rather than the best of all of them, because each step costs a whole placement
    search and an explanation is printed for every row a reader asked about. The step is
    also the one a reader would take: somebody who will not run Q4 tries Q3 before IQ1.
    """
    best: Quant | None = None
    best_size = 0.0
    for candidate in quants:
        if candidate.name == current.name:
            continue
        sizes = _comparable_sizes(current, candidate)
        if sizes is None:
            continue
        mine, theirs = sizes
        if theirs >= mine:
            continue
        if best is None or theirs > best_size:
            best, best_size = candidate, theirs
    return best


def _swap(
    model: CatalogModel,
    smaller: Quant,
    host: Host,
    placement: Placement,
    *,
    needs: Needs,
    vision: bool,
    gen_tps: float | None,
    working_context: int | None,
) -> QuantSwap:
    """Plan one step down, and say from that placement alone what it would change."""
    other, speed, problem = _sized(
        model, smaller, host, needs=needs, vision=vision, working_context=working_context
    )
    if other is None:
        return QuantSwap(
            quant=smaller.name, download_bytes=smaller.bytes_, unplaceable_because=problem
        )
    theirs = None if speed is None else speed.gen_tps
    placed = other.mode != "unsupported"
    return QuantSwap(
        quant=smaller.name,
        download_bytes=smaller.bytes_,
        mode=other.mode,
        verdict=other.budget.verdict,
        max_context_fit=other.max_context_fit,
        gen_tps=theirs,
        entirely_on_card=other.mode == "gpu",
        better_verdict=(
            placed and _SEVERITY[other.budget.verdict] < _SEVERITY[placement.budget.verdict]
        ),
        more_context=placed and other.max_context_fit > placement.max_context_fit,
        faster=(
            placed
            and theirs is not None
            and gen_tps is not None
            and gen_tps > 0
            and theirs >= gen_tps * MEANINGFUL_SPEEDUP
        ),
    )


def counterfactuals(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    placement: Placement,
    *,
    quants: Sequence[Quant],
    needs: Needs | None = None,
    vision: bool = True,
    gen_tps: float | None = None,
    working_context: int | None = None,
) -> Counterfactuals:
    """What a reader could change, with every answer costed rather than reasoned about.

    Args:
        model: The catalog entry.
        quant: The quantisation being explained.
        host: The machine it was planned on.
        placement: The placement being explained, whose ladder supplies the context rung.
        quants: Everything the model publishes, for the step down.
        needs: The request the placement was made for, so the alternative is planned
            against the same one.
        vision: Whether vision was wanted, for the same reason.
        gen_tps: What the placement being explained is estimated to generate, so the
            alternative can be called faster or not; ``None`` skips that comparison.
        working_context: The context to estimate the alternative at. A board estimates
            every row at one fixed context so the column compares like with like, and an
            alternative estimated at some other one would be the single figure on the page
            that does not.

    Returns:
        The context rung and the quantisation step, each ``None`` when this candidate has
        no such answer.

    This costs one whole placement search per call, which is why nothing calls it for every
    row of every board. ``--explain`` is where section 12.3 puts it and where a reader has
    already asked for the working.
    """
    reach, exhausted = _next_rung(placement)
    smaller = next_quant_down(quants, quant)
    swap = (
        None
        if smaller is None
        else _swap(
            model,
            smaller,
            host,
            placement,
            needs=needs or Needs(),
            vision=vision,
            gen_tps=gen_tps,
            working_context=working_context,
        )
    )
    return Counterfactuals(context=reach, every_context_fits=exhausted, quant=swap)
