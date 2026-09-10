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
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath

from pydantic import BaseModel, ConfigDict, Field

from llamafit.budget import compute
from llamafit.models.catalog import CatalogModel, Extra, Measured, Quant
from llamafit.models.host import Host
from llamafit.models.plan import Budget, ContextTier, Needs, Placement, SpeedEstimate
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
from llamafit.placement.modes import projector_of, thread_count
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
) -> Placement:
    """Rebuild a placement around settings a user chose by hand, keeping everything true.

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        host: The scanned machine.
        settings: The knobs to cost.
        requested_context: What the user asked to be sized for, for the notes.
        vision: Whether vision was wanted at all, for the notes.

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
        max_context_fit=max_context_fit(model, quant, host, settings, budget_for=budget_for),
        tiers=context_tiers(model, quant, host, settings, budget_for=budget_for),
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
    placement = plan_model(model, quant, host, needs=needs, vision=vision)
    wanted = max(needs.requested_context or placement.context, needs.min_context)

    if micro_batch is not None and micro_batch != placement.micro_batch:
        placement = replan(
            model,
            quant,
            host,
            settings_of(placement).with_micro_batch(micro_batch),
            requested_context=wanted,
            vision=vision,
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
    )
