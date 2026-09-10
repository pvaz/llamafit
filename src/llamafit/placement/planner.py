# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The placement search: where a model's bytes go, and the settings that put them there.

Section 9.2 describes a search, not a list of preferences. The planner evaluates modes in
order, keeps the first configuration whose verdict is better than Too Tight, and then
keeps looking, because a later mode can still reach a context an earlier one could not.
Stopping at the first acceptable answer would recommend a working configuration that is
not the best working configuration, which is the kind of failure nobody ever reports: it
runs, so it looks right.

The search is pruned, and every prune is only sound because it matches the order of the
parts of :func:`~llamafit.placement.modes.rank`. Each one is written down where it
happens. ``tests/unit/test_placement_planner.py`` checks the pruned search against an
exhaustive one over the same space, so the two can never quietly disagree.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from llamafit.constants import (
    DEFAULT_REQUESTED_CONTEXT,
    KV_TYPE_DEFAULT,
    MICRO_BATCH_LADDER,
    MIN_CONTEXT_TOKENS,
)
from llamafit.i18n import _
from llamafit.logging import get_logger
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.host import Host
from llamafit.models.plan import Budget, Needs, Placement, Verdict
from llamafit.placement.context_tiers import context_tiers, max_context_fit
from llamafit.placement.flags import prose_quirks
from llamafit.placement.modes import (
    BudgetFn,
    PlacementSettings,
    available_modes,
    context_ceiling,
    context_ladder,
    has_margin,
    initial_settings,
    is_acceptable,
    kv_ladder,
    layer_count,
    layer_ladder,
    projector_ladder,
    projector_of,
    rank,
    shared_expert_ladder,
    thread_count,
)
from llamafit.units import format_bytes, format_grouped

_log = get_logger("placement.planner")

_Gate = Callable[[Verdict], bool]


def plan_placement(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    *,
    budget_for: BudgetFn,
    needs: Needs | None = None,
    vision: bool = True,
) -> Placement:
    """Find the best placement for one model and quantisation on one machine.

    Args:
        model: The catalog entry.
        quant: The quantisation to place.
        host: The scanned machine.
        budget_for: What costs a configuration. Injected rather than imported so the
            planner and section 8's budget can be built, tested and changed apart; any
            callable taking a model, a quant, a host and a
            :class:`~llamafit.placement.modes.PlacementSettings` satisfies it.
        needs: What the user asked for. Defaults to the project's own defaults.
        vision: Whether to try to keep the vision projector at all; ``plan --no-vision``
            passes ``False``.

    Returns:
        The best placement found, or one whose mode is ``unsupported`` carrying the
        budget of the smallest configuration tried, so a reader is told how far off the
        machine is rather than only that it failed.
    """
    needs = needs or Needs()
    ceiling = context_ceiling(model, needs)
    # A minimum above what the user asked to be sized for is still a minimum: size for it.
    asked = max(needs.requested_context or DEFAULT_REQUESTED_CONTEXT, needs.min_context)
    requested = min(asked, ceiling)
    threads = thread_count(host.cpu)

    if needs.min_context > ceiling:
        note = _("%(model)s holds %(native)s tokens, fewer than the %(minimum)s asked for.") % {
            "model": model.name,
            "native": format_grouped(ceiling),
            "minimum": format_grouped(needs.min_context),
        }
        return _unsupported(model, quant, host, budget_for=budget_for, threads=threads, note=note)

    contexts = context_ladder(requested, needs.min_context)
    kv_types = kv_ladder(model, allow_kv_quant=needs.allow_kv_quant)
    projectors = projector_ladder(model, quant, vision=vision)
    layers = layer_count(quant.gguf_facts)

    best: tuple[tuple[int, ...], PlacementSettings, Budget] | None = None
    evaluations = 0

    for mode in available_modes(model, quant, host):
        rungs = layer_ladder(mode, layers)
        shared_experts = shared_expert_ladder(mode, quant.gguf_facts)
        for context in contexts:
            found_at_this_context = False
            for kv_type in kv_types:
                for micro_batch in MICRO_BATCH_LADDER:
                    for projector in projectors:
                        placed = False
                        for shared in shared_experts:
                            base = initial_settings(
                                mode,
                                context=context,
                                micro_batch=micro_batch,
                                kv_type=kv_type,
                                projector_pool=projector,
                            ).with_shared_experts(shared)
                            # The projector earns the card only with margin (section 9.2);
                            # everywhere else the plain acceptability gate applies.
                            gate: _Gate = has_margin if projector == "vram" else is_acceptable
                            chosen, count = _search_layers(
                                model, quant, host, base, rungs, budget_for=budget_for, gate=gate
                            )
                            evaluations += count
                            if chosen is None:
                                continue
                            settings, budget = chosen
                            key = rank(settings, budget, requested_context=requested)
                            if best is None or key > best[0]:
                                best = (key, settings, budget)
                            placed = True
                            # Prune: keeping the shared experts with their layers outranks
                            # moving them, so once the first rung places the model there is
                            # nothing the second could win on.
                            break
                        if not placed:
                            continue
                        found_at_this_context = True
                        # Prune: the projector ladder runs best first and sits above the
                        # shared experts in the rank, so no later placement of it can win.
                        break
                    if found_at_this_context:
                        # Prune: micro-batches run largest first and outrank the
                        # projector, which is section 9.2's "largest first, dropping when
                        # the compute buffer breaks the fit".
                        break
                if found_at_this_context:
                    # Prune: f16 outranks a quantised cache, so once it fits there is
                    # nothing q8_0 could win on.
                    break
            if found_at_this_context:
                # Prune: contexts run largest first and the context reached is the first
                # part of the rank, so every shorter one in this mode loses outright.
                break
        if best is not None:
            # Prune: the mode is the first part of the rank and the modes are walked
            # fastest first, so once one of them has produced anything acceptable, no
            # later mode can beat it whatever context it reaches. This is the prune that
            # makes the preference in section 9.2's mode order actually bite, and the
            # floor under the context ladder is what keeps it honest: everything reached
            # here already holds a context worth having.
            break

    if best is None:
        return _unsupported(
            model,
            quant,
            host,
            budget_for=budget_for,
            threads=threads,
            note=_("No configuration of this model fits this machine."),
        )

    _key, settings, budget = best
    _log.debug(
        "%s %s: %s at %d tokens after %d budgets",
        model.id,
        quant.name,
        settings.mode,
        settings.context,
        evaluations,
    )
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
            requested_context=requested,
            threads=threads,
            vision=vision,
        ),
    )


def _search_layers(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    base: PlacementSettings,
    rungs: Sequence[int],
    *,
    budget_for: BudgetFn,
    gate: _Gate,
) -> tuple[tuple[PlacementSettings, Budget] | None, int]:
    """Choose the mode's own parameter for one point of the search.

    Args:
        model: The catalog entry.
        quant: The quantisation being placed.
        host: The scanned machine.
        base: Everything already decided: mode, context, KV type, micro-batch, projector.
        rungs: The mode's ladder from :func:`~llamafit.placement.modes.layer_ladder`.
        budget_for: The injected budget function.
        gate: Which verdicts count as acceptable here.

    Returns:
        The best acceptable configuration and its budget, or ``None``, and how many
        budgets were computed getting there.

    The two ladders that have more than one rung run in opposite directions, and each
    stops where section 9.2 says it stops.

    ``moe-offload`` starts with every routed expert in system memory and moves them back
    onto the card one layer at a time, so each rung needs more VRAM than the last: the
    best acceptable rung is the last one, and the scan ends when VRAM runs out — the
    design's own "while VRAM allows". A first rung that fails on *system* memory rather
    than on VRAM does not end it, because moving experts to the card is exactly what
    would fix that.

    ``hybrid`` starts with the most layers on the card and gives them up one at a time,
    so each rung needs less VRAM and more system memory: the first acceptable rung is the
    best one — "the largest ``-ngl`` that fits" — and the scan ends when system memory
    runs out.
    """
    best: tuple[PlacementSettings, Budget] | None = None
    evaluations = 0
    for value in rungs:
        settings = base.with_layer_choice(value)
        budget = budget_for(model, quant, host, settings)
        evaluations += 1
        if gate(budget.verdict):
            if base.mode != "moe-offload":
                return (settings, budget), evaluations
            best = (settings, budget)
        if base.mode == "moe-offload" and budget.vram_required > budget.vram_available:
            break
        if base.mode == "hybrid" and budget.ram_required > budget.ram_available:
            break
    return best, evaluations


def placement_notes(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    settings: PlacementSettings,
    budget: Budget,
    *,
    requested_context: int,
    threads: int,
    vision: bool,
) -> tuple[str, ...]:
    """What a reader has to be told about a placement that the numbers do not say.

    Every note is a template with values filled in, never free text, so the CLI, the
    terminal interface and the web page all say the same thing in the reader's own
    language (section 12.3).
    """
    notes: list[str] = []
    if settings.mode == "moe-offload" and settings.shared_experts_pool == "ram":
        notes.append(
            _(
                "Routed experts are held in system memory, and so are the always-on "
                "shared experts, which is what -ot ffn_.*_shexp=CPU does; attention and "
                "the KV cache stay on the card."
            )
        )
    elif settings.mode == "moe-offload":
        notes.append(
            _(
                "Routed experts are held in system memory; attention, the KV cache and "
                "the shared experts stay on the card."
            )
        )
    elif settings.mode == "hybrid":
        notes.append(
            _("%(layers)s layers are on the card and the rest run on the processor.")
            % {"layers": format_grouped(settings.gpu_layers)}
        )
    elif settings.mode == "cpu":
        notes.append(_("Nothing runs on a graphics card: every weight is in system memory."))

    if settings.context < requested_context:
        notes.append(
            _("Sized for %(context)s tokens rather than the %(requested)s asked for.")
            % {
                "context": format_grouped(settings.context),
                "requested": format_grouped(requested_context),
            }
        )
    if settings.kv_type != KV_TYPE_DEFAULT:
        notes.append(
            _("The KV cache is quantised to %(type)s to reach this context.")
            % {"type": settings.kv_type}
        )

    has_projector = projector_of(model, quant) is not None
    if has_projector and vision and settings.projector_pool is None:
        notes.append(
            _("Vision is off: the projector did not fit, on the card or in system memory.")
        )
    elif has_projector and settings.projector_pool == "ram":
        notes.append(
            _(
                "The vision projector is in system memory (--no-mmproj-offload), which "
                "costs nothing measurable for text and makes images slower."
            )
        )

    available = host.cpu.logical_cores
    if threads < available:
        notes.append(
            _(
                "%(threads)s threads, not the %(available)s this processor has: the "
                "efficiency cores are left out because generation measures slower with "
                "them, and what is left is the threads the performance cores provide."
            )
            % {"threads": format_grouped(threads), "available": format_grouped(available)}
        )
    if budget.verdict == "tight":
        notes.append(
            _(
                "Tight: %(vram)s of the graphics card is free and this needs %(needed)s, "
                "so another program can push it over. The context tier table is how a "
                "launch script recovers."
            )
            % {
                "vram": format_bytes(budget.vram_available),
                "needed": format_bytes(budget.vram_required),
            }
        )
    notes.extend(prose_quirks(model))
    return tuple(notes)


def _unsupported(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    *,
    budget_for: BudgetFn,
    threads: int,
    note: str,
) -> Placement:
    """A placement for a candidate that does not fit, carrying the cheapest budget tried.

    The budget is the smallest configuration the planner can express — everything in
    system memory, the shortest context, the smallest micro-batch, no vision — so a
    reader is told how far off the machine is instead of only that it failed.
    """
    settings = initial_settings(
        "cpu",
        context=MIN_CONTEXT_TOKENS,
        micro_batch=MICRO_BATCH_LADDER[-1],
        kv_type=KV_TYPE_DEFAULT,
        projector_pool=None,
    )
    budget = budget_for(model, quant, host, settings)
    placement = settings.to_placement(
        budget, threads=threads, max_context_fit=0, tiers=(), notes=(note,)
    )
    return placement.model_copy(update={"mode": "unsupported"})
