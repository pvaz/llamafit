# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The join between section 8's budget and section 9's search.

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
"""

from __future__ import annotations

from llamafit.budget import compute
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.host import Host
from llamafit.models.plan import Budget, Needs, Placement
from llamafit.placement import PlacementSettings, plan_placement
from llamafit.placement.modes import projector_of


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
