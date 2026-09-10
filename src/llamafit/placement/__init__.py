# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The placement planner: where a model's bytes go, and the settings that put them there.

Section 9 of the design. Four things live here and each is one of its subsections: the run
modes and the ladders the search walks (:mod:`~llamafit.placement.modes`, 9.1 and 9.2),
the search itself (:mod:`~llamafit.placement.planner`, 9.2 and 9.4), the context ladder a
launch script chooses from at start time (:mod:`~llamafit.placement.context_tiers`, 9.3)
and the ``llama-server`` arguments (:mod:`~llamafit.placement.flags`, 9.5).

What is *not* here is the budget. Sizing a configuration is section 8's job, and the
planner takes it as an injected callable — :class:`~llamafit.placement.modes.BudgetFn` —
the way the rest of the project injects its command runner and its HTTP client, so the
two can be read, tested and changed apart.
"""

from llamafit.placement.context_tiers import context_tiers, max_context_fit
from llamafit.placement.flags import LaunchOptions, command_line, render_flags
from llamafit.placement.modes import (
    BudgetFn,
    PlacementSettings,
    available_modes,
    context_ladder,
    kv_ladder,
    layer_ladder,
    projector_ladder,
    rank,
    thread_count,
)
from llamafit.placement.planner import placement_notes, plan_placement

__all__ = [
    "BudgetFn",
    "LaunchOptions",
    "PlacementSettings",
    "available_modes",
    "command_line",
    "context_ladder",
    "context_tiers",
    "kv_ladder",
    "layer_ladder",
    "max_context_fit",
    "placement_notes",
    "plan_placement",
    "projector_ladder",
    "rank",
    "render_flags",
    "thread_count",
]
