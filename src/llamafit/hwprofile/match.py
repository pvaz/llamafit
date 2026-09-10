# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Which profile, if any, describes the machine LlamaFit is running on.

A profile's ``match`` block exists so that a machine somebody has benchmarked picks its
own calibration up again without being told to (section 4.4, and section 16 for what
writes it). Matching is deliberately dull: every rule given must hold, and a profile with
no rules matches nothing at all. A rule that is absent is not a rule that passes -- it is
a profile saying it does not claim to be any particular machine, which is exactly what a
profile of somebody else's hardware should say.

Nothing here reads a matched profile's figures over the live scan's. The scan measured
this machine and the profile did not; what a match is for is the ``calibration`` block,
which phase 3 fills in and phase 3 will apply.
"""

from __future__ import annotations

from collections.abc import Sequence

from llamafit.hwprofile.loader import LoadedProfile
from llamafit.models.host import Host
from llamafit.models.hwprofile import HardwareProfile


def matches(profile: HardwareProfile, host: Host) -> bool:
    """Whether every rule in ``profile.match`` holds for ``host``.

    Returns:
        ``False`` for a profile that states no rules, whatever the host is.
    """
    rules = profile.match
    if rules.empty:
        return False
    if rules.gpu_name_contains is not None:
        wanted = rules.gpu_name_contains.casefold()
        if not any(wanted in gpu.name.casefold() for gpu in host.gpus):
            return False
    if (
        rules.cpu_model_contains is not None
        and rules.cpu_model_contains.casefold() not in host.cpu.model.casefold()
    ):
        return False
    return not (rules.total_ram_min is not None and host.memory.total_bytes < rules.total_ram_min)


def _specificity(profile: HardwareProfile) -> int:
    """How many rules a profile's match block states, which is how exact it claims to be."""
    rules = profile.match
    return sum(
        value is not None
        for value in (rules.gpu_name_contains, rules.cpu_model_contains, rules.total_ram_min)
    )


def best_match(profiles: Sequence[LoadedProfile], host: Host) -> LoadedProfile | None:
    """The profile that fits ``host`` most exactly, or ``None`` when none does.

    Two ties are broken in the order somebody would expect to win them: a profile the
    user wrote beats one that shipped in the package, and among equals the first loaded
    wins so the answer does not depend on how a directory happens to be sorted.
    """
    candidates = [loaded for loaded in profiles if matches(loaded.profile, host)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda loaded: (_specificity(loaded.profile), not loaded.bundled),
    )
