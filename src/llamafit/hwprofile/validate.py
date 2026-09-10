# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Check profile files directly, without composing them into a set.

The schema in :mod:`llamafit.models.hwprofile` catches everything that is wrong with a
profile on its own terms: a missing field, a size that is not a size, a bandwidth with no
label. What is left is everything that is only wrong in company, and this module is where
those live -- the same division ``llamafit.catalog.validate`` draws, and for the same
reason: a check that needs a second document cannot be a field validator.

Three of them, and each exists because it caught something a person would otherwise have
shipped:

* Two files claiming one name. Loaded into a set they shadow one another silently.
* A card's figures typed in against the bundled table's. If a profile says an RTX 4060
  moves 500 GB/s, one of the two is wrong and neither the file nor the table will say so.
  Only a real disagreement is reported: a profile that leaves the figures out gets them
  from the table and can never disagree with it, which is why leaving them out is what
  ``docs/hardware-profiles.md`` tells an author to do.
* Match rules that do not fit the machine the profile itself describes. Those rules exist
  to recognise this machine when LlamaFit next runs on it; rules that cannot recognise
  the profile's own figures will never fire, and nothing else would ever say so.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from llamafit.hardware.gputable import lookup_gpu
from llamafit.hwprofile.loader import LoadedProfile, Problem, load_profile_file
from llamafit.hwprofile.match import matches
from llamafit.hwprofile.simulate import host_from_profile
from llamafit.models.hwprofile import HardwareProfile

SPEC_TOLERANCE = 0.10
"""How far a stated GPU figure may sit from the bundled table's before it is reported.

Ten percent is wider than a rounding difference and narrower than a mistake. Vendors
quote memory bandwidth to three figures and the table rounds; a card genuinely binned
differently is a different card and belongs in the table under its own name.
"""


def _spec_problems(path: Path, profile: HardwareProfile) -> list[Problem]:
    """Report GPU figures that contradict the bundled specification table."""
    problems: list[Problem] = []
    for index, card in enumerate(profile.gpus):
        spec = lookup_gpu(card.name)
        if spec is None:
            continue
        for field, stated, tabled in (
            ("bandwidth_gbps", card.bandwidth_gbps, spec.bandwidth_gbps),
            ("compute_tflops_fp16", card.compute_tflops_fp16, spec.compute_tflops_fp16),
        ):
            if stated is None or tabled <= 0:
                continue
            if abs(stated - tabled) / tabled <= SPEC_TOLERANCE:
                continue
            problems.append(
                Problem(
                    file=str(path),
                    profile=profile.name,
                    location=f"gpus.{index}.{field}",
                    message=(
                        f"{field} is {stated:g}, but the bundled table gives {tabled:g} for a "
                        f"card matching {card.name!r}. Drop the field to use the table's "
                        "figure, or say in `provenance` where this one was measured."
                    ),
                )
            )
        if spec.unified and not profile.unified_memory:
            problems.append(
                Problem(
                    file=str(path),
                    profile=profile.name,
                    location="unified_memory",
                    message=(
                        f"unified_memory is false, but {card.name!r} matches a table row for "
                        "a part whose graphics share the system memory pool"
                    ),
                )
            )
    return problems


def _match_problem(path: Path, profile: HardwareProfile) -> Problem | None:
    """Report match rules that would not recognise the profile's own machine.

    Returns:
        The problem, or ``None`` when the profile states no rules (which is a profile
        declining to claim a machine, not a mistake) or when its rules fit it.
    """
    if profile.match.empty:
        return None
    loaded = LoadedProfile(profile=profile, path=path, bundled=False)
    if matches(profile, host_from_profile(loaded)):
        return None
    return Problem(
        file=str(path),
        profile=profile.name,
        location="match",
        message=(
            "these match rules do not fit the machine this profile describes, so they can "
            "never select it: check gpu_name_contains, cpu_model_contains and total_ram_min "
            "against the fields above them"
        ),
    )


def validate_files(paths: Sequence[Path]) -> list[Problem]:
    """Validate each file in ``paths`` independently and report every problem.

    A syntax error or a failed validation is never an exception here, only a
    :class:`~llamafit.hwprofile.loader.Problem`. A name claimed by two of the given files
    is reported too, since loaded together one would silently shadow the other.

    Returns:
        One problem per thing wrong, in the order the files were given.
    """
    problems: list[Problem] = []
    seen: dict[str, str] = {}
    for path in paths:
        profile, file_problems = load_profile_file(path)
        problems.extend(file_problems)
        if profile is None:
            continue
        first_seen_in = seen.get(profile.name)
        if first_seen_in is not None:
            problems.append(
                Problem(
                    file=str(path),
                    profile=profile.name,
                    location="name",
                    message=f"duplicate profile name {profile.name!r}, first seen in "
                    f"{first_seen_in}",
                )
            )
        else:
            seen[profile.name] = str(path)
        problems.extend(_spec_problems(path, profile))
        match_problem = _match_problem(path, profile)
        if match_problem is not None:
            problems.append(match_problem)
    return problems
