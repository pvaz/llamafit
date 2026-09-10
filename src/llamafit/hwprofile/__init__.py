# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Hardware profiles: load them, check them, match them, and score against them.

A profile answers the question a scan cannot: what would run on a machine that is not
this one -- a card somebody is thinking of buying, or the twenty identical machines a
model is being chosen for from a build server that is none of them. It is a
:class:`~llamafit.models.hwprofile.HardwareProfile` in a JSON file, and
:func:`host_from_profile` turns it into the same
:class:`~llamafit.models.host.Host` a scan produces, so nothing downstream needs to know
which it was handed.

Nothing downstream needs to know, and every reader is told anyway: a host that did not
come from the probes carries :class:`~llamafit.models.host.Simulation`, and
``host.simulated`` is in the serialised output of every command that prints a host.
"""

from llamafit.hwprofile.loader import (
    PROFILE_SUFFIX,
    LoadedProfile,
    Problem,
    bundled_profiles_dir,
    describe,
    load_profile_file,
    load_profiles,
    profile_files,
    resolve_profile,
    user_profiles_dir,
)
from llamafit.hwprofile.match import best_match, matches
from llamafit.hwprofile.simulate import host_from_profile, override_host, resolve_host
from llamafit.hwprofile.validate import validate_files

__all__ = [
    "PROFILE_SUFFIX",
    "LoadedProfile",
    "Problem",
    "best_match",
    "bundled_profiles_dir",
    "describe",
    "host_from_profile",
    "load_profile_file",
    "load_profiles",
    "matches",
    "override_host",
    "profile_files",
    "resolve_host",
    "resolve_profile",
    "user_profiles_dir",
    "validate_files",
]
