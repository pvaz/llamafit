# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Read hardware profiles from JSON: the bundled ones, and the user's own.

This is the catalog loader's habit applied to a second kind of data, deliberately and
not by accident. Nothing here raises on a malformed file. A JSON syntax error, a
document that is not an object, a field of the wrong shape, a ``name`` that disagrees
with the file it is in -- each becomes a :class:`Problem` naming the file and the field,
and the caller decides what to do about it. ``llamafit hardware validate`` prints them;
``llamafit hardware list`` warns that there are some and shows what did load.

The one thing that *is* raised is a name that resolves to nothing, because
``--profile nosuch`` is a question with no answer rather than a file with a mistake in
it. That is a :class:`~llamafit.errors.ConfigError`, exit code 1, and it names the
profiles that were there.

:class:`Problem` is its own dataclass rather than the catalog's, and carries ``profile``
where the catalog's carries ``model_id``. The shape a ``--json`` consumer sees is the
same four keys; the second one names the thing that is actually wrong, which for a
profile is a profile.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from llamafit.data import packaged_dir
from llamafit.errors import ConfigError
from llamafit.i18n import _
from llamafit.models.hwprofile import HardwareProfile
from llamafit.paths import get_paths

PROFILE_SUFFIX = ".json"
"""What a profile file is called. One machine per file, named for the profile."""

_PROFILES_ENV_VAR = "LLAMAFIT_PROFILES"
_PROFILES_DIRNAME = "profiles"


@dataclass(frozen=True)
class Problem:
    """One thing wrong with a hardware profile file.

    Attributes:
        file: The file the problem was found in.
        profile: The profile's name, when it could be read; ``None`` otherwise.
        location: Where in the document the problem is, dotted, for example
            ``memory.bandwidth_source``, or ``"file"`` for a problem with the file itself.
        message: A human-readable description of the problem.
    """

    file: str
    profile: str | None
    location: str
    message: str


@dataclass(frozen=True)
class LoadedProfile:
    """A profile and where it came from.

    Attributes:
        profile: The parsed document.
        path: The file it was read from.
        bundled: True when it shipped inside the package rather than being the user's.
    """

    profile: HardwareProfile
    path: Path
    bundled: bool

    @property
    def name(self) -> str:
        """The profile's name, which is what ``--profile`` takes."""
        return self.profile.name


def bundled_profiles_dir() -> Path:
    """The packaged directory holding the profiles that ship with LlamaFit.

    Raises:
        PackagedDataError: The directory did not ship in this installation, or shipped
            with no profile in it. Checked for the same reason the catalog directory is:
            an empty directory reads as "this build bundles no profiles", which is
            indistinguishable from a broken wheel and is the wrong answer to give.
    """
    return packaged_dir(
        "llamafit.data.profiles", what="its hardware profiles", contains=f"*{PROFILE_SUFFIX}"
    )


def user_profiles_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where a user's own profiles live, which is what ``hardware path`` prints.

    ``LLAMAFIT_PROFILES`` names the directory outright when set; otherwise it is
    ``profiles`` under the data directory, beside the custom models file.
    """
    env = os.environ if env is None else env
    override = env.get(_PROFILES_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return get_paths(env).data_dir / _PROFILES_DIRNAME


def load_profile_file(path: Path) -> tuple[HardwareProfile | None, list[Problem]]:
    """Parse and validate one profile file.

    A file that cannot be read, is not JSON, is not an object, or fails validation comes
    back as ``None`` and one or more problems; this function never raises for malformed
    input. A profile whose ``name`` is not the file's stem is returned -- it is a usable
    machine -- along with a problem, because the loader finds profiles by file name and
    a disagreement there means one of the two names cannot be typed.

    Returns:
        The profile, or ``None`` when the document could not be understood at all, and
        every problem found in the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return None, [Problem(file=str(path), profile=None, location="file", message=str(exc))]

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [
            Problem(file=str(path), profile=None, location="file", message=f"invalid JSON: {exc}")
        ]

    if not isinstance(raw, dict):
        return None, [
            Problem(
                file=str(path),
                profile=None,
                location="file",
                message="expected an object describing one machine",
            )
        ]

    name = raw.get("name") if isinstance(raw.get("name"), str) else None
    try:
        profile = HardwareProfile.model_validate(raw)
    except ValidationError as exc:
        return None, [
            Problem(
                file=str(path),
                profile=name,
                location=".".join(str(part) for part in error["loc"]) or "(root)",
                message=error["msg"],
            )
            for error in exc.errors()
        ]

    problems: list[Problem] = []
    if profile.name != path.stem:
        problems.append(
            Problem(
                file=str(path),
                profile=profile.name,
                location="name",
                message=(
                    f"the profile is named {profile.name!r} but the file is "
                    f"{path.name!r}; `--profile {profile.name}` looks for "
                    f"{profile.name}{PROFILE_SUFFIX}, so rename one to match the other"
                ),
            )
        )
    return profile, problems


def profile_files(directory: Path) -> list[Path]:
    """Every profile file in one directory, in name order; empty when there is none."""
    if not directory.is_dir():
        return []
    return sorted(directory.glob(f"*{PROFILE_SUFFIX}"))


def load_profiles(
    *,
    bundled_dir: Path | None = None,
    user_dir: Path | None = None,
) -> tuple[list[LoadedProfile], list[Problem]]:
    """Load every profile: the bundled ones first, then the user's.

    A user profile whose name matches a bundled one replaces it in place, so somebody
    who recalibrates the machine LlamaFit ships a profile of keeps the name they already
    use. Two bundled files claiming one name is a problem and the second is discarded,
    the way the catalog treats a duplicate id.

    Returns:
        The profiles in load order, and every problem found in any file.
    """
    directory = bundled_dir if bundled_dir is not None else bundled_profiles_dir()
    user = user_dir if user_dir is not None else user_profiles_dir()

    problems: list[Problem] = []
    by_name: dict[str, LoadedProfile] = {}
    order: list[str] = []

    for path in profile_files(directory):
        profile, file_problems = load_profile_file(path)
        problems.extend(file_problems)
        if profile is None:
            continue
        if profile.name in by_name:
            problems.append(
                Problem(
                    file=str(path),
                    profile=profile.name,
                    location="name",
                    message=f"duplicate profile name {profile.name!r}",
                )
            )
            continue
        by_name[profile.name] = LoadedProfile(profile=profile, path=path, bundled=True)
        order.append(profile.name)

    for path in profile_files(user):
        profile, file_problems = load_profile_file(path)
        problems.extend(file_problems)
        if profile is None:
            continue
        if profile.name not in by_name:
            order.append(profile.name)
        by_name[profile.name] = LoadedProfile(profile=profile, path=path, bundled=False)

    return [by_name[name] for name in order], problems


def _looks_like_a_path(reference: str) -> bool:
    """Whether ``--profile`` was given a file rather than a name.

    A name is lowercase letters, digits, hyphens and dots (see
    :data:`~llamafit.models.hwprofile.NAME_PATTERN`), so anything carrying a separator,
    a suffix or a capital was meant as a path. Deciding by shape rather than by trying
    the filesystem first keeps ``--profile reference-rtx4060-128gb`` from picking up a
    file of that name that happens to sit in the working directory.
    """
    return (
        reference.endswith(PROFILE_SUFFIX)
        or "/" in reference
        or "\\" in reference
        or reference.startswith("~")
    )


def resolve_profile(
    reference: str,
    *,
    bundled_dir: Path | None = None,
    user_dir: Path | None = None,
) -> tuple[LoadedProfile, list[Problem]]:
    """Find the profile ``--profile`` asked for, by name or by path.

    Args:
        reference: A bundled or user profile's name, or a path to a profile file.
        bundled_dir: Where the packaged profiles are; for tests.
        user_dir: Where the user's profiles are; for tests.

    Returns:
        The profile, and any problems its own file had that did not stop it loading.

    Raises:
        ConfigError: The name matches no profile, or the file named does not exist or
            could not be read as a profile. Every problem found is in the message, so a
            user who wrote a file with a mistake in it is told what the mistake is
            rather than being told the file is not a profile.
    """
    if _looks_like_a_path(reference):
        path = Path(reference).expanduser()
        if not path.is_file():
            raise ConfigError(
                _("no hardware profile file at %(path)s") % {"path": str(path)},
                hint=_("Run `llamafit hardware list` to see the profiles LlamaFit has."),
            )
        profile, problems = load_profile_file(path)
        if profile is None:
            raise ConfigError(
                _("%(path)s is not a valid hardware profile: %(details)s")
                % {"path": str(path), "details": describe(problems)},
                hint=_("Run `llamafit hardware validate FILE` for the whole list."),
            )
        return LoadedProfile(profile=profile, path=path, bundled=False), problems

    profiles, problems = load_profiles(bundled_dir=bundled_dir, user_dir=user_dir)
    for loaded in profiles:
        if loaded.name == reference:
            return loaded, [p for p in problems if p.file == str(loaded.path)]
    known = ", ".join(loaded.name for loaded in profiles)
    raise ConfigError(
        _("no hardware profile named %(name)s") % {"name": repr(reference)},
        hint=(
            _("Profiles LlamaFit has: %(names)s") % {"names": known}
            if known
            else _("Run `llamafit hardware path` to see where your own profiles go.")
        ),
    )


def describe(problems: Sequence[Problem]) -> str:
    """Every problem on one line, for an error message that has to be a sentence.

    The locations name JSON fields, which is a profile author's language rather than a
    reader's, so nothing here is translated. See ``docs/translations.md``.
    """
    return "; ".join(f"{p.file}: {p.location}: {p.message}" for p in problems)
