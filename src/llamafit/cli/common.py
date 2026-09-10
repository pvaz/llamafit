# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What every command that reads the catalog or the machine needs before it can answer.

Four commands now load the catalog, three of them look a model up by id, and three take a
use case or a capability from the command line and have to reject a typo by name. Written
once here rather than four times, because the interesting part of each is the error: a
mistyped id that names the closest real ones, an unknown capability that lists the ones
that exist, a catalog file with a problem in it that says which command will explain it.
Those are the sentences a person actually meets, and they should not differ between two
commands because two people wrote them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast, get_args

from rich.console import Console
from rich.text import Text

from llamafit.catalog.loader import load_catalog
from llamafit.cli.app import CliState
from llamafit.errors import CatalogError
from llamafit.i18n import _, ngettext
from llamafit.models.catalog import Capability, Catalog, CatalogModel, UseCase
from llamafit.models.host import Host
from llamafit.models.llamacpp import LocalModel
from llamafit.models.report import SystemReport
from llamafit.services.scan import scan_system
from llamafit.units import parse_size

VALID_CAPABILITIES: tuple[str, ...] = get_args(Capability)
"""Every capability the catalog knows, taken from the catalog's own type."""

VALID_USE_CASES: tuple[str, ...] = get_args(UseCase)
"""Every use case the catalog knows, taken from the catalog's own type."""


def closest_ids(
    catalog_ids: Sequence[str], target: str, *, min_prefix: int = 2, limit: int = 3
) -> list[str]:
    """The catalog ids that share the longest prefix with ``target``.

    "Closest" is deliberately narrow: the longest common prefix, case-insensitive,
    counted from the first character. A typo like ``qwen3-coder`` for
    ``qwen3-coder-next`` shares an eleven-character prefix and is an obvious match;
    unrelated input shares nothing worth naming. Below ``min_prefix`` characters of
    overlap, nothing is suggested at all, so nonsense input gets an honest "not
    found" instead of a handful of unrelated ids picked only because they had to
    come from somewhere. At most ``limit`` ids are returned, alphabetically, so a
    genuine tie does not turn one bad guess into a wall of suggestions.
    """

    def prefix_len(candidate: str) -> int:
        length = 0
        for a, b in zip(target.casefold(), candidate.casefold(), strict=False):
            if a != b:
                break
            length += 1
        return length

    scored = [(prefix_len(candidate), candidate) for candidate in catalog_ids]
    # Named rather than thrown away as `_`: this module imports the translator under
    # that name, and a throwaway inside a function rebinds it to a string with nothing
    # to warn you until the next translated call in the same function is not callable.
    best = max((score for score, _candidate in scored), default=0)
    if best < min_prefix:
        return []
    return sorted(candidate for score, candidate in scored if score == best)[:limit]


def find_model(catalog: Catalog, model_id: str) -> CatalogModel:
    """The model with ``model_id``, or a :class:`CatalogError` naming close ids instead.

    An id is a name, not a password: matching strips surrounding whitespace (from
    a pasted value) and folds case, so ``" QWEN3-CODER-NEXT"`` finds
    ``qwen3-coder-next`` the same way the closest-id suggestion already would.
    Every catalog id is lowercase by construction, so folding the input is enough;
    no separate case-insensitive index is needed.
    """
    normalized = model_id.strip().casefold()
    model = catalog.by_id.get(normalized)
    if model is not None:
        return model
    suggestions = closest_ids(sorted(catalog.by_id), normalized)
    hint = (
        _("Did you mean: %(ids)s?") % {"ids": ", ".join(suggestions)}
        if suggestions
        else _("Run `llamafit list` to see every model id.")
    )
    raise CatalogError(
        _("no model named %(id)s in the catalog") % {"id": repr(model_id)}, hint=hint
    )


def load_catalog_or_warn(state: CliState) -> Catalog:
    """Load the catalog, warning to stderr once when any file had a problem.

    A user who adds a model to their custom catalog and mistypes a field would
    otherwise just be told the model does not exist, with a hint pointing at
    ``list``, which silently does not show it either: the loader already knows
    exactly what went wrong, and this is the one place every command reaches it
    from. The warning is one line, not fatal, and never dumped into a table:
    browsing keeps working with whatever loaded, and ``catalog validate`` is
    where the detail lives.
    """
    catalog, problems = load_catalog()
    if problems:
        stderr = Console(stderr=True, no_color=state.no_color, highlight=False)
        stderr.print(
            Text(
                ngettext(
                    "%(count)d catalog problem found; run `llamafit catalog validate` for details.",
                    "%(count)d catalog problems found; "
                    "run `llamafit catalog validate` for details.",
                    len(problems),
                )
                % {"count": len(problems)},
                style="yellow",
            )
        )
    return catalog


def check_use_case(value: str) -> UseCase:
    """Return the use case, or refuse it by name and list the six that exist.

    Typer can enumerate a single ``Literal`` option's choices itself, but doing
    that would make an unknown value a bare Click usage error: a different exit
    code and no list of what would have worked, for the same kind of mistake
    the capability options already report as a clean, listed :class:`CatalogError`.

    The message names ``--use-case`` outright because both commands that take one spell
    it that way; the capability check below cannot, which is why it is told the flag.
    """
    if value not in VALID_USE_CASES:
        raise CatalogError(
            _("invalid --use-case %(value)s") % {"value": repr(value)},
            hint=_("Valid use cases: %(values)s") % {"values": ", ".join(VALID_USE_CASES)},
        )
    return cast("UseCase", value)


def check_capabilities(
    values: Sequence[str], *, option: str = "--capability"
) -> tuple[Capability, ...]:
    """Return the capabilities, or refuse the unknown ones by name.

    Args:
        values: What the user typed.
        option: The flag they typed it after, which the message names. ``list`` spells
            this ``--capability`` and ``recommend`` spells it ``--require``, and an error
            that named the wrong one would send a reader looking for a flag they did not
            use.

    Returns:
        The same values, as catalog capabilities.

    Raises:
        CatalogError: If any value is not a capability the catalog knows.

    Typer cannot enumerate a ``list[Literal[...]]`` option at all -- a list of a
    "complex" sub-type is one of the few shapes it refuses outright -- so this has to be
    checked by hand whatever else is done, and checking the use case the same way is what
    makes the two fail alike.
    """
    invalid = [value for value in values if value not in VALID_CAPABILITIES]
    if invalid:
        raise CatalogError(
            ngettext(
                "invalid %(option)s value: %(values)s",
                "invalid %(option)s values: %(values)s",
                len(invalid),
            )
            % {"option": option, "values": ", ".join(invalid)},
            hint=_("Valid capabilities: %(values)s") % {"values": ", ".join(VALID_CAPABILITIES)},
        )
    return cast("tuple[Capability, ...]", tuple(values))


def check_size(value: str, *, option: str) -> int:
    """Parse a human size such as ``120G`` or ``7.5GiB``, or say what a size looks like."""
    try:
        return parse_size(value)
    except ValueError as exc:
        raise CatalogError(
            _("%(option)s is not a size: %(value)s") % {"option": option, "value": repr(value)},
            hint=_("Sizes look like 8G, 7.5GiB or 512M; a bare number is bytes."),
        ) from exc


def scan(*, measure_bandwidth: bool = True) -> SystemReport:
    """The machine and its llama.cpp installation, for a command that has to size something.

    One call rather than two so that every command sizes against the same scan: the free
    memory a budget is computed from and the local files a command line names have to come
    from one moment, or a plan can name a file the same run decided was missing.
    """
    return scan_system(measure_bandwidth=measure_bandwidth)


def host_of(report: SystemReport) -> Host:
    """The scanned machine out of a report, for a caller that wants only that half."""
    return report.host


def local_models_of(report: SystemReport) -> list[LocalModel]:
    """The GGUF files llama.cpp already has, for the installed marker and the model path."""
    return list(report.llamacpp.local_models)
