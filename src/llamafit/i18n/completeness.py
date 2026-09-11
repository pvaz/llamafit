# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""How much of a language LlamaFit actually speaks, and the sentence that admits it.

Thirty-seven catalogs ship and six of them are finished. The other thirty-one carry
between a fifth and a quarter of the messages, and because a catalog is written from the
top of the template down, what they carry is the short fragments -- a column heading, a
verdict, a unit -- and what they are missing is the prose. The effect on a screen is a
tool that labels its columns in the reader's language and then explains itself in English,
which reads less like a translation in progress than like a translation that has gone
wrong. Nothing used to say otherwise, and the thing a reader cannot be expected to work
out from the screen is exactly the thing an interface has to say out loud.

So one sentence, once, when the catalog that was loaded is materially incomplete, in the
same place and by the same route as the notice for a language LlamaFit does not speak at
all: :attr:`~llamafit.i18n.select.LanguageChoice.notice`, printed to stderr at start-up.

**The sentence is in English on purpose**, like every other message in this package. It is
the one place where that is not a gap: a notice saying a language is a quarter translated
cannot be written in the quarter of that language which is translated, and a reader who
meets it in English has already been shown what it is about.

**The figure is counted at run time, not written down.** A catalog is complete relative to
``messages.pot``, and only the template knows how many messages there are. Reading it cost
eleven milliseconds where this was measured, which took ``--version`` -- the shortest
command there is -- from 396 to 410, and it is paid once in a process and only by a reader
who is not reading English: :func:`~llamafit.i18n.translator.set_language` installs English
without opening a catalog at all and never asks this module anything, so the common run
pays nothing. Both ways of not paying it were worse. A number generated into a source file
is a thing no generator here does -- they write data, never code. A number written down by
hand is a build that breaks the next time somebody adds an English message, and adding one
is the thing that has to stay cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

from llamafit.errors import LlamaFitError
from llamafit.i18n.catalogs import TEMPLATE_NAME, catalog_dir
from llamafit.i18n.po import PoCatalog, read_po
from llamafit.logging import get_logger

_log = get_logger("i18n")

NEARLY_COMPLETE = 0.9
"""The share of the template a catalog must carry before nothing is said about it.

Nine tenths, because the catalogs that ship are nowhere near it from either side: six sit
at ninety-nine or a hundred per cent and the rest between twenty-two and twenty-four, so
the line falls in open ground and neither one English message being added nor one sentence
being translated can move a language across it.

It is not a standard for a finished translation -- a language missing a tenth of what it
could say is not finished, and ``docs/translations.md`` asks for rather more than that. It
is the point past which repeating the shortfall on every single run would cost the reader
more attention than the fact is worth.
"""

SHORTFALL_HINT = "Finishing it is a pull request: docs/translations.md says how."
"""What to do about it, for the reader who would rather fix this than be told about it."""


@dataclass(frozen=True)
class Completeness:
    """How many of the messages a language could say it can actually say.

    Attributes:
        translated: Entries in the catalog that somebody has filled in.
        total: Entries in the template, which is every message LlamaFit has; zero when
            no template shipped, which makes the whole question unanswerable rather than
            answerable with a zero.
    """

    translated: int
    total: int

    @property
    def known(self) -> bool:
        """Whether there was a template to measure against at all."""
        return self.total > 0

    @property
    def share(self) -> float:
        """The fraction of the template this catalog carries, never above one.

        A catalog holding an entry the template does not is dead weight rather than extra
        coverage -- the test suite fails a shipped catalog for it -- so it is capped here
        instead of being allowed to report a hundred and two per cent.
        """
        if not self.known:
            return 1.0
        return min(1.0, self.translated / self.total)

    @property
    def percent(self) -> int:
        """The share as a whole number, rounded down.

        Down rather than to nearest, because every rounding in this sentence should err
        towards admitting more of the gap and not less.
        """
        return int(self.share * 100)

    @property
    def nearly_complete(self) -> bool:
        """Whether this catalog is close enough to finished to say nothing about it."""
        return not self.known or self.share >= NEARLY_COMPLETE


def template_message_count(*, directory: Path | None = None) -> int:
    """How many messages LlamaFit has, counted from the template a translator copies.

    Args:
        directory: Where the catalogs live; the packaged directory when ``None``.

    Returns:
        The number of entries in ``messages.pot``, or zero when there is no template to
        read. Zero is not a count: it is what :attr:`Completeness.known` reports as
        nothing being known, and it makes this whole layer say nothing rather than accuse
        a perfectly good catalog of being empty. A packaging gap must not turn into a
        remark on every run, let alone a wrong one.
    """
    return _count(directory)


def completeness(catalog: PoCatalog, *, directory: Path | None = None) -> Completeness:
    """Measure one loaded catalog against the template.

    Args:
        catalog: The catalog that was loaded and installed.
        directory: Where the catalogs live; the packaged directory when ``None``.

    Returns:
        What it carries and what there was to carry.

    An entry is counted when somebody has filled it in, which is
    :attr:`~llamafit.i18n.po.Message.translated` and not whether a lookup would use it: a
    translation dropped for a broken placeholder has already been blanked by the reader by
    the time this runs, so it counts as missing here without anything special being done
    about it, and the notice for *that* says something else.
    """
    filled = sum(1 for message in catalog.messages.values() if message.translated)
    return Completeness(translated=filled, total=template_message_count(directory=directory))


def shortfall_notice(language: str, measured: Completeness) -> str:
    """The sentence said when the loaded catalog is materially incomplete.

    Args:
        language: What to call the language -- its ``Language-Team`` where the catalog
            declares one, so a reader sees ``Japanese`` rather than ``ja``, and the tag
            where it does not.
        measured: What the catalog carries.

    Returns:
        One sentence, in English, naming the share and what falls back.
    """
    return (
        f"LlamaFit's {language} translation is about {measured.percent}% written, "
        "so much of what it says will be in English."
    )


def shortfall_after(notice: str, measured: Completeness) -> str:
    """The same fact, tacked onto a notice that has already named the catalog.

    A request for a region no catalog covers is answered with another region's, and that
    other region's catalog may itself be part written: two things wrong with one run, and
    only one line to say them in. The substitution notice has already said which catalog
    it settled on, so naming it a second time would read as a second catalog; this points
    back at the one just named instead.

    Args:
        notice: What the choice already says, ending in a full stop.
        measured: What the catalog carries.

    Returns:
        Both sentences, in that order.
    """
    return (
        f"{notice} That one is about {measured.percent}% written, "
        "so much of what it says will be in English."
    )


@cache
def _count(directory: Path | None) -> int:
    """Read and count the template, at most once per directory in a process.

    Cached because the answer is a property of the release and cannot change under a
    running process, and because the read is the only part of this that costs anything.

    Nothing here raises. A template that did not ship, cannot be opened or will not parse
    is a broken installation, and a broken installation is already going to show every
    message in English; refusing to start over the file that would have said so out loud
    would be the one way to make it worse.
    """
    try:
        folder = catalog_dir() if directory is None else directory
        return len(read_po(folder / TEMPLATE_NAME).messages)
    except (LlamaFitError, OSError) as exc:
        _log.debug("could not count the message template: %s", exc)
        return 0
