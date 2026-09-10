# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Which of the board's columns fit, which of its rows are showing, and in what order.

Pure functions over the rows :func:`llamafit.services.recommend.build_board` returned.
Nothing here plans, sizes, estimates or scores anything: a sort key reads a figure the
scorer already produced, a filter reads a verdict the budget already reached, and a cell
spells a number somebody else worked out. Keeping that in one module with no Textual in it
is what makes it testable at a hundred widths without a terminal.

Two rules are this file's own.

**The rank never moves.** Sorting by speed reorders the rows on the screen and leaves the
``#`` column showing the place :func:`~llamafit.scoring.rank.rank` gave each one. A column
sort is a way of looking at an answer; it is not a second opinion about it, and a table
that renumbered itself would quietly claim it was.

**A speed never appears without its label.** ``Tok/s`` and ``How`` are admitted to the
table together or not at all, so a narrow terminal loses the pair rather than keeping the
figure and dropping the word that says what kind of figure it is. The command line can drop
the label because it prints a sentence under the table saying the same thing; this screen
keeps the sentence *and* the column, since a figure and its provenance separated by a
scroll are a figure without provenance.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from rich.text import Text

from llamafit.cli.render_board import (
    confidence_label,
    mode_label,
    verdict_label,
    verdict_style,
)
from llamafit.i18n import _, isolate, pgettext
from llamafit.services.recommend import MIN_FIT_VERDICTS, BoardRow
from llamafit.tui.format import context, number, size
from llamafit.units import format_bytes

Column = Literal[
    "rank",
    "model",
    "quant",
    "score",
    "gen",
    "confidence",
    "verdict",
    "mode",
    "context",
    "quality",
    "vram",
    "have",
    "prompt",
    "size",
    "ram",
]

Sort = Literal["score", "speed", "quality", "context", "size"]
"""What ``s`` cycles through. ``score`` is the board's own order and the default."""

SORTS: tuple[Sort, ...] = ("score", "speed", "quality", "context", "size")

FitFilter = Literal["all", "runs", "roomy"]
"""What ``f`` cycles through, from everything to only the configurations with room."""

FILTERS: tuple[FitFilter, ...] = ("all", "runs", "roomy")

REQUIRED: tuple[Column, ...] = ("rank", "model", "quant", "score")
"""The four columns a row cannot be told apart or acted on without."""

MIXED_ORDER: tuple[tuple[Column, ...], ...] = (
    ("gen", "confidence"),
    ("verdict",),
    ("mode",),
    ("context",),
    ("have",),
    ("quality",),
    ("vram",),
    ("prompt",),
    ("size",),
    ("ram",),
)
"""What is admitted next when the rows disagree about how their speeds were arrived at.

How fast it runs and how that figure was arrived at, then how well it fits, then how it
runs at all, then how much context it holds, then whether the file is already here, then
the quality behind the score, then the memory, the prompt speed and the download.

``gen`` and ``confidence`` are one group. When two rows carry different labels, no sentence
under the table is true of both, so the only place the label can live is beside the figure,
and a width that cannot hold the pair holds neither.
"""

UNIFORM_ORDER: tuple[tuple[Column, ...], ...] = (
    ("gen",),
    ("verdict",),
    ("mode",),
    ("context",),
    ("have",),
    ("quality",),
    ("confidence",),
    ("vram",),
    ("prompt",),
    ("size",),
    ("ram",),
)
"""The same order when every row agrees, which is the only case the pair may be split.

The band above the table then says, in a whole sentence that does not scroll, what kind of
number every speed on it is -- which is what the command line's caption says and what
section 10.3 asks of an interface. The column is still admitted when there is room for it,
because a label beside the figure is better than a label above the table; it is simply no
longer the thing a narrow terminal must keep at the cost of the verdict.
"""

WIDTHS: dict[Column, int] = {
    "rank": 3,
    "model": 24,
    "quant": 11,
    "score": 5,
    "gen": 6,
    "confidence": 11,
    "verdict": 7,
    "mode": 14,
    "context": 7,
    "have": 5,
    "quality": 5,
    "vram": 9,
    "prompt": 6,
    "size": 9,
    "ram": 9,
}
"""What each column costs in terminal cells before its padding.

``model`` is the widest catalog id plus room, because section 13.2's own note on this
screen is that it degrades by hiding the least important columns and never by truncating a
model's name: a reader who cannot read the name cannot ask for the model.
"""

_CELL_PADDING = 2
"""What a table cell costs beyond its content: one cell of padding on either side."""


def columns_for_width(width: int, *, uniform_confidence: bool = False) -> tuple[Column, ...]:
    """The columns that fit in ``width`` cells, in a fixed priority.

    Args:
        width: How many terminal cells the table has.
        uniform_confidence: Whether every row's speed carries the same label, which is
            what decides whether the label may be a sentence above the table instead of a
            column beside each figure.

    Returns:
        The columns to draw, in order.

    The four in :data:`REQUIRED` are never dropped, even when they do not fit: a table too
    narrow for them is a table that scrolls, and a row a reader cannot identify is worse
    than one they have to scroll to read. Everything else is admitted only while its whole
    group fits, and the first group that does not ends the list rather than being shrunk --
    a figure missing a digit is worse than a column that is honestly absent.
    """
    remaining = width - sum(WIDTHS[name] + _CELL_PADDING for name in REQUIRED)
    included: list[Column] = list(REQUIRED)
    for group in UNIFORM_ORDER if uniform_confidence else MIXED_ORDER:
        cost = sum(WIDTHS[name] + _CELL_PADDING for name in group)
        if cost > remaining:
            break
        included.extend(group)
        remaining -= cost
    return tuple(included)


def one_confidence(rows: Sequence[BoardRow]) -> bool:
    """Whether every speed on the board was arrived at the same way.

    A board with no speeds at all counts as agreeing: there is no figure whose label could
    be lost, and the column would be a row of dashes bought with the verdict's width.
    """
    labels = {row.candidate.speed.confidence for row in rows if row.candidate.speed is not None}
    return len(labels) <= 1


def heading(column: Column) -> str:
    """One column's heading, in the words the command line's board already uses."""
    headings: dict[Column, str] = {
        "rank": pgettext("column heading", "#"),
        "model": pgettext("column heading", "Model"),
        "quant": pgettext("column heading", "Quant"),
        "score": pgettext("column heading", "Score"),
        "gen": pgettext("column heading", "Tok/s"),
        "confidence": pgettext("column heading", "How"),
        "verdict": pgettext("column heading", "Fit"),
        "mode": pgettext("column heading", "Runs"),
        "context": pgettext("column heading", "Ctx"),
        "quality": pgettext("column heading", "Qual"),
        "vram": pgettext("column heading", "Card"),
        "prompt": pgettext("column heading", "PP/s"),
        "size": pgettext("column heading", "Size"),
        "ram": pgettext("column heading", "RAM"),
        "have": pgettext("column heading", "Have"),
    }
    return headings[column]


def cell(row: BoardRow, column: Column) -> Text:
    """One cell of one row, as text no style tag can be read out of.

    A figure that could not be produced comes back as the word for a size nobody could
    read rather than as a blank, because a blank cell reads as a zero and a zero here
    would be a claim.
    """
    candidate = row.candidate
    score = candidate.score
    placement = candidate.placement
    speed = candidate.speed
    budget = placement.budget if placement is not None else None
    unknown = format_bytes(None)
    if column == "verdict":
        if budget is None:
            return Text(unknown)
        return Text(verdict_label(budget.verdict), style=verdict_style(budget.verdict))
    values: dict[Column, str] = {
        "rank": isolate(str(row.rank)) if row.rank is not None else unknown,
        "model": isolate(row.model_id),
        "quant": isolate(row.quant),
        "score": number(score.total) if score is not None else unknown,
        "gen": number(speed.gen_tps) if speed is not None else unknown,
        "confidence": confidence_label(speed.confidence) if speed is not None else unknown,
        "mode": mode_label(placement.mode) if placement is not None else unknown,
        "context": context(placement.max_context_fit) if placement is not None else unknown,
        "quality": number(score.quality, 0) if score is not None else unknown,
        "vram": size(budget.vram_required) if budget is not None else unknown,
        "prompt": number(speed.pp_tps, 0) if speed is not None else unknown,
        "size": size(row.download_bytes),
        "ram": size(budget.ram_required) if budget is not None else unknown,
        "have": (
            pgettext("model file on this machine", "yes")
            if row.local_path
            else pgettext("model file on this machine", "no")
        ),
    }
    return Text(values[column])


def sort_label(sort: Sort) -> str:
    """What the rows are ordered by, in a word the state line can carry."""
    labels: dict[Sort, str] = {
        "score": pgettext("board sort", "score"),
        "speed": pgettext("board sort", "speed"),
        "quality": pgettext("board sort", "quality"),
        "context": pgettext("board sort", "context"),
        "size": pgettext("board sort", "download size"),
    }
    return labels[sort]


def filter_label(fit: FitFilter) -> str:
    """Which rows are showing, in a phrase the state line can carry."""
    labels: dict[FitFilter, str] = {
        "all": pgettext("board filter", "every candidate"),
        "runs": pgettext("board filter", "the ones that run"),
        "roomy": pgettext("board filter", "the ones with room to spare"),
    }
    return labels[fit]


def _sort_key(row: BoardRow, sort: Sort) -> tuple[float, int]:
    """How far up the screen a row goes, with its rank breaking every tie.

    Every figure read here was produced by a service. The rank is the tie-breaker because
    it is the one ordering this program actually stands behind, so two rows that a column
    sort cannot tell apart come back in the order the ranking put them.
    """
    candidate = row.candidate
    rank = row.rank if row.rank is not None else 10**6
    if sort == "speed":
        speed = candidate.speed
        return (-(speed.gen_tps if speed is not None else -1.0), rank)
    if sort == "quality":
        score = candidate.score
        return (-(score.quality if score is not None else -1.0), rank)
    if sort == "context":
        placement = candidate.placement
        return (-(placement.max_context_fit if placement is not None else -1), rank)
    if sort == "size":
        # An unknown download size sorts last rather than first: "nobody has filled this
        # in" is not the same claim as "this is the smallest one".
        return (float(row.download_bytes) if row.download_bytes is not None else float("inf"), rank)
    return (float(rank), rank)


def sorted_rows(rows: Sequence[BoardRow], sort: Sort) -> list[BoardRow]:
    """The rows in the order the current column sort puts them."""
    return sorted(rows, key=lambda row: _sort_key(row, sort))


def _matches_search(row: BoardRow, search: str) -> bool:
    """Whether the typed text appears in the id, the name or the quantisation."""
    wanted = search.strip().casefold()
    if not wanted:
        return True
    return any(wanted in field.casefold() for field in (row.model_id, row.name, row.quant))


def _passes_fit(row: BoardRow, fit: FitFilter) -> bool:
    """Whether a row survives the fit filter.

    The verdicts are read from :data:`~llamafit.services.recommend.MIN_FIT_VERDICTS`,
    which is the service's own list, best first. Writing the order out again here would
    be a second opinion about which verdicts count as running.
    """
    if fit == "all":
        return True
    placement = row.candidate.placement
    if placement is None:
        return False
    allowed = MIN_FIT_VERDICTS if fit == "runs" else MIN_FIT_VERDICTS[:2]
    return placement.budget.verdict in allowed


def visible_rows(
    rows: Sequence[BoardRow],
    *,
    search: str = "",
    fit: FitFilter = "all",
    installed_only: bool = False,
    sort: Sort = "score",
) -> list[BoardRow]:
    """The rows on screen: what survives the filters, in the order the sort asks for."""
    kept = [
        row
        for row in rows
        if _matches_search(row, search)
        and _passes_fit(row, fit)
        and (not installed_only or row.local_path is not None)
    ]
    return sorted_rows(kept, sort)


def state_line(shown: int, total: int, *, sort: Sort, fit: FitFilter, installed_only: bool) -> str:
    """One line saying what is on the screen and why it is in that order.

    Two shapes, because "only the ones already downloaded" is a claim about the list that
    the other words do not make, and a reader who cannot see the model they came for needs
    to be told which of the filters is hiding it.
    """
    counted = _("%(shown)d of %(total)d shown") % {"shown": shown, "total": total}
    if installed_only:
        return _("%(counted)s, by %(sort)s, showing %(filter)s already on this machine.") % {
            "counted": counted,
            "sort": sort_label(sort),
            "filter": filter_label(fit),
        }
    return _("%(counted)s, by %(sort)s, showing %(filter)s.") % {
        "counted": counted,
        "sort": sort_label(sort),
        "filter": filter_label(fit),
    }
