# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Why a row is where it is: the command line's own ``--explain``, plus what checks it.

Section 12.3 says every recommendation expands into the inputs that produced it, and that
the terminal dashboard shows the same text the command line prints with ``--explain``. It
is taken literally here: :func:`llamafit.cli.render_board.render_explanation` builds the
renderable and this module hands it to a pane. The two interfaces cannot drift, because
there is only one of them.

One thing is added. Every speed on this screen is a formula on default constants -- nothing
has been benchmarked on any machine yet -- and the only check a reader has on a formula
today is a run somebody recorded on a machine of their own. The catalog carries those, the
board deliberately does not let them become the estimate, and
:func:`~llamafit.services.recommend.recorded_measurements` is the service that hands them
over to be shown *beside* it. On a command line printing them under every row would bury
the answer; on a screen showing one row at a time it costs nothing, so this is where the
comparison section 20 asks for actually becomes available to a reader.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.text import Text

from llamafit.cli.render_board import render_explanation, render_measurements, render_reasons
from llamafit.i18n import _, isolate
from llamafit.models.catalog import Catalog
from llamafit.services.recommend import Board, BoardRow, recorded_measurements


def identity_line(row: BoardRow, catalog: Catalog | None) -> str:
    """The line above an explanation: what this model is, beyond its place on the board.

    The board carries an id, a name, a quantisation and a download size, and none of those
    answer "may I use this and what is it for". The licence and the capabilities are one
    lookup away in the catalog the board was built from, so a reader is not sent to another
    command for the two facts most likely to rule a model out.
    """
    model = None if catalog is None else catalog.by_id.get(row.model_id)
    if model is None:
        return _("%(name)s %(quant)s") % {
            "name": isolate(row.name),
            "quant": isolate(row.quant),
        }
    return _("%(name)s %(quant)s, %(vendor)s, licensed %(license)s; it can: %(capabilities)s") % {
        "name": isolate(row.name),
        "quant": isolate(row.quant),
        "vendor": isolate(model.vendor),
        "license": isolate(model.license.spdx),
        "capabilities": isolate(", ".join(model.capabilities)),
    }


def explanation(row: BoardRow, board: Board, catalog: Catalog | None) -> RenderableType:
    """One row expanded into everything that produced it, and everything that checks it.

    The middle of this is the command line's ``--explain`` output, unchanged: the score
    with its four parts and their weights, the quality it was built from, the budget line
    by line with the source of every line, the context ladder with what each rung costs,
    and where a token's time goes.
    """
    pieces: list[RenderableType] = [
        Text(identity_line(row, catalog), style="bold"),
        render_explanation(row, board),
    ]
    model = None if catalog is None else catalog.by_id.get(row.model_id)
    if model is not None:
        recorded = render_measurements(recorded_measurements(model, row.quant))
        if recorded is not None:
            pieces.append(Text(""))
            pieces.append(recorded)
    return Group(*pieces)


def nothing_selected() -> RenderableType:
    """What the pane says when no row is under the cursor, which is not the same as empty."""
    return Text(_("Choose a row to see what put it there."))


def not_ranked(board: Board | None) -> RenderableType:
    """The candidates that did not qualify, grouped by the reason, each reason said once.

    A board with nothing on it is the case this matters most for: the answer to "why is
    there nothing here" is the whole of this list, and a screen that showed an empty
    table and stopped would be telling a reader less than the command line does. The
    rows themselves are on the table, dimmed, with every figure that was computed for
    them; this is the command line's own paragraph under that table.
    """
    if board is None:
        return Text(_("Nothing has been ranked yet."))
    reasons = render_reasons(board)
    if reasons is None:
        return Text(_("Every candidate in the catalog was ranked; none was excluded."))
    return reasons
