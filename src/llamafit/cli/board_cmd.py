# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""``llamafit fit`` and ``llamafit recommend``: the two boards.

``fit`` asks one question — how well does each model use this machine — and answers it for
every model in the catalog, including the ones built for a job nobody asked about.
``recommend`` asks the question the whole program exists for: given what I want to do, what
should I run? It plans, estimates, scores and orders, and it shows the candidates it
excluded with the reason each was excluded, because a shorter list tells a reader nothing.

``--explain`` is the flag that keeps the project's standing promise. A recommendation
expands into the four scores, the weights that combined them, the quality it was built
from, the budget line by line with the source of every line, the context ladder and where
a token's time goes. Nothing in this program should ever have to be taken on faith, and
this is the command where that is hardest and matters most.
"""

from __future__ import annotations

from typing import cast

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.common import (
    check_capabilities,
    check_size,
    check_use_case,
    load_catalog_or_warn,
    scan,
)
from llamafit.cli.render_board import (
    render_board,
    render_excluded,
    render_explanation,
    render_fit,
    render_fit_excluded,
)
from llamafit.errors import CatalogError
from llamafit.i18n import _, lazy_gettext
from llamafit.models.plan import Needs, Verdict
from llamafit.scoring import PREFERENCES
from llamafit.services.recommend import MIN_FIT_VERDICTS, build_board, build_fit_board

# Module-level singletons rather than inline ``typer.Option(...)`` calls: ruff's B008 does
# not recognise every annotation shape (a ``list``, an optional string) as safe to call in
# a default position, even though Typer only ever reads these once, at import time. Every
# ``help=`` is deferred for the reason ``app.py`` gives: a decorator runs while the module
# is imported, long before a language has been chosen, so an eager ``_()`` would freeze the
# whole interface in English with nothing failing.
_USE_CASE_OPTION: str = typer.Option(
    "general",
    "--use-case",
    help=cast(
        str,
        lazy_gettext(
            "What the model is for. It sets the score weights, the speed target and the "
            "context a candidate is measured against."
        ),
    ),
)
_REQUIRE_OPTION: list[str] = typer.Option(
    [],
    "--require",
    help=cast(
        str,
        lazy_gettext("Capability the model must have; a model without it is excluded "),
    ),
)
_LICENSE_OPTION: list[str] = typer.Option(
    [],
    "--license",
    help=cast(str, lazy_gettext("Keep only these licence identifiers (repeatable).")),
)
_PREFER_OPTION: str = typer.Option(
    "balanced",
    "--prefer",
    help=cast(
        str,
        lazy_gettext(
            "balanced, quality or speed: moves a tenth of the weight between quality and "
            "speed. It leans the board; it does not replace the weights."
        ),
    ),
)
_MAX_DOWNLOAD_OPTION: str | None = typer.Option(
    None,
    "--max-download",
    metavar="SIZE",
    help=cast(str, lazy_gettext("Exclude anything larger to download, for example 40G or 7.5GiB.")),
)
_MIN_FIT_OPTION: str = typer.Option(
    "tight",
    "--min-fit",
    help=cast(str, lazy_gettext("The worst verdict to list: comfortable, fits or tight.")),
)


def _check_min_fit(value: str) -> Verdict:
    """Return the verdict ``--min-fit`` named, or refuse it and list the three."""
    for verdict in MIN_FIT_VERDICTS:
        if value == verdict:
            return verdict
    raise CatalogError(
        _("invalid --min-fit %(value)s") % {"value": repr(value)},
        hint=_("Valid values: %(values)s") % {"values": ", ".join(MIN_FIT_VERDICTS)},
    )


def _check_prefer(value: str) -> str:
    """Return the preference ``--prefer`` named, or refuse it and list the three."""
    if value not in PREFERENCES:
        raise CatalogError(
            _("invalid --prefer %(value)s") % {"value": repr(value)},
            hint=_("Valid values: %(values)s") % {"values": ", ".join(PREFERENCES)},
        )
    return value


@app.command(
    "recommend",
    help=cast(
        str,
        lazy_gettext(
            "Rank the catalog for what you want to do on this machine.\n\n"
            "Every candidate is planned, sized and estimated here, so the first run scans "
            "the machine. Speeds compare candidates at 8K tokens of context; --explain "
            "expands a row into the scores, the weights, the budget and the speed "
            "breakdown that produced it."
        ),
    ),
)
def recommend_command(
    ctx: typer.Context,
    use_case: str = _USE_CASE_OPTION,
    require: list[str] = _REQUIRE_OPTION,
    prefer: str = _PREFER_OPTION,
    license_: list[str] = _LICENSE_OPTION,
    max_download: str | None = _MAX_DOWNLOAD_OPTION,
    min_context: int = typer.Option(
        0,
        "--min-context",
        min=0,
        help=cast(
            str,
            lazy_gettext("Exclude candidates that cannot hold at least this many tokens."),
        ),
    ),
    limit: int = typer.Option(
        10, "--limit", min=1, help=cast(str, lazy_gettext("Show at most this many rows."))
    ),
    all_quants: bool = typer.Option(
        False,
        "--all-quants",
        help=cast(str, lazy_gettext("Show every quantisation instead of the best one per model.")),
    ),
    explain: bool = typer.Option(
        False,
        "--explain",
        help=cast(
            str,
            lazy_gettext(
                "Expand every row shown into the scores, weights, budget and speed "
                "breakdown behind it. Combine with --limit 1 for one model."
            ),
        ),
    ),
    no_vision: bool = typer.Option(
        False,
        "--no-vision",
        help=cast(str, lazy_gettext("Plan without a vision projector, freeing its memory.")),
    ),
) -> None:
    """Rank the catalog for what you want to do on this machine."""
    state: CliState = ctx.obj
    catalog = load_catalog_or_warn(state)
    report = scan()
    needs = Needs(
        use_case=check_use_case(use_case),
        capabilities=check_capabilities(require, option="--require"),
        min_context=min_context,
        max_download_bytes=(
            None if max_download is None else check_size(max_download, option="--max-download")
        ),
    )
    board = build_board(
        catalog,
        report.host,
        needs,
        all_quants=all_quants,
        limit=limit,
        local_models=report.llamacpp.local_models,
        licenses=license_,
        prefer=_check_prefer(prefer),
        vision=not no_vision,
    )
    if state.json_output:
        typer.echo(board.model_dump_json(indent=2, by_alias=True))
        return

    console = state.console
    if board.rows:
        console.print(render_board(board, console_width=console.width))
    else:
        console.print(
            _(
                "Nothing was ranked. Every candidate is listed below with the reason; "
                "widen the request or free some memory."
            )
        )
    # The explanations come before the exclusions, because they belong to the rows above
    # them: a reader who asked for the working behind row 1 should not have to scroll past
    # a table of models that are not on the board to reach it.
    if explain:
        for row in board.rows:
            console.print(render_explanation(row, board))
    excluded = render_excluded(board.excluded)
    if excluded is not None:
        console.print()
        console.print(excluded)


@app.command(
    "fit",
    help=cast(
        str,
        lazy_gettext(
            "Rank every model by how well it uses this machine, whatever it is for.\n\n"
            "No use case, no weights and no score: this is the memory question on its own. "
            "Models are sized for 32K tokens of context, and the context column is the "
            "largest each one holds in the mode shown."
        ),
    ),
)
def fit_command(
    ctx: typer.Context,
    perfect: bool = typer.Option(
        False,
        "--perfect",
        help=cast(
            str,
            lazy_gettext(
                "Only the configurations that use the machine well: between half and four "
                "fifths of the tightest pool."
            ),
        ),
    ),
    min_fit: str = _MIN_FIT_OPTION,
    limit: int | None = typer.Option(
        None, "--limit", min=1, help=cast(str, lazy_gettext("Show at most this many rows."))
    ),
    all_quants: bool = typer.Option(
        False,
        "--all-quants",
        help=cast(str, lazy_gettext("Show every quantisation instead of the best one per model.")),
    ),
) -> None:
    """Rank every model by how well it uses this machine, whatever it is for."""
    state: CliState = ctx.obj
    catalog = load_catalog_or_warn(state)
    report = scan()
    board = build_fit_board(
        catalog,
        report.host,
        min_fit=_check_min_fit(min_fit),
        perfect=perfect,
        limit=limit,
        all_quants=all_quants,
        local_models=report.llamacpp.local_models,
    )
    if state.json_output:
        typer.echo(board.model_dump_json(indent=2, by_alias=True))
        return

    console = state.console
    if board.rows:
        console.print(render_fit(board, console_width=console.width))
    else:
        console.print(
            _(
                "Nothing fits this machine at that threshold. Try --min-fit tight, or "
                "drop --perfect."
            )
        )
    excluded = render_fit_excluded(board.excluded)
    if excluded is not None:
        console.print()
        console.print(excluded)
