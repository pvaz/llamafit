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

Two families of flag, kept apart on purpose. A **request** flag -- ``--use-case``,
``--require``, ``--min-tps``, ``--max-download`` and the rest -- changes what is ranked,
and the board says so under itself: a candidate it excludes is listed with the reason.
A **view** flag -- ``--sort``, ``--search``, ``--installed``, ``--runs``, ``--min-fit``
on ``recommend``, ``--columns``, ``--wide``, ``--hide-excluded`` -- changes only what is
drawn: the ranking, the count that qualified and the ``--json`` document are untouched,
except that ``--sort`` reorders the document's rows and leaves their ranks alone, as the
web API's ``sort=`` does. The numeric boxes the web page has under every heading are not
flags here. ``--min-tps`` excludes and says so; a ``--min-speed`` that merely hid would
sit one line under it in ``--help`` and differ in a word, and a reader who picked the
wrong one would get a board whose caption disagreed with its rows for a reason nothing
named. The page needed boxes because a pointer cannot type a flag; a terminal has
``--sort``, ``--limit`` and ``--json``, and the dashboard's ``/`` box takes the terms.
"""

from __future__ import annotations

from typing import cast, get_args

import typer

from llamafit.cli.app import CliState, app
from llamafit.cli.common import (
    check_capabilities,
    check_size,
    check_use_case,
    checked_max_context,
    load_catalog_or_warn,
    machine,
)
from llamafit.cli.render_board import (
    BOARD_ORDER,
    FIT_ORDER,
    FIT_SORT_KEYS,
    SORT_KEYS,
    Column,
    Filters,
    View,
    column_budget,
    parse_columns,
    parse_sort,
    render_board,
    render_explanation,
    render_fit,
    render_fit_reasons,
    render_reasons,
    sorted_rows,
    visible_rows,
)
from llamafit.errors import CatalogError
from llamafit.i18n import _, lazy_gettext
from llamafit.models.plan import Needs, RunMode, Verdict
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
# Optional, so it is a module-level singleton for the reason the block above gives: ruff's
# B008 does not read ``float | None`` in a default position as safe to call.
_MIN_TPS_OPTION: float | None = typer.Option(
    None,
    "--min-tps",
    min=0.0,
    help=cast(
        str,
        lazy_gettext(
            "Exclude candidates generating fewer tokens per second than this, in place of "
            "the speed a person reads at. Pass 0 when nobody is waiting on the tokens."
        ),
    ),
)

# The view flags. None of them changes what is ranked; each changes what is drawn.
_SORT_OPTION: str = typer.Option(
    "score",
    "--sort",
    metavar="KEY[:asc|:desc]",
    help=cast(
        str,
        lazy_gettext(
            "Order the rows drawn: score, speed, quality, context, size, prompt, card, "
            "ram, fit, model or quant, largest or best first unless :asc says otherwise. "
            "The # column keeps the ranking."
        ),
    ),
)
_SEARCH_OPTION: str | None = typer.Option(
    None,
    "--search",
    metavar="TEXT",
    help=cast(
        str, lazy_gettext("Draw only rows whose id, name or quantisation contains this text.")
    ),
)
_INSTALLED_OPTION: bool = typer.Option(
    False,
    "--installed",
    help=cast(str, lazy_gettext("Draw only rows whose file is already on this machine.")),
)
_RUNS_OPTION: str | None = typer.Option(
    None,
    "--runs",
    metavar="MODE",
    help=cast(
        str,
        lazy_gettext("Draw only rows that run this way: gpu, moe-offload, hybrid or cpu."),
    ),
)
_VIEW_MIN_FIT_OPTION: str | None = typer.Option(
    None,
    "--min-fit",
    help=cast(
        str,
        lazy_gettext(
            "Draw only rows at or above this verdict: comfortable, fits or tight. The "
            "ranking is unchanged and a hidden row is still counted."
        ),
    ),
)
_COLUMNS_OPTION: str | None = typer.Option(
    None,
    "--columns",
    metavar="LIST",
    help=cast(
        str,
        lazy_gettext(
            "Draw exactly these columns, comma-separated, in this order, named as --json "
            "names them: rank, model, quant, size, have, score, quality, gen, prompt, "
            "confidence, mode, vram, ram, verdict, context."
        ),
    ),
)
_WIDE_OPTION: bool = typer.Option(
    False,
    "--wide",
    help=cast(
        str,
        lazy_gettext(
            "Draw every column whatever the terminal's width; a cell that does not fit folds."
        ),
    ),
)
_HIDE_EXCLUDED_OPTION: bool = typer.Option(
    False,
    "--hide-excluded",
    help=cast(
        str,
        lazy_gettext(
            "Leave the candidates that were not ranked off the table. The reasons are "
            "still printed under it."
        ),
    ),
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


def _check_runs(value: str) -> RunMode:
    """Return the run mode ``--runs`` named, or refuse it and list the four."""
    modes = [mode for mode in get_args(RunMode) if mode != "unsupported"]
    if value not in modes:
        raise CatalogError(
            _("invalid --runs %(value)s") % {"value": repr(value)},
            hint=_("Valid values: %(values)s") % {"values": ", ".join(modes)},
        )
    return cast("RunMode", value)


def _check_view(
    *,
    sort: str,
    sort_keys: tuple[str, ...],
    search: str | None,
    installed: bool,
    runs: str | None,
    min_fit: str | None,
    columns: str | None,
    column_keys: tuple[Column, ...],
    wide: bool,
    hide_excluded: bool,
) -> View:
    """The view flags as one :class:`~llamafit.cli.render_board.View`, each checked by name.

    Every refusal lists what would have been accepted, in the words the other checkers
    in this module use, because a flag that says "invalid" and stops is a flag that
    sends a reader to the documentation for a list this program already holds.
    """
    try:
        key, descending = parse_sort(sort, allowed=sort_keys)
    except ValueError as exc:
        raise CatalogError(
            _("invalid --sort %(value)s") % {"value": repr(sort)},
            hint=_("Valid values: %(values)s") % {"values": ", ".join(sort_keys)},
        ) from exc
    chosen: tuple[Column, ...] | None = None
    if columns is not None:
        try:
            chosen = parse_columns(columns, allowed=column_keys)
        except ValueError as exc:
            raise CatalogError(
                _("invalid --columns %(value)s") % {"value": repr(str(exc))},
                hint=_("Valid values: %(values)s") % {"values": ", ".join(column_keys)},
            ) from exc
    return View(
        sort=key,
        descending=descending,
        filters=Filters(
            search=search or "",
            min_fit=None if min_fit is None else _check_min_fit(min_fit),
            installed=installed,
            mode=None if runs is None else _check_runs(runs),
        ),
        columns=chosen,
        wide=wide,
        excluded=not hide_excluded,
    )


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
    min_tps: float | None = _MIN_TPS_OPTION,
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
    sort: str = _SORT_OPTION,
    search: str | None = _SEARCH_OPTION,
    installed: bool = _INSTALLED_OPTION,
    runs: str | None = _RUNS_OPTION,
    min_fit: str | None = _VIEW_MIN_FIT_OPTION,
    columns: str | None = _COLUMNS_OPTION,
    wide: bool = _WIDE_OPTION,
    hide_excluded: bool = _HIDE_EXCLUDED_OPTION,
) -> None:
    """Rank the catalog for what you want to do on this machine."""
    state: CliState = ctx.obj
    view = _check_view(
        sort=sort,
        sort_keys=SORT_KEYS,
        search=search,
        installed=installed,
        runs=runs,
        min_fit=min_fit,
        columns=columns,
        column_keys=BOARD_ORDER,
        wide=wide,
        hide_excluded=hide_excluded,
    )
    catalog = load_catalog_or_warn(state)
    report = machine(state)
    needs = Needs(
        use_case=check_use_case(use_case),
        capabilities=check_capabilities(require, option="--require"),
        min_context=min_context,
        min_tps=min_tps,
        max_context=checked_max_context(state, min_context=min_context),
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
        # The one view flag the document honours, the way the API's ``sort=`` honours it:
        # the rows reordered, every rank left where the scorer put it.
        if view.reordered:
            board = board.model_copy(
                update={"rows": sorted_rows(board.rows, view.sort, view.descending)}
            )
        typer.echo(board.model_dump_json(indent=2, by_alias=True))
        return

    # Printed through the renderer whether or not anything ranked. The sentence for an
    # empty board lives in `render_board` with the table it replaces, because the red line
    # saying these rows are not this machine has to be above both of them, and a command
    # that chose between a renderer and a bare `print` was the one place it was not.
    console = state.console
    console.print(
        render_board(board, console_width=console.width, view=view, budget=column_budget(catalog))
    )
    # The explanations come before the reasons, because they belong to the rows above
    # them: a reader who asked for the working behind row 1 should not have to scroll past
    # every candidate that is not on the board to reach it. They follow the view -- the
    # rows drawn, in the order drawn -- so ``--sort speed --explain`` explains the fastest
    # first and a row a filter hid is not expanded.
    if explain:
        for row in visible_rows(board.rows, view):
            console.print(render_explanation(row, board))
    reasons = render_reasons(board)
    if reasons is not None:
        console.print()
        console.print(reasons)


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
                "Only the configurations that use the machine well: at least half its "
                "memory held in weights, and no pool past four fifths."
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
    sort: str = _SORT_OPTION,
    search: str | None = _SEARCH_OPTION,
    installed: bool = _INSTALLED_OPTION,
    runs: str | None = _RUNS_OPTION,
    columns: str | None = _COLUMNS_OPTION,
    wide: bool = _WIDE_OPTION,
    hide_excluded: bool = _HIDE_EXCLUDED_OPTION,
) -> None:
    """Rank every model by how well it uses this machine, whatever it is for."""
    state: CliState = ctx.obj
    # ``--min-fit`` is a request flag here -- it decides what qualifies and the count says
    # so -- which is why the view takes none: the same word cannot mean two things on one
    # command.
    view = _check_view(
        sort=sort,
        sort_keys=FIT_SORT_KEYS,
        search=search,
        installed=installed,
        runs=runs,
        min_fit=None,
        columns=columns,
        column_keys=FIT_ORDER,
        wide=wide,
        hide_excluded=hide_excluded,
    )
    catalog = load_catalog_or_warn(state)
    report = machine(state)
    board = build_fit_board(
        catalog,
        report.host,
        needs=Needs(max_context=checked_max_context(state)),
        min_fit=_check_min_fit(min_fit),
        perfect=perfect,
        limit=limit,
        all_quants=all_quants,
        local_models=report.llamacpp.local_models,
    )
    if state.json_output:
        if view.reordered:
            board = board.model_copy(
                update={"rows": sorted_rows(board.rows, view.sort, view.descending)}
            )
        typer.echo(board.model_dump_json(indent=2, by_alias=True))
        return

    console = state.console
    console.print(
        render_fit(board, console_width=console.width, view=view, budget=column_budget(catalog))
    )
    reasons = render_fit_reasons(board)
    if reasons is not None:
        console.print()
        console.print(reasons)
