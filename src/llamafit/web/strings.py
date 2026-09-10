# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Every word the page draws, translated in Python and handed to the browser as JSON.

The problem this module exists to solve: LlamaFit speaks thirty-seven languages out of
GNU ``.po`` catalogs that only Python can read, and a page of static JavaScript cannot
read them. The three ways out are all worse than this one. Shipping an English page
gives most of the project's readers nothing. Translating in JavaScript means a second
catalog format, a second extractor and two sets of strings that drift. Compiling the
catalogs into JavaScript at build time means a build step, and there is not one.

So the page holds no words at all. Every label it draws is a key into a dictionary this
module builds, on the server, at request time, through the same ``_()`` and
``pgettext()`` calls every other message in LlamaFit goes through -- which means
``scripts/gen_messages.py`` extracts them from this file exactly as it extracts them from
the command line's, and a translator who fills in a ``.po`` file translates the dashboard
without knowing it exists. ``GET /api/v1/ui`` returns the dictionary; ``app.js`` looks
words up in it.

The tables of *values* -- verdicts, run modes, confidence labels, memory pools, budget
components -- are not written here at all. They are imported from
:mod:`llamafit.cli.render_board`, which is the interface layer's vocabulary and already
public, so the page cannot call ``too-tight`` anything but what the terminal calls it. The
capability names are the one table copied rather than imported, because the command line
keeps its copy private; the copy uses the identical ``pgettext`` context and messages, so
both read the same catalog entry, and ``tests/unit/test_web_strings.py`` fails if the two
tables ever disagree about a name.

One language per process. The translator is chosen once, at start-up, from ``--language``
or the environment or the system locale, exactly as it is for the command line; there is
no per-request negotiation and ``Accept-Language`` is ignored. That is honest for a server
whose whole design is one person at one keyboard, and it is what keeps the dashboard's
words and the terminal's words the same words.
"""

from __future__ import annotations

from typing import get_args

from llamafit import __version__
from llamafit.cli.render_board import (
    component_label,
    confidence_label,
    confidence_sentence,
    mode_label,
    pool_label,
    source_label,
    verdict_label,
    verdict_sentence,
)
from llamafit.i18n import _, current_language, is_rtl, pgettext
from llamafit.models.catalog import Capability
from llamafit.models.plan import Confidence, Pool, RunMode, Verdict
from llamafit.units import decimal_separator, format_bytes, group_separator

SOURCE_URL = "https://github.com/pvaz/llamafit"
"""Where this program's source is, which the page links to in its footer.

Section 13 of the AGPL is why the link is there rather than in a README nobody serving
this would think to publish: running a modified version as something people reach over a
network is the case the clause covers, and offering them the source is the condition.
"""

BUDGET_COMPONENTS: tuple[str, ...] = (
    "dense-weights",
    "shared-expert-weights",
    "expert-weights",
    "token-embedding",
    "output-head",
    "global-weights",
    "lazy-tables",
    "kv-cache",
    "kv-cache-unaccounted",
    "recurrent-state",
    "compute-buffer",
    "output-buffer",
    "vision-projector",
    "vision-projector-compute",
    "cuda-context",
    "process-overhead",
)
"""Section 8.1's components, in the order a reader meets them in a budget.

There is no enumeration to read these off: a component is a string a budget line carries.
A line whose component is not here still draws, under its own identifier, because
:func:`llamafit.cli.render_board.component_label` falls back to the identifier and the
page falls back to the same thing.
"""


def capability_label(capability: str) -> str:
    """One of a model's capabilities, in the reader's language.

    Args:
        capability: The catalog's own enum value.

    Returns:
        The word for it, or the value unchanged when this table has not met it.

    The command line keeps its copy of this table private, so it is written out again
    here rather than imported. Every entry uses the same ``pgettext`` context and the
    same English message, which means both look up the same catalog entry and no
    translator is asked for the word twice; ``tests/unit/test_web_strings.py`` compares
    the two tables entry by entry and fails if they ever part company.
    """
    labels = {
        "coding": pgettext("model capability", "coding"),
        "thinking": pgettext("model capability", "thinking"),
        "vision": pgettext("model capability", "vision"),
        "tools": pgettext("model capability", "tools"),
        "multilingual": pgettext("model capability", "multilingual"),
        "long-context": pgettext("model capability", "long-context"),
        "embeddings": pgettext("model capability", "embeddings"),
        "audio": pgettext("model capability", "audio"),
    }
    return labels.get(capability, capability)


def value_labels() -> dict[str, dict[str, str]]:
    """The words for every enumerated value the page draws, keyed by value.

    Returns:
        One table per enumeration: ``verdict``, ``verdict_sentence``, ``mode``,
        ``confidence``, ``confidence_sentence``, ``pool``, ``component``, ``source``
        and ``capability``. The keys are the identifiers ``--json`` carries, so the
        page looks a value up by the same name the document gives it.

    Every table but the last comes from :mod:`llamafit.cli.render_board`, so the page
    and the terminal cannot disagree about what a value is called.
    """
    return {
        "verdict": {value: verdict_label(value) for value in get_args(Verdict)},
        "verdict_sentence": {value: verdict_sentence(value) for value in get_args(Verdict)},
        "mode": {value: mode_label(value) for value in get_args(RunMode)},
        "confidence": {value: confidence_label(value) for value in get_args(Confidence)},
        "confidence_sentence": {
            value: confidence_sentence(value) for value in get_args(Confidence)
        },
        "pool": {value: pool_label(value) for value in get_args(Pool)},
        "component": {value: component_label(value) for value in BUDGET_COMPONENTS},
        "source": {"exact": source_label(True), "formula": source_label(False)},
        "capability": {value: capability_label(value) for value in get_args(Capability)},
        "score_part": {
            "quality": pgettext("score part", "quality"),
            "speed": pgettext("score part", "speed"),
            "fit": pgettext("score part", "fit"),
            "context": pgettext("score part", "context"),
        },
        "token_time": {
            "vram_seconds_per_token": pgettext("token time", "reading the card"),
            "ram_seconds_per_token": pgettext("token time", "reading system memory"),
            "overhead_seconds_per_token": pgettext("token time", "everything else"),
        },
    }


def column_headings() -> dict[str, str]:
    """The board's column headings, and the other tables', which are the terminal's own.

    Returns:
        Heading text keyed by the column name the page asks for it under.

    Every entry repeats a ``pgettext("column heading", ...)`` call that
    :mod:`llamafit.cli.render_board` already makes, so both interfaces read one catalog
    entry per heading and a column meaning the same thing is spelled the same way. That
    is why ``Tok/s`` was renamed in three places at once rather than one: a reader who
    could not find the tokens-per-second column behind ``Gen/s`` could not find it in the
    terminal either.

    Every heading the terminal can draw is served, whether or not the board draws it
    today. A heading is a word, and which words a table spends its width on is a decision
    ``app.js`` makes -- the page's board keeps seven columns and puts the rest inside the
    row -- while this table's job is only that the word exists and is the terminal's own.
    """
    return {
        "rank": pgettext("column heading", "#"),
        "model": pgettext("column heading", "Model"),
        "quant": pgettext("column heading", "Quant"),
        "score": pgettext("column heading", "Score"),
        "gen": pgettext("column heading", "Tok/s"),
        "confidence": pgettext("column heading", "How"),
        "verdict": pgettext("column heading", "Fit"),
        "mode": pgettext("column heading", "Runs"),
        "context": pgettext("column heading", "Ctx"),
        # The board is a table of many columns and abbreviates; the context ladder has
        # three and does not. Both spellings are the terminal's, in the same two places.
        "context_full": pgettext("column heading", "Context"),
        "quality": pgettext("column heading", "Qual"),
        "vram": pgettext("column heading", "Card"),
        "prompt": pgettext("column heading", "Prompt tok/s"),
        "size": pgettext("column heading", "Size"),
        "ram": pgettext("column heading", "RAM"),
        "component": pgettext("column heading", "Component"),
        "where": pgettext("column heading", "Where"),
        "from": pgettext("column heading", "From"),
        "why_not": pgettext("column heading", "Why not"),
        "part": pgettext("column heading", "Part"),
        "weight": pgettext("column heading", "Weight"),
        "adds": pgettext("column heading", "Adds"),
        "card_needs": pgettext("column heading", "Card needs"),
        "on_this_machine": pgettext("column heading", "On this machine"),
        "token_time": pgettext("column heading", "A token's time"),
        "seconds": pgettext("column heading", "Seconds"),
        "share": pgettext("column heading", "Share"),
        "run": pgettext("column heading", "Run"),
        "date": pgettext("column heading", "Date"),
    }


def number_format() -> dict[str, str]:
    """The punctuation this language writes numbers with, for the page to format with.

    Returns:
        The group separator, the decimal separator, and the word for a size nobody could
        read.

    The page draws byte counts and token counts, and it gets them as integers because the
    API serialises the models rather than pictures of them. Formatting them is the one
    piece of :mod:`llamafit.units` that ``app.js`` repeats, and it repeats only the
    arithmetic -- which unit, one decimal place -- while every character that varies by
    language comes from here. ``tests/unit/test_web_strings.py`` pins the unit ladder and
    a handful of sizes against :func:`llamafit.units.format_bytes`, so a change to how
    LlamaFit writes a size fails a test that names the file to change with it.
    """
    return {
        "group": group_separator(),
        "decimal": decimal_separator(),
        "unknown_size": format_bytes(None),
    }


def page_strings() -> dict[str, str]:
    """Every word the page draws that is not the name of a value or a column.

    Returns:
        The text keyed by the name ``app.js`` asks for it under.

    Written out one literal at a time, like every other message in LlamaFit, because
    ``scripts/gen_messages.py`` reads the syntax tree and can only extract a literal.
    """
    return {
        # The five panels, which are section 13.2's screens laid out for a browser.
        "panel.board": _("Board"),
        "panel.needs": _("Needs"),
        "panel.host": _("Host"),
        "panel.plan": _("Plan"),
        "panel.simulate": _("Simulate"),
        # The board.
        "board.title": _("Recommended"),
        "board.excluded": _("Not ranked"),
        "board.empty": _(
            "Nothing was ranked. Every candidate is listed below with the reason; "
            "widen the request or free some memory."
        ),
        "board.explain": _("Why this row"),
        "board.plan_this": _("Plan this"),
        "board.installed": _("Already on this machine"),
        "board.score_parts": _("Score"),
        # Sentences the terminal prints under the same tables, reused here by writing the
        # identical message: one catalog entry serves both, and a translator who filled it
        # in for the command line has already translated the page.
        "board.caption": _(
            "Speeds are for %(working)s tokens of context so every row compares like with "
            "like; the context column is the largest each one holds. Sized and scored for "
            "%(use_case)s at %(requested)s tokens."
        ),
        "board.caption_split": _(
            "Speeds are for %(working)s tokens of context so every row compares like with "
            "like; the context column is the largest each one holds. Sized for %(planned)s "
            "tokens and scored for %(use_case)s against %(requested)s."
        ),
        # The terminal's own version of this sentence names --limit, which is the control
        # there. Here the control is the "Rows to show" box above the table, so the page
        # gets its own wording: a message that points at a flag the reader cannot see is
        # a message that tells them to go and find one.
        "board.truncated": _(
            "Showing the best %(shown)s of %(total)s that qualified; raise “Rows to "
            "show” for the rest, which are neither worse-behaved nor hidden, only "
            "further down."
        ),
        "board.weights": _("Weights: %(weights)s."),
        "board.score_total": _("Score %(total)s"),
        "plan.speed_headline": _(
            "%(gen)s tokens per second generated and %(prompt)s read, at %(context)s tokens "
            "of context (%(confidence)s)."
        ),
        "plan.tiers_caption": _(
            "A launch script compares the free memory it sees against the card column and "
            "takes the largest rung that fits (see docs/cli.md)."
        ),
        "plan.card_totals": _("Card: %(required)s of %(available)s free, %(share)s used."),
        "plan.system_totals": _(
            "System memory: %(required)s of %(available)s free, %(share)s used."
        ),
        # The needs form, whose fields are the options `llamafit recommend` takes.
        "needs.use_case": _("What the model is for"),
        "needs.require": _("Capabilities it must have"),
        "needs.prefer": _("Lean the weights"),
        "needs.min_context": _("Smallest context worth having"),
        "needs.max_download": _("Most you will download"),
        "needs.all_quants": _("Show every quantisation"),
        "needs.vision": _("Keep the vision projector"),
        "needs.limit": _("Rows to show"),
        "needs.apply": _("Apply"),
        "needs.reset": _("Reset"),
        # The host panel.
        "host.rescan": _("Scan again"),
        "host.llamacpp": _("llama.cpp"),
        "host.findings": _("Findings"),
        "host.machine": _("Host"),
        # The row labels are the terminal's own, written as the same literals so both
        # interfaces read one catalog entry per row.
        "host.os": _("OS"),
        "host.cpu": _("CPU"),
        "host.memory": _("Memory"),
        "host.gpu": _("GPU"),
        "host.disk": _("Disk"),
        "host.installed": _("Installed"),
        # The plan panel.
        "plan.quant": _("Quantisation"),
        "plan.context": _("Context"),
        "plan.micro_batch": _("Micro-batch"),
        "plan.build": _("Plan it"),
        "plan.command": _("Command line"),
        "plan.copy": _("Copy"),
        "plan.copied": _("Copied"),
        "plan.flags": _("Flags"),
        "plan.budget": _("Memory budget"),
        "plan.tiers": _("Context tiers"),
        "plan.speed": _("Speed"),
        "plan.notes": _("Notes"),
        "plan.measurements": _("Recorded elsewhere"),
        "plan.file_here": _("Model file: %(path)s."),
        "plan.file_missing": _(
            "Model file: %(path)s, which is not on this machine yet; that is where a "
            "download would put it. %(size)s to fetch."
        ),
        "plan.requested_cost": _("The %(context)s tokens asked for would have cost"),
        "plan.pick_first": _("Choose a row on the board first."),
        # The simulate panel.
        "simulate.profile": _("Hardware profile"),
        "simulate.live": _("This machine, as scanned"),
        "simulate.gpu_memory": _("Graphics memory"),
        "simulate.ram": _("System memory"),
        "simulate.cpu_cores": _("Processor cores"),
        "simulate.apply": _("Score against that machine"),
        "simulate.clear": _("Back to this machine"),
        "simulate.badge": _("SIMULATED"),
        "simulate.sizes": _("Sizes look like 8G, 7.5GiB or 512M; a bare number is bytes."),
        # Chrome shared by every panel.
        "app.subtitle": _("What runs here, how fast, and with which settings."),
        "app.loading": _("Working…"),
        "app.error": _("Something went wrong"),
        "app.hint": _("Hint"),
        "app.command": _("Command"),
        "app.none": _("none"),
        "app.source": _("Source code (AGPL-3.0-or-later)"),
        "app.local_only": _(
            "This dashboard is served from your own machine and sends nothing anywhere."
        ),
    }


def ui_payload() -> dict[str, object]:
    """Everything the page needs before it can draw a single word.

    Returns:
        The language tag, the reading direction, the sentence saying no figure here has
        been measured, and the three string tables.

    The estimate sentence is not written here. It is
    :func:`llamafit.cli.render_board.confidence_sentence` for ``estimated``, which is the
    sentence the terminal prints under the same table: nothing has been benchmarked on
    this machine, so every speed on the page is the formula's own answer, and the page
    has to say that where a reader will see it rather than leave it to be inferred from a
    one-word column.
    """
    return {
        "language": current_language(),
        "direction": "rtl" if is_rtl(current_language()) else "ltr",
        "version": __version__,
        "source_url": SOURCE_URL,
        "estimate_notice": confidence_sentence("estimated"),
        "format": number_format(),
        "strings": page_strings(),
        "columns": column_headings(),
        "labels": value_labels(),
    }
