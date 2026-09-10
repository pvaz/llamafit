# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Rich tables for the board, the fit listing, the memory budget and one model's plan.

A second rendering module rather than more of :mod:`llamafit.cli.render`, which draws the
host, llama.cpp and the catalog. These tables answer a different question and share almost
nothing with those: what they share is the habits, and those live in the first module's
documentation and are followed here — a counted noun through :func:`ngettext`, a short word
under a :func:`pgettext` context naming its row, a whole sentence per shape rather than
fragments joined in Python, and every identifier, figure and size through
:func:`llamafit.i18n.isolate` on its way into a line that a right-to-left reader may meet.

One rule is this file's own, and it is the reason most of the code below exists.

**No number appears without saying how it was arrived at.** A budget line says whether its
bytes come from the file's own tensor table or from a fitted formula, and the lines that
are modelled carry the note that says what was modelled. A speed says whether it is a
benchmark, a correction of the formula by one, or the formula on its defaults. A score
expands into the four parts and the weights that combined them. A context tier says whether
it fits, whether it would page, or whether nothing would absorb it. The alternative — a
table of thirteen bare figures — is what a tool whose whole claim is honesty about
uncertainty must never print, and it is the easiest thing in the world to print by accident.

The second rule follows from the first. **A configuration that overflows the card is shown,
not skipped.** Section 8.4: an NVIDIA driver does not refuse the allocation, it pages to
system memory and the speed collapses while the server starts and the log looks healthy. So
``pages`` is a state of its own in the tier table, the tier table is printed for every plan,
and a context the planner had to retreat from is costed and shown beside the one it chose.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from llamafit.cli.render import render_simulation
from llamafit.i18n import _, for_display, isolate, mirror_justify, ngettext, pgettext, reading_order
from llamafit.models.catalog import Measured
from llamafit.models.plan import (
    Budget,
    Candidate,
    ContextTier,
    Placement,
    QualityBreakdown,
    SpeedEstimate,
)
from llamafit.scoring.fit_score import (
    IDEAL_HIGH,
    IDEAL_LOW,
    resident_model_bytes,
    worst_pool_utilisation,
)
from llamafit.scoring.speed_score import prompt_penalty, target_tps
from llamafit.services.plan import PlanReport, TargetCheck
from llamafit.services.recommend import Board, BoardRow, FitBoard, FitRow
from llamafit.units import format_bytes, format_grouped, localise_number

Cell = str | Text
"""What one cell of a table may be: markup-free text, or a plain string of labels."""

_ID_COLUMN_MAX_WIDTH = 28
"""How wide the model column may grow, in terminal cells, before it folds."""

_COLUMN_OVERHEAD = 3
"""What one more column costs beyond its content: a border and the padding either side."""

_TABLE_OVERHEAD = 1
"""The table's own leading border, charged once."""


def _cell(text: str) -> Text:
    """One cell's finished text, prepared for the reader's writing direction.

    ``Text`` rather than a markup string because a model name, a path or a flag may hold
    square brackets Rich would otherwise try to parse as a tag.
    """
    return Text(for_display(text))


def _add_columns(table: Table, columns: Sequence[Mapping[str, Any]]) -> None:
    """Add a table's columns in reading order, with their alignment mirrored to match."""
    for column in reading_order(columns):
        options = dict(column)
        header = str(options.pop("header"))
        options["justify"] = mirror_justify(str(options.get("justify", "left")))
        table.add_column(for_display(header), **options)


def _add_row(table: Table, *cells: Cell) -> None:
    """Add one row, its cells in the same order :func:`_add_columns` put the columns."""
    prepared: list[Cell] = [c if isinstance(c, Text) else for_display(c) for c in cells]
    table.add_row(*reading_order(prepared))


def _size(n: int | None) -> str:
    """A byte size as one directional island, or the word for a size nobody could read."""
    return isolate(format_bytes(n)) if n is not None else format_bytes(n)


def _number(value: float, digits: int = 1) -> str:
    """A figure with its separators in this language's punctuation, as one island."""
    return isolate(localise_number(f"{value:,.{digits}f}"))


def _tps(value: float) -> str:
    """A tokens-per-second figure, whole when it is whole, as one directional island.

    Six reads better in a sentence than 6.0, and a figure somebody typed as 7.5 has to
    keep its fraction or the caption would name a boundary that is not the one in force.
    """
    return _number(value, 0 if value == int(value) else 1)


def _percent(share: float | None) -> str:
    """A utilisation as a percentage, or the word for one nobody could compute."""
    if share is None:
        return pgettext("utilisation", "n/a")
    return isolate(localise_number(f"{share * 100:.0f}%"))


def _context(tokens: int) -> str:
    """A context length compactly, for example ``40K``, and plainly when it is not round."""
    if tokens % 1024 == 0:
        return isolate(f"{tokens // 1024}K")
    return isolate(format_grouped(tokens))


def mode_label(mode: str) -> str:
    """How the weights are divided, in the reader's language.

    The five run modes of section 9.1 are the identifiers ``--json`` and the documentation
    use, and they are words a reader meets in a column heading's worth of space. Each is
    written out as its own call so the extractor can find it; a mode this table does not
    know yet comes back in English rather than disappearing from the row.
    """
    labels = {
        "gpu": pgettext("run mode", "GPU"),
        "moe-offload": pgettext("run mode", "experts in RAM"),
        "hybrid": pgettext("run mode", "split"),
        "cpu": pgettext("run mode", "CPU"),
        "unsupported": pgettext("run mode", "nowhere"),
    }
    return labels.get(mode, mode)


def verdict_label(verdict: str) -> str:
    """How well a configuration fits, in one word narrow enough for a column.

    The long forms are section 8.3's own: Comfortable, Fits, Tight, Too Tight, Does Not
    Fit. These are the short forms a table can carry, and :func:`verdict_sentence` is what
    says what each of them means where there is room to say it.

    """
    labels = {
        "comfortable": pgettext("fit verdict", "roomy"),
        "fits": pgettext("fit verdict", "fits"),
        "tight": pgettext("fit verdict", "tight"),
        "too-tight": pgettext("fit verdict", "pages"),
        "does-not-fit": pgettext("fit verdict", "no room"),
    }
    return labels.get(verdict, verdict)


def verdict_style(verdict: str) -> str:
    """The colour a verdict is drawn in: green while it runs well, red once it does not."""
    return {
        "comfortable": "green",
        "fits": "green",
        "tight": "yellow",
        "too-tight": "red",
        "does-not-fit": "red",
    }.get(verdict, "")


def verdict_sentence(verdict: str) -> str:
    """What a verdict means, spelled out, including the one that reads as success.

    ``too-tight`` is the sentence this whole file exists for. It is not a failure a reader
    would notice: the server starts, the log is clean, and the driver quietly pages the
    overflow to system memory while generation runs at a fraction of its speed.
    """
    sentences = {
        "comfortable": _("Comfortable: room for a longer context or a second model."),
        "fits": _("Fits: the intended configuration runs as planned."),
        "tight": _("Tight: it runs, but a browser or a second process can push it over."),
        "too-tight": _(
            "Pages: this needs more of the graphics card than is free. The driver will not "
            "refuse it — it moves the overflow to system memory, the server starts, the log "
            "looks healthy, and generation runs at a fraction of its speed with nothing "
            "saying why."
        ),
        "does-not-fit": _("No room: more memory than this machine has, with nothing to page to."),
    }
    return sentences.get(verdict, verdict)


def confidence_label(confidence: str) -> str:
    """How much weight a figure deserves, in the reader's language (section 10.3)."""
    labels = {
        "measured": pgettext("confidence", "measured"),
        "calibrated": pgettext("confidence", "calibrated"),
        "estimated": pgettext("confidence", "estimated"),
        "unsupported": pgettext("confidence", "no estimate"),
    }
    return labels.get(confidence, confidence)


def confidence_sentence(confidence: str) -> str:
    """What a confidence label claims, said once under a table rather than per row."""
    sentences = {
        "measured": _("Speeds are benchmarks taken on this machine, not predictions."),
        "calibrated": _(
            "Speeds are section 10's formula corrected by benchmarks taken on this machine."
        ),
        "estimated": _(
            "Speeds are section 10's formula on its default constants. Nothing has been "
            "benchmarked on this machine yet, so no figure here is a measurement."
        ),
        "unsupported": _("No speed could be estimated."),
    }
    return sentences.get(confidence, confidence)


def pool_label(pool: str) -> str:
    """Where a component's bytes live, in the reader's language."""
    labels = {
        "vram": pgettext("memory pool", "card"),
        "ram": pgettext("memory pool", "system"),
        "disk": pgettext("memory pool", "disk"),
    }
    return labels.get(pool, pool)


def component_label(component: str) -> str:
    """One budget line's component, as words rather than as the identifier ``--json`` uses.

    Section 8.1's own table, one call per row so the extractor can find each. A component
    this table has not met comes back as its identifier, which is ugly and truthful, rather
    than as a blank cell.
    """
    labels = {
        "dense-weights": pgettext("budget component", "dense weights"),
        "shared-expert-weights": pgettext("budget component", "shared experts"),
        "expert-weights": pgettext("budget component", "routed experts"),
        "token-embedding": pgettext("budget component", "token embedding"),
        "output-head": pgettext("budget component", "output head"),
        "global-weights": pgettext("budget component", "other weights"),
        "lazy-tables": pgettext("budget component", "streamed tables"),
        "kv-cache": pgettext("budget component", "KV cache"),
        "kv-cache-unaccounted": pgettext("budget component", "KV cache, undeclared"),
        "recurrent-state": pgettext("budget component", "recurrent state"),
        "compute-buffer": pgettext("budget component", "compute buffer"),
        "output-buffer": pgettext("budget component", "output buffer"),
        "vision-projector": pgettext("budget component", "vision projector"),
        "vision-projector-compute": pgettext("budget component", "vision compute"),
        "cuda-context": pgettext("budget component", "backend overhead"),
        "process-overhead": pgettext("budget component", "process overhead"),
    }
    return labels.get(component, component)


def source_label(exact: bool) -> str:
    """Whether a budget line is arithmetic on the file's own bytes, or a model of them."""
    if exact:
        return pgettext("budget line source", "file")
    return pgettext("budget line source", "formula")


def render_budget(budget: Budget, *, title: str | None = None, notes: bool = True) -> Group:
    """One configuration's memory budget, line by line, with the source of every line.

    Args:
        budget: What the configuration needs.
        title: A heading for the table, when the caller has one.
        notes: Print the note behind each modelled line. A second budget for the same
            model at another context carries the same notes word for word, and printing
            them twice buries the one thing that differs between the two tables.

    Returns:
        The table, the two totals, the verdict spelled out, and the note behind every line
        that is a model rather than a measurement.

    The ``From`` column is the point of the table. Weights and the key-value cache are
    arithmetic on sizes the file itself reports; the compute buffer is a formula fitted to
    ten measurements on one graphics card. Both end up as byte figures in the same column,
    and without a word saying which is which the second borrows the authority of the first.
    """
    table = Table(title=for_display(title) if title else None)
    _add_columns(
        table,
        [
            {"header": pgettext("column heading", "Component")},
            {"header": pgettext("column heading", "Where"), "no_wrap": True},
            {"header": pgettext("column heading", "Size"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "From"), "no_wrap": True},
        ],
    )
    for line in budget.lines:
        _add_row(
            table,
            _cell(component_label(line.component)),
            _cell(pool_label(line.pool)),
            _cell(_size(line.bytes_)),
            _cell(source_label(line.exact)),
        )
    return Group(table, *_budget_totals(budget), *(_budget_notes(budget) if notes else []))


def _budget_totals(budget: Budget) -> list[RenderableType]:
    """The two pools' totals, each as one sentence with what is free and the share taken."""
    lines: list[RenderableType] = [
        _cell(
            _("Card: %(required)s of %(available)s free, %(share)s used.")
            % {
                "required": _size(budget.vram_required),
                "available": _size(budget.vram_available),
                "share": _percent(budget.vram_utilisation),
            }
        ),
        _cell(
            _("System memory: %(required)s of %(available)s free, %(share)s used.")
            % {
                "required": _size(budget.ram_required),
                "available": _size(budget.ram_available),
                "share": _percent(budget.ram_utilisation),
            }
        ),
        Text(for_display(verdict_sentence(budget.verdict)), style=verdict_style(budget.verdict)),
    ]
    return lines


def _budget_notes(budget: Budget) -> list[RenderableType]:
    """The note under every line that says something the figure alone does not.

    Only the lines that carry one, and each named by its component, so a reader who wants
    to know why the backend overhead is 120 MiB rather than 300 can find out without being
    made to read fifteen notes to get there.
    """
    noted = [line for line in budget.lines if line.note]
    if not noted:
        return []
    out: list[RenderableType] = [Text("")]
    for line in noted:
        out.append(
            Text(
                for_display(
                    _("%(component)s: %(note)s")
                    % {"component": component_label(line.component), "note": line.note}
                ),
                style="dim",
            )
        )
    return out


def tier_state(tier: ContextTier) -> str:
    """What happens at one rung of the ladder, in the word its own verdict earned.

    Three answers, not two. A rung that fits is a rung a launch script may take; a rung
    that overflows the card is one the driver will accept and then page, silently, which
    is the case section 8.4 exists to name; a rung that overflows system memory is one that
    will not start. A table showing only "yes" and "no" would file the second under the
    third and lose the only one a reader could not have worked out for themselves.
    """
    return verdict_label(tier.verdict)


def render_tiers(placement: Placement) -> Group:
    """The context ladder, with what each rung costs the card and what happens there.

    This is the most useful thing a plan produces. A recommendation sized for the free
    memory of one moment stops being true when a browser opens; this table stays true,
    because a launch script reads the free VRAM it actually sees and picks the largest rung
    whose ``vram_required`` fits under it (section 15.3).
    """
    table = Table(title=for_display(_("Context tiers")))
    _add_columns(
        table,
        [
            {"header": pgettext("column heading", "Context"), "justify": "right", "no_wrap": True},
            {
                "header": pgettext("column heading", "Card needs"),
                "justify": "right",
                "no_wrap": True,
            },
            {"header": pgettext("column heading", "On this machine"), "no_wrap": True},
        ],
    )
    pages = any(tier.verdict == "too-tight" for tier in placement.tiers)
    for tier in placement.tiers:
        _add_row(
            table,
            _cell(_context(tier.tokens)),
            _cell(_size(tier.vram_required)),
            Text(for_display(tier_state(tier)), style=verdict_style(tier.verdict)),
        )
    caption = _(
        "A launch script compares the free memory it sees against the card column and takes "
        "the largest rung that fits (see docs/cli.md)."
    )
    below: list[RenderableType] = [_cell(caption)]
    if pages:
        below.append(Text(for_display(verdict_sentence("too-tight")), style="red"))
    return Group(table, *below)


def render_speed(speed: SpeedEstimate, *, context: int) -> Group:
    """The two speeds, where each one's time goes, and how the figures were arrived at.

    Both figures get the same treatment, because both were owed it. A trio of shares
    always adds up to one over the figure it belongs to, whichever rung of section 10.3's
    ladder that figure came from, so a reader who doubts a number is owed terms that add
    up to it and say which one dominates.

    Prompt processing is the one that pays for this. Its second term, the expert set
    crossing the link, is charged at a rate section 10.2 fitted on the single model whose
    expert set does not fit in system memory; a model whose set stays in the page cache
    streams about three times faster and this project has one constant for both paths.
    That is a known gap, written down rather than fixed here, and it is why the term is a
    row of its own with the note underneath rather than a share of a total: the reader
    looking at the terms is exactly the one who might catch it before we do.
    """
    headline = _cell(
        _(
            "%(gen)s tokens per second generated and %(prompt)s read, at %(context)s tokens "
            "of context (%(confidence)s)."
        )
        % {
            "gen": _number(speed.gen_tps),
            "prompt": _number(speed.pp_tps, 0),
            "context": _context(context),
            "confidence": confidence_label(speed.confidence),
        }
    )
    tables = [
        _time_table(
            pgettext("column heading", "A token's time"),
            (
                (pgettext("token time", "reading the card"), speed.vram_seconds_per_token),
                (pgettext("token time", "reading system memory"), speed.ram_seconds_per_token),
                (pgettext("token time", "everything else"), speed.overhead_seconds_per_token),
            ),
        ),
        _time_table(
            pgettext("column heading", "A prompt token's time"),
            (
                (
                    pgettext("prompt token time", "doing the arithmetic"),
                    speed.prompt_compute_seconds_per_token,
                ),
                (
                    pgettext("prompt token time", "streaming the experts across the link"),
                    speed.prompt_link_seconds_per_token,
                ),
                (
                    pgettext("prompt token time", "reading the experts from system memory"),
                    speed.prompt_ram_seconds_per_token,
                ),
            ),
        ),
    ]
    drawn = [table for table in tables if table is not None]
    return Group(headline, *drawn, *_speed_notes(speed))


def _time_table(heading: str, parts: Sequence[tuple[str, float]]) -> Table | None:
    """One figure's time, term by term, or ``None`` when there is no breakdown to show.

    Args:
        heading: What the first column is called, which is what the trio adds up to.
        parts: Each term's label and its seconds, in the order a reader should meet them.

    Returns:
        The table, or ``None`` when the terms add up to nothing -- an estimate that
        carries no breakdown at all, which is not the same as one whose terms are zero.
    """
    total = sum(seconds for _label, seconds in parts)
    if total <= 0:
        return None
    table = Table(show_header=True)
    _add_columns(
        table,
        [
            {"header": heading},
            {
                "header": pgettext("column heading", "Seconds"),
                "justify": "right",
                "no_wrap": True,
            },
            {"header": pgettext("column heading", "Share"), "justify": "right", "no_wrap": True},
        ],
    )
    digits = _seconds_digits(total)
    for label, seconds in parts:
        _add_row(
            table, _cell(label), _cell(_number(seconds, digits)), _cell(_percent(seconds / total))
        )
    return table


def _seconds_digits(total: float) -> int:
    """Decimals enough that the smallest term of a trio is still a number.

    Four for anything a person waits on, which is every generation token and nearly every
    prompt token. A small dense model held on the card reads a prompt token in forty-six
    microseconds, and four decimals would print that as ``0.0000`` three times over: a
    breakdown that adds up to zero is worse than no breakdown, because it looks like one.
    """
    digits = 4
    while digits < 9 and total < 10 ** -(digits - 2):
        digits += 1
    return digits


def _speed_notes(speed: SpeedEstimate) -> list[RenderableType]:
    """What the estimator applied and to which bytes, one line each."""
    lines: list[RenderableType] = []
    if speed.measured_on is not None:
        note = _("Measured on %(date)s.") % {"date": isolate(speed.measured_on.isoformat())}
        lines.append(Text(for_display(note), style="dim"))
    lines.extend(Text(for_display(n), style="dim") for n in speed.notes)
    return lines


def render_measurements(measurements: Sequence[Measured]) -> Group | None:
    """The runs somebody has recorded for this model, for comparing against the estimate.

    These are not this machine's benchmarks and are never allowed to become the estimate:
    section 10.3's ``measured`` means a run taken here, and phase 3 is what will store one.
    What they are is the only check a reader has today on whether the formula is telling
    the truth, so they are printed beside it with the flags, the context and the date that
    make them checkable.
    """
    if not measurements:
        return None
    table = Table(title=for_display(_("Recorded elsewhere")))
    _add_columns(
        table,
        [
            {"header": pgettext("column heading", "Run")},
            {"header": pgettext("column heading", "Gen/s"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "Prompt/s"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "Context"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "Date"), "no_wrap": True},
        ],
    )
    for run in measurements:
        _add_row(
            table,
            _cell(isolate(run.profile)),
            _cell(_number(run.gen_tps) if run.gen_tps else format_bytes(None)),
            _cell(_number(run.pp_tps, 0) if run.pp_tps else format_bytes(None)),
            _cell(_context(run.context) if run.context else format_bytes(None)),
            _cell(isolate(run.date.isoformat()) if run.date else format_bytes(None)),
        )
    below: list[RenderableType] = [
        _cell(
            _(
                "These are the curator's own runs on their machine, not benchmarks of this "
                "one, so nothing above is labelled measured. Compare them with the estimate; "
                "`llamafit bench` will measure this machine in phase 3."
            )
        )
    ]
    for run in measurements:
        if run.flags:
            below.append(
                Text(
                    for_display(
                        _("%(profile)s: %(flags)s")
                        % {"profile": isolate(run.profile), "flags": isolate(run.flags)}
                    ),
                    style="dim",
                )
            )
    return Group(table, *below)


def _quality_sentence(quality: QualityBreakdown, use_case: str) -> str:
    """The three parts of a quality figure, as one sentence naming each.

    Two shapes rather than a fragment bolted on, because the bonus is either there or it
    is not and a translator can move a word across a join they can see.
    """
    if quality.alignment_bonus:
        return _(
            "%(baseline)s from the curator, less %(penalty)s for this quantisation, plus "
            "%(bonus)s because %(use_case)s is the job it was built for."
        ) % {
            "baseline": _number(quality.baseline, 0),
            "penalty": _number(quality.quant_penalty),
            "bonus": _number(quality.alignment_bonus, 0),
            "use_case": use_case,
        }
    return _(
        "%(baseline)s from the curator, less %(penalty)s for this quantisation; it lists "
        "%(use_case)s but was not built for it."
    ) % {
        "baseline": _number(quality.baseline, 0),
        "penalty": _number(quality.quant_penalty),
        "use_case": use_case,
    }


def _speed_sentence(speed: SpeedEstimate, use_case: str) -> str:
    """What the speed score was measured against, and what the prompt cost it.

    Two shapes, because the prompt-processing modifier of section 11.4 either applied or
    it did not, and a reader who is told a score of 80 out of a model comfortably past its
    target deserves to be told that twenty points went on how slowly it reads.
    """
    penalty = prompt_penalty(speed.pp_tps, use_case)
    if penalty:
        return _(
            "%(gen)s tokens per second against the %(target)s this use case asks for, less "
            "%(penalty)s because it reads a prompt at %(prompt)s."
        ) % {
            "gen": _number(speed.gen_tps),
            "target": _number(target_tps(use_case), 0),
            "penalty": _number(penalty, 0),
            "prompt": _number(speed.pp_tps, 0),
        }
    return _("%(gen)s tokens per second against the %(target)s this use case asks for.") % {
        "gen": _number(speed.gen_tps),
        "target": _number(target_tps(use_case), 0),
    }


def _fit_sentence(budget: Budget) -> str:
    """The two questions the fit score asks, and this candidate's answer to each.

    Section 11.3 scores the worse of them, so a reader shown only one number would have no
    way to tell which of the two took the points off. Both are named: how much of the
    machine's memory the model itself holds, and how full the tightest pool ended up.
    """
    return _(
        "Its weights hold %(weights)s of this machine's %(memory)s and the tightest pool "
        "is %(share)s full. The score wants at least %(low)s of the memory taken and no "
        "more than %(high)s of any one pool."
    ) % {
        "weights": isolate(format_bytes(resident_model_bytes(budget))),
        "memory": isolate(format_bytes(budget.vram_available + budget.ram_available)),
        "share": _percent(worst_pool_utilisation(budget)),
        "low": _percent(IDEAL_LOW),
        "high": _percent(IDEAL_HIGH),
    }


def _context_sentence(max_context_fit: int, requested: int) -> str:
    """How much of the context the request wanted this candidate actually holds."""
    return _("Holds %(fit)s tokens of the %(requested)s this request is scored against.") % {
        "fit": isolate(format_grouped(max_context_fit)),
        "requested": isolate(format_grouped(requested)),
    }


def render_score(candidate: Candidate, *, use_case: str, requested_context: int) -> Group | None:
    """The composite score expanded into its four parts, their weights and what each added.

    Args:
        candidate: The scored candidate, with the placement and speed the parts came from.
        use_case: What the request asked for, which set the weights and the speed target.
        requested_context: The context the context score was measured against.

    Returns:
        The table and one sentence per part, or ``None`` for a candidate with no score.

    Section 12.3's standing promise, and the one table in this program that exists purely
    so that a ranking is never asked to be taken on faith. The ``Adds`` column is the
    part's score times its weight, which is what actually moved the total, because a part
    scoring 100 under a weight of 0.05 has told the reader almost nothing. And each part
    gets a sentence saying what it was measured against: a bare 46 for fit is a number on
    faith however carefully the weight beside it is printed.
    """
    score = candidate.score
    if score is None:
        return None
    table = Table(
        title=for_display(
            _("Score %(total)s") % {"total": _number(score.total)},
        )
    )
    _add_columns(
        table,
        [
            {"header": pgettext("column heading", "Part")},
            {"header": pgettext("column heading", "Score"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "Weight"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "Adds"), "justify": "right", "no_wrap": True},
        ],
    )
    parts = {
        "quality": (pgettext("score part", "quality"), score.quality),
        "speed": (pgettext("score part", "speed"), score.speed),
        "fit": (pgettext("score part", "fit"), score.fit),
        "context": (pgettext("score part", "context"), score.context),
    }
    for name, (label, value) in parts.items():
        weight = score.weights.get(name, 0.0)
        _add_row(
            table,
            _cell(label),
            _cell(_number(value)),
            _cell(_number(weight, 2)),
            _cell(_number(value * weight)),
        )
    below: list[RenderableType] = []
    if candidate.quality is not None:
        below.append(_cell(_quality_sentence(candidate.quality, use_case)))
    if candidate.speed is not None:
        below.append(_cell(_speed_sentence(candidate.speed, use_case)))
    if candidate.placement is not None:
        below.append(_cell(_fit_sentence(candidate.placement.budget)))
        below.append(
            _cell(_context_sentence(candidate.placement.max_context_fit, requested_context))
        )
    return Group(table, *below)


def render_notes(notes: Sequence[str]) -> Group | None:
    """The placement's own notes: what a reader has to be told that the numbers do not say."""
    if not notes:
        return None
    return Group(*[_cell(note) for note in notes])


def _speed_cell(speed: SpeedEstimate | None) -> str:
    """The generation figure for a board row, or the mark for a row that has none."""
    return format_bytes(None) if speed is None else _number(speed.gen_tps)


def _board_columns(
    rows: Sequence[BoardRow], console_width: int, *, show_confidence: bool
) -> tuple[int, int, list[str]]:
    """Decide which of the board's optional columns fit, in a fixed priority.

    The rank, the model, the quantisation and the score are never dropped: without them a
    row cannot be told apart from another or acted on. Everything else is admitted only
    while its whole content fits, in the order below, and the first column that does not
    fit ends the list rather than being shrunk — a figure missing a digit is worse than a
    column that is honestly absent, and a wider terminal shows every one of them.

    The order is what a reader decides on. How fast it runs, then how well it fits, then
    how it runs at all, then how much context it holds, then the quality behind the score,
    then the memory, the prompt speed and the download.
    """
    model_width = min(
        _ID_COLUMN_MAX_WIDTH,
        max(
            [cell_len(pgettext("column heading", "Model"))]
            + [cell_len(row.model_id) for row in rows]
        ),
    )
    quant_width = max(
        [cell_len(pgettext("column heading", "Quant"))] + [cell_len(row.quant) for row in rows]
    )
    fixed = (
        _TABLE_OVERHEAD
        + (2 + _COLUMN_OVERHEAD)
        + (model_width + _COLUMN_OVERHEAD)
        + (quant_width + _COLUMN_OVERHEAD)
        + (5 + _COLUMN_OVERHEAD)
    )
    optional: list[tuple[str, int]] = [("gen", 5)]
    if show_confidence:
        optional.append(("confidence", cell_len(confidence_label("calibrated"))))
    optional += [
        ("verdict", cell_len(verdict_label("does-not-fit"))),
        ("mode", cell_len(mode_label("moe-offload"))),
        ("context", 7),
        ("quality", 4),
        ("vram", 9),
        ("prompt", 6),
        ("size", 9),
        ("ram", 9),
    ]
    remaining = console_width - fixed
    included: list[str] = []
    for name, width in optional:
        cost = max(width, 4) + _COLUMN_OVERHEAD
        if cost > remaining:
            break
        included.append(name)
        remaining -= cost
    return model_width, quant_width, included


def render_board(board: Board, *, console_width: int = 80) -> Group:
    """The ranked board: the answer a person came for, with its provenance attached.

    Args:
        board: The ranked candidates and the request they answer.
        console_width: The console's width; 80, the narrowest this table is designed for,
            when the caller does not know.

    Returns:
        The table, and the caption saying what the speed column is and what it is not.
    """
    model_width, quant_width, included = _board_columns(
        board.rows, console_width, show_confidence=_mixed_confidence(board.rows)
    )
    table = Table(title=for_display(_("Recommended")))
    columns: list[Mapping[str, Any]] = [
        {"header": pgettext("column heading", "#"), "justify": "right", "no_wrap": True},
        {
            "header": pgettext("column heading", "Model"),
            "style": "bold",
            "max_width": model_width,
            "overflow": "fold",
        },
        {"header": pgettext("column heading", "Quant"), "max_width": quant_width},
        {"header": pgettext("column heading", "Score"), "justify": "right", "no_wrap": True},
    ]
    headings = {
        "gen": pgettext("column heading", "Gen/s"),
        "confidence": pgettext("column heading", "How"),
        "verdict": pgettext("column heading", "Fit"),
        "mode": pgettext("column heading", "Runs"),
        "context": pgettext("column heading", "Ctx"),
        "quality": pgettext("column heading", "Qual"),
        "vram": pgettext("column heading", "Card"),
        "prompt": pgettext("column heading", "PP/s"),
        "size": pgettext("column heading", "Size"),
        "ram": pgettext("column heading", "RAM"),
    }
    numeric = {"gen", "context", "quality", "vram", "prompt", "size", "ram"}
    for name in included:
        columns.append(
            {
                "header": headings[name],
                "justify": "right" if name in numeric else "left",
                "no_wrap": True,
            }
        )
    _add_columns(table, columns)

    for row in board.rows:
        _add_row(table, *_board_cells(row, included))
    banner = render_simulation(board.simulation)
    parts: list[RenderableType] = [] if banner is None else [banner]
    return Group(*parts, table, *_board_captions(board))


def _mixed_confidence(rows: Sequence[BoardRow]) -> bool:
    """Whether the rows disagree about how their speeds were arrived at.

    When every row carries the same label there is nothing a column per row can say that
    one sentence under the table does not say better, and the space buys a column that
    does vary. When they differ, the label has to be on the row it belongs to.
    """
    labels = {row.candidate.speed.confidence for row in rows if row.candidate.speed is not None}
    return len(labels) > 1


def _board_cells(row: BoardRow, included: Sequence[str]) -> list[Cell]:
    """One board row's cells, in the order :func:`render_board` put its columns."""
    candidate = row.candidate
    score = candidate.score
    placement = candidate.placement
    speed = candidate.speed
    budget = placement.budget if placement is not None else None
    cells: list[Cell] = [
        _cell(isolate(str(row.rank))),
        _cell(isolate(row.model_id)),
        _cell(isolate(row.quant)),
        _cell(_number(score.total) if score is not None else format_bytes(None)),
    ]
    values: dict[str, Cell] = {
        "gen": _cell(_speed_cell(speed)),
        "confidence": _cell(
            confidence_label(speed.confidence) if speed is not None else format_bytes(None)
        ),
        "verdict": Text(
            for_display(verdict_label(budget.verdict)) if budget is not None else "",
            style=verdict_style(budget.verdict) if budget is not None else "",
        ),
        "mode": _cell(mode_label(placement.mode) if placement is not None else ""),
        "context": _cell(_context(placement.max_context_fit) if placement is not None else ""),
        "quality": _cell(_number(score.quality, 0) if score is not None else ""),
        "vram": _cell(_size(budget.vram_required) if budget is not None else ""),
        "prompt": _cell(_number(speed.pp_tps, 0) if speed is not None else ""),
        "size": _cell(_size(row.download_bytes)),
        "ram": _cell(_size(budget.ram_required) if budget is not None else ""),
    }
    cells += [values[name] for name in included]
    return cells


def _min_tps_caption(min_tps: float | None) -> str | None:
    """Say that this request moved section 11.4's speed floor, when it did.

    Silence is the default and means the floor is the one the specification derived from
    reading rates. A request that relaxed it has to say so on the board itself: the whole
    point of the exclusion is that a board must not quietly offer something nobody can
    wait for, and a board that quietly stopped applying the rule would fail in exactly the
    same way, only harder to notice.

    Args:
        min_tps: What the request asked for, or ``None`` when it asked for nothing.

    Returns:
        The sentence to print under the table, or ``None`` when there is nothing to say.
    """
    if min_tps is None:
        return None
    if min_tps <= 0.0:
        return _(
            "--min-tps 0: nobody is waiting on these tokens, so nothing was excluded for "
            "generating more slowly than a person reads."
        )
    return _(
        "--min-tps %(tps)s: this board wanted at least that many tokens per second, and "
        "anything slower is excluded below rather than ranked here."
    ) % {"tps": _tps(min_tps)}


def _board_captions(board: Board) -> list[RenderableType]:
    """What the board's numbers are, said once under the table rather than per row."""
    labels = {row.candidate.speed.confidence for row in board.rows if row.candidate.speed}
    if board.planned_context == board.requested_context:
        first = _(
            "Speeds are for %(working)s tokens of context so every row compares like with "
            "like; the context column is the largest each one holds. Sized and scored for "
            "%(use_case)s at %(requested)s tokens."
        ) % {
            "working": _context(board.working_context),
            "use_case": isolate(board.needs.use_case),
            "requested": _context(board.requested_context),
        }
    else:
        first = _(
            "Speeds are for %(working)s tokens of context so every row compares like with "
            "like; the context column is the largest each one holds. Sized for %(planned)s "
            "tokens and scored for %(use_case)s against %(requested)s."
        ) % {
            "working": _context(board.working_context),
            "planned": _context(board.planned_context),
            "use_case": isolate(board.needs.use_case),
            "requested": _context(board.requested_context),
        }
    captions: list[RenderableType] = [_cell(first)]
    floor = _min_tps_caption(board.needs.min_tps)
    if floor is not None:
        captions.append(_cell(floor))
    if len(labels) == 1:
        captions.append(_cell(confidence_sentence(next(iter(labels)))))
    captions.append(
        _cell(
            _("Weights: %(weights)s.")
            % {
                "weights": ", ".join(
                    f"{part} {isolate(localise_number(f'{weight:.2f}'))}"
                    for part, weight in board.weights.items()
                )
            }
        )
    )
    return captions


def render_excluded(rows: Sequence[BoardRow]) -> Group | None:
    """The candidates that did not qualify, each with the reason and what to change.

    A shorter list tells a reader nothing. "Llama 3.1 8B was excluded because its entry
    lists general, chat and reasoning, not coding" tells them why a model they expected is
    missing and what would bring it back — the request, or a one-line catalog change.
    """
    if not rows:
        return None
    table = Table(title=for_display(_("Not ranked")))
    _add_columns(
        table,
        [
            {"header": pgettext("column heading", "Model"), "style": "bold"},
            {"header": pgettext("column heading", "Quant"), "no_wrap": True},
            {"header": pgettext("column heading", "Why not")},
        ],
    )
    for row in rows:
        _add_row(
            table,
            _cell(isolate(row.model_id)),
            _cell(isolate(row.quant)),
            _cell(row.candidate.excluded_because or ""),
        )
    return Group(table)


def render_explanation(row: BoardRow, board: Board) -> Group:
    """One row expanded into everything that produced it.

    Section 12.3, and the standing promise this command exists to keep: the score with its
    parts and weights, the quality it was built from, the budget line by line with the
    source of each, the context ladder, and where a token's time goes.
    """
    candidate = row.candidate
    pieces: list[RenderableType] = [
        Text(""),
        Text(
            for_display(
                _("%(rank)s. %(name)s %(quant)s")
                % {
                    "rank": isolate(str(row.rank)),
                    "name": isolate(row.name),
                    "quant": isolate(row.quant),
                }
            ),
            style="bold",
        ),
    ]
    scored = render_score(
        candidate, use_case=board.needs.use_case, requested_context=board.requested_context
    )
    if scored is not None:
        pieces.append(scored)
    if candidate.speed is not None and candidate.placement is not None:
        pieces.append(render_speed(candidate.speed, context=board.working_context))
    if candidate.placement is not None:
        pieces.append(render_budget(candidate.placement.budget))
        notes = render_notes(candidate.placement.notes)
        if notes is not None:
            pieces.append(notes)
        pieces.append(render_tiers(candidate.placement))
    if candidate.excluded_because:
        pieces.append(_cell(candidate.excluded_because))
    return Group(*pieces)


def render_fit(board: FitBoard, *, console_width: int = 80) -> Group:
    """Every model ranked by how well it uses this machine, and nothing else.

    ``fit`` asks a narrower question than ``recommend``, so the table is narrower: no
    score, no weights, no use case. What it does carry is the verdict and both pools, since
    "how well does it fit" is the only question being asked and those are the answer.
    """
    model_width = min(
        _ID_COLUMN_MAX_WIDTH,
        max(
            [cell_len(pgettext("column heading", "Model"))]
            + [cell_len(row.model_id) for row in board.rows]
        ),
    )
    # The heading is the one place the simulated banner would have been contradicted in
    # its own words: "Fit on this machine" one line under "they are not this machine".
    title = _("Fit on the simulated machine") if board.simulation else _("Fit on this machine")
    table = Table(title=for_display(title))
    columns: list[Mapping[str, Any]] = [
        {"header": pgettext("column heading", "#"), "justify": "right", "no_wrap": True},
        {
            "header": pgettext("column heading", "Model"),
            "style": "bold",
            "max_width": model_width,
            "overflow": "fold",
        },
        {"header": pgettext("column heading", "Quant"), "no_wrap": True},
        {"header": pgettext("column heading", "Fit"), "no_wrap": True},
    ]
    optional = console_width >= 80
    if optional:
        columns += [
            {"header": pgettext("column heading", "Runs"), "no_wrap": True},
            {"header": pgettext("column heading", "Ctx"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "Card"), "justify": "right", "no_wrap": True},
            {"header": pgettext("column heading", "RAM"), "justify": "right", "no_wrap": True},
        ]
    _add_columns(table, columns)
    for row in board.rows:
        placement = row.placement
        cells: list[Cell] = [
            _cell(isolate(str(row.rank))),
            _cell(isolate(row.model_id)),
            _cell(isolate(row.quant)),
            Text(
                for_display(verdict_label(placement.budget.verdict)) if placement else "",
                style=verdict_style(placement.budget.verdict) if placement else "",
            ),
        ]
        if optional and placement is not None:
            cells += [
                _cell(mode_label(placement.mode)),
                _cell(_context(placement.max_context_fit)),
                _cell(_size(placement.budget.vram_required)),
                _cell(_size(placement.budget.ram_required)),
            ]
        elif optional:
            cells += ["", "", "", ""]
        _add_row(table, *cells)
    caption = _cell(
        _(
            "Sized for %(context)s tokens. The context column is the largest each one holds "
            "in the mode shown."
        )
        % {"context": _context(board.planned_context)}
    )
    banner = render_simulation(board.simulation)
    parts: list[RenderableType] = [] if banner is None else [banner]
    return Group(*parts, table, caption)


def render_fit_excluded(rows: Sequence[FitRow]) -> Group | None:
    """The models with no placement at all, each carrying the reason."""
    if not rows:
        return None
    table = Table(title=for_display(_("Not placed")))
    _add_columns(
        table,
        [
            {"header": pgettext("column heading", "Model"), "style": "bold"},
            {"header": pgettext("column heading", "Quant"), "no_wrap": True},
            {"header": pgettext("column heading", "Why not")},
        ],
    )
    for row in rows:
        _add_row(
            table,
            _cell(isolate(row.model_id)),
            _cell(isolate(row.quant)),
            _cell(row.excluded_because or ""),
        )
    return Group(table)


def _placement_sentence(placement: Placement) -> str:
    """The chosen placement in one line: the mode, the context and the cache."""
    return _(
        "%(mode)s at %(context)s tokens, micro-batch %(ub)s, %(kv)s cache, %(threads)s threads."
    ) % {
        "mode": mode_label(placement.mode),
        "context": isolate(format_grouped(placement.context)),
        "ub": isolate(format_grouped(placement.micro_batch)),
        "kv": isolate(placement.kv_type),
        "threads": isolate(format_grouped(placement.threads)),
    }


def _target_sentence(target: TargetCheck) -> str:
    """The ``--target-tps`` answer, as one whole sentence per outcome.

    Three shapes, written out rather than assembled, because they say three different
    things: the target is already met, a shorter context would meet it, or nothing this
    machine can do with this model and quantisation will.
    """
    if target.reached:
        return _("This reaches the %(target)s tokens per second asked for.") % {
            "target": _number(target.target_tps, 0)
        }
    if target.best_context is not None:
        return _(
            "It does not reach %(target)s tokens per second here, but %(context)s tokens of "
            "context would: pass --context %(tokens)s."
        ) % {
            "target": _number(target.target_tps, 0),
            "context": _context(target.best_context),
            "tokens": isolate(str(target.best_context)),
        }
    return _(
        "Nothing this model and quantisation can do on this machine reaches %(target)s tokens "
        "per second; the fastest configuration tried is %(best)s. A smaller model or a "
        "smaller quantisation is the change that would."
    ) % {"target": _number(target.target_tps, 0), "best": _number(target.best_tps)}


def render_plan(report: PlanReport) -> Group:
    """One model's plan: where it goes, what it costs, how fast it runs and what to paste.

    The command line comes last and on its own line, because it is the one thing here a
    reader will copy, and it must not be wrapped into a table cell where a shell would take
    a broken line.
    """
    placement = report.placement
    banner = render_simulation(report.simulation)
    pieces: list[RenderableType] = [] if banner is None else [banner]
    pieces += [
        Text(
            for_display(
                _("%(name)s %(quant)s")
                % {"name": isolate(report.name), "quant": isolate(report.quant)}
            ),
            style="bold",
        ),
        _cell(_placement_sentence(placement)),
    ]
    notes = render_notes(placement.notes)
    if notes is not None:
        pieces.append(notes)
    pieces.append(Text(""))
    pieces.append(render_budget(placement.budget, title=_("Memory budget")))
    if report.requested_budget is not None:
        pieces.append(Text(""))
        pieces.append(
            render_budget(
                report.requested_budget,
                title=_("The %(context)s tokens asked for would have cost")
                % {"context": _context(report.requested_context)},
                notes=False,
            )
        )
    pieces.append(Text(""))
    pieces.append(render_tiers(placement))
    if report.speed is not None:
        pieces.append(Text(""))
        pieces.append(render_speed(report.speed, context=placement.context))
    if report.target is not None:
        pieces.append(_cell(_target_sentence(report.target)))
    recorded = render_measurements(report.measurements)
    if recorded is not None:
        pieces.append(Text(""))
        pieces.append(recorded)
    pieces.append(Text(""))
    pieces.extend(_file_lines(report))
    pieces.append(Text(for_display(_("Command line"))))
    pieces.append(Text(" ".join(report.command)))
    return Group(*pieces)


def _file_lines(report: PlanReport) -> list[RenderableType]:
    """Whether the file this command line names is on the machine, said before the command.

    A plan for a model nobody has downloaded is still a plan worth reading, and the path in
    it is where ``llamafit install model`` will put the file. Saying which of the two a
    reader is looking at is what stops somebody pasting a command line believing a download
    has happened.
    """
    if report.model_present:
        line = _("Model file: %(path)s.") % {"path": isolate(report.model_path)}
    else:
        line = _(
            "Model file: %(path)s, which is not on this machine yet; that is where a download "
            "would put it. %(size)s to fetch."
        ) % {"path": isolate(report.model_path), "size": _size(report.download_bytes)}
    return [_cell(line)]


def count_sentence(shown: int, total: int) -> str:
    """How many rows are on screen out of how many there are, as a counted noun."""
    return ngettext(
        "%(shown)d of %(total)d candidate shown.",
        "%(shown)d of %(total)d candidates shown.",
        total,
    ) % {"shown": shown, "total": total}
