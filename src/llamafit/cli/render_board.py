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

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar, cast, get_args

from rich.cells import cell_len
from rich.console import Group, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from llamafit.cli.render import bandwidth_sentence, simulated_answer
from llamafit.i18n import (
    _,
    for_display,
    isolate,
    mirror_justify,
    ngettext,
    pgettext,
    pgettext_literal,
    reading_order,
)
from llamafit.models.catalog import Capability, Catalog, Measured
from llamafit.models.host import MachineFacts
from llamafit.models.plan import (
    Budget,
    Candidate,
    Confidence,
    ContextTier,
    Needs,
    Placement,
    QualityBreakdown,
    RunMode,
    SpeedEstimate,
    Verdict,
)
from llamafit.scoring.fit_score import (
    IDEAL_HIGH,
    IDEAL_LOW,
    resident_model_bytes,
    worst_pool_utilisation,
)
from llamafit.scoring.speed_score import floor_tps, prompt_penalty, target_tps
from llamafit.services.plan import PlanReport, TargetCheck
from llamafit.services.recommend import Board, BoardRow, FitBoard, FitRow, quant_entries
from llamafit.units import format_bytes, format_grouped, localise_number, parse_size

Cell = str | Text
"""What one cell of a table may be: markup-free text, or a plain string of labels."""


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


def _add_row(table: Table, *cells: Cell, style: str | None = None) -> None:
    """Add one row, its cells in the same order :func:`_add_columns` put the columns."""
    prepared: list[Cell] = [c if isinstance(c, Text) else for_display(c) for c in cells]
    table.add_row(*reading_order(prepared), style=style)


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

    The sentence opens with the word in this table and not with section 8.3's, for every
    verdict. ``too-tight`` reads *Pages* in both places and ``does-not-fit`` reads *No
    room* in both; ``comfortable`` once read *roomy* in the column and *Comfortable* in
    the sentence, which asked a reader to work out that two words were one verdict and
    asked thirty-seven translators to keep two unrelated words in step.
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
        "comfortable": _("Roomy: there is room for a longer context or a second model."),
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
            {"header": pgettext("column heading", "Tok/s"), "justify": "right", "no_wrap": True},
            {
                "header": pgettext("column heading", "Prompt tok/s"),
                "justify": "right",
                "no_wrap": True,
            },
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


# --- the vocabulary the three interfaces share ----------------------------------------
#
# Column keys, sort keys and filter terms, and the words for each. ``board_cmd`` reads the
# tuples for its flags, ``llamafit.tui.screens.board`` for its keys, ``llamafit.web.api``
# for ``sort=`` and ``llamafit.web.strings`` for the page's headings, so a word added here
# reaches all three interfaces and a word spelled differently in one of them cannot exist.

Column = Literal[
    "rank",
    "model",
    "quant",
    "size",
    "have",
    "score",
    "quality",
    "gen",
    "prompt",
    "confidence",
    "mode",
    "vram",
    "ram",
    "verdict",
    "context",
]
"""One column of a board, named as ``--json`` names the figure it holds."""

BOARD_ORDER: tuple[Column, ...] = (
    "rank",
    "model",
    "quant",
    "size",
    "have",
    "score",
    "quality",
    "gen",
    "prompt",
    "confidence",
    "mode",
    "vram",
    "ram",
    "verdict",
    "context",
)
"""Every column the board can draw, in the order it draws them.

The web page's order, which is the reference, with ``have`` beside ``size`` because the
page keeps that fact inside the row. Which columns a width admits is decided by
:data:`BOARD_PRIORITY`; where an admitted column goes is decided here, so a column keeps
its place whatever the width. Widening a terminal by twenty cells puts ``Qual`` between
``Score`` and ``Tok/s``, where the page has it, rather than on the far right, and a reader
who knows where a figure lives at one width knows where it lives at every width.
"""

BOARD_REQUIRED: tuple[Column, ...] = ("rank", "model", "quant", "score")
"""The four columns a row cannot be told apart or acted on without; never dropped."""

BOARD_PRIORITY: tuple[Column, ...] = (
    "gen",
    "verdict",
    "mode",
    "context",
    "quality",
    "vram",
    "size",
    "have",
    "confidence",
    "prompt",
    "ram",
)
"""What a narrow terminal admits next, one whole column at a time.

How fast it runs, then whether it runs, then how, then how much context it holds, the
quality behind the score, what it takes on the card, what it costs to fetch, whether the
file is already here, the confidence word, the prompt speed and the system memory. The
download moved up from where the command line once had it: a download figure decides
more choices than a prompt speed does, and the page's ``≤ GiB`` box is on it.

``confidence`` sits here for a board whose rows all carry the same label, which the
caption under the table then says once. When the rows disagree no sentence under the
table is true of all of them, so the word is admitted together with ``gen`` or not at
all: a figure without the word that says what kind of figure it is would be the thing
the rule at the top of this file forbids.
"""

FIT_ORDER: tuple[Column, ...] = (
    "rank",
    "model",
    "quant",
    "size",
    "have",
    "mode",
    "vram",
    "ram",
    "verdict",
    "context",
)
"""The columns ``llamafit fit`` can draw, in :data:`BOARD_ORDER`'s order.

No score, no speed, no quality: ``fit`` asks the memory question on its own.
"""

FIT_REQUIRED: tuple[Column, ...] = ("rank", "model", "quant", "verdict")
"""What a fit row cannot be read without: its identity and the one answer it carries."""

FIT_PRIORITY: tuple[Column, ...] = ("mode", "context", "vram", "ram", "size", "have")
"""What a fit listing admits next, one whole column at a time."""

ID_COLUMN_MAX_WIDTH = 28
"""How wide the model column may grow, in terminal cells, before a name folds.

Six of the sixty-two bundled ids are longer than this and the longest is thirty-four; a
column wide enough for that one would spend nearly half an eighty-cell terminal on it.
So this is where a name *folds* onto a second line, which is the one thing this column
may do. A name cut at a width is not a shorter name but a different and wrong one:
``nemotron-3.5-lightning-30b-a3b`` arriving as ``nemotron-3.5-lightning-3`` is a model
nobody can look up or type. ``llamafit.cli.render`` holds the same number under its own
name for a different table.
"""

SortKey = Literal[
    "score",
    "speed",
    "quality",
    "context",
    "size",
    "prompt",
    "card",
    "ram",
    "fit",
    "model",
    "quant",
]
"""What a board can be ordered by: ``--sort`` on the command line, ``s`` on the dashboard,
``sort=`` on the API. One word each, and the same word in all three places."""

SORT_KEYS: tuple[SortKey, ...] = (
    "score",
    "speed",
    "quality",
    "context",
    "size",
    "prompt",
    "card",
    "ram",
    "fit",
    "model",
    "quant",
)
"""Every sort key, ``score`` -- the board's own order -- first, then in column order."""

FIT_SORT_KEYS: tuple[SortKey, ...] = (
    "score",
    "context",
    "size",
    "card",
    "ram",
    "fit",
    "model",
    "quant",
)
"""The sort keys a fit row has a value for. ``score`` is the fit score, the listing's own order."""

_FIGURE_SORTS: frozenset[str] = frozenset(
    {"score", "speed", "quality", "context", "size", "prompt", "card", "ram", "fit"}
)
"""The keys that order a figure, which a reader wants largest -- or best -- first."""

_SORT_DIRECTIONS: dict[str, bool] = {"asc": False, "desc": True}
"""The two words ``--sort KEY:asc`` and ``KEY:desc`` accept, and what each means."""


def parse_sort(text: str, *, allowed: Sequence[str] = SORT_KEYS) -> tuple[SortKey, bool | None]:
    """Read ``KEY``, ``KEY:asc`` or ``KEY:desc`` into a key and a direction.

    Args:
        text: What the flag or the query parameter carried.
        allowed: The keys this listing has a value for; ``fit`` has fewer than the board.

    Returns:
        The key, and ``True`` for largest first, ``False`` for smallest first, or ``None``
        when nothing was said and the key's own default applies.

    Raises:
        ValueError: The key is not one of ``allowed``, or the suffix is not one of the
            two words. The message is the offending text; the caller says which flag or
            parameter it came from and lists what would have been accepted.
    """
    key, _colon, direction = text.partition(":")
    if key not in allowed or (direction and direction not in _SORT_DIRECTIONS):
        raise ValueError(text)
    return cast("SortKey", key), _SORT_DIRECTIONS.get(direction)


def default_descending(key: SortKey) -> bool:
    """Which way a key runs when nobody said: largest first for a figure, A to Z for a word.

    The page's rule (``app.js``), kept here so a heading clicked on the page and a flag
    typed on the command line put the same row first. ``fit`` counts as a figure whose
    largest value is ``comfortable``, so the best verdict comes first.
    """
    return key in _FIGURE_SORTS


def sort_label(key: SortKey) -> str:
    """What the rows are ordered by, in a word the state line can carry."""
    labels: dict[SortKey, str] = {
        "score": pgettext("board sort", "score"),
        "speed": pgettext("board sort", "speed"),
        "quality": pgettext("board sort", "quality"),
        "context": pgettext("board sort", "context"),
        "size": pgettext("board sort", "download size"),
        "prompt": pgettext("board sort", "prompt speed"),
        "card": pgettext("board sort", "card memory"),
        "ram": pgettext("board sort", "system memory"),
        "fit": pgettext("board sort", "fit"),
        "model": pgettext("board sort", "model id"),
        "quant": pgettext("board sort", "quantisation"),
    }
    return labels[key]


def column_heading(column: Column) -> str:
    """One column's heading, in the reader's language.

    The one table of headings for the command line, the dashboard and, through
    :mod:`llamafit.web.strings`, the page. ``Tok/s`` was once renamed in three places at
    once because each interface spelled it for itself; now there is one place.
    """
    headings: dict[Column, str] = {
        "rank": pgettext("column heading", "#"),
        "model": pgettext("column heading", "Model"),
        "quant": pgettext("column heading", "Quant"),
        "size": pgettext("column heading", "Size"),
        "have": pgettext("column heading", "Have"),
        "score": pgettext("column heading", "Score"),
        "quality": pgettext("column heading", "Qual"),
        "gen": pgettext("column heading", "Tok/s"),
        "prompt": pgettext("column heading", "Prompt tok/s"),
        "confidence": pgettext("column heading", "How"),
        "mode": pgettext("column heading", "Runs"),
        "vram": pgettext("column heading", "Card"),
        "ram": pgettext("column heading", "RAM"),
        "verdict": pgettext("column heading", "Fit"),
        "context": pgettext("column heading", "Ctx"),
    }
    return headings[column]


def parse_columns(text: str, *, allowed: Sequence[Column] = BOARD_ORDER) -> tuple[Column, ...]:
    """Read ``--columns``' comma-separated list into column keys, in the order typed.

    Args:
        text: What was typed, for example ``model,quant,gen,size``.
        allowed: The columns this listing can draw.

    Returns:
        The columns, in the order given.

    Raises:
        ValueError: A name that is not a column of this listing. The message is the name;
            the caller lists what would have been accepted.
    """
    columns: list[Column] = []
    for name in (part.strip() for part in text.split(",")):
        if name not in allowed:
            raise ValueError(name)
        columns.append(cast("Column", name))
    return tuple(columns)


@dataclass(frozen=True)
class ColumnBudget:
    """What the two identity columns whose width varies cost on this catalog.

    Attributes:
        model: The model column's width, capped at :data:`ID_COLUMN_MAX_WIDTH`.
        quant: The quantisation column's width.

    Measured from the catalog, never from the rows a limit left on the board. The
    admission of every optional column depends on these two, and a width that admitted
    nine columns under ``--limit 2`` and seven under the default ten was a table that
    changed shape for a reason no reader could see.
    """

    model: int = ID_COLUMN_MAX_WIDTH
    quant: int = 10


def column_budget(catalog: Catalog | None) -> ColumnBudget:
    """The identity columns' widths for this catalog, or the defaults without one.

    Args:
        catalog: The models the board was built from, or ``None`` when the caller has no
            catalog at hand, in which case the bundled catalog's own figures serve.
    """
    if catalog is None or not catalog.models:
        return ColumnBudget()
    ids = [cell_len(model.id) for model in catalog.models]
    quants = [cell_len(quant.name) for model in catalog.models for quant in quant_entries(model)]
    return ColumnBudget(
        model=min(ID_COLUMN_MAX_WIDTH, max([cell_len(column_heading("model")), *ids])),
        quant=max([cell_len(column_heading("quant")), *quants]),
    )


def column_widths(budget: ColumnBudget | None = None) -> dict[Column, int]:
    """What each column costs in terminal cells before its padding, in this language.

    Measured from the labels a column can hold rather than written down, because a
    translated word is not the width of its English: ``experts in RAM`` is fourteen
    cells and its Portuguese is longer, and a table budgeted on the English admitted a
    column the Portuguese then overflowed. The figures are the widest a figure of that
    kind gets on any board: a score is ``100.0``, a size ``123.4 GiB``.
    """
    words = {
        "have": (
            pgettext("model file on this machine", "yes"),
            pgettext("model file on this machine", "no"),
        ),
        "confidence": tuple(confidence_label(value) for value in get_args(Confidence)),
        "mode": tuple(mode_label(value) for value in get_args(RunMode)),
        "verdict": tuple(verdict_label(value) for value in get_args(Verdict)),
    }
    figures: dict[Column, int] = {
        "rank": 2,
        "size": 9,
        "score": 5,
        "quality": 3,
        "gen": 5,
        "prompt": 5,
        "vram": 9,
        "ram": 9,
        "context": 7,
    }
    budget = budget or ColumnBudget()
    widths: dict[Column, int] = {}
    for column in BOARD_ORDER:
        heading = cell_len(column_heading(column))
        if column == "model":
            widths[column] = max(heading, budget.model)
        elif column == "quant":
            widths[column] = max(heading, budget.quant)
        elif column in words:
            widths[column] = max(heading, *(cell_len(word) for word in words[column]))
        else:
            widths[column] = max(heading, figures[column])
    return widths


_COLUMN_OVERHEAD = 3
"""What one more column costs beyond its content: a border and the padding either side."""

_TABLE_OVERHEAD = 1
"""The table's own leading border, charged once."""


def choose_columns(
    width: int,
    *,
    order: Sequence[Column],
    required: Sequence[Column],
    priority: Sequence[Column],
    widths: Mapping[Column, int],
    mixed_confidence: bool = False,
) -> tuple[Column, ...]:
    """The columns that fit in ``width`` cells, admitted by priority and drawn in order.

    Args:
        width: How many terminal cells the table has.
        order: Every column the listing can draw, in the order it draws them.
        required: The columns never dropped, even when they do not fit.
        priority: The rest, in the order they are admitted.
        widths: What each column costs before its padding.
        mixed_confidence: Whether the rows disagree about how their speeds were arrived
            at, which is what binds ``confidence`` to ``gen``.

    Returns:
        The columns to draw, in ``order``'s order.

    The required columns are never dropped: a table too narrow for them is a table that
    folds its names harder, and a row a reader cannot identify is worse than one that
    takes two lines. Everything else is admitted only while its whole cost fits, and the
    first column that does not fit ends the list rather than being shrunk -- a figure
    missing a digit is worse than a column that is honestly absent, and a wider terminal
    shows every one of them.

    The cost is a Rich table's: a leading border, then a border and a cell of padding on
    either side of every column. The dashboard's table is cheaper by a cell a column and
    budgets with the same figures anyway, so that the two interfaces draw the same
    columns at the same width and the only difference is a cell or two of slack.
    """

    def cost(column: Column) -> int:
        return widths[column] + _COLUMN_OVERHEAD

    remaining = width - _TABLE_OVERHEAD - sum(cost(column) for column in required)
    admitted: set[Column] = set(required)
    for column in priority:
        if column == "confidence" and mixed_confidence:
            continue
        group: tuple[Column, ...] = (
            ("gen", "confidence") if column == "gen" and mixed_confidence else (column,)
        )
        needed = sum(cost(member) for member in group)
        if needed > remaining:
            break
        admitted.update(group)
        remaining -= needed
    return tuple(column for column in order if column in admitted)


def board_columns(
    width: int, *, mixed_confidence: bool, budget: ColumnBudget | None = None
) -> tuple[Column, ...]:
    """The board's columns at this width: the one chooser every interface draws with."""
    return choose_columns(
        width,
        order=BOARD_ORDER,
        required=BOARD_REQUIRED,
        priority=BOARD_PRIORITY,
        widths=column_widths(budget),
        mixed_confidence=mixed_confidence,
    )


def fit_columns(width: int, *, budget: ColumnBudget | None = None) -> tuple[Column, ...]:
    """The fit listing's columns at this width, chosen the way the board's are."""
    return choose_columns(
        width,
        order=FIT_ORDER,
        required=FIT_REQUIRED,
        priority=FIT_PRIORITY,
        widths=column_widths(budget),
    )


@dataclass(frozen=True)
class RowFacts:
    """What one row of either listing carries, read once so every cell reads the same.

    A :class:`~llamafit.services.recommend.BoardRow` holds its figures inside a
    :class:`~llamafit.models.plan.Candidate` and a
    :class:`~llamafit.services.recommend.FitRow` holds them on itself. Everything that
    draws, sorts or filters a row reads this instead, so the two shapes are one shape
    from here on and nothing below has to know which listing a row came from.

    Attributes:
        rank: Its place, or ``None`` when it was not ranked.
        model_id: The catalog id.
        name: The display name.
        quant: The quantisation.
        download_bytes: What fetching it would cost, when known.
        local_path: Where it already is on disk, when it is.
        placement: Where its bytes would go, when a placement was found.
        speed: How fast it would run there, when one was estimated.
        score: The board's composite, or the fit score on a fit row, or ``None``.
        quality: The quality behind the score, when one was worked out.
        excluded_because: Why it was not ranked, as a sentence, when it was not.
        excluded_tag: The same in a word or two, or ``None`` when the service gave none.
    """

    rank: int | None
    model_id: str
    name: str
    quant: str
    download_bytes: int | None
    local_path: str | None
    placement: Placement | None
    speed: SpeedEstimate | None
    score: float | None
    quality: float | None
    excluded_because: str | None
    excluded_tag: str | None

    @property
    def ranked(self) -> bool:
        """Whether this row is on the board proper rather than under it."""
        return self.rank is not None


def facts_of(row: BoardRow | FitRow) -> RowFacts:
    """One row's facts, whichever listing it belongs to."""
    if isinstance(row, BoardRow):
        candidate = row.candidate
        return RowFacts(
            rank=row.rank,
            model_id=row.model_id,
            name=row.name,
            quant=row.quant,
            download_bytes=row.download_bytes,
            local_path=row.local_path,
            placement=candidate.placement,
            speed=candidate.speed,
            score=candidate.score.total if candidate.score is not None else None,
            quality=candidate.quality.quality if candidate.quality is not None else None,
            excluded_because=candidate.excluded_because,
            excluded_tag=candidate.excluded_tag,
        )
    return RowFacts(
        rank=row.rank,
        model_id=row.model_id,
        name=row.name,
        quant=row.quant,
        download_bytes=row.download_bytes,
        local_path=row.local_path,
        placement=row.placement,
        speed=None,
        score=row.fit,
        quality=None,
        excluded_because=row.excluded_because,
        excluded_tag=_fit_tag(row),
    )


def _fit_tag(row: FitRow) -> str | None:
    """The one-cell reason a fit row was not placed, read off the sentence it carries.

    ``fit`` has two reasons and no tag field. One reason is a fixed sentence the service
    writes for a model no placement of fits; the other is whatever the planner said
    when it refused to size the file, which is a sentence about the file's facts. The
    same catalog entry is read here as the service read there, so the comparison holds
    in every language.
    """
    if row.rank is not None or row.excluded_because is None:
        return None
    if row.excluded_because == _("no placement of it fits this machine at any context"):
        return pgettext("exclusion tag", "no room")
    return pgettext("exclusion tag", "no facts")


def exclusion_tag(facts: RowFacts) -> str | None:
    """The word for an unranked row's cell, with a plain one for a reason that has none.

    Every exclusion the scorer makes carries a tag; the licence refusal and the planner's
    refusal to size a file are made elsewhere and carry only a sentence. A cell for those
    reads *not ranked* rather than nothing, because a blank in a column of scores reads
    as a zero, and the sentence is under the table and in the row's explanation.
    """
    if facts.ranked:
        return None
    if facts.excluded_tag:
        return facts.excluded_tag
    if facts.excluded_because:
        return pgettext("exclusion tag", "not ranked")
    return None


def mixed_confidence(rows: Iterable[BoardRow]) -> bool:
    """Whether the rows disagree about how their speeds were arrived at.

    When every row carries the same label there is nothing a column per row can say that
    one sentence under the table does not say better, and the space buys a column that
    does vary. When they differ, the label has to be on the row it belongs to. A board
    with no speeds at all counts as agreeing: there is no figure whose label could be
    lost.
    """
    labels = {row.candidate.speed.confidence for row in rows if row.candidate.speed is not None}
    return len(labels) > 1


def _no_figure() -> str:
    """The mark for a cell whose figure was never computed, one cell wide.

    Not a blank, which reads as a zero, and not the word for an unknown size, which is
    seven cells and would widen every column an unplanned row has nothing for. A row
    that was never planned -- refused for its licence, say -- has nothing in seven of
    these columns, and the reason is in the one column that says why.
    """
    return pgettext_literal("board cell with no figure", "–")  # noqa: RUF001  - a dash, on purpose


def board_cell(facts: RowFacts, column: Column, *, tag_column: Column = "score") -> Text:
    """One cell of one row, as text no style tag can be read out of.

    Args:
        facts: The row.
        column: Which cell.
        tag_column: Where an unranked row's reason goes in a word: the score column on
            the board, which such a row has no score for, and the verdict column on a
            fit listing, which has no score column at all.

    Returns:
        The cell, coloured for a verdict and dimmed for an unranked row.
    """
    placement = facts.placement
    speed = facts.speed
    budget = placement.budget if placement is not None else None
    # An unsupported placement carries a speed of zero rather than none and a budget of
    # nothing on the card, and a zero is a figure where "nothing was placed" is the truth.
    estimated = speed if speed is not None and speed.confidence != "unsupported" else None
    placed = placement if placement is not None and placement.mode != "unsupported" else None
    sized = placed.budget if placed is not None else None
    none = _no_figure()
    if column == tag_column and not facts.ranked:
        return Text(for_display(exclusion_tag(facts) or none), style="dim")
    values: dict[Column, str] = {
        "rank": isolate(str(facts.rank)) if facts.rank is not None else "",
        "model": isolate(facts.model_id),
        "quant": isolate(facts.quant),
        "size": _size(facts.download_bytes) if facts.download_bytes is not None else none,
        "have": (
            pgettext("model file on this machine", "yes")
            if facts.local_path
            else pgettext("model file on this machine", "no")
        ),
        "score": _number(facts.score) if facts.score is not None else none,
        "quality": _number(facts.quality, 0) if facts.quality is not None else none,
        "gen": _number(estimated.gen_tps) if estimated is not None else none,
        "prompt": _number(estimated.pp_tps, 0) if estimated is not None else none,
        "confidence": confidence_label(speed.confidence) if speed is not None else none,
        "mode": mode_label(placement.mode) if placement is not None else none,
        "vram": _size(sized.vram_required) if sized is not None else none,
        "ram": _size(sized.ram_required) if sized is not None else none,
        "verdict": verdict_label(budget.verdict) if budget is not None else none,
        "context": _context(placed.max_context_fit) if placed is not None else none,
    }
    style = ""
    if column == "verdict" and budget is not None:
        style = verdict_style(budget.verdict)
    elif column == "model":
        style = "bold"
    if not facts.ranked:
        style = f"{style} dim".strip()
    return Text(for_display(values[column]), style=style)


# --- filters, sorts and the line that says what is on the screen ----------------------


@dataclass(frozen=True)
class Filters:
    """What a reader asked to see of the rows a board already produced.

    None of these change the ranking or ``ranked_total``: a *request* filter such as
    ``--min-tps`` changes what is ranked and is said on the board, while a filter here
    changes only what is drawn. The page has a box under every heading for the same
    purpose; the dashboard's ``/`` box takes the same terms as text, and the command
    line takes the four a person types most as flags.

    Attributes:
        search: Text the id, the name or the quantisation must contain.
        min_fit: The worst verdict to draw, or ``None`` for every verdict.
        installed: Only rows whose file is already on this machine.
        mode: Only rows that run in this mode.
        min_speed: At least this many tokens per second generated.
        max_size: At most this many bytes to download.
        max_card: At most this many bytes on the card.
        max_ram: At most this many bytes of system memory.
        min_context: Holding at least this many tokens.
        min_quality: A quality figure at least this high.
    """

    search: str = ""
    min_fit: Verdict | None = None
    installed: bool = False
    mode: RunMode | None = None
    min_speed: float | None = None
    max_size: int | None = None
    max_card: int | None = None
    max_ram: int | None = None
    min_context: int | None = None
    min_quality: float | None = None

    @property
    def active(self) -> bool:
        """Whether anything at all is being hidden."""
        return self != Filters()


FILTER_TERMS: tuple[str, ...] = (
    "fit>=VERDICT",
    "speed>=N",
    "size<=SIZE",
    "card<=SIZE",
    "ram<=SIZE",
    "ctx>=N",
    "quality>=N",
    "runs=MODE",
    "have",
)
"""The terms :func:`parse_filters` reads, as a reader is shown them when one is refused.

Each names the page's box under the same heading, with the sign the box shows: a floor
on a figure a reader wants more of, a ceiling on one they pay for. Anything else typed
is text the model's id, name or quantisation must contain.
"""

_VERDICT_ORDER: tuple[Verdict, ...] = get_args(Verdict)
"""The verdicts best first, which is the order the type declares them in."""


def parse_filters(text: str) -> Filters:
    """Read the dashboard's filter box, or a ``--filter`` string, into :class:`Filters`.

    Args:
        text: Terms separated by spaces, from :data:`FILTER_TERMS`; whatever is not a
            term is text to search for.

    Returns:
        The filters.

    Raises:
        ValueError: A term's value cannot be read -- a verdict that is not one of the
            five, a run mode that is not one of the four, a size that is not a size, a
            number that is not a number. The message names the term and the values it
            takes, and nothing is applied, because a filter half applied hides rows for
            a reason the state line cannot then name.
    """
    search: list[str] = []
    fields: dict[str, Any] = {}
    for term in text.split():
        key, sign, value = _split_term(term)
        if key is None:
            if term.casefold() == "have":
                fields["installed"] = True
            else:
                search.append(term)
            continue
        if key == "fit" and sign == ">=":
            fields["min_fit"] = _checked_verdict(value)
        elif key == "runs" and sign == "=":
            fields["mode"] = _checked_mode(value)
        elif key == "speed" and sign == ">=":
            fields["min_speed"] = _checked_number(term, value)
        elif key == "quality" and sign == ">=":
            fields["min_quality"] = _checked_number(term, value)
        elif key == "ctx" and sign == ">=":
            fields["min_context"] = int(_checked_number(term, value.upper().removesuffix("K")))
            if value.upper().endswith("K"):
                fields["min_context"] *= 1024
        elif key in ("size", "card", "ram") and sign == "<=":
            fields[f"max_{key}"] = _checked_size(term, value)
        else:
            raise ValueError(
                _(
                    "%(term)s is not a filter; the terms are %(terms)s, and anything else "
                    "is text to look for."
                )
                % {"term": term, "terms": ", ".join(FILTER_TERMS)}
            )
    return Filters(search=" ".join(search), **fields)


def _split_term(term: str) -> tuple[str | None, str, str]:
    """A term as key, sign and value, or no key at all for plain text."""
    for sign in (">=", "<=", "="):
        key, found, value = term.partition(sign)
        if found and key.isalpha():
            return key.casefold(), sign, value
    return None, "", term


def _checked_verdict(value: str) -> Verdict:
    """The verdict a term named, or the five it could have."""
    if value not in _VERDICT_ORDER:
        raise ValueError(_("fit>= takes one of %(values)s") % {"values": ", ".join(_VERDICT_ORDER)})
    return value


def _checked_mode(value: str) -> RunMode:
    """The run mode a term named, or the ones it could have."""
    modes = [mode for mode in get_args(RunMode) if mode != "unsupported"]
    if value not in modes:
        raise ValueError(_("runs= takes one of %(values)s") % {"values": ", ".join(modes)})
    return cast("RunMode", value)


def _checked_number(term: str, value: str) -> float:
    """A figure typed into a term, in either decimal punctuation."""
    try:
        return float(value.replace(",", "."))
    except ValueError as exc:
        raise ValueError(_("%(term)s needs a number") % {"term": term}) from exc


def _checked_size(term: str, value: str) -> int:
    """A size typed into a term, read the way every size flag is read."""
    try:
        return parse_size(value)
    except ValueError as exc:
        raise ValueError(
            _("%(term)s needs a size such as 8G, 7.5GiB or 512M") % {"term": term}
        ) from exc


def filter_terms(filters: Filters) -> str:
    """The filters as the text :func:`parse_filters` would read them back from.

    What the dashboard's box shows when it opens, so a filter set by a key is one a
    reader can see and edit, and an empty box on Enter clears every one of them.
    """
    terms: list[str] = []
    if filters.search.strip():
        terms.append(filters.search.strip())
    if filters.min_fit is not None:
        terms.append(f"fit>={filters.min_fit}")
    if filters.mode is not None:
        terms.append(f"runs={filters.mode}")
    if filters.min_speed is not None:
        terms.append(f"speed>={filters.min_speed:g}")
    if filters.max_size is not None:
        terms.append(f"size<={filters.max_size}")
    if filters.max_card is not None:
        terms.append(f"card<={filters.max_card}")
    if filters.max_ram is not None:
        terms.append(f"ram<={filters.max_ram}")
    if filters.min_context is not None:
        terms.append(f"ctx>={filters.min_context}")
    if filters.min_quality is not None:
        terms.append(f"quality>={filters.min_quality:g}")
    if filters.installed:
        terms.append("have")
    return " ".join(terms)


def passes(facts: RowFacts, filters: Filters) -> bool:
    """Whether one row survives every filter.

    A row with no figure at all is not hidden by a threshold on that figure: it has not
    failed the test, it was never given one, and hiding it would be the board quietly
    deciding. The fit filter is the exception, because "does it run" is a question about
    a placement and a row with none has no answer to it that puts it on a list of things
    that run.
    """
    placement = facts.placement
    budget = placement.budget if placement is not None else None
    wanted = filters.search.strip().casefold()
    if wanted and not any(
        wanted in field.casefold() for field in (facts.model_id, facts.name, facts.quant)
    ):
        return False
    if filters.min_fit is not None and (
        budget is None
        or _VERDICT_ORDER.index(budget.verdict) > _VERDICT_ORDER.index(filters.min_fit)
    ):
        return False
    if filters.installed and facts.local_path is None:
        return False
    if filters.mode is not None and placement is not None and placement.mode != filters.mode:
        return False
    speed = facts.speed
    checks: list[tuple[float | None, float | None, bool]] = [
        (speed.gen_tps if speed is not None else None, filters.min_speed, True),
        (facts.download_bytes, filters.max_size, False),
        (budget.vram_required if budget is not None else None, filters.max_card, False),
        (budget.ram_required if budget is not None else None, filters.max_ram, False),
        (placement.max_context_fit if placement is not None else None, filters.min_context, True),
        (facts.quality, filters.min_quality, True),
    ]
    for value, limit, floor in checks:
        if value is None or limit is None:
            continue
        if (value < limit) if floor else (value > limit):
            return False
    return True


def filter_label(min_fit: Verdict | None) -> str:
    """Which verdicts are showing, in a phrase the state line can carry."""
    if min_fit is None:
        return pgettext("board filter", "every candidate")
    labels: dict[str, str] = {
        "comfortable": pgettext("board filter", "the ones with room to spare"),
        "fits": pgettext("board filter", "the ones that fit"),
        "tight": pgettext("board filter", "the ones that run"),
    }
    return labels.get(min_fit, filter_label(None))


def filter_phrases(filters: Filters) -> list[str]:
    """Every active term but the fit filter and the on-disk one, each as a phrase."""
    phrases: list[str] = []
    if filters.search.strip():
        phrases.append(_("matching %(text)s") % {"text": isolate(filters.search.strip())})
    if filters.mode is not None:
        phrases.append(_("running as %(mode)s") % {"mode": mode_label(filters.mode)})
    if filters.min_speed is not None:
        phrases.append(_("at least %(tps)s tokens per second") % {"tps": _tps(filters.min_speed)})
    if filters.max_size is not None:
        phrases.append(_("no more than %(size)s to download") % {"size": _size(filters.max_size)})
    if filters.max_card is not None:
        phrases.append(_("no more than %(size)s on the card") % {"size": _size(filters.max_card)})
    if filters.max_ram is not None:
        phrases.append(
            _("no more than %(size)s of system memory") % {"size": _size(filters.max_ram)}
        )
    if filters.min_context is not None:
        phrases.append(
            _("holding at least %(context)s tokens")
            % {"context": isolate(format_grouped(filters.min_context))}
        )
    if filters.min_quality is not None:
        phrases.append(
            _("a quality of at least %(quality)s") % {"quality": _number(filters.min_quality, 0)}
        )
    return phrases


_Row = TypeVar("_Row", BoardRow, FitRow)


def sort_value(facts: RowFacts, key: SortKey) -> float | str | None:
    """The figure or word one key orders a row by, or ``None`` when the row has none."""
    placement = facts.placement
    budget = placement.budget if placement is not None else None
    speed = facts.speed
    if key == "score":
        return facts.score
    if key == "speed":
        return speed.gen_tps if speed is not None else None
    if key == "prompt":
        return speed.pp_tps if speed is not None else None
    if key == "quality":
        return facts.quality
    if key == "context":
        return float(placement.max_context_fit) if placement is not None else None
    if key == "size":
        return float(facts.download_bytes) if facts.download_bytes is not None else None
    if key == "card":
        return float(budget.vram_required) if budget is not None else None
    if key == "ram":
        return float(budget.ram_required) if budget is not None else None
    if key == "fit":
        # Best first is largest first, so the order the type declares is turned round.
        return float(len(_VERDICT_ORDER) - _VERDICT_ORDER.index(budget.verdict)) if budget else None
    if key == "model":
        return facts.model_id.casefold()
    return facts.quant.casefold()


def sorted_rows(rows: Sequence[_Row], key: SortKey, descending: bool | None = None) -> list[_Row]:
    """The rows in the order one key puts them, with the ranking breaking every tie.

    Args:
        rows: The rows, in the order the service ranked them.
        key: What to order by.
        descending: Largest first, or ``None`` for the key's own default.

    Returns:
        A new list. The rank travels with each row and is never rewritten: a column sort
        is a way of looking at an answer, not a second opinion about it, and a table that
        renumbered itself would quietly claim it was. A row with nothing to order by --
        no placement, no speed, no size filled in -- goes last whichever way the key
        runs: "nobody has filled this in" is not the same claim as "this is the
        smallest one".

    Every value read here was produced by a service. The sort is stable, so two rows a
    key cannot tell apart come back in the order the ranking put them.
    """
    if descending is None:
        descending = default_descending(key)
    valued: list[tuple[float | str, _Row]] = []
    unvalued: list[_Row] = []
    for row in rows:
        value = sort_value(facts_of(row), key)
        if value is None:
            unvalued.append(row)
        else:
            valued.append((value, row))
    valued.sort(key=lambda pair: pair[0], reverse=descending)
    return [row for _value, row in valued] + unvalued


@dataclass(frozen=True)
class View:
    """How the drawn listing differs from the ranked one: order, filters, columns, rows.

    Attributes:
        sort: What the rows are ordered by.
        descending: Largest first, or ``None`` for the key's own default.
        filters: What is hidden.
        columns: Exactly these columns, in this order, or ``None`` to let the width choose.
        wide: Every column the listing has, whatever the width.
        excluded: Draw the rows that were not ranked, dimmed, under the ranked ones.
    """

    sort: SortKey = "score"
    descending: bool | None = None
    filters: Filters = Filters()
    columns: tuple[Column, ...] | None = None
    wide: bool = False
    excluded: bool = True

    @property
    def reordered(self) -> bool:
        """Whether the rows are in any order but the ranking's own."""
        return self.sort != "score" or (
            self.descending is not None and self.descending != default_descending(self.sort)
        )


def visible_rows(rows: Sequence[_Row], view: View) -> list[_Row]:
    """The rows on screen: what survives the filters, in the order the sort asks for."""
    kept = [row for row in rows if passes(facts_of(row), view.filters)]
    return sorted_rows(kept, view.sort, view.descending)


def state_line(shown: int, total: int, view: View) -> str:
    """One line saying what is on the screen, why it is in that order and what is hidden.

    The shapes are the dashboard's own, kept because six catalogs carry them. "Already
    on this machine" is a claim about the list the other words do not make, so it keeps
    the sentence it had; every other term joins the phrase after "showing", and a reader
    who cannot see the model they came for is told which of the filters is hiding it.
    """
    counted = _("%(shown)d of %(total)d shown") % {"shown": shown, "total": total}
    sort = sort_label(view.sort)
    if view.descending is not None and view.descending != default_descending(view.sort):
        sort = _("%(sort)s in reverse") % {"sort": sort}
    filters = view.filters
    phrases = [filter_label(filters.min_fit), *filter_phrases(filters)]
    if filters.installed and len(phrases) == 1:
        return _("%(counted)s, by %(sort)s, showing %(filter)s already on this machine.") % {
            "counted": counted,
            "sort": sort,
            "filter": phrases[0],
        }
    if filters.installed:
        phrases.append(pgettext("board filter", "already on this machine"))
    return _("%(counted)s, by %(sort)s, showing %(filter)s.") % {
        "counted": counted,
        "sort": sort,
        "filter": ", ".join(phrases),
    }


# --- the board ------------------------------------------------------------------------


def _column_spec(column: Column, budget: ColumnBudget) -> dict[str, Any]:
    """How Rich should lay one column out: its heading, alignment and what may fold.

    Only the model and the quantisation may wrap, and the model column has the cap
    that makes a long id fold rather than push every other column off the screen. The
    figures never wrap: a number split across two lines is two numbers. When an
    unranked row's tag is wider than the score column, Rich takes the difference from
    the widest column that may wrap, which is the model column -- so a tag costs a few
    more folded names and never a column.
    """
    numeric = column in {
        "rank",
        "size",
        "score",
        "quality",
        "gen",
        "prompt",
        "vram",
        "ram",
        "context",
    }
    spec: dict[str, Any] = {
        "header": column_heading(column),
        "justify": "right" if numeric else "left",
        "no_wrap": column not in {"model", "quant", "mode"},
    }
    if column == "model":
        spec |= {"max_width": budget.model, "overflow": "fold"}
    if column == "quant":
        spec["max_width"] = budget.quant
    return spec


def _drawn_columns(
    view: View, console_width: int, *, mixed: bool, budget: ColumnBudget, order: Sequence[Column]
) -> tuple[Column, ...]:
    """The columns a view draws: the ones it named, all of them, or what the width admits."""
    if view.columns is not None:
        return view.columns
    if view.wide:
        return tuple(order)
    if order is FIT_ORDER:
        return fit_columns(console_width, budget=budget)
    return board_columns(console_width, mixed_confidence=mixed, budget=budget)


def _draw_listing(
    title: str,
    rows: Sequence[RowFacts],
    columns: Sequence[Column],
    budget: ColumnBudget,
    *,
    tag_column: Column,
) -> Table:
    """One table of rows, ranked and unranked alike, the unranked ones dimmed."""
    table = Table(title=for_display(title))
    _add_columns(table, [_column_spec(column, budget) for column in columns])
    for facts in rows:
        cells = [board_cell(facts, column, tag_column=tag_column) for column in columns]
        _add_row(table, *cells, style=None if facts.ranked else "dim")
    return table


def render_board(
    board: Board,
    *,
    console_width: int = 80,
    view: View | None = None,
    budget: ColumnBudget | None = None,
) -> Group:
    """The ranked board: the answer a person came for, with its provenance attached.

    Args:
        board: The ranked candidates and the request they answer.
        console_width: The console's width; 80, the narrowest this table is designed for,
            when the caller does not know.
        view: How to draw it -- the order, the filters, the columns, whether the unranked
            rows are on it -- or the default, which is the ranking's own order with every
            column the width admits.
        budget: The identity columns' widths, measured from the catalog; the bundled
            catalog's when the caller has none.

    Returns:
        The table, the caption saying what the speed column is and what it is not, and
        a line saying what the view hid or reordered when it did either. A board that
        ranked nothing is that sentence instead of the table, rendered here rather than
        by the command, so that it is drawn under the same banner and beside the same
        record of the machine every other shape of this answer carries.

    The candidates that were not ranked are rows of the same table, dimmed, every column
    filled with whatever was computed for them and the word for the reason where the
    score would be. A row excluded for running at four tokens a second was placed, sized
    and estimated first, and those figures are what tell a reader whether a smaller
    quantisation would rescue it; a second table of sentences told them none of that
    and said one sentence twenty-six times. The sentence is under the table now, once
    per reason, from :func:`render_reasons`.
    """
    view = view or View()
    budget = budget or ColumnBudget()
    if not board.rows:
        return simulated_answer(
            board.simulation,
            _cell(
                _(
                    "Nothing was ranked. Every candidate is listed below with the reason; "
                    "widen the request or free some memory."
                )
            ),
            *_machine_captions(board.machine, speeds=False),
        )
    columns = _drawn_columns(
        view, console_width, mixed=mixed_confidence(board.rows), budget=budget, order=BOARD_ORDER
    )
    carried = [*board.rows, *(board.excluded if view.excluded else [])]
    drawn = [facts_of(row) for row in visible_rows(carried, view)]
    table = _draw_listing(_("Recommended"), drawn, columns, budget, tag_column="score")
    captions = _board_captions(board)
    if view.filters.active or view.reordered:
        captions.append(_cell(state_line(len(drawn), len(carried), view)))
    return simulated_answer(board.simulation, table, *captions)


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


def _truncation_caption(shown: int, ranked_total: int) -> str | None:
    """Say that a limit cut the list, when it did.

    A limit that says nothing is a limit that hides the catalog. With five models nothing
    was ever cut and the sentence never appeared; with fifty, the default keeps ten and a
    reader who is not told has no reason to believe the other forty exist. The point of
    ranking a catalog is to offer choices, including the ones the reader will turn down.

    Args:
        shown: How many rows the table actually drew.
        ranked_total: How many qualified before the slice.

    Returns:
        The sentence to print under the table, or ``None`` when nothing was cut.
    """
    if ranked_total <= shown:
        return None
    return _(
        "Showing the best %(shown)s of %(total)s that qualified; --limit sets how many, "
        "and the rest are neither worse-behaved nor hidden, only further down."
    ) % {
        "shown": isolate(localise_number(str(shown))),
        "total": isolate(localise_number(str(ranked_total))),
    }


def _unsized_caption(unsized_gpus: Sequence[str]) -> str | None:
    """Say that a card was on the machine and none of these figures used it.

    This is the caption this module's first rule was written for. Every row here has a
    speed, and on a machine whose card could not be sized every one of those speeds is a
    CPU-only speed -- arrived at not by measuring the machine but by assuming away the
    part of it nobody could read. At eighty columns even the ``Runs`` column that would
    have shown ``CPU`` is dropped for want of room, so without this sentence the table is
    a column of speeds produced from an assumption with nothing beside them saying so,
    which is the shape the rule at the top of this file exists to forbid.

    Args:
        unsized_gpus: The cards the scan found and could not size.

    Returns:
        The sentence to print under the table, or ``None`` when every card was sized --
        which is to say, on nearly every machine, nothing at all.
    """
    if not unsized_gpus:
        return None
    return _(
        "%(gpus)s is on this machine and nothing could read how much of it is free, so "
        "every speed above is a CPU-only speed. `llamafit doctor` says how to fix that."
    ) % {"gpus": ", ".join(isolate(name) for name in unsized_gpus)}


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
    unsized = _unsized_caption(board.unsized_gpus)
    if unsized is not None:
        captions.append(_cell(unsized))
    captions += _machine_captions(board.machine)
    cut = _truncation_caption(len(board.rows), board.ranked_total)
    if cut is not None:
        captions.append(_cell(cut))
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


def _machine_captions(machine: MachineFacts | None, *, speeds: bool = True) -> list[RenderableType]:
    """Which machine produced the rows above, in the figures that make one answer differ.

    Args:
        machine: What the board recorded, or ``None`` on a board built before this
            existed or by hand.
        speeds: Whether there are speeds above to attribute. False on a listing that
            ranked nothing, where the pools are still the reason and the bandwidth is
            the divisor of an empty set.

    Returns:
        One line for the pools and, when there are speeds, one for the bandwidth, plus a
        line saying the bandwidth was kept from an earlier run when it was.

    This file's own rule -- no number appears without saying how it was arrived at -- was
    being kept for every figure on the table and broken for the one underneath it. The
    speeds are all divided by a memory bandwidth the scan times, and the fits are all
    measured against memory that was free at the moment somebody asked; run the same
    command twice and both move, and the board moves with them while saying nothing. So
    the board says what it used, in the same voice every other line here uses.
    """
    if machine is None:
        return []
    if machine.gpu_name is None:
        first = _("Computed with %(ram_free)s of %(ram_total)s system memory free, no card.") % {
            "ram_free": _size(machine.ram_available_bytes),
            "ram_total": _size(machine.ram_total_bytes),
        }
    else:
        first = _(
            "Computed with %(ram_free)s of %(ram_total)s system memory free and %(vram_free)s "
            "of %(vram_total)s free on the %(gpu)s."
        ) % {
            "ram_free": _size(machine.ram_available_bytes),
            "ram_total": _size(machine.ram_total_bytes),
            "vram_free": _size(machine.vram_free_bytes),
            "vram_total": _size(machine.vram_total_bytes),
            "gpu": isolate(machine.gpu_name),
        }
    captions: list[RenderableType] = [_cell(first)]
    if not speeds:
        return captions
    captions.append(
        _cell(
            _("Every speed above is derived from %(bandwidth)s of memory bandwidth.")
            % {
                "bandwidth": bandwidth_sentence(
                    machine.ram_bandwidth_gbps,
                    machine.ram_bandwidth_source,
                    cached=machine.ram_bandwidth_cached,
                )
            }
        )
    )
    if machine.ram_bandwidth_cached:
        captions.append(
            _cell(
                _(
                    "That bandwidth was kept from an earlier run on this machine; "
                    "`llamafit system --refresh-bandwidth` times it again."
                )
            )
        )
    return captions


_Figure = Callable[[RowFacts], str | None]
"""What one exclusion's figure is for one row: the speed that was too slow, the size that
was too big, the context that was too short, or nothing for a reason with no figure."""


def _no_figure_for(_facts: RowFacts) -> str | None:
    """The figure of a reason that has none: a licence, a run mode, a missing capability."""
    return None


def _reason_groups(needs: Needs) -> dict[str, tuple[str, _Figure]]:
    """One sentence per exclusion the scorer makes, keyed by the tag it puts in the cell.

    Args:
        needs: The request, whose floor, ceiling and minimum the sentences name.

    Returns:
        The sentence and the figure-reader for each tag, in this language.

    The tags are read from the same catalog entries :mod:`llamafit.scoring.rank` reads,
    so a tag matches its group in every language. The sentences are the scorer's own
    with the row's figure lifted out, because the figure is what differed between the
    twenty-six copies of the sentence the old table printed; it goes beside the id
    instead. A tag this table has not met -- a reason made outside the scorer, or one
    added after this was written -- is not dropped: :func:`_render_reasons` lists its
    rows under the tag with each row's own sentence.
    """
    floor = _tps(floor_tps(needs.use_case, needs.min_tps))
    if needs.min_tps is not None:
        slow = _(
            "generates fewer than the %(floor)s tokens per second this request asks for; "
            "lower --min-tps or choose a smaller model or quantisation"
        ) % {"floor": floor}
    else:
        slow = _(
            "generates fewer than the %(floor)s tokens per second a person reads at: a "
            "batch tool on this machine and not one to sit in front of; a smaller model "
            "or quantisation would keep up"
        ) % {"floor": floor}
    ceiling = needs.max_download_bytes

    def tps(facts: RowFacts) -> str | None:
        if facts.speed is None:
            return None
        return _("%(tps)s tok/s") % {"tps": _number(facts.speed.gen_tps)}

    def size(facts: RowFacts) -> str | None:
        return None if facts.download_bytes is None else _size(facts.download_bytes)

    def context(facts: RowFacts) -> str | None:
        return None if facts.placement is None else _context(facts.placement.max_context_fit)

    groups: dict[str, tuple[str, _Figure]] = {
        pgettext("exclusion tag", "too slow"): (slow, tps),
        pgettext("exclusion tag", "too big"): (
            _("larger to download than the %(limit)s this request allows")
            % {"limit": _size(int(ceiling)) if ceiling is not None else format_bytes(None)},
            size,
        ),
        pgettext("exclusion tag", "unknown quant"): (
            _("a quantisation whose cost in quality is not known, so it cannot be scored"),
            _no_figure_for,
        ),
        pgettext("exclusion tag", "no room"): (
            _("needs more memory than this machine has, even at its smallest context"),
            _no_figure_for,
        ),
        pgettext("exclusion tag", "unsupported"): (
            _("no run mode supports this model on this machine"),
            _no_figure_for,
        ),
        pgettext("exclusion tag", "short context"): (
            _(
                "holds fewer than the %(minimum)s tokens asked for; lower the minimum "
                "context or free memory to see it ranked"
            )
            % {"minimum": isolate(format_grouped(needs.min_context))},
            context,
        ),
        pgettext("exclusion tag", "no estimate"): (
            _("no speed estimate, so it cannot be ranked against models that have one"),
            _no_figure_for,
        ),
    }
    for capability in get_args(Capability):
        tag = pgettext("exclusion tag", "no %(capability)s") % {"capability": capability}
        groups[tag] = (
            _(
                "no %(capability)s capability; drop it from the request, or ask for a use "
                "case that does not need it"
            )
            % {"capability": capability},
            _no_figure_for,
        )
    return groups


def render_reasons(board: Board) -> Group | None:
    """Why each unranked candidate is not on the board, one sentence per reason.

    A shorter list tells a reader nothing, and so does a longer one that says the same
    thing twenty-six times. The old ``Not ranked`` table was a hundred and eighty lines
    under a twenty-line board, one row per candidate, most of them carrying the same
    forty words with one figure changed. This says each reason once, then the ids with
    the figure that failed: "too slow (26): generates fewer than the 6 tokens per
    second a person reads at … — deepseek-v4-flash-0731 UD-IQ2_XXS (4.6 tok/s), …". The
    row itself is on the board above, dimmed, with every figure that was computed for
    it, and ``--explain`` still prints the whole sentence under an expanded row.
    """
    if not board.excluded:
        return None
    facts = [facts_of(row) for row in board.excluded]
    return _render_reasons(_("Not ranked"), facts, _reason_groups(board.needs))


def render_fit_reasons(board: FitBoard) -> Group | None:
    """Why each model with no placement is under the fit listing rather than on it."""
    if not board.excluded:
        return None
    groups: dict[str, tuple[str, _Figure]] = {
        pgettext("exclusion tag", "no room"): (
            _("no placement of it fits this machine at any context"),
            _no_figure_for,
        ),
    }
    return _render_reasons(_("Not placed"), [facts_of(row) for row in board.excluded], groups)


def _render_reasons(
    title: str, rows: Sequence[RowFacts], groups: Mapping[str, tuple[str, _Figure]]
) -> Group:
    """The reasons, grouped by tag, the largest group first.

    A group this file knows gets its sentence once and its ids on the line below, each
    with the figure that failed. A group it does not know -- a reason the scorer did not
    make, or a tag added after this was written -- gets its ids one per line, each with
    the sentence it actually carries, which is longer and never wrong.
    """
    by_tag: dict[str, list[RowFacts]] = {}
    for facts in rows:
        by_tag.setdefault(exclusion_tag(facts) or "", []).append(facts)
    pieces: list[RenderableType] = [Text(for_display(title), style="bold")]
    for tag, members in sorted(by_tag.items(), key=lambda item: (-len(item[1]), item[0])):
        count = isolate(localise_number(str(len(members))))
        group = groups.get(tag)
        if group is None:
            heading = _cell(_("%(tag)s (%(count)s):") % {"tag": tag, "count": count})
            heading.highlight_words([tag], "bold")
            pieces.append(heading)
            for facts in members:
                line = _("%(model)s %(quant)s: %(reason)s") % {
                    "model": isolate(facts.model_id),
                    "quant": isolate(facts.quant),
                    "reason": facts.excluded_because or "",
                }
                pieces.append(Padding(_cell(line), (0, 0, 0, 2)))
            continue
        sentence, figure = group
        heading = _cell(
            _("%(tag)s (%(count)s): %(sentence)s.")
            % {"tag": tag, "count": count, "sentence": sentence}
        )
        heading.highlight_words([tag], "bold")
        pieces.append(heading)
        entries: list[str] = []
        for facts in members:
            value = figure(facts)
            entries.append(
                _("%(name)s %(quant)s")
                % {"name": isolate(facts.model_id), "quant": isolate(facts.quant)}
                if value is None
                else _("%(model)s %(quant)s (%(figure)s)")
                % {"model": isolate(facts.model_id), "quant": isolate(facts.quant), "figure": value}
            )
        pieces.append(Padding(_cell(", ".join(entries)), (0, 0, 0, 2)))
    return Group(*pieces)


def render_explanation(row: BoardRow, board: Board) -> Group:
    """One row expanded into everything that produced it.

    Section 12.3, and the standing promise this command exists to keep: the score with its
    parts and weights, the quality it was built from, the budget line by line with the
    source of each, the context ladder, and where a token's time goes.
    """
    candidate = row.candidate
    if row.rank is None:
        # An unranked row has no place to lead with, so it leads with the reason.
        heading = _("%(name)s %(quant)s (%(tag)s)") % {
            "name": isolate(row.name),
            "quant": isolate(row.quant),
            "tag": exclusion_tag(facts_of(row)) or pgettext("exclusion tag", "not ranked"),
        }
    else:
        heading = _("%(rank)s. %(name)s %(quant)s") % {
            "rank": isolate(str(row.rank)),
            "name": isolate(row.name),
            "quant": isolate(row.quant),
        }
    pieces: list[RenderableType] = [Text(""), Text(for_display(heading), style="bold")]
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


def render_fit(
    board: FitBoard,
    *,
    console_width: int = 80,
    view: View | None = None,
    budget: ColumnBudget | None = None,
) -> Group:
    """Every model ranked by how well it uses this machine, and nothing else.

    ``fit`` asks a narrower question than ``recommend``, so the table is narrower: no
    score, no weights, no use case. What it does carry is the verdict and both pools, since
    "how well does it fit" is the only question being asked and those are the answer. The
    columns are chosen the way the board's are, from :data:`FIT_PRIORITY`, and the models
    with no placement are rows of the same table, dimmed, with the word for the reason
    where the verdict would be.

    A listing where nothing passed the threshold is the sentence saying so, drawn here for
    the reason :func:`render_board` gives: an empty answer about somebody else's machine
    has to say whose machine it was as loudly as a full one does.
    """
    view = view or View()
    budget = budget or ColumnBudget()
    if not board.rows:
        return simulated_answer(
            board.simulation,
            _cell(
                _(
                    "Nothing fits this machine at that threshold. Try --min-fit tight, or "
                    "drop --perfect."
                )
            ),
            *_machine_captions(board.machine, speeds=False),
        )
    # The heading is the one place the simulated banner would have been contradicted in
    # its own words: "Fit on this machine" one line under "they are not this machine".
    title = _("Fit on the simulated machine") if board.simulation else _("Fit on this machine")
    columns = _drawn_columns(view, console_width, mixed=False, budget=budget, order=FIT_ORDER)
    carried = [*board.rows, *(board.excluded if view.excluded else [])]
    drawn = [facts_of(row) for row in visible_rows(carried, view)]
    table = _draw_listing(title, drawn, columns, budget, tag_column="verdict")
    captions: list[RenderableType] = [
        _cell(
            _(
                "Sized for %(context)s tokens. The context column is the largest each one "
                "holds in the mode shown."
            )
            % {"context": _context(board.planned_context)}
        )
    ]
    # A whole sentence of its own rather than the board's, because ``fit`` prints no
    # speeds: what an unsized card costs here is the card column, which reads 0 B down
    # the page for a machine that has one.
    if board.unsized_gpus:
        captions.append(
            _cell(
                _(
                    "%(gpus)s is on this machine and nothing could read how much of it is "
                    "free, so every row was sized as if there were no card. `llamafit "
                    "doctor` says how to fix that."
                )
                % {"gpus": ", ".join(isolate(name) for name in board.unsized_gpus)}
            )
        )
    cut = _truncation_caption(len(board.rows), board.ranked_total)
    if cut is not None:
        captions.append(_cell(cut))
    captions += _machine_captions(board.machine, speeds=False)
    if view.filters.active or view.reordered:
        captions.append(_cell(state_line(len(drawn), len(carried), view)))
    return simulated_answer(board.simulation, table, *captions)


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
    pieces: list[RenderableType] = [
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
    return simulated_answer(report.simulation, *pieces)


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
