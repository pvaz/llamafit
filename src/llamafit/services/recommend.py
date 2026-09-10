# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The board: every candidate planned, estimated, scored and put in order.

This is section 3.3's data flow with nothing left out. For each model and each
quantisation the catalog publishes, the planner searches for a placement (section 9), the
estimator turns that placement into a speed (section 10), and the scorer turns the two
into a composite (section 11). :func:`build_board` is the one call that walks all of it,
and it returns pydantic models so the command, the terminal interface and the web API can
render the same answer without any of them recomputing it.

Three decisions here are worth reading before the code.

**Nothing is dropped.** A candidate the scorer excludes comes back in
:attr:`Board.excluded` carrying the reason, and so does one the planner could not size at
all. A shorter list tells a reader nothing; "excluded because its entry does not offer it
for coding" tells them what to change.

**Nothing is labelled ``measured``.** The estimator can be handed benchmarks and will
prefer them to its own formula, and section 10.3 reserves that label for "a stored
benchmark *on this host*". A catalog entry's ``measured`` block is a record from whichever
machine its curator ran it on, and phase 3 is what will store this machine's own. So the
board runs the formula, every figure is ``estimated``, and the records the catalog does
carry are handed to the caller separately, through :func:`recorded_measurements`, to be
shown beside the estimate rather than in place of it. Section 20's own definition of done
asks for estimates *within tolerance of* the measurements, which is a comparison that only
exists while the two are different numbers.

**Every candidate is compared at the same context.** Section 10.1 fixes the board's
working context at 8K tokens so the speed column compares like with like; ``plan`` is
where a user's own context is estimated for.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePath
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field

from llamafit.constants import DEFAULT_REQUESTED_CONTEXT, DEFAULT_WORKING_CONTEXT
from llamafit.errors import LlamaFitError
from llamafit.i18n import _
from llamafit.models.catalog import Catalog, CatalogModel, Measured, Quant
from llamafit.models.host import Host, Simulation
from llamafit.models.llamacpp import LocalModel
from llamafit.models.plan import (
    Candidate,
    Needs,
    Placement,
    Score,
    SpeedEstimate,
    Verdict,
)
from llamafit.scoring import evaluate, rank, shift_preference, weights_for
from llamafit.scoring.context_score import requested_context
from llamafit.scoring.fit_score import fit_score_for
from llamafit.services.plan import plan_model
from llamafit.speed import estimate_speed

MIN_FIT_VERDICTS: tuple[Verdict, ...] = ("comfortable", "fits", "tight")
"""The verdicts ``llamafit fit --min-fit`` accepts, best first."""

_VERDICT_RANK: dict[Verdict, int] = {
    "comfortable": 4,
    "fits": 3,
    "tight": 2,
    "too-tight": 1,
    "does-not-fit": 0,
}

PERFECT_FIT = 100.0
"""The fit score of a configuration inside section 11.3's ideal band, which ``--perfect`` keeps."""


class BoardRow(BaseModel):
    """One line of the board: a scored candidate and what a reader needs to name it.

    The candidate carries the placement, the speed, the quality and the score with every
    part it was made from. What it does not carry is anything about the *file* — the
    catalog only knows a model by its id, and a reader picks a row by its name, its
    download size and whether they already have it on disk.

    Attributes:
        rank: Position on the board, one-based, or ``None`` for an excluded candidate.
            An excluded candidate has no rank because a list of things that did not
            qualify has no best.
        model_id: The catalog id.
        name: The model's display name.
        quant: The quantisation this row is about.
        download_bytes: What fetching it would cost, or ``None`` before a refresh has
            filled the size in.
        local_path: Where this quant already is on disk, when llama.cpp's own directories
            hold it, and ``None`` otherwise.
        candidate: The placement, speed, quality and score, or the reason there is none.
    """

    model_config = ConfigDict(extra="forbid")

    rank: int | None
    model_id: str
    name: str
    quant: str
    download_bytes: int | None = None
    local_path: str | None = None
    candidate: Candidate


class Board(BaseModel):
    """A ranked recommendation for one request, with the candidates that did not qualify.

    Attributes:
        needs: What was asked for, including the defaults that were filled in.
        weights: What each part of the score counted for, after any config override.
        requested_context: The context the context score was measured against, which
            section 11.2 defaults by use case.
        planned_context: The context the *planner* was asked to size for, which section
            9.2 defaults to 32,768 whatever the use case. The two are different questions
            and, for a general or chat request, different numbers; a caption that printed
            one of them under the other's name would be exactly the kind of quiet
            mislabelling this program exists to avoid.
        working_context: The context every speed was estimated at, so the column
            compares like with like.
        rows: The ranked candidates, best first.
        excluded: The candidates that were not ranked, each carrying its reason.
        simulation: What was substituted for the machine these rows were computed on, or
            ``None`` when they were computed on the machine the reader is sitting at.
    """

    model_config = ConfigDict(extra="forbid")

    needs: Needs
    weights: dict[str, float]
    requested_context: int
    planned_context: int
    working_context: int
    rows: list[BoardRow] = Field(default_factory=list)
    excluded: list[BoardRow] = Field(default_factory=list)
    simulation: Simulation | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def simulated(self) -> bool:
        """Whether these rows describe a machine other than the one running LlamaFit.

        The same pair of fields :class:`~llamafit.models.host.Host` carries, for the same
        reason and with the same guarantee: computed, so it cannot drift from
        ``simulation`` and cannot be left out of the serialised board. A board does not
        carry the host, so without this a program reading ``recommend --json`` would have
        no way at all to tell an answer about this machine from an answer about a
        profile -- the terminal has a red line above the table and a script has nothing.
        """
        return self.simulation is not None


class FitRow(BaseModel):
    """One line of ``llamafit fit``: how well a model uses this machine, and nothing else.

    ``fit`` asks a narrower question than ``recommend`` and so answers with a narrower
    row. There is no score, because nothing was weighted, and no use case, because a
    model is not excluded here for being built for another job: "all models ranked by
    fit" means all of them.

    Attributes:
        rank: Position, one-based, or ``None`` when the model could not be placed.
        model_id: The catalog id.
        name: The model's display name.
        quant: The quantisation this row is about.
        download_bytes: What fetching it would cost, when known.
        local_path: Where this quant already is on disk, when it is.
        placement: The best placement found, or ``None``.
        fit: Section 11.3's fit score for that placement, or ``None``.
        excluded_because: Why there is no placement, when there is none.
    """

    model_config = ConfigDict(extra="forbid")

    rank: int | None
    model_id: str
    name: str
    quant: str
    download_bytes: int | None = None
    local_path: str | None = None
    placement: Placement | None = None
    fit: Score | None = None
    excluded_because: str | None = None


class FitBoard(BaseModel):
    """Every model ranked by how well it uses this machine.

    Attributes:
        planned_context: The context every placement was sized for.
        rows: The ranked models, best fit first.
        excluded: The ones with no placement at all, each carrying its reason.
        simulation: What was substituted for the machine these rows were computed on, or
            ``None`` when they were computed on the machine the reader is sitting at.
    """

    model_config = ConfigDict(extra="forbid")

    planned_context: int
    rows: list[FitRow] = Field(default_factory=list)
    excluded: list[FitRow] = Field(default_factory=list)
    simulation: Simulation | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def simulated(self) -> bool:
        """Whether these rows describe a machine other than the one running LlamaFit."""
        return self.simulation is not None


def local_index(local_models: Iterable[LocalModel]) -> dict[str, str]:
    """The local GGUF files keyed by file name, for matching a quant against the disk.

    Args:
        local_models: What llama.cpp's directories hold, from the scan.

    Returns:
        File name to full path. The name is what a catalog quant records, and the path is
        what a reader needs to launch it.
    """
    return {PurePath(local.path).name: local.path for local in local_models}


def local_path_for(quant: Quant, index: Mapping[str, str]) -> str | None:
    """Where this quant's first file already is on disk, or ``None``.

    The first shard is the one ``llama-server -m`` is given, and llama.cpp finds the rest
    beside it, so a split quant is "on disk" exactly when its first shard is.

    Both sides are reduced to a bare file name before they are compared. A catalog's
    ``files`` entry is a path inside the publishing repository and often carries a
    directory — ``UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`` is one of
    the seeded entries — while a local file is wherever its owner put it. Matching the two
    whole would report every sharded model as missing from a machine that has it.
    """
    for name in quant.files:
        found = index.get(PurePath(name).name)
        if found is not None:
            return found
    return None


def recorded_measurements(model: CatalogModel, quant: str | None = None) -> tuple[Measured, ...]:
    """The benchmarks the catalog records for this model, for showing beside an estimate.

    Args:
        model: The catalog entry.
        quant: Keep only measurements of this quantisation, or ``None`` for all of them.

    Returns:
        The entry's own ``measured`` block, filtered.

    These are *not* passed to the estimator. See this module's own documentation: a
    catalog measurement was taken on the curator's machine, and section 10.3's ``measured``
    label means a benchmark taken on the machine being planned for. Showing them beside
    the estimate lets a reader compare the two, which is what section 20 asks of phase 1;
    letting one replace the other would relabel somebody else's machine as this one.
    """
    return tuple(m for m in model.measured if quant is None or m.quant == quant)


def _speed_for(
    model: CatalogModel,
    quant: Quant,
    host: Host,
    placement: Placement,
    *,
    working_context: int,
) -> SpeedEstimate | None:
    """Estimate one placement's speed, or ``None`` when the file's facts are missing."""
    facts = quant.gguf_facts
    if facts is None:
        return None
    return estimate_speed(
        placement,
        facts,
        host,
        working_context=working_context,
        active_params=model.params.active_b * 1e9,
        quant=quant.name,
    )


def _placement_for(
    model: CatalogModel, quant: Quant, host: Host, needs: Needs, *, vision: bool
) -> tuple[Placement | None, str | None]:
    """Plan one configuration, turning a refusal to size it into a reason a reader can act on.

    The budget raises rather than guesses when a quant has no GGUF facts, which is the
    right answer — a size nobody has read is not a size — and it reaches a board as one
    row that says so instead of as a traceback that loses the other thirty.
    """
    try:
        return plan_model(model, quant, host, needs=needs, vision=vision), None
    except LlamaFitError as exc:
        return None, exc.message


def planned_context(needs: Needs) -> int:
    """The context the planner will actually size for, which is not the score's context.

    Section 9.2 sizes for what the request names and falls back to 32,768 tokens; section
    11.2 scores the context that comes out against a figure that varies by use case, 8,192
    for general and chat. Both are right about their own question and they are frequently
    different numbers, so anything that prints a sizing context reads it from here rather
    than from :func:`~llamafit.scoring.context_score.requested_context`.

    ``max_context`` caps both, which is why the caption a board prints under this figure
    never names a context the same board refused to plan.
    """
    wanted = max(needs.requested_context or DEFAULT_REQUESTED_CONTEXT, needs.min_context)
    return wanted if needs.max_context is None else min(wanted, needs.max_context)


def quant_entries(model: CatalogModel) -> list[Quant]:
    """Every quantisation the model's sources publish, in catalog order, without repeats.

    Two sources republishing the same quant name is one candidate, not two: the reader
    chooses a quantisation, not a repository.
    """
    seen: set[str] = set()
    quants: list[Quant] = []
    for source in model.sources:
        for quant in source.quants:
            if quant.name not in seen:
                seen.add(quant.name)
                quants.append(quant)
    return quants


def best_quant(model: CatalogModel, host: Host, needs: Needs | None = None) -> Quant:
    """The quantisation of one model that scores best on one machine.

    Args:
        model: The catalog entry.
        host: The scanned machine.
        needs: What to score against; the job the entry leads with, otherwise.

    Returns:
        The best-scoring quantisation, or the first the catalog lists when none of them can
        be scored at all.

    Raises:
        IndexError: If the model publishes no quantisations, which the catalog schema does
            not allow and a caller reading a hand-written file should check for anyway.

    This is what ``llamafit plan <model>`` uses when the user names no quantisation, and it
    is deliberately the same machinery :func:`build_board` uses rather than a rule of its
    own. Two commands that chose differently would disagree about the same model on the
    same machine, which is exactly the kind of difference nobody thinks to check.
    """
    quants = quant_entries(model)
    needs = needs or Needs(use_case=model.use_cases[0])
    candidates: list[Candidate] = []
    for quant in quants:
        placement, problem = _placement_for(model, quant, host, needs, vision=True)
        speed = (
            None
            if placement is None
            else _speed_for(model, quant, host, placement, working_context=DEFAULT_WORKING_CONTEXT)
        )
        candidates.append(
            Candidate(model_id=model.id, quant=quant.name, excluded_because=problem)
            if problem is not None
            else evaluate(model, quant, needs, placement, speed)
        )
    best = rank(candidates)[0]
    by_name = {quant.name: quant for quant in quants}
    return by_name.get(best.quant, quants[0])


def build_board(
    catalog: Catalog,
    host: Host,
    needs: Needs,
    *,
    all_quants: bool = False,
    limit: int | None = None,
    local_models: Sequence[LocalModel] = (),
    weight_overrides: Mapping[str, Mapping[str, float]] | None = None,
    licenses: Sequence[str] = (),
    prefer: str = "balanced",
    vision: bool = True,
) -> Board:
    """Plan, estimate, score and order every candidate in the catalog for one request.

    Args:
        catalog: The models to consider.
        host: The scanned machine.
        needs: What the user asked for.
        all_quants: Show every quantisation rather than the best one per model.
        limit: Keep at most this many ranked rows; the excluded ones are never cut,
            because the reason a model is missing is the thing a limit must not hide.
        local_models: The GGUF files llama.cpp already has, for the installed marker.
        weight_overrides: The config file's weights, keyed by use case and then by part.
        licenses: SPDX identifiers the request will accept; empty accepts any.
        prefer: ``balanced``, ``quality`` or ``speed``, which leans the weights.
        vision: Whether to try to keep a vision projector.

    Returns:
        The board, ranked, with the excluded candidates kept and explained.

    Raises:
        ConfigError: If the use case or the preference is unknown, or a weight override is
            malformed.

    The licence is checked before anything else about a candidate, and before section
    11's own exclusions, because it is the most absolute of the constraints a request can
    carry: a licence the user will not accept makes every other question about that model
    moot. It is still an exclusion and not a filter, so the model appears with the licence
    it actually has and the reader can decide whether to widen the request.
    """
    weights = shift_preference(weights_for(needs.use_case, weight_overrides), prefer)
    wanted = requested_context(needs)
    index = local_index(local_models)
    allowed = {spdx.casefold() for spdx in licenses}

    rows: list[BoardRow] = []
    for model in catalog.models:
        refused = (
            None
            if not allowed or model.license.spdx.casefold() in allowed
            else _("licensed %(spdx)s, which this request does not allow")
            % {"spdx": model.license.spdx}
        )
        for quant in quant_entries(model):
            placement, problem = (
                (None, refused)
                if refused is not None
                else _placement_for(model, quant, host, needs, vision=vision)
            )
            speed = (
                None
                if placement is None
                else _speed_for(
                    model, quant, host, placement, working_context=DEFAULT_WORKING_CONTEXT
                )
            )
            candidate = (
                Candidate(model_id=model.id, quant=quant.name, excluded_because=problem)
                if problem is not None
                else evaluate(model, quant, needs, placement, speed, weights=weights)
            )
            rows.append(
                BoardRow(
                    rank=None,
                    model_id=model.id,
                    name=model.name,
                    quant=quant.name,
                    download_bytes=quant.bytes_,
                    local_path=local_path_for(quant, index),
                    candidate=candidate,
                )
            )

    ordered = _rank_rows(rows)
    scored = [row for row in ordered if row.candidate.score is not None]
    excluded = [row for row in ordered if row.candidate.score is None]
    if not all_quants:
        scored = _best_per_model(scored)
        excluded = _best_per_model(excluded)
    if limit is not None:
        scored = scored[:limit]
    for position, row in enumerate(scored, start=1):
        row.rank = position

    return Board(
        needs=needs,
        weights=weights,
        requested_context=wanted,
        planned_context=planned_context(needs),
        working_context=DEFAULT_WORKING_CONTEXT,
        rows=scored,
        excluded=excluded,
        simulation=host.simulation,
    )


def build_fit_board(
    catalog: Catalog,
    host: Host,
    *,
    needs: Needs | None = None,
    min_fit: Verdict = "tight",
    perfect: bool = False,
    limit: int | None = None,
    all_quants: bool = False,
    local_models: Sequence[LocalModel] = (),
) -> FitBoard:
    """Rank every model by how well it uses this machine, and nothing else.

    Args:
        catalog: The models to consider.
        host: The scanned machine.
        needs: The context and cache preferences to plan against; the defaults otherwise.
        min_fit: The worst verdict a ranked row may carry.
        perfect: Keep only the configurations inside section 11.3's ideal band.
        limit: Keep at most this many ranked rows.
        all_quants: Show every quantisation rather than the best-fitting one per model.
        local_models: The GGUF files llama.cpp already has, for the installed marker.

    Returns:
        The models ordered by fit score, with the unplaceable ones kept and explained.

    ``fit`` deliberately does not go through section 11's exclusions. Those answer "is
    this a good answer to my request", and this command has no request: a coding model is
    not excluded from a fit listing for being a coding model. The only reasons a row is
    unranked here are that nobody has read the file's header, or that no placement of it
    fits this machine at all.
    """
    needs = needs or Needs()
    index = local_index(local_models)
    rows: list[FitRow] = []
    excluded: list[FitRow] = []

    for model in catalog.models:
        for quant in quant_entries(model):
            placement, problem = _placement_for(model, quant, host, needs, vision=True)
            row = FitRow(
                rank=None,
                model_id=model.id,
                name=model.name,
                quant=quant.name,
                download_bytes=quant.bytes_,
                local_path=local_path_for(quant, index),
                placement=placement,
                fit=None if placement is None else fit_score_for(placement.budget),
                excluded_because=problem,
            )
            if problem is not None:
                excluded.append(row)
            elif placement is None or placement.mode == "unsupported":
                row.fit = None
                row.excluded_because = _("no placement of it fits this machine at any context")
                excluded.append(row)
            else:
                rows.append(row)

    rows.sort(key=_fit_ordering)
    if not all_quants:
        rows = _best_per_model(rows)
        excluded = _best_per_model(excluded)
    kept = [row for row in rows if _passes_fit_filters(row, min_fit=min_fit, perfect=perfect)]
    if limit is not None:
        kept = kept[:limit]
    for position, row in enumerate(kept, start=1):
        row.rank = position
    return FitBoard(
        planned_context=planned_context(needs),
        rows=kept,
        excluded=excluded,
        simulation=host.simulation,
    )


def _passes_fit_filters(row: FitRow, *, min_fit: Verdict, perfect: bool) -> bool:
    """Whether a fit row survives ``--min-fit`` and ``--perfect``."""
    if row.placement is None:
        return False
    if _VERDICT_RANK[row.placement.budget.verdict] < _VERDICT_RANK[min_fit]:
        return False
    return not (perfect and (row.fit or 0.0) < PERFECT_FIT)


def _fit_ordering(row: FitRow) -> tuple[float, int, str, str]:
    """Sort key for a fit row: the score, then the verdict, then identity, best first."""
    verdict = 0 if row.placement is None else _VERDICT_RANK[row.placement.budget.verdict]
    return (-(row.fit or 0.0), -verdict, row.model_id, row.quant)


def _rank_rows(rows: Sequence[BoardRow]) -> list[BoardRow]:
    """Order board rows by ordering the candidates inside them.

    :func:`~llamafit.scoring.rank.rank` is the one place the ordering is written down, so
    the rows follow the candidates rather than repeating the rule. Two rows can hold the
    same model and quant only when the catalog does, which it does not, so the identity of
    a candidate is enough to find its row again.
    """
    by_identity = {(row.candidate.model_id, row.candidate.quant): row for row in rows}
    ordered = rank(row.candidate for row in rows)
    return [by_identity[(candidate.model_id, candidate.quant)] for candidate in ordered]


_Row = TypeVar("_Row", BoardRow, FitRow)


def _best_per_model(rows: Sequence[_Row]) -> list[_Row]:
    """Keep the first row each model contributes, the list already being in order.

    Section 12.2's "only the best quant per model appears by default". The list arrives
    ranked, so the best one is the first one seen, and ``--all-quants`` is the flag that
    skips this entirely.
    """
    seen: set[str] = set()
    kept: list[_Row] = []
    for row in rows:
        if row.model_id in seen:
            continue
        seen.add(row.model_id)
        kept.append(row)
    return kept
