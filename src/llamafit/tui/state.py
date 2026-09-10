# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What the dashboard is asking for, and the one place it asks the services.

Nothing in this module imports Textual, and that is the point. Every question a screen can
be asked -- what does this machine have, what would run on it, why is that row where it is,
what would the command line be -- is answered by :mod:`llamafit.services`, and this is the
only file in the package that calls them. A screen holds a :class:`Dashboard` and reads
fields off it; no screen computes anything, and none of them can, because none of them has
a catalog or a host to compute from.

Two consequences are worth saying out loud.

**The services are injected.** :class:`Dashboard` takes the scan and the catalog loader as
callables, defaulting to the real ones, so a test drives the whole interface against a
recorded machine and a known catalog without a probe running or a file being read. That is
also what makes the empty catalog, the machine with no card and the scan whose probes
mostly failed testable at all: they are arguments, not accidents.

**A failure is a state, not an exception.** ``llamafit doctor`` exists because probes fail
on real machines, and a dashboard that fell over when one did would be worse than the
command line it replaces. Every service call here is wrapped, a :class:`LlamaFitError` is
kept in :attr:`Dashboard.problem` with the message and hint it was raised with, and the
screens draw what they still have beside it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llamafit.catalog.loader import Problem as CatalogProblem
from llamafit.catalog.loader import load_catalog
from llamafit.errors import LlamaFitError
from llamafit.hwprofile.simulate import resolve_host
from llamafit.models.catalog import Catalog
from llamafit.models.host import Host
from llamafit.models.llamacpp import LocalModel
from llamafit.models.plan import Needs
from llamafit.models.report import SystemReport
from llamafit.services.recommend import Board, BoardRow, build_board
from llamafit.services.scan import scan_system

Scanner = Callable[[], SystemReport]
"""What produces the machine: :func:`llamafit.services.scan.scan_system`, or a fake."""

Loader = Callable[[], tuple[Catalog, list[CatalogProblem]]]
"""What produces the catalog: :func:`llamafit.catalog.loader.load_catalog`, or a fake."""


@dataclass(frozen=True)
class Request:
    """What the Needs screen has asked for, which is what the board answers.

    Section 12.1's ``Needs`` plus the three things :func:`build_board` takes beside it.
    They are separate there because they are different kinds of thing -- what the model
    must be, versus how the board should be drawn -- and they are together here because
    they are one form on one screen.

    Attributes:
        needs: The use case, required capabilities, minimum context and download ceiling.
        prefer: ``balanced``, ``quality`` or ``speed``; leans the weights by a tenth.
        licenses: SPDX identifiers the request will accept; empty accepts any.
        vision: Whether to try to keep a vision projector.
        all_quants: Show every quantisation rather than the best one per model.
    """

    needs: Needs = field(default_factory=Needs)
    prefer: str = "balanced"
    licenses: tuple[str, ...] = ()
    vision: bool = True
    all_quants: bool = False


@dataclass(frozen=True)
class Substitution:
    """What the Simulate screen has put in place of the machine that was scanned.

    The fields are exactly :func:`llamafit.hwprofile.simulate.resolve_host`'s arguments,
    because this dataclass is a form and that function is what the form calls. Nothing
    here decides what an override *means*; section 4.4 wrote that down once and the
    dashboard is not going to write it down a second time.

    Attributes:
        profile: A bundled or user profile's name, which replaces the machine outright.
        gpu_memory: VRAM the card should be pretended to have, in bytes.
        ram: System memory the machine should be pretended to have, in bytes.
        cpu_cores: Physical cores the processor should be pretended to have.
    """

    profile: str | None = None
    gpu_memory: int | None = None
    ram: int | None = None
    cpu_cores: int | None = None

    @property
    def active(self) -> bool:
        """Whether anything at all was substituted, which is what the badge shows."""
        return any(
            value is not None for value in (self.profile, self.gpu_memory, self.ram, self.cpu_cores)
        )


class Dashboard:
    """The services' answers, held together, and the only caller of them in this package.

    Attributes:
        request: What is being asked for; the Needs screen replaces it, and
            ``llamafit --max-context`` seeds it.
        substitution: What stands in for the scanned machine; the Simulate screen
            replaces it, and section 13.1's ``--profile``, ``--memory``, ``--ram`` and
            ``--cpu-cores`` seed it, so the dashboard opens on the machine the command
            line named with its badge already showing.
        report: The scan, once it has been taken, and ``None`` before.
        catalog: The models, once loaded, and ``None`` before.
        catalog_problems: Every file the loader could not read, which the Host screen
            lists rather than a one-line warning to a stderr no window has.
        board: The ranked answer, and ``None`` when nothing could be ranked yet.
        problem: The last :class:`LlamaFitError` any of it raised, kept so a screen can
            show its message and its hint instead of a traceback.
    """

    def __init__(
        self,
        *,
        scanner: Scanner = scan_system,
        loader: Loader = load_catalog,
        request: Request | None = None,
        substitution: Substitution | None = None,
    ) -> None:
        self._scanner = scanner
        self._loader = loader
        self.request = request or Request()
        self.substitution = substitution or Substitution()
        self.report: SystemReport | None = None
        self.catalog: Catalog | None = None
        self.catalog_problems: list[CatalogProblem] = []
        self.board: Board | None = None
        self.problem: LlamaFitError | None = None

    # --- what the services were asked --------------------------------------------------

    def scan(self) -> None:
        """Take the scan, or record why it could not be taken.

        The catalog is loaded here too, because the first paint needs both and a screen
        that had one of them would have nothing to draw with it.
        """
        self.problem = None
        try:
            self.catalog, self.catalog_problems = self._loader()
        except LlamaFitError as exc:
            self.problem = exc
            self.catalog, self.catalog_problems = None, []
        try:
            self.report = self._scanner()
        except LlamaFitError as exc:
            self.problem = self.problem or exc
            self.report = None

    def rebuild(self) -> None:
        """Rank the catalog for the current request against the current machine.

        Every candidate is planned, sized, estimated and scored inside
        :func:`~llamafit.services.recommend.build_board`; this hands it the arguments and
        keeps what comes back. The board is set to ``None`` rather than left stale when
        there is nothing to build it from, so a screen shows "no answer yet" instead of
        the answer to a question that is no longer being asked.
        """
        host, catalog = self.host, self.catalog
        if host is None or catalog is None:
            self.board = None
            return
        try:
            self.board = build_board(
                catalog,
                host,
                self.request.needs,
                all_quants=self.request.all_quants,
                local_models=self.local_models,
                licenses=self.request.licenses,
                prefer=self.request.prefer,
                vision=self.request.vision,
            )
        except LlamaFitError as exc:
            self.problem = exc
            self.board = None

    def refresh(self) -> None:
        """Scan and rank: the whole of what the dashboard does, in the order it does it."""
        self.scan()
        self.rebuild()

    # --- what the screens read ---------------------------------------------------------

    @property
    def host(self) -> Host | None:
        """The machine to answer against: the scan, a profile, or either with a pool moved.

        ``resolve_host`` is handed the scan as a callable it will not call when a profile
        was named, which is section 4.4's own rule and not this module's: a run scoring
        against somebody else's machine has no business reading this one's.
        """
        report = self.report
        if report is None and self.substitution.profile is None:
            return None
        try:
            return resolve_host(
                scan_host=lambda: _require(report),
                profile=self.substitution.profile,
                gpu_memory=self.substitution.gpu_memory,
                ram=self.substitution.ram,
                cpu_cores=self.substitution.cpu_cores,
            )
        except LlamaFitError as exc:
            self.problem = exc
            return None if report is None else report.host

    @property
    def local_models(self) -> Sequence[LocalModel]:
        """The GGUF files llama.cpp already has, for the installed marker."""
        return () if self.report is None else self.report.llamacpp.local_models

    @property
    def rows(self) -> Sequence[BoardRow]:
        """The ranked rows, or nothing when nothing was ranked."""
        return () if self.board is None else self.board.rows

    def row_at(self, index: int) -> BoardRow | None:
        """The ranked row at ``index``, or ``None`` when the board is shorter than that."""
        rows = self.rows
        return rows[index] if 0 <= index < len(rows) else None

    # --- what the forms write ----------------------------------------------------------

    def ask(self, request: Request) -> None:
        """Replace the request and re-rank against it."""
        self.request = request
        self.rebuild()

    def substitute(self, substitution: Substitution) -> LlamaFitError | None:
        """Replace the stand-in machine and re-rank against it.

        Returns:
            The error the substitution raised, when it raised one, with the previous
            machine put back. A card cannot be resized on a machine that has none, and a
            profile can name a file nobody wrote; either way the form has to be able to
            say so, and a dashboard that quietly went on answering about the old machine
            under a badge saying it was simulating another would be lying twice.
        """
        previous = self.substitution
        self.substitution = substitution
        self.problem = None
        self.host  # noqa: B018  - evaluated for the error it records, not for its value
        if self.problem is not None:
            failed, self.problem = self.problem, None
            self.substitution = previous
            return failed
        self.rebuild()
        return None


def _require(report: SystemReport | None) -> Host:
    """The scanned host, insisting it exists.

    :attr:`Dashboard.host` only reaches this when it has a report or a profile, and
    ``resolve_host`` only calls it when there is no profile, so the two together mean the
    report is there. Saying so with a raise rather than an ``assert`` keeps the promise
    under ``python -O``, where an assertion is not run at all.
    """
    if report is None:  # pragma: no cover - unreachable via Dashboard.host
        raise RuntimeError("the scan was asked for before it was taken")
    return report.host
