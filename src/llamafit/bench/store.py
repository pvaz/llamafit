# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Where measurements live, and what the store refuses to keep.

Section 14 puts benchmarks in ``benchmarks.sqlite`` in the data directory, and section 16
says what a row carries: host fingerprint, llama.cpp build, model, quant, flags. This is
that, with three rules that are the reason the module is worth reading.

**A row is refused when its conditions are not knowable.** A run whose command line asked
for one micro-batch and whose own report says it used another did not measure what the row
would claim it measured. It is not stored and the caller is told which setting disagreed.
A run on a simulated host is refused outright: a hardware profile describes a machine
nobody is sitting at, and there is no honest way for a measurement to have come from one.

**A row is never edited and never merged.** Two runs of identical conditions are two rows
that happen to share a ``conditions_hash``; nothing averages them on the way in, because
the median of a set is a thing a reader should be able to watch being taken. Aggregation
happens on the way out, in :meth:`BenchStore.measurements`, where it can be explained.

**The whole result is stored as itself.** The columns exist to find rows; the row's
content is the serialised :class:`~llamafit.bench.types.BenchRun`, so a field added to that
model cannot drift away from a column somebody forgot to add. The version the row was
written under travels with it, and a database written by a newer LlamaFit is refused whole
rather than read half-way.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from types import TracebackType

from llamafit import __version__
from llamafit.bench.fingerprint import conditions_conflicts, conditions_hash
from llamafit.bench.types import (
    BENCH_SCHEMA_VERSION,
    BenchKind,
    BenchRun,
    Calibration,
    PagingCheck,
    RunConditions,
    RunTraffic,
)
from llamafit.errors import ProbeError
from llamafit.i18n import _
from llamafit.logging import get_logger
from llamafit.models.catalog import Measured
from llamafit.paths import get_paths

DATABASE_NAME = "benchmarks.sqlite"
"""The file section 14 names, in the application data directory."""

_log = get_logger("bench.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bench_runs (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    conditions_hash TEXT NOT NULL,
    host_fingerprint TEXT NOT NULL,
    model_id TEXT NOT NULL,
    quant TEXT NOT NULL,
    kind TEXT NOT NULL,
    llama_cpp_build INTEGER,
    flag_string TEXT NOT NULL,
    document TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS bench_runs_lookup
    ON bench_runs (host_fingerprint, model_id, quant);
CREATE INDEX IF NOT EXISTS bench_runs_conditions
    ON bench_runs (conditions_hash);
CREATE TABLE IF NOT EXISTS calibrations (
    id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    fitted_at TEXT NOT NULL,
    host_fingerprint TEXT NOT NULL,
    document TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS calibrations_host ON calibrations (host_fingerprint);
"""


def database_path() -> Path:
    """Where the benchmark database lives on this installation."""
    return get_paths().data_dir / DATABASE_NAME


class BenchStore:
    """The benchmark database, opened for the life of one command.

    Args:
        path: The database file. Its directory is created if it is not there yet.
    """

    def __init__(self, path: Path) -> None:
        """Open the database, creating it and its schema when it does not exist."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._check_version()

    def __enter__(self) -> BenchStore:
        """Return the open store, for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the connection, whatever happened inside the block."""
        self.close()

    def close(self) -> None:
        """Close the connection."""
        self._connection.close()

    def _check_version(self) -> None:
        """Refuse a database a newer LlamaFit wrote, rather than reading it half-way."""
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(BENCH_SCHEMA_VERSION),),
            )
            self._connection.commit()
            return
        stored = int(row["value"])
        if stored > BENCH_SCHEMA_VERSION:
            # Closed before raising: the constructor is where the connection was opened,
            # and nobody who never got an object back can be expected to close it.
            self._connection.close()
            raise ProbeError(
                _(
                    "%(path)s was written by a newer LlamaFit (benchmark format %(found)d,"
                    " this build reads %(known)d)"
                )
                % {"path": str(self.path), "found": stored, "known": BENCH_SCHEMA_VERSION},
                hint=_("Upgrade LlamaFit, or move the file aside to start a new one."),
            )

    def record(
        self,
        *,
        kind: BenchKind,
        conditions: RunConditions,
        gen_tps: float | None = None,
        pp_tps: float | None = None,
        ttft_ms: float | None = None,
        peak_vram_bytes: int | None = None,
        vram_total_bytes: int | None = None,
        traffic: RunTraffic | None = None,
        estimated_gen_tps: float | None = None,
        estimated_pp_tps: float | None = None,
        paging: PagingCheck | None = None,
        tool_call_ok: bool | None = None,
        buffer_bytes: dict[str, int] | None = None,
    ) -> BenchRun:
        """Store one measurement, refusing it if its conditions do not describe the run.

        Args:
            kind: What was measured.
            conditions: The run that happened.
            gen_tps: Generated tokens per second, when this kind produces one.
            pp_tps: Prompt tokens per second, when this kind produces one.
            ttft_ms: Time to first token, for a server request.
            peak_vram_bytes: The highest VRAM reading taken during the run.
            vram_total_bytes: The card's total at the time.
            traffic: What the estimator believed the run would read.
            estimated_gen_tps: What the estimate said *before* the run.
            estimated_pp_tps: The same for prompt processing.
            paging: The paging verdict, when one could be formed.
            tool_call_ok: Whether a tool-call request came back well-formed.
            buffer_bytes: The buffer sizes the server's log reported.

        Returns:
            The stored run, with the identifier and timestamp the store gave it.

        Raises:
            ProbeError: If the host is simulated, or if any setting the command line asked
                for disagrees with what the tool says it did. Both are refusals to write a
                row whose flags describe something other than what ran, which is the one
                way a stored benchmark can do more harm than no benchmark at all.
        """
        if conditions.host_fingerprint.startswith("sim-"):
            raise ProbeError(
                _("a benchmark cannot be recorded against a simulated machine"),
                hint=_("Run `llamafit bench` on the machine itself, without --profile."),
            )
        conflicts = conditions_conflicts(conditions)
        if conflicts:
            raise ProbeError(
                _("the run did not use the settings it was given: %(problems)s")
                % {"problems": "; ".join(conflicts)},
                hint=_(
                    "Nothing was stored. A result filed under flags it was not taken with"
                    " would be believed later by everything that reads it."
                ),
            )
        run = BenchRun(
            id=uuid.uuid4().hex,
            schema_version=BENCH_SCHEMA_VERSION,
            recorded_at=datetime.now(timezone.utc),
            kind=kind,
            conditions=conditions,
            conditions_hash=conditions_hash(kind, conditions),
            llamafit_version=__version__,
            gen_tps=gen_tps,
            pp_tps=pp_tps,
            ttft_ms=ttft_ms,
            peak_vram_bytes=peak_vram_bytes,
            vram_total_bytes=vram_total_bytes,
            traffic=traffic,
            estimated_gen_tps=estimated_gen_tps,
            estimated_pp_tps=estimated_pp_tps,
            paging=paging,
            tool_call_ok=tool_call_ok,
            buffer_bytes=buffer_bytes or {},
        )
        self.add(run)
        return run

    def add(self, run: BenchRun) -> None:
        """Write a run that has already been built, without re-checking its conditions.

        Used by :meth:`record` and by tests that build a row on purpose. Anything arriving
        from a real benchmark goes through :meth:`record`, which is where the refusals are.
        """
        self._connection.execute(
            "INSERT INTO bench_runs (id, schema_version, recorded_at, conditions_hash,"
            " host_fingerprint, model_id, quant, kind, llama_cpp_build, flag_string, document)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run.id,
                run.schema_version,
                run.recorded_at.isoformat(),
                run.conditions_hash,
                run.conditions.host_fingerprint,
                run.conditions.model_id,
                run.conditions.quant,
                run.kind,
                run.conditions.llama_cpp_build,
                run.conditions.flag_string,
                run.model_dump_json(),
            ),
        )
        self._connection.commit()
        _log.debug("stored %s run %s for %s", run.kind, run.id, run.conditions.model_id)

    def runs(
        self,
        *,
        host_fingerprint: str | None = None,
        model_id: str | None = None,
        quant: str | None = None,
    ) -> list[BenchRun]:
        """Every stored run matching the filters, oldest first.

        Args:
            host_fingerprint: Only runs from this machine.
            model_id: Only runs of this model.
            quant: Only runs of this quantisation.

        Returns:
            The runs, in the order they were taken.
        """
        clauses: list[str] = []
        values: list[object] = []
        for column, wanted in (
            ("host_fingerprint", host_fingerprint),
            ("model_id", model_id),
            ("quant", quant),
        ):
            if wanted is not None:
                clauses.append(f"{column} = ?")
                values.append(wanted)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            f"SELECT document FROM bench_runs{where} ORDER BY recorded_at, id", values
        ).fetchall()
        return [BenchRun.model_validate_json(row["document"]) for row in rows]

    def measurements(
        self, *, host_fingerprint: str, model_id: str, quant: str | None = None
    ) -> list[Measured]:
        """The stored runs in the shape the speed estimator already understands.

        Args:
            host_fingerprint: The machine asking. Only its own runs come back, because
                section 10.3 reserves ``measured`` for a benchmark taken *here*.
            model_id: The model being estimated.
            quant: The quantisation, or ``None`` for every one of them.

        Returns:
            One :class:`~llamafit.models.catalog.Measured` per distinct set of conditions,
            newest first within a set. Repeats of one configuration are reduced to their
            median rather than to their best: a benchmark is a claim about what a machine
            does, and the fastest of five runs is a claim about what it did once.

        A run the paging detector caught is left out. Its number is real, and it is kept in
        the database as the evidence that the configuration pages, but it is not a
        measurement of the speed that configuration runs at and must never be handed to
        something that will label an estimate ``measured`` with it.
        """
        grouped: dict[str, list[BenchRun]] = {}
        for run in self.runs(host_fingerprint=host_fingerprint, model_id=model_id, quant=quant):
            if not run.trustworthy:
                continue
            grouped.setdefault(run.conditions_hash, []).append(run)
        results: list[Measured] = []
        for repeats in grouped.values():
            newest = max(repeats, key=lambda item: item.recorded_at)
            measurement = newest.as_measurement()
            if len(repeats) > 1:
                measurement = measurement.model_copy(
                    update={
                        "gen_tps": _median_of(repeats, "gen_tps"),
                        "pp_tps": _median_of(repeats, "pp_tps"),
                    }
                )
            results.append(measurement)
        results.sort(key=_measurement_order, reverse=True)
        return results

    def save_calibration(self, calibration: Calibration) -> None:
        """Store one fit, keeping every earlier one so a change can be seen.

        Args:
            calibration: What was fitted, and what was refused.
        """
        self._connection.execute(
            "INSERT INTO calibrations (id, schema_version, fitted_at, host_fingerprint,"
            " document) VALUES (?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                BENCH_SCHEMA_VERSION,
                calibration.fitted_at.isoformat(),
                calibration.host_fingerprint,
                calibration.model_dump_json(),
            ),
        )
        self._connection.commit()

    def latest_calibration(self, host_fingerprint: str) -> Calibration | None:
        """The most recent fit for one machine, or ``None`` when it has never been fitted."""
        row = self._connection.execute(
            "SELECT document FROM calibrations WHERE host_fingerprint = ?"
            " ORDER BY fitted_at DESC, id DESC LIMIT 1",
            (host_fingerprint,),
        ).fetchone()
        return None if row is None else Calibration.model_validate_json(row["document"])


def _median_of(runs: Sequence[BenchRun], field: str) -> float | None:
    """The median of one field across repeats, or ``None`` when none of them has it."""
    values = [value for run in runs if (value := getattr(run, field)) is not None]
    return float(median(values)) if values else None


def _measurement_order(item: Measured) -> tuple[int, str]:
    """Sort key putting the dated measurements newest-first and the undated ones last."""
    return (0 if item.date is None else 1, item.date.isoformat() if item.date else "")


@contextmanager
def open_store(path: Path | None = None) -> Iterator[BenchStore]:
    """Open the benchmark database for the length of a command.

    Args:
        path: A database file, or ``None`` for the one this installation uses.

    Yields:
        The open store, closed when the block ends however it ends.
    """
    store = BenchStore(path or database_path())
    try:
        yield store
    finally:
        store.close()
