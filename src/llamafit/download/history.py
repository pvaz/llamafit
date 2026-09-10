# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What has been downloaded, when, how big it was and whether it finished.

Section 15's installer asks for a downloads history, and the reason it is worth keeping is
smaller than it sounds: somebody who has fetched nine models over three months has no other
way to answer "which quantisation did I actually pull, and is that the one I benchmarked".
The file is small, human-readable JSON, and losing it costs nothing but the answer.

Each record is appended whether the install finished or not. A failed one is the more
useful of the two: it says which file it stopped on, so the sentence a user got at the time
is still there when they come back to it a week later.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

HISTORY_NAME = "downloads.json"
"""The file, under the data directory."""

MAX_HISTORY_ENTRIES = 200
"""How many records are kept. Older ones fall off the front; this is a convenience,
not an audit log, and an unbounded file that nothing ever prunes is a slow leak."""


class DownloadRecord(BaseModel):
    """One attempt at installing one model.

    Attributes:
        model_id: The catalog id.
        quant: The quantisation.
        repo: The repository the files came from.
        directory: Where they went.
        files: The file names, in plan order.
        total_bytes: What the model weighs.
        fetched_bytes: What this run actually pulled over the network.
        finished: Whether every file arrived and was checked.
        error: The sentence the user was given, when it did not finish.
        at: When the run ended, in UTC.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str
    quant: str
    repo: str
    directory: str
    files: list[str] = Field(default_factory=list)
    total_bytes: int = 0
    fetched_bytes: int = 0
    finished: bool = False
    error: str | None = None
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def history_path(data_dir: Path) -> Path:
    """Where the history lives under ``data_dir``."""
    return data_dir / HISTORY_NAME


def read_history(path: Path) -> list[DownloadRecord]:
    """Every record in the file, oldest first, or nothing at all when it cannot be read.

    A history that will not parse is not worth a failure: it is a convenience file, and
    refusing to download a model because of it would be absurd.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(document, list):
        return []
    records: list[DownloadRecord] = []
    for entry in document:
        try:
            records.append(DownloadRecord.model_validate(entry))
        except ValueError:
            continue
    return records


def append_record(path: Path, record: DownloadRecord) -> None:
    """Add ``record`` to the history, keeping the last :data:`MAX_HISTORY_ENTRIES`.

    Written to a sibling temporary path and moved into place, the way every other file
    this package writes is, so an interrupted write leaves the old history rather than
    half of a new one. A failure to write is swallowed: the model is on disk and correct,
    and refusing to say so because a convenience file could not be updated would be the
    wrong trade.
    """
    records = [*read_history(path), record][-MAX_HISTORY_ENTRIES:]
    payload = json.dumps([r.model_dump(mode="json") for r in records], indent=2, ensure_ascii=False)
    temporary = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    except OSError:
        return


def recent(records: Sequence[DownloadRecord], limit: int) -> list[DownloadRecord]:
    """The last ``limit`` records, newest first."""
    return list(reversed(list(records)[-limit:]))
