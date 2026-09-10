# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Checking that what arrived is what the catalog described.

This is the reason the catalog carries a checksum per file at all. A model that downloaded
wrong does not fail loudly: llama.cpp loads it and it answers nonsense, or it crashes in a
place nobody connects to a download that finished cleanly two hours earlier. The one moment
that mistake is cheap to catch is here, before the file is moved into place.

Hugging Face publishes the SHA-256 of each LFS object, which is the checksum of the file's
own bytes, so no manifest format or hashing scheme of ours is involved: the number in the
catalog and the number computed here are the same kind of number.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

from llamafit.download.errors import ChecksumError, DownloadCancelledError, DownloadError
from llamafit.i18n import _

HASH_BLOCK_BYTES = 1 << 20
"""How much is read at a time while hashing: 1 MiB."""


def sha256_of(
    path: Path,
    *,
    block_bytes: int = HASH_BLOCK_BYTES,
    on_bytes: Callable[[int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    """The SHA-256 of a file, read a block at a time.

    Args:
        path: The file to hash.
        block_bytes: How much to read per iteration.
        on_bytes: Called with each block's length, for a progress display. Hashing a
            sixty-gigabyte file takes minutes, and a bar that stops moving during them
            looks exactly like a bar that has hung.
        should_stop: Asked between blocks whether the user has cancelled.

    Returns:
        The digest, lowercase hexadecimal.

    Raises:
        DownloadError: If the file cannot be read.
        DownloadCancelledError: If ``should_stop`` said so.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                if should_stop is not None and should_stop():
                    raise DownloadCancelledError(_("Stopped before the checksum was finished."))
                block = handle.read(block_bytes)
                if not block:
                    break
                digest.update(block)
                if on_bytes is not None:
                    on_bytes(len(block))
    except OSError as exc:
        raise DownloadError(
            _("could not read %(path)s: %(error)s") % {"path": path, "error": exc},
            hint=_("Check that the file exists and is readable."),
        ) from exc
    return digest.hexdigest()


def check_size(path: Path, expected: int, *, name: str) -> None:
    """Refuse a file whose length is not the length the catalog recorded.

    Checked before the checksum because it is free and because it names the failure
    better: a file that is short was truncated, and saying so is more use than saying
    two hexadecimal strings differ.

    Args:
        path: The file to measure.
        expected: How many bytes the catalog says it has.
        name: What to call the file in the message.

    Raises:
        ChecksumError: If the size does not match.
        DownloadError: If the file cannot be measured.
    """
    try:
        actual = path.stat().st_size
    except OSError as exc:
        raise DownloadError(
            _("could not read %(path)s: %(error)s") % {"path": path, "error": exc},
            hint=_("Check that the file exists and is readable."),
        ) from exc
    if actual != expected:
        raise ChecksumError(
            _("%(file)s is %(actual)d bytes, but the catalog says %(expected)d")
            % {"file": name, "actual": actual, "expected": expected},
            hint=_("The transfer was cut short. Run the command again to fetch it properly."),
        )


def check_sha256(path: Path, expected: str, *, name: str, actual: str | None = None) -> str:
    """Refuse a file whose checksum is not the catalog's.

    Args:
        path: The file to hash.
        expected: The checksum the catalog holds.
        name: What to call the file in the message.
        actual: An already-computed digest, to save hashing twice.

    Returns:
        The digest that matched.

    Raises:
        ChecksumError: If the digests differ.
    """
    digest = actual if actual is not None else sha256_of(path)
    if digest.lower() != expected.lower():
        raise ChecksumError(
            _("%(file)s does not match the checksum the catalog holds.") % {"file": name},
            hint=_(
                "Expected %(expected)s but the file hashes to %(actual)s. The bad copy has "
                "been deleted; run the command again to fetch it."
            )
            % {"expected": expected.lower(), "actual": digest.lower()},
        )
    return digest
