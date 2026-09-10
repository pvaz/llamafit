# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""What can go wrong while fetching weights.

Every one of these is a :class:`~llamafit.errors.LlamaFitError`, so the command line
prints a sentence and a hint rather than a traceback, and exits 1. None of them is an
environment problem in the sense :mod:`llamafit.cli.app` reserves exit code 2 for: a
disk that is too small, a checksum that did not match and a transfer somebody stopped
are all things the person at the keyboard can act on.
"""

from __future__ import annotations

from llamafit.errors import LlamaFitError


class DownloadError(LlamaFitError):
    """A download could not be completed."""


class DiskSpaceError(DownloadError):
    """The target volume does not have room for what was asked for.

    Raised before the first byte is written. A hundred-gigabyte download that fails at
    ninety percent has cost hours and left nothing behind, and the arithmetic that
    prevents it is a subtraction.
    """


class ChecksumError(DownloadError):
    """A file arrived whole but is not the file the catalog describes.

    The worst failure this package has, because it is the one that does not announce
    itself: a model with a flipped bit loads and answers nonsense, or crashes somewhere
    nobody connects to the download. The file is deleted rather than kept.
    """


class DownloadCancelledError(DownloadError):
    """Somebody stopped the download; everything that had arrived is still on disk."""


class RangeNotSupportedError(DownloadError):
    """The server answered a range request with the whole file.

    Not shown to a user: the file driver catches it and restarts that one file as a
    single sequential stream. Writing a full body into a chunk's slot would corrupt the
    file in a way only the checksum would catch, hours later.
    """


class RateLimitedError(DownloadError):
    """The server asked us to slow down.

    Attributes:
        retry_after: How many seconds the server asked for, when it said.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        """Record the message and the delay the server asked for."""
        super().__init__(message)
        self.retry_after = retry_after
