# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Fetching model weights: planning it, checking the disk, doing it, proving it arrived whole.

The files here are tens or hundreds of gigabytes, so everything in this package is about
doing it once. :mod:`~llamafit.download.plan` decides what a model is before anything is
fetched; :mod:`~llamafit.download.state` is what makes an interrupted transfer resume
rather than restart; :mod:`~llamafit.download.engine` fetches and refuses to hand over a
file that does not match the catalog's checksum; :mod:`~llamafit.download.install` puts
them in order and writes the record that says the model is complete.
"""

from __future__ import annotations

from llamafit.download.engine import (
    DEFAULT_WORKERS,
    MAX_WORKERS,
    DownloadOptions,
    FileOutcome,
    download_file,
    download_plan,
    probe_size,
)
from llamafit.download.errors import (
    ChecksumError,
    DiskSpaceError,
    DownloadCancelledError,
    DownloadError,
    RangeNotSupportedError,
    RateLimitedError,
)
from llamafit.download.history import (
    DownloadRecord,
    append_record,
    history_path,
    read_history,
    recent,
)
from llamafit.download.install import InstallOutcome, install_model, prepare, write_manifest
from llamafit.download.plan import (
    DISK_HEADROOM_BYTES,
    MANIFEST_NAME,
    DiskCheck,
    DownloadPlan,
    FileRequest,
    build_plan,
    check_disk_space,
    hugging_face_url,
    read_manifest,
    refuse_unverifiable,
)
from llamafit.download.progress import NullProgress, ProgressReporter, RichProgress
from llamafit.download.state import DEFAULT_CHUNK_BYTES, PartFile, open_part
from llamafit.download.throttle import Governor, RateLimiter
from llamafit.download.transport import HttpRangeReader, RangeReader, RangeResponse

__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_WORKERS",
    "DISK_HEADROOM_BYTES",
    "MANIFEST_NAME",
    "MAX_WORKERS",
    "ChecksumError",
    "DiskCheck",
    "DiskSpaceError",
    "DownloadCancelledError",
    "DownloadError",
    "DownloadOptions",
    "DownloadPlan",
    "DownloadRecord",
    "FileOutcome",
    "FileRequest",
    "Governor",
    "HttpRangeReader",
    "InstallOutcome",
    "NullProgress",
    "PartFile",
    "ProgressReporter",
    "RangeNotSupportedError",
    "RangeReader",
    "RangeResponse",
    "RateLimitedError",
    "RateLimiter",
    "RichProgress",
    "append_record",
    "build_plan",
    "check_disk_space",
    "download_file",
    "download_plan",
    "history_path",
    "hugging_face_url",
    "install_model",
    "open_part",
    "prepare",
    "probe_size",
    "read_history",
    "read_manifest",
    "recent",
    "refuse_unverifiable",
    "write_manifest",
]
