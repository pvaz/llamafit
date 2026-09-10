# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Fetching one file, in parallel, resumably, and refusing to hand over a wrong one.

A file is cut into fixed chunks and the chunks are fetched by a pool of workers, each
writing into its own range of a part file created at the final size. What makes that safe
rather than merely fast is the sidecar in :mod:`llamafit.download.state`: a chunk is
recorded only once its last byte is written, so an interrupted transfer resumes at the
first chunk that was not, and never at zero.

Four situations get more care than the happy path, because each of them is a way to end up
with a file that looks finished and is not.

**A server that stops honouring ranges.** A redirect to a content network that answers a
range request with the whole body would, written into a chunk's slot, corrupt the file in a
way only the checksum catches, hours later. The status is checked before a single byte is
written, and the file restarts as one sequential stream instead.

**A server that rate-limits.** A 429 or a 503 is not retried harder. The delay the server
asked for is honoured, and :class:`~llamafit.download.throttle.Governor` retires a permit,
so the pressure actually comes down instead of the same sixteen workers arriving together
after the pause.

**A short chunk.** A body that ends early leaves a hole. The chunk is not recorded, the
progress it claimed is taken back, and it is fetched again.

**A checksum that does not match.** The part file and its record are both deleted. Leaving
them would mean the next run resumed into the same wrong bytes forever.
"""

from __future__ import annotations

import random
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from llamafit.download.errors import (
    ChecksumError,
    DownloadCancelledError,
    DownloadError,
    RangeNotSupportedError,
    RateLimitedError,
)
from llamafit.download.plan import DownloadPlan, FileRequest
from llamafit.download.progress import NullProgress, ProgressReporter
from llamafit.download.state import DEFAULT_CHUNK_BYTES, PartFile, open_part
from llamafit.download.throttle import Governor, RateLimiter
from llamafit.download.transport import RangeReader, RangeResponse
from llamafit.download.verify import check_sha256, check_size, sha256_of
from llamafit.errors import NetworkError
from llamafit.i18n import _
from llamafit.units import format_bytes

STOP_POLL_SECONDS = 0.2
"""How often a waiting worker looks up to see whether the user has stopped the run."""

DEFAULT_WORKERS = 8
"""How many requests are in flight by default.

The specification's range is sixteen to thirty-two, and ``--workers`` reaches all of it.
Eight is the default anyway, because the machine this runs on is somebody's own and the
line it is using is shared with whoever else is in the house. Sixteen streams from a
content network will take every bit of a domestic connection, and the person who typed the
command did not ask for their video call to stop working. Someone who wants the line
saturated can say so; someone who does not should not have to find out that they should
have.
"""

MAX_WORKERS = 32
"""The ceiling the specification names. More connections stop buying throughput and
start looking, from the far end, like something worth blocking."""

RETRYABLE_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
"""Statuses worth trying again: a timeout, a too-early, a rate limit, a server wobble."""

RATE_LIMIT_STATUSES = frozenset({429, 503})
"""Statuses that mean "you are asking too often", which lower the ceiling as well as wait."""

MAX_BACKOFF_SECONDS = 60.0
"""However many times a transfer has failed, it is retried within the minute."""


@dataclass(frozen=True)
class DownloadOptions:
    """How to fetch: how many at once, how big a bite, how patiently, how politely.

    Attributes:
        workers: How many requests may be in flight, capped at :data:`MAX_WORKERS`.
        chunk_bytes: How much one request asks for.
        limit_rate: A ceiling in bytes per second across every worker, or ``None``.
        max_attempts: How many times one chunk is tried before the file fails.
        backoff_seconds: The first delay after a failure; it doubles, with jitter.
        recheck: Re-hash files that are already on disk instead of trusting their size.
    """

    workers: int = DEFAULT_WORKERS
    chunk_bytes: int = DEFAULT_CHUNK_BYTES
    limit_rate: int | None = None
    max_attempts: int = 5
    backoff_seconds: float = 1.0
    recheck: bool = False

    def effective_workers(self) -> int:
        """The worker count actually used, held between one and :data:`MAX_WORKERS`."""
        return max(1, min(self.workers, MAX_WORKERS))


@dataclass(frozen=True)
class FileOutcome:
    """What happened to one file.

    Attributes:
        name: The file's bare name.
        path: Where it now is.
        size: Its size in bytes.
        fetched: How many bytes this run pulled over the network, which is zero for a
            file that was already here and less than ``size`` for a resumed one.
        resumed: Whether this run continued a transfer an earlier one had started.
        already_present: Whether nothing had to be fetched at all.
        verified: Whether a checksum was checked and matched.
        sha256: The digest that matched, when one was computed.
    """

    name: str
    path: Path
    size: int
    fetched: int
    resumed: bool
    already_present: bool
    verified: bool
    sha256: str | None = None


class _Tally:
    """The running byte count for one file, so a restart can rewind the bar exactly."""

    def __init__(self, reporter: ProgressReporter, name: str, start: int) -> None:
        self.reporter = reporter
        self.name = name
        self.value = start
        self.fetched = 0

    def add(self, count: int) -> None:
        """Move the bar by ``count``, which is negative when an attempt is unwound."""
        self.value += count
        if count > 0:
            self.fetched += count
        self.reporter.bytes_received(self.name, count)

    def rewind_to(self, value: int) -> None:
        """Take the bar back to ``value``, after a restart threw work away."""
        if value != self.value:
            self.add(value - self.value)


class _Stop:
    """Cancellation from either direction: the user, or the first chunk that failed."""

    def __init__(self, outer: threading.Event) -> None:
        self.outer = outer
        self.inner = threading.Event()

    def is_set(self) -> bool:
        """Whether anything has asked for this to stop."""
        return self.outer.is_set() or self.inner.is_set()

    def fail(self) -> None:
        """Ask the other workers on this file to stop, because one of them failed."""
        self.inner.set()

    def cancelled_by_user(self) -> bool:
        """Whether it was the user who stopped it, rather than a failure."""
        return self.outer.is_set()

    def wait(self, seconds: float) -> None:
        """Sleep for ``seconds``, but come back early if anything asks to stop.

        Two events cannot be waited on at once, so the shorter one is waited on and the
        longer one polled. A fifth of a second is far below what anybody notices and far
        above what the polling costs.
        """
        deadline = time.monotonic() + seconds
        while not self.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.inner.wait(min(remaining, STOP_POLL_SECONDS))


def _retry_after(response: RangeResponse) -> float | None:
    """The delay the server asked for, in seconds, when it asked in seconds."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        # The header also allows an HTTP date. Parsing one to find out we should wait
        # roughly as long as our own backoff already says is not worth the code.
        return None
    return max(0.0, min(seconds, MAX_BACKOFF_SECONDS))


def _content_range_total(response: RangeResponse) -> int | None:
    """The file's total size out of a ``Content-Range`` header, when it is there."""
    value = response.headers.get("content-range", "")
    total = value.rsplit("/", 1)[-1] if "/" in value else ""
    return int(total) if total.isdigit() else None


def _classify(response: RangeResponse, *, name: str, url: str, whole_file: bool) -> None:
    """Turn a response status into nothing at all, or into the right kind of failure.

    Raises:
        RangeNotSupportedError: The server answered a partial request with a full body.
        RateLimitedError: The server asked us to slow down.
        NetworkError: Something transient; the caller retries.
        DownloadError: Something that will not get better by trying again.
    """
    status = response.status
    if status == 206 or (status == 200 and whole_file):
        return
    if status == 200:
        raise RangeNotSupportedError(
            _("%(url)s answered a range request with the whole file.") % {"url": url}
        )
    if status in RATE_LIMIT_STATUSES:
        raise RateLimitedError(
            _("%(url)s is rate-limiting this download (HTTP %(status)d).")
            % {"url": url, "status": status},
            retry_after=_retry_after(response),
        )
    if status in RETRYABLE_STATUSES:
        raise NetworkError(
            _("%(url)s returned HTTP %(status)d.") % {"url": url, "status": status},
            hint=_("Check your network connection and run the command again to resume."),
        )
    if status in (401, 403):
        raise DownloadError(
            _("%(file)s needs authorisation (HTTP %(status)d).") % {"file": name, "status": status},
            hint=_(
                "This repository is gated. Accept its terms on Hugging Face and set "
                "HF_TOKEN to a token that can read it."
            ),
        )
    if status == 404:
        raise DownloadError(
            _("%(file)s is not at %(url)s any more (HTTP 404).") % {"file": name, "url": url},
            hint=_("Run `llamafit catalog refresh` to pick up the repository's current files."),
        )
    raise DownloadError(
        _("%(url)s returned HTTP %(status)d.") % {"url": url, "status": status},
        hint=_("Confirm the URL points at a downloadable file."),
    )


def probe_size(
    reader: RangeReader, request: FileRequest, *, options: DownloadOptions | None = None
) -> int:
    """Ask the server how big the file actually is, with a one-byte range request.

    The catalog records one size for a whole quantisation and none at all for shard three
    of four, so for a split model this is the only honest source of the number a part file
    has to be created at. It costs one tiny request per file, it happens before anything is
    allocated, and it follows the redirect to the content network once so the workers do
    not each discover it separately.

    A ``200`` here is not yet a failure: it means the server ignored the range, which is
    worth knowing but not worth stopping for, since ``Content-Length`` still says how big
    the file is and :func:`download_file` has a sequential path for exactly this server.

    Args:
        reader: Where bytes come from.
        request: The file being sized.
        options: How patiently to retry a server that is busy or rate-limiting.

    Returns:
        The file's size in bytes.

    Raises:
        DownloadError: If the server will not say, or says something that contradicts a
            size the catalog stated outright.
        NetworkError: If the request kept failing.
    """
    options = options or DownloadOptions()
    size = _probe_with_retries(reader, request, options)
    if size is None:
        raise DownloadError(
            _("%(url)s did not say how big %(file)s is.")
            % {"url": request.url, "file": request.name},
            hint=_("Confirm the URL points at a downloadable file."),
        )
    if request.size is not None and request.size != size:
        raise DownloadError(
            _(
                "%(file)s is %(actual)s on the server but the catalog says %(expected)s; "
                "the repository has been re-uploaded since the catalog was refreshed."
            )
            % {
                "file": request.name,
                "actual": format_bytes(size),
                "expected": format_bytes(request.size),
            },
            hint=_("Run `llamafit catalog refresh` and try again."),
        )
    return size


def _probe_with_retries(
    reader: RangeReader, request: FileRequest, options: DownloadOptions
) -> int | None:
    """One tiny range request, retried while the server is busy or rate-limiting."""
    for attempt_number in range(1, options.max_attempts + 1):
        try:
            with reader.open(request.url, 0, 0) as response:
                _classify(response, name=request.name, url=request.url, whole_file=True)
                size = _content_range_total(response)
                if size is None:
                    length = response.headers.get("content-length")
                    if length is not None and length.isdigit() and response.status == 200:
                        size = int(length)
            return size
        except (RateLimitedError, NetworkError) as exc:
            if attempt_number >= options.max_attempts:
                raise
            time.sleep(_backoff_delay(exc, options, attempt_number))
    return None


@dataclass
class _Attempt:
    """How much one try at one chunk managed to write before it went wrong."""

    written: int = 0


def _write_chunk(
    part: PartFile,
    index: int,
    reader: RangeReader,
    limiter: RateLimiter,
    tally: _Tally,
    stop: _Stop,
    attempt: _Attempt,
    *,
    name: str,
    whole_file: bool,
) -> None:
    """One try at one chunk: request the range, write it, and insist on its full length.

    Raises:
        DownloadCancelledError: If anything asked to stop mid-stream.
        NetworkError: If the body ended early, which is a transfer to try again.
        DownloadError: For any other refusal; see :func:`_classify`.
    """
    start, end = part.bounds(index)
    expected = part.length_of(index)
    with reader.open(part.url, start, end) as response:
        _classify(response, name=name, url=part.url, whole_file=whole_file)
        with part.path.open("r+b") as handle:
            handle.seek(start)
            for piece in response.body:
                if stop.is_set():
                    raise DownloadCancelledError(_("Stopped."))
                # A server that sends more than the range it was asked for would write
                # over the next chunk's slot, which nothing downstream would notice.
                room = expected - attempt.written
                block = piece if len(piece) <= room else piece[:room]
                if not block:
                    break
                limiter.take(len(block))
                handle.write(block)
                attempt.written += len(block)
                tally.add(len(block))
            handle.flush()
    if attempt.written != expected:
        raise NetworkError(
            _("%(file)s: the server sent %(actual)d bytes of a %(expected)d byte range.")
            % {"file": name, "actual": attempt.written, "expected": expected},
            hint=_("Run the command again; the parts that did arrive are kept."),
        )


def _fetch_chunk(
    part: PartFile,
    index: int,
    reader: RangeReader,
    options: DownloadOptions,
    limiter: RateLimiter,
    governor: Governor,
    reporter: ProgressReporter,
    tally: _Tally,
    stop: _Stop,
    *,
    name: str,
    whole_file: bool,
) -> None:
    """Fetch one chunk, retrying a transient failure and backing off a rate limit.

    Raises:
        DownloadCancelledError: If the user stopped the run.
        RangeNotSupportedError: If the server will not do ranges; the file driver handles it.
        DownloadError: If the chunk could not be fetched within ``max_attempts``.
    """
    for attempt_number in range(1, options.max_attempts + 1):
        if stop.is_set():
            raise DownloadCancelledError(_("Stopped."))
        attempt = _Attempt()
        governor.acquire()
        try:
            _write_chunk(
                part,
                index,
                reader,
                limiter,
                tally,
                stop,
                attempt,
                name=name,
                whole_file=whole_file,
            )
        except (RateLimitedError, NetworkError) as exc:
            tally.add(-attempt.written)
            if isinstance(exc, RateLimitedError) and governor.back_off():
                reporter.note(
                    _("%(file)s: the server asked for fewer connections; using %(count)d.")
                    % {"file": name, "count": governor.capacity}
                )
            if attempt_number >= options.max_attempts:
                raise
            delay = _backoff_delay(exc, options, attempt_number)
            stop.wait(delay)
            continue
        except Exception:
            tally.add(-attempt.written)
            raise
        finally:
            governor.release()
        part.mark_done(index)
        return


def _backoff_delay(exc: BaseException, options: DownloadOptions, attempt_number: int) -> float:
    """How long to wait before trying a chunk again.

    The server's own ``Retry-After`` wins when it sent one; otherwise the delay doubles
    with each attempt. The jitter is not decoration: without it, sixteen workers that were
    all refused at the same moment come back at the same moment too.
    """
    if isinstance(exc, RateLimitedError) and exc.retry_after is not None:
        return exc.retry_after
    growth = 2.0 ** (attempt_number - 1)
    base = min(options.backoff_seconds * growth, MAX_BACKOFF_SECONDS)
    return base * (0.5 + random.random() / 2)


def _run_chunks(
    part: PartFile,
    reader: RangeReader,
    options: DownloadOptions,
    limiter: RateLimiter,
    governor: Governor,
    reporter: ProgressReporter,
    tally: _Tally,
    outer_stop: threading.Event,
    *,
    name: str,
    workers: int,
    whole_file: bool,
) -> None:
    """Fetch every chunk that is not already recorded, in parallel.

    The first failure stops the rest rather than letting fifteen more workers spend a
    minute each discovering the same thing. A cancellation is only reported as one when
    nothing else went wrong, so a real failure is never hidden behind the stop it caused.

    Raises:
        DownloadError: Whatever the first failing chunk raised.
    """
    pending = part.pending()
    if not pending:
        return
    stop = _Stop(outer_stop)
    failures: list[BaseException] = []
    pool_size = max(1, min(workers, len(pending)))
    with ThreadPoolExecutor(max_workers=pool_size, thread_name_prefix="llamafit-dl") as pool:
        futures: list[Future[None]] = [
            pool.submit(
                _fetch_chunk,
                part,
                index,
                reader,
                options,
                limiter,
                governor,
                reporter,
                tally,
                stop,
                name=name,
                whole_file=whole_file,
            )
            for index in pending
        ]
        for future in as_completed(futures):
            failure = future.exception()
            if failure is not None:
                failures.append(failure)
                stop.fail()
    if failures:
        raise _best_failure(failures)


def _best_failure(failures: Sequence[BaseException]) -> BaseException:
    """The failure worth reporting: a real one over the cancellations it caused.

    Fifteen workers stopping because a sixteenth failed produce fifteen cancellations and
    one useful sentence, and it is the sentence a reader needs.
    """
    for failure in failures:
        if not isinstance(failure, DownloadCancelledError):
            return failure
    return failures[0]


def download_file(
    request: FileRequest,
    reader: RangeReader,
    *,
    options: DownloadOptions | None = None,
    reporter: ProgressReporter | None = None,
    limiter: RateLimiter | None = None,
    stop: threading.Event | None = None,
) -> FileOutcome:
    """Fetch one file, resume it if it was started, and verify it before it is kept.

    Args:
        request: What to fetch and where to put it.
        reader: Where bytes come from.
        options: How many workers, how big a chunk, how patient to be.
        reporter: Who to tell about progress.
        limiter: A shared rate limit, built from ``options`` when not given.
        stop: Set to cancel; whatever has arrived stays on disk.

    Returns:
        What happened to the file.

    Raises:
        ChecksumError: If the file arrived whole but is not the file the catalog
            describes. Nothing is left behind for a later run to resume into.
        DownloadCancelledError: If ``stop`` was set.
        DownloadError: For anything else that went wrong.
    """
    options = options or DownloadOptions()
    reporter = reporter or NullProgress()
    limiter = limiter or RateLimiter(options.limit_rate)
    stop = stop or threading.Event()
    governor = Governor(options.effective_workers())

    if request.already_present() and not options.recheck:
        size = request.expected_size or 0
        reporter.file_started(request.name, size, size)
        reporter.file_finished(request.name)
        return FileOutcome(
            name=request.name,
            path=request.target,
            size=size,
            fetched=0,
            resumed=False,
            already_present=True,
            verified=False,
        )
    if request.already_present() and options.recheck:
        outcome = _recheck(request, reporter)
        if outcome is not None:
            return outcome

    size = probe_size(reader, request, options=options)
    part = open_part(request.target, request.url, size, chunk_size=options.chunk_bytes)
    part.allocate()
    already = part.bytes_done()
    resumed = already > 0
    reporter.file_started(request.name, size, already)
    tally = _Tally(reporter, request.name, already)

    try:
        _run_chunks(
            part,
            reader,
            options,
            limiter,
            governor,
            reporter,
            tally,
            stop,
            name=request.name,
            workers=options.effective_workers(),
            whole_file=part.chunk_count == 1,
        )
    except RangeNotSupportedError:
        reporter.note(
            _("%(file)s: this server will not serve byte ranges; fetching it in one stream.")
            % {"file": request.name}
        )
        part.discard()
        part = open_part(request.target, request.url, size, chunk_size=max(size, 1))
        part.allocate()
        tally.rewind_to(0)
        _run_chunks(
            part,
            reader,
            options,
            limiter,
            governor,
            reporter,
            tally,
            stop,
            name=request.name,
            workers=1,
            whole_file=True,
        )

    digest = _verify(request, part, reporter, stop, size)
    part.finish()
    reporter.file_finished(request.name)
    return FileOutcome(
        name=request.name,
        path=request.target,
        size=size,
        fetched=tally.fetched,
        resumed=resumed,
        already_present=False,
        verified=digest is not None,
        sha256=digest,
    )


def _recheck(request: FileRequest, reporter: ProgressReporter) -> FileOutcome | None:
    """Re-hash a file that is already here; return its outcome, or ``None`` to refetch."""
    size = request.expected_size or 0
    if request.sha256 is None:
        reporter.file_started(request.name, size, size)
        reporter.file_finished(request.name)
        return FileOutcome(
            name=request.name,
            path=request.target,
            size=size,
            fetched=0,
            resumed=False,
            already_present=True,
            verified=False,
        )
    reporter.file_started(request.name, size, 0)
    reporter.file_verifying(request.name, size)
    digest = sha256_of(
        request.target, on_bytes=lambda count: reporter.bytes_received(request.name, count)
    )
    if digest.lower() == request.sha256.lower():
        reporter.file_finished(request.name)
        return FileOutcome(
            name=request.name,
            path=request.target,
            size=size,
            fetched=0,
            resumed=False,
            already_present=True,
            verified=True,
            sha256=digest,
        )
    reporter.note(
        _("%(file)s does not match its checksum; fetching it again.") % {"file": request.name}
    )
    request.target.unlink(missing_ok=True)
    return None


def _verify(
    request: FileRequest,
    part: PartFile,
    reporter: ProgressReporter,
    stop: threading.Event,
    size: int,
) -> str | None:
    """Check the finished part file, deleting it outright when it is not the right file.

    Raises:
        ChecksumError: If the size or the digest is wrong. The part file and its record
            are gone by then: a bad copy that stayed would be resumed into forever, and a
            user who ran the command again would get the same wrong file faster.
    """
    reporter.file_verifying(request.name, size)
    try:
        check_size(part.path, size, name=request.name)
    except ChecksumError:
        part.discard()
        raise
    if request.sha256 is None:
        return None
    digest = sha256_of(
        part.path,
        on_bytes=lambda count: reporter.bytes_received(request.name, count),
        should_stop=stop.is_set,
    )
    try:
        check_sha256(part.path, request.sha256, name=request.name, actual=digest)
    except ChecksumError:
        part.discard()
        raise
    return digest


def download_plan(
    plan: DownloadPlan,
    reader: RangeReader,
    *,
    options: DownloadOptions | None = None,
    reporter: ProgressReporter | None = None,
    stop: threading.Event | None = None,
) -> list[FileOutcome]:
    """Fetch every file in a plan, in order, and hand back what happened to each.

    Files are taken one at a time and the workers are spent on the chunks of the file in
    hand. A model's shards are tens of gigabytes each; there is nothing to gain from
    having four of them half-finished at once, and something real to lose, since a run
    stopped halfway then has four part files instead of one.

    Args:
        plan: What to fetch.
        reader: Where bytes come from.
        options: How to fetch it.
        reporter: Who to tell.
        stop: Set to cancel.

    Returns:
        One outcome per file, in plan order.

    Raises:
        DownloadError: As soon as any file fails. The files already finished stay where
            they are, verified, and the next run picks up from there.
    """
    options = options or DownloadOptions()
    limiter = RateLimiter(options.limit_rate)
    outcomes: list[FileOutcome] = []
    for request in plan.files:
        outcomes.append(
            download_file(
                request,
                reader,
                options=options,
                reporter=reporter,
                limiter=limiter,
                stop=stop,
            )
        )
    return outcomes
