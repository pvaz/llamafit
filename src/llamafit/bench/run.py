# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Running the benchmark: ``llama-bench``, then a real server, then what it all cost.

Section 16.1, in order. ``llama-bench`` at the planned flags for a prompt figure and a
generation figure; then ``llama-server`` started with the same placement and asked three
fixed questions; then the VRAM the card was holding while it happened and the buffer sizes
the server wrote to its log.

Everything external goes through a protocol, the way ``hardware/runner.py`` and
``llamacpp/server.py`` already do it: a command runner, a server launcher, an HTTP client
and a VRAM sampler. The fakes for all four live here beside the real ones, so a test can
replay a recorded machine without a card, and so the shape a test substitutes is the shape
the real code was written against rather than something adjacent to it.

Two things about the payloads are worth saying out loud.

**The three requests are fixed and are never translated.** A benchmark whose prompt changed
with the reader's language would produce numbers that could not be compared with anybody
else's, or with the same machine's own from last month. These strings are inputs to a
program, the way the flags are.

**The speeds come from the server's own ``timings`` block, not from a stopwatch around the
call.** A stopwatch measures the network, the JSON and whatever else the machine was doing;
the server counts the tokens it evaluated and the milliseconds it spent on them. Same rule
as everywhere else here: ask the thing that did the work what it did.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import httpx

from llamafit.bench.fingerprint import (
    conditions_hash,
    flags_from_argv,
    host_fingerprint,
    host_summary,
)
from llamafit.bench.paging import detect_paging
from llamafit.bench.parse import (
    BenchRow,
    parse_buffer_sizes,
    parse_llama_bench_json,
    parse_llama_bench_markdown,
    parse_used_vram,
)
from llamafit.bench.store import BenchStore
from llamafit.bench.types import (
    BENCH_SCHEMA_VERSION,
    BenchKind,
    BenchReport,
    BenchRun,
    PagingCheck,
    RunConditions,
    RunTraffic,
    measured_context,
)
from llamafit.constants import EFF_PCIE, KV_TYPE_DEFAULT, SHARED_EXPERT_OVERRIDE
from llamafit.errors import ProbeError
from llamafit.hardware.runner import Runner
from llamafit.i18n import _
from llamafit.models.gguf import GgufFacts
from llamafit.models.host import Host
from llamafit.models.plan import Placement
from llamafit.services.plan import PlanReport
from llamafit.speed import (
    expert_bytes_in_ram,
    formula_estimate,
    per_token_traffic,
    resolve_bandwidths,
)
from llamafit.speed.traffic import streamed_expert_fraction

SHORT_PROMPT = "Write one short paragraph about why local inference is useful."
"""The first request: a small prompt, a real generation, on a server that has just started.

Deliberately the first thing asked. The calibration record shows this request coming in at
eight tokens per second where the next one manages fourteen, because a large model's data
is still arriving from disk while it runs. That gap is worth measuring rather than warming
away, and it is filed under its own kind so nothing averages it with the warm figure.
"""

_FILLER = (
    "The memory system reads every weight a token touches, once, from wherever the "
    "placement put it, and the arithmetic on top of those bytes is small enough to hide "
    "behind them. "
)

TOOL_SCHEMA: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
"""The third request. A tool call either comes back well-formed or it does not.

It is not a speed measurement. It is the one thing in the benchmark that answers a question
speed cannot: whether this model, at this quantisation, with this chat template, is
actually usable for the job somebody is about to point it at.
"""


def long_prompt(words: int = 1000) -> str:
    """A prompt of about ``words`` words, the same every time it is asked for.

    Args:
        words: Roughly how many words to produce.

    Returns:
        The prompt. Built by repetition from one sentence rather than from a corpus,
        because the number that matters is the token count and a benchmark that quietly
        changed its own input would produce a series nobody could read.
    """
    per_copy = len(_FILLER.split())
    return (_FILLER * max(1, words // per_copy)).strip()


class ServerHandle(Protocol):
    """A ``llama-server`` that has been started and can be stopped and read."""

    def is_running(self) -> bool:
        """Whether the process is still alive."""
        ...

    def stop(self) -> None:
        """Stop it, without raising if it has already gone."""
        ...

    def log(self) -> str:
        """Everything it has written so far."""
        ...


class ServerLauncher(Protocol):
    """Something that can start a ``llama-server`` and hand back a handle to it."""

    def start(self, argv: Sequence[str]) -> ServerHandle:
        """Start the server described by ``argv``."""
        ...


@dataclass
class _SubprocessHandle:
    """A real server process, with its output going to a file so it can be read back."""

    process: subprocess.Popen[bytes]
    log_path: Path

    def is_running(self) -> bool:
        """Whether the process has not exited yet."""
        return self.process.poll() is None

    def stop(self) -> None:
        """Ask it to stop, then insist; a server that has already gone is not an error."""
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:  # a server wedged on shutdown must not wedge us
            self.process.kill()
            self.process.wait(timeout=15)

    def log(self) -> str:
        """Everything written to the log file so far, or nothing if it cannot be read."""
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:  # the file may not exist if the process never started
            return ""


@dataclass
class SubprocessServerLauncher:
    """Starts a real ``llama-server``, with its whole output captured to ``log_path``.

    Attributes:
        log_path: Where the server's output goes. It is the only place the buffer sizes
            appear, so it is captured rather than discarded even when nobody asks for it.
    """

    log_path: Path

    def start(self, argv: Sequence[str]) -> ServerHandle:
        """Start the server, writing everything it says into the log file.

        Raises:
            ProbeError: If the process could not be started at all.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            sink = self.log_path.open("wb")
        except OSError as exc:
            raise ProbeError(
                _("could not open %(path)s for the server log") % {"path": str(self.log_path)},
                hint=str(exc),
            ) from exc
        try:
            process = subprocess.Popen(list(argv), stdout=sink, stderr=subprocess.STDOUT)
        except OSError as exc:
            sink.close()
            raise ProbeError(
                _("could not start llama-server"),
                hint=str(exc),
                command=" ".join(argv),
            ) from exc
        return _SubprocessHandle(process=process, log_path=self.log_path)


@dataclass
class _FakeHandle:
    """A handle over a recorded log, for tests."""

    text: str
    alive: bool = True
    stopped: bool = False

    def is_running(self) -> bool:
        """Whether the recorded server is meant to look alive."""
        return self.alive and not self.stopped

    def stop(self) -> None:
        """Mark it stopped."""
        self.stopped = True

    def log(self) -> str:
        """The recorded log."""
        return self.text


@dataclass
class FakeServerLauncher:
    """A launcher that starts nothing and replays a recorded server log.

    Attributes:
        log_text: What the recorded server wrote.
        started: Every command line it was asked to start, for assertions.
        alive: Whether the handle should claim the process is running.
    """

    log_text: str = ""
    started: list[list[str]] = field(default_factory=list)
    alive: bool = True

    def start(self, argv: Sequence[str]) -> ServerHandle:
        """Record the command line and hand back a handle over the recorded log."""
        self.started.append(list(argv))
        return _FakeHandle(text=self.log_text, alive=self.alive)


class BenchHttp(Protocol):
    """The two HTTP calls a benchmark makes, neither of which may raise."""

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Fetch ``url`` and parse JSON; ``None`` for any failure."""
        ...

    def post_json(
        self, url: str, payload: Mapping[str, Any], *, timeout: float = 120.0
    ) -> Any | None:
        """Post ``payload`` to ``url`` and parse JSON; ``None`` for any failure."""
        ...


class HttpxBenchClient:
    """The real client. A refused connection or a bad status is ``None``, never an exception."""

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Fetch ``url``; ``None`` for network errors, non-200 or invalid JSON."""
        try:
            response = httpx.get(url, timeout=timeout)
            if response.status_code != 200:
                return None
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None

    def post_json(
        self, url: str, payload: Mapping[str, Any], *, timeout: float = 120.0
    ) -> Any | None:
        """Post ``payload``; ``None`` for network errors, non-200 or invalid JSON."""
        try:
            response = httpx.post(url, json=dict(payload), timeout=timeout)
            if response.status_code != 200:
                return None
            return response.json()
        except (httpx.HTTPError, ValueError):
            return None


@dataclass
class FakeBenchHttp:
    """Canned responses keyed by URL; posts are answered from a queue per URL.

    Attributes:
        gets: What each URL returns for a ``GET``.
        posts: A list of responses per URL, consumed in order, so the three fixed requests
            can each be given a different recorded answer.
        calls: Every call made, for assertions.
    """

    gets: dict[str, Any] = field(default_factory=dict)
    posts: dict[str, list[Any]] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)

    def get_json(self, url: str, *, timeout: float = 1.5) -> Any | None:
        """Return the canned response for ``url``."""
        self.calls.append(("GET", url))
        return self.gets.get(url)

    def post_json(
        self, url: str, payload: Mapping[str, Any], *, timeout: float = 120.0
    ) -> Any | None:
        """Return the next canned response for ``url``, or ``None`` when they run out."""
        self.calls.append(("POST", url))
        queue = self.posts.get(url)
        if not queue:
            return None
        return queue.pop(0)


class VramSampler(Protocol):
    """Something that can say how much VRAM the card is holding right now."""

    def sample(self) -> int | None:
        """The reading in bytes, or ``None`` when nothing answered."""
        ...


@dataclass
class NvidiaSmiSampler:
    """Reads VRAM in use from ``nvidia-smi``.

    Attributes:
        runner: The command runner.
        index: Which card to ask about.
    """

    runner: Runner
    index: int = 0

    def sample(self) -> int | None:
        """One reading, or ``None`` when the tool is absent or answered ``[N/A]``."""
        result = self.runner.run(
            [
                "nvidia-smi",
                f"--id={self.index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            timeout=5.0,
        )
        if not result.ok:
            return None
        return parse_used_vram(result.stdout)


@dataclass
class FakeVramSampler:
    """Replays recorded readings, then repeats the last one.

    Attributes:
        readings: The readings in order, in bytes. An empty list samples ``None``, which
            is what a machine with no vendor tool does.
        taken: How many readings have been handed out.
    """

    readings: list[int] = field(default_factory=list)
    taken: int = 0

    def sample(self) -> int | None:
        """The next recorded reading."""
        if not self.readings:
            return None
        index = min(self.taken, len(self.readings) - 1)
        self.taken += 1
        return self.readings[index]


@dataclass(frozen=True)
class BenchOptions:
    """What one invocation of the benchmark should do.

    Attributes:
        n_prompt: Prompt tokens for the ``llama-bench`` prompt row.
        n_gen: Generated tokens for the ``llama-bench`` generation row.
        repetitions: How many times ``llama-bench`` repeats each row before taking its own
            median. Three, from ``docs/benchmarking.md``.
        micro_batches: Extra micro-batches to sweep, which is what makes the prompt half of
            the calibration identifiable at all: section 10.2 has two free parameters and
            one micro-batch is one equation.
        with_server: Whether to start a real server and make the three fixed requests.
        bench_timeout_s: How long ``llama-bench`` may take.
        health_timeout_s: How long to wait for the server to answer ``/health``. Loading a
            hundred gigabytes off a disk is minutes, not seconds.
        request_timeout_s: How long one request may take.
    """

    n_prompt: int = 2048
    n_gen: int = 128
    repetitions: int = 3
    micro_batches: tuple[int, ...] = ()
    with_server: bool = True
    bench_timeout_s: float = 1800.0
    health_timeout_s: float = 900.0
    request_timeout_s: float = 600.0


def llama_bench_argv(
    placement: Placement,
    *,
    executable: str,
    model_path: str,
    options: BenchOptions,
) -> list[str]:
    """The ``llama-bench`` command line for one placement.

    Args:
        placement: Where the bytes go.
        executable: The ``llama-bench`` binary.
        model_path: The GGUF file, or the first shard of a split one.
        options: What to measure.

    Returns:
        The whole command, program name first.

    Every setting the placement carries is named, including ones ``llama-bench`` would
    default to anyway. That is not tidiness. A benchmark's stored flags are what a later
    reader matches against a placement; a flag the record does not mention cannot disagree
    with anything; and the one gap this project has already found came from exactly that. A
    record that says ``-ngl 99`` and nothing about ``--n-cpu-moe`` matches every offload
    with that layer count, including the ones it describes nothing about.
    """
    batches = ",".join(str(size) for size in (placement.micro_batch, *options.micro_batches))
    args = [
        executable,
        "-m",
        model_path,
        "-p",
        str(options.n_prompt),
        "-n",
        str(options.n_gen),
        "-ngl",
        str(placement.gpu_layers),
    ]
    if placement.cpu_moe_layers is not None:
        args += ["--n-cpu-moe", str(placement.cpu_moe_layers)]
    if placement.shared_experts_pool == "ram":
        args += ["-ot", SHARED_EXPERT_OVERRIDE]
    args += ["-ub", batches, "-b", str(placement.batch), "-t", str(placement.threads)]
    if placement.kv_type != KV_TYPE_DEFAULT:
        args += ["-ctk", placement.kv_type, "-ctv", placement.kv_type]
    args += ["-fa", "1", "-r", str(options.repetitions), "-o", "json"]
    return args


def run_llama_bench(runner: Runner, argv: Sequence[str], *, timeout: float) -> list[BenchRow]:
    """Run ``llama-bench`` and read its rows.

    Args:
        runner: The command runner.
        argv: The command line.
        timeout: How long it may take.

    Returns:
        One row per result the tool produced.

    Raises:
        ProbeError: If the tool could not run, exited non-zero, or produced output no row
            could be read from. A benchmark that produced no number did not happen, and the
            alternative to raising is storing a row with nothing in it.
    """
    result = runner.run(list(argv), timeout=timeout)
    if result.error is not None:
        raise ProbeError(result.error, command=" ".join(argv))
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or _("no output")
        raise ProbeError(
            _("llama-bench exited with code %(code)d: %(detail)s")
            % {"code": result.returncode, "detail": detail},
            command=" ".join(argv),
        )
    try:
        return parse_llama_bench_json(result.stdout)
    except ValueError:
        # A build too old for `-o json` prints the table instead, and the table is still a
        # measurement. What it is not is a complete record of itself, which is why JSON is
        # asked for first and why a row read from a table says less about its own settings.
        try:
            return parse_llama_bench_markdown(result.stdout)
        except ValueError as exc:
            raise ProbeError(
                _("llama-bench output could not be read: %(error)s") % {"error": exc},
                command=" ".join(argv),
            ) from exc


@dataclass(frozen=True)
class RequestResult:
    """What one fixed request to a running server produced.

    Attributes:
        kind: Which of the three it was.
        gen_tps: Generated tokens per second, from the server's own timings.
        pp_tps: Prompt tokens per second, likewise.
        ttft_ms: Time to first token, taken as the milliseconds spent on the prompt.
        n_prompt: Prompt tokens the server said it evaluated.
        n_gen: Tokens it said it predicted.
        tool_call_ok: Whether a tool call came back well-formed, for that request only.
    """

    kind: BenchKind
    gen_tps: float | None = None
    pp_tps: float | None = None
    ttft_ms: float | None = None
    n_prompt: int | None = None
    n_gen: int | None = None
    tool_call_ok: bool | None = None


def wait_for_health(
    http: BenchHttp,
    base_url: str,
    *,
    timeout: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    handle: ServerHandle | None = None,
) -> bool:
    """Wait until the server answers ``/health``, or give up.

    Args:
        http: The HTTP client.
        base_url: The server's address.
        timeout: How long to wait.
        sleep: How to wait between attempts.
        clock: Where the time comes from.
        handle: The process, so a server that has already exited is not waited for.

    Returns:
        True when it answered, false when the time ran out or the process died first.
    """
    deadline = clock() + timeout
    while clock() < deadline:
        if handle is not None and not handle.is_running():
            return False
        health = http.get_json(f"{base_url}/health", timeout=2.0)
        if isinstance(health, dict) and health.get("status") == "ok":
            return True
        sleep(0.5)
    return False


def _completion(
    http: BenchHttp, base_url: str, prompt: str, *, n_predict: int, kind: BenchKind, timeout: float
) -> RequestResult:
    """One ``/completion`` request, read out of the server's own ``timings`` block."""
    answer = http.post_json(
        f"{base_url}/completion",
        {"prompt": prompt, "n_predict": n_predict, "cache_prompt": False, "stream": False},
        timeout=timeout,
    )
    if not isinstance(answer, dict):
        return RequestResult(kind=kind)
    raw = answer.get("timings")
    timings: dict[str, Any] = raw if isinstance(raw, dict) else {}
    return RequestResult(
        kind=kind,
        gen_tps=_positive(timings.get("predicted_per_second")),
        pp_tps=_positive(timings.get("prompt_per_second")),
        ttft_ms=_positive(timings.get("prompt_ms")),
        n_prompt=_count(answer.get("tokens_evaluated")),
        n_gen=_count(answer.get("tokens_predicted")),
    )


def _tool_call(http: BenchHttp, base_url: str, *, timeout: float) -> RequestResult:
    """One tool-call request, answered yes or no rather than in tokens per second."""
    answer = http.post_json(
        f"{base_url}/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "What is the weather in Lisbon?"}],
            "tools": TOOL_SCHEMA,
            "tool_choice": "auto",
            "max_tokens": 128,
            "stream": False,
        },
        timeout=timeout,
    )
    return RequestResult(kind="server-toolcall", tool_call_ok=_tool_call_is_well_formed(answer))


def _tool_call_is_well_formed(answer: Any) -> bool:
    """Whether a chat response really carries a tool call somebody could act on.

    Not merely whether the server answered. A model that writes the call as prose in the
    message body has failed at the thing being tested, and a check that looked only for a
    200 would call that a success.
    """
    if not isinstance(answer, dict):
        return False
    choices = answer.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return False
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return False
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], dict):
        return False
    function = calls[0].get("function")
    if not isinstance(function, dict) or not function.get("name"):
        return False
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return True
    if not isinstance(arguments, str):
        return False
    try:
        return isinstance(json.loads(arguments), dict)
    except ValueError:
        return False


def _positive(value: object) -> float | None:
    """A JSON number when it is one and is above zero, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value > 0 else None


def _count(value: object) -> int | None:
    """A JSON token count when it is one, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if value >= 0 else None


def probe_server(
    http: BenchHttp,
    base_url: str,
    *,
    options: BenchOptions,
    sampler: VramSampler,
) -> tuple[list[RequestResult], int | None, int | None]:
    """Make the three fixed requests, sampling VRAM around them.

    Args:
        http: The HTTP client.
        base_url: The server's address.
        options: The timeouts.
        sampler: Where the VRAM readings come from.

    Returns:
        The three results, the highest VRAM reading taken, and the context the server said
        it was running at.

    The readings are taken before the first request and after each one. llama.cpp allocates
    its weights, its cache and its compute buffers while the model loads, so the peak is
    reached before any token is generated; sampling around the requests rather than during
    them costs nothing real and keeps this out of a second thread.
    """
    readings: list[int] = []

    def take() -> None:
        reading = sampler.sample()
        if reading is not None:
            readings.append(reading)

    take()
    results = [
        _completion(
            http,
            base_url,
            SHORT_PROMPT,
            n_predict=128,
            kind="server-short",
            timeout=options.request_timeout_s,
        )
    ]
    take()
    results.append(
        _completion(
            http,
            base_url,
            long_prompt(),
            n_predict=64,
            kind="server-1k",
            timeout=options.request_timeout_s,
        )
    )
    take()
    results.append(_tool_call(http, base_url, timeout=options.request_timeout_s))
    take()
    return results, (max(readings) if readings else None), _server_context(http, base_url)


def _server_context(http: BenchHttp, base_url: str) -> int | None:
    """The context the server says it is running at, which is not always the one asked for."""
    props = http.get_json(f"{base_url}/props", timeout=5.0)
    if not isinstance(props, dict):
        return None
    settings = props.get("default_generation_settings")
    if isinstance(settings, dict):
        return _count(settings.get("n_ctx"))
    return _count(props.get("n_ctx"))


def traffic_of(
    placement: Placement,
    facts: GgufFacts,
    host: Host,
    *,
    active_params: float | None,
    working_context: int,
    micro_batch: int,
) -> RunTraffic:
    """What the estimator believes one run reads, in the shape a result stores.

    Args:
        placement: Where the bytes go.
        facts: The file's derived facts.
        host: The machine.
        active_params: Parameters active per token, from the catalog.
        working_context: Tokens of key-value cache this run's generated tokens read
            against, which is what the run filled and not what the server allocated.
        micro_batch: The micro-batch this run's prompt figures belong to, which for one
            row of a ``--sweep`` is one of the several the command line asked for.

    Returns:
        The traffic, with the raw bandwidths it would be charged against.

    Recorded with the run rather than recomputed at calibration time. A placement rebuilt
    months later out of a stored command line, against a catalog and a set of GGUF facts
    that have both moved since, is the same "a record matched a configuration it did not
    describe" failure arriving through a different door.

    **The last two arguments have no defaults on purpose.** They used to be read off the
    placement, so every row of a benchmark was recorded as having read the key-value cache
    of a 32,768-token context it never filled, and every row of a micro-batch sweep was
    recorded at the planned micro-batch. The first made the generation comparison a
    measurement of context rather than of the formula; the second gave the prompt fit
    three rows whose only free column was identical, which is a rank-deficient system and
    the one thing ``--sweep`` exists to avoid. A caller has to say what the run did.
    """
    bandwidths = resolve_bandwidths(host)
    traffic = per_token_traffic(placement, facts, working_context=working_context)
    streamed = expert_bytes_in_ram(placement, facts) * streamed_expert_fraction(facts, micro_batch)
    return RunTraffic(
        device_bytes=traffic.device_bytes,
        sequential_bytes=traffic.sequential_bytes,
        scattered_bytes=traffic.scattered_bytes,
        ram_gbps=bandwidths.ram_gbps,
        device_gbps=bandwidths.device_gbps,
        streamed_expert_bytes=round(streamed),
        active_params=active_params or 0.0,
        compute_flops=bandwidths.compute_flops,
        # `EffectiveBandwidths.pcie` already carries section 10.2's efficiency; the fit
        # wants the rated link, because that efficiency is the thing being fitted.
        pcie_gbps=bandwidths.pcie / (EFF_PCIE * 1e9),
        micro_batch=micro_batch,
        working_context=working_context,
    )


@dataclass
class BenchInputs:
    """Everything :func:`run_benchmark` needs that is not injected machinery.

    Attributes:
        plan: The plan being benchmarked, which supplies the placement and the flags.
        host: The machine.
        facts: The file's derived facts, for the traffic record.
        active_params: Parameters active per token, from the catalog.
        bench_executable: The ``llama-bench`` binary.
        server_argv: The ``llama-server`` command line, exactly as ``plan`` rendered it.
        base_url: Where the server will answer.
        llama_cpp_build: The build number the installation reported for itself.
        llama_cpp_commit: The commit it reported.
    """

    plan: PlanReport
    host: Host
    facts: GgufFacts | None
    active_params: float | None
    bench_executable: str
    server_argv: Sequence[str]
    base_url: str
    llama_cpp_build: int | None = None
    llama_cpp_commit: str | None = None


@dataclass(frozen=True)
class Prediction:
    """What the formula says about one set of conditions, before anything has run.

    Attributes:
        context: Tokens of key-value cache the figures are for.
        micro_batch: The micro-batch the prompt figure is for.
        traffic: The bytes the estimator believes a token at that context reads.
        gen_tps: Generated tokens per second, or ``None`` when there is no estimate.
        pp_tps: Prompt tokens per second, likewise.
    """

    context: int
    micro_batch: int
    traffic: RunTraffic | None
    gen_tps: float | None
    pp_tps: float | None


@dataclass
class Estimator:
    """Section 10's formula, asked again for each set of conditions a run turns out to have.

    This class is the answer to the question the ``bench`` finding posed, and the question
    was which of three honest comparisons to make: run ``llama-bench`` at the planned
    context, estimate at the context ``llama-bench`` used, or print both figures and
    compare neither.

    **It estimates at the context each measurement reached, and it is not a compromise.**
    Section 10.1 charges ``kv_bytes_per_token x working_context`` for the cache a token
    reads, and a token reads the cache that exists, not the one that was allocated: the
    quantity in the formula is the depth the run filled. So the comparable estimate is the
    formula at that depth, and this project already works that way everywhere else:
    ``docs/calibration/`` gives "generation, short context, ``llama-bench tg128``" and
    "generation at 32K tokens of context" as two measurements on two lines, and the four
    reference runs the estimator's constants were identified from are stated at 128, 128,
    128 and about a thousand tokens against placements sized for 4,096 and 262,144.

    The alternative -- ``llama-bench -d 32768``, which prefills the cache before it times
    anything -- was rejected on three counts. It needs a build new enough to have the
    flag, and a benchmark that silently measures something else on an older one is the
    failure this package exists to refuse. It costs a full 32,768-token prefill per row,
    minutes on a large model, for a figure the same run already yields at a depth nobody
    waited for. And it cannot be done at all for the three server requests, which are
    fixed questions of fixed length and would have to be lengthened into something no
    longer comparable with last month's. What it would buy is the one thing this choice
    gives up: at 128 tokens the key-value term is a rounding error, so a benchmark here
    checks every term of section 10.1 *except* the growth of the cache. That term is worth
    checking and is not checked; :attr:`BenchReport.planned_gen_tps` is where the
    unchecked figure is at least shown.

    Answers are cached because a benchmark asks for a handful of distinct conditions and
    repeats them -- a micro-batch sweep runs ``tg128`` once per rung, all at 128 tokens --
    while each call rebuilds the whole traffic breakdown and every note that goes with it.
    """

    inputs: BenchInputs
    _answers: dict[tuple[int, int], Prediction] = field(default_factory=dict)

    def at(self, *, context: int | None, micro_batch: int) -> Prediction:
        """The estimate for one context and micro-batch, capped at the placement's own.

        Args:
            context: Tokens of key-value cache the run filled, or ``None`` when the tool
                reported no token counts and the depth is therefore unknown.
            micro_batch: The micro-batch the run used.

        Returns:
            The prediction. Empty -- no traffic, no figures -- when the file's facts are
            missing or the depth is unknown, because an estimate is a statement about
            conditions and a guess at the conditions is not one. An empty prediction leaves
            a measurement with no ratio, which is the honest shape of "not compared".
        """
        placement = self.inputs.plan.placement
        facts = self.inputs.facts
        if facts is None or context is None or placement.mode == "unsupported":
            return Prediction(
                context=context or 0,
                micro_batch=micro_batch,
                traffic=None,
                gen_tps=None,
                pp_tps=None,
            )
        # A run cannot fill more cache than the server allocated, and `estimate_speed`
        # clamps the same way, so the two agree about the planned context.
        key = (min(context, placement.context), micro_batch)
        answer = self._answers.get(key)
        if answer is None:
            depth, ubatch = key
            formula = formula_estimate(
                placement,
                facts,
                resolve_bandwidths(self.inputs.host),
                working_context=depth,
                micro_batch=ubatch,
                active_params=self.inputs.active_params,
            )
            answer = Prediction(
                context=depth,
                micro_batch=ubatch,
                traffic=traffic_of(
                    placement,
                    facts,
                    self.inputs.host,
                    active_params=self.inputs.active_params,
                    working_context=depth,
                    micro_batch=ubatch,
                ),
                gen_tps=formula.gen_tps or None,
                pp_tps=formula.pp_tps or None,
            )
            self._answers[key] = answer
        return answer


def run_benchmark(
    inputs: BenchInputs,
    *,
    runner: Runner,
    launcher: ServerLauncher,
    http: BenchHttp,
    sampler: VramSampler,
    options: BenchOptions | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> BenchReport:
    """Measure one plan, and return every result with the estimate it is judged against.

    Args:
        inputs: The plan, the machine and the two command lines.
        runner: Runs ``llama-bench`` and the VRAM sampler's tool.
        launcher: Starts the server.
        http: Talks to it.
        sampler: Reads VRAM.
        options: What to measure; the defaults of :class:`BenchOptions` when omitted.
        sleep: How to wait while the server loads.
        clock: Where the time comes from.

    Returns:
        The report, with results that have not been stored yet. Storing is the caller's
        decision and is made after the paging check, because a run that paged is stored as
        evidence rather than as a measurement of the configuration somebody asked about.

    Raises:
        ProbeError: If ``llama-bench`` could not run or produced nothing readable, or if
            the server never became healthy.

    Every row gets its own estimate, from :class:`Estimator`, at the context that row
    filled and the micro-batch it ran at. One estimate for the whole benchmark is what
    there used to be, and a benchmark is five measurements of at least three different
    configurations: ``tg128`` at 128 tokens of cache, a thousand-token request at about a
    thousand, and with ``--sweep`` a prompt row per micro-batch on the ladder.
    """
    options = options or BenchOptions()
    plan = inputs.plan
    placement = plan.placement
    estimator = Estimator(inputs)
    vram_total = _card_total(inputs.host)

    argv = llama_bench_argv(
        placement,
        executable=inputs.bench_executable,
        model_path=plan.model_path,
        options=options,
    )
    runs: list[BenchRun] = []
    for row in run_llama_bench(runner, argv, timeout=options.bench_timeout_s):
        generated = row.kind == "llama-bench-tg"
        micro_batch = _int_or_none(row.settings.get("ub")) or placement.micro_batch
        estimate = estimator.at(
            context=measured_context(row.n_prompt, row.n_gen), micro_batch=micro_batch
        )
        runs.append(
            _pending(
                kind=row.kind,
                conditions=_conditions(
                    inputs=inputs,
                    build=row.build or inputs.llama_cpp_build,
                    commit=row.commit or inputs.llama_cpp_commit,
                    argv=argv,
                    reported=row.settings,
                    context=None,
                    micro_batch=micro_batch,
                    n_prompt=row.n_prompt,
                    n_gen=row.n_gen,
                ),
                gen_tps=row.tokens_per_second if generated else None,
                pp_tps=None if generated else row.tokens_per_second,
                traffic=estimate.traffic,
                estimated_gen_tps=estimate.gen_tps if generated else None,
                estimated_pp_tps=None if generated else estimate.pp_tps,
                vram_total_bytes=vram_total,
            )
        )

    paging: PagingCheck | None = None
    if options.with_server:
        server_runs, paging = _server_phase(
            inputs,
            launcher=launcher,
            http=http,
            sampler=sampler,
            options=options,
            estimator=estimator,
            vram_total=vram_total,
            sleep=sleep,
            clock=clock,
        )
        runs.extend(server_runs)

    return BenchReport(
        model_id=plan.model_id,
        quant=plan.quant,
        host_fingerprint=host_fingerprint(inputs.host),
        command=list(argv),
        server_command=list(inputs.server_argv) if options.with_server else [],
        runs=runs,
        paging=paging,
        planned_context=placement.context,
        planned_gen_tps=plan.speed.gen_tps if plan.speed else None,
    )


def _server_phase(
    inputs: BenchInputs,
    *,
    launcher: ServerLauncher,
    http: BenchHttp,
    sampler: VramSampler,
    options: BenchOptions,
    estimator: Estimator,
    vram_total: int | None,
    sleep: Callable[[float], None],
    clock: Callable[[], float],
) -> tuple[list[BenchRun], PagingCheck | None]:
    """Start the server, ask it the three questions, and stop it whatever happens.

    The three questions fill three different amounts of cache -- a couple of dozen tokens,
    about a thousand, and whatever a tool call costs -- so each gets its estimate from
    ``estimator`` at its own depth rather than all three sharing the plan's.
    """
    handle = launcher.start(inputs.server_argv)
    try:
        healthy = wait_for_health(
            http,
            inputs.base_url,
            timeout=options.health_timeout_s,
            sleep=sleep,
            clock=clock,
            handle=handle,
        )
        if not healthy:
            raise ProbeError(
                _("llama-server did not become healthy"),
                hint=_("Its whole output is in the log file `bench` named; the end says why."),
                command=" ".join(inputs.server_argv),
            )
        results, peak, context = probe_server(
            http, inputs.base_url, options=options, sampler=sampler
        )
        log_text = handle.log()
    finally:
        handle.stop()

    buffers = parse_buffer_sizes(log_text)
    micro_batch = inputs.plan.placement.micro_batch
    warm = next((item for item in results if item.kind == "server-1k"), None)
    # Section 16.4's speed half compares the warm request with the estimate, and the two
    # have to be for the same cache or the ratio measures the context rather than the
    # driver -- upwards, at that: a plan's estimate at 32,768 tokens is lower than the
    # truth at one thousand, so a configuration that really was paging would clear the
    # threshold on the difference alone.
    warm_estimate = estimator.at(
        context=measured_context(warm.n_prompt, warm.n_gen) if warm else None,
        micro_batch=micro_batch,
    )
    paging = detect_paging(
        peak_vram_bytes=peak,
        vram_total_bytes=vram_total,
        measured_gen_tps=warm.gen_tps if warm else None,
        estimated_gen_tps=warm_estimate.gen_tps,
        tiers=inputs.plan.placement.tiers,
    )
    runs: list[BenchRun] = []
    for result in results:
        estimate = estimator.at(
            context=measured_context(result.n_prompt, result.n_gen), micro_batch=micro_batch
        )
        runs.append(
            _pending(
                kind=result.kind,
                conditions=_conditions(
                    inputs=inputs,
                    build=inputs.llama_cpp_build,
                    commit=inputs.llama_cpp_commit,
                    argv=inputs.server_argv,
                    reported={},
                    context=context or inputs.plan.placement.context,
                    micro_batch=micro_batch,
                    n_prompt=result.n_prompt,
                    n_gen=result.n_gen,
                ),
                gen_tps=result.gen_tps,
                pp_tps=result.pp_tps,
                ttft_ms=result.ttft_ms,
                traffic=estimate.traffic,
                estimated_gen_tps=estimate.gen_tps if result.gen_tps else None,
                estimated_pp_tps=estimate.pp_tps if result.pp_tps else None,
                peak_vram_bytes=peak,
                vram_total_bytes=vram_total,
                paging=paging,
                tool_call_ok=result.tool_call_ok,
                buffer_bytes=buffers,
            )
        )
    return runs, paging


def store_report(store: BenchStore, report: BenchReport) -> BenchReport:
    """Write every result of a report to the store and return it with the stored rows.

    Args:
        store: The open database.
        report: What the benchmark produced.

    Returns:
        The same report with its runs replaced by the stored ones, which carry the
        identifier, the timestamp and the conditions hash the store gave them.

    Raises:
        ProbeError: If any run's conditions do not describe the run. Nothing partial is
            left behind that a reader could mistake for a complete benchmark: the first
            refusal stops the whole thing, because a benchmark half of whose rows are
            missing is a benchmark whose comparison table is wrong.
    """
    stored = [
        store.record(
            kind=run.kind,
            conditions=run.conditions,
            gen_tps=run.gen_tps,
            pp_tps=run.pp_tps,
            ttft_ms=run.ttft_ms,
            peak_vram_bytes=run.peak_vram_bytes,
            vram_total_bytes=run.vram_total_bytes,
            traffic=run.traffic,
            estimated_gen_tps=run.estimated_gen_tps,
            estimated_pp_tps=run.estimated_pp_tps,
            paging=run.paging,
            tool_call_ok=run.tool_call_ok,
            buffer_bytes=run.buffer_bytes,
        )
        for run in report.runs
    ]
    return report.model_copy(update={"runs": stored, "stored": True})


def _conditions(
    *,
    inputs: BenchInputs,
    build: int | None,
    commit: str | None,
    argv: Sequence[str],
    reported: Mapping[str, str],
    context: int | None,
    micro_batch: int | None,
    n_prompt: int | None,
    n_gen: int | None,
) -> RunConditions:
    """Assemble the conditions of one run: the command line, and what the tool said it did.

    The settings start from the command line and are then overwritten by whatever the tool
    reported about itself, so a value llama.cpp chose or clamped replaces the one it was
    offered. That ordering is the whole point: the request is the fallback, the report is
    the record.
    """
    settings = flags_from_argv(argv)
    settings.update({key: value for key, value in reported.items() if value})
    if context:
        settings["c"] = str(context)
    return RunConditions(
        host_fingerprint=host_fingerprint(inputs.host),
        host_summary=host_summary(inputs.host),
        llama_cpp_build=build,
        llama_cpp_commit=commit,
        model_id=inputs.plan.model_id,
        quant=inputs.plan.quant,
        model_file=inputs.plan.model_path,
        model_bytes=_file_size(inputs.plan.model_path),
        argv=tuple(argv),
        settings=settings,
        context=context,
        micro_batch=micro_batch,
        n_prompt=n_prompt,
        n_gen=n_gen,
    )


def _pending(
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
    """A result that has not been stored yet, with a placeholder identity.

    The store is what gives a run its identifier, because having been recorded is what an
    identifier is a fact about. A report can be printed without storing anything --
    ``--no-store`` does exactly that -- so the object has to exist before then, and this is
    what it exists as until it does.
    """
    return BenchRun(
        id="pending",
        schema_version=BENCH_SCHEMA_VERSION,
        recorded_at=datetime.now(timezone.utc),
        kind=kind,
        conditions=conditions,
        conditions_hash=conditions_hash(kind, conditions),
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


def _file_size(path: str) -> int | None:
    """The file's size, or ``None`` when it is not there to be measured."""
    try:
        return Path(path).stat().st_size
    except OSError:
        return None


def _card_total(host: Host) -> int | None:
    """The primary card's total VRAM, or ``None`` when there is no card or no figure."""
    gpu = host.primary_gpu
    return None if gpu is None else gpu.vram_total_bytes


def _int_or_none(text: str | None) -> int | None:
    """A settings value as an integer, or ``None`` when it is not one."""
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None
