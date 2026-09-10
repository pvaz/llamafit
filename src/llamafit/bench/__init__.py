# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Measuring, instead of calculating: section 16's verifier and calibration.

Everything else in LlamaFit predicts. This package is where a prediction gets checked
against the machine, and it is the only place a number can earn section 10.3's ``measured``
label. That makes it the part with the most to lose, because the label is believed: a
figure marked ``measured`` outranks every formula in the tool and will be trusted long
after whoever took it has forgotten what they were doing.

So four rules run through the whole package, and each one is a refusal.

**A result records what ran, not what was asked for** (:mod:`llamafit.bench.types`,
:mod:`llamafit.bench.fingerprint`). llama.cpp clamps a context, expands a micro-batch sweep
into a run apiece, and quietly ignores a flag it was not compiled for. The record carries
the command line, the tool's own report of what it did, and a comparison of the two; a run
whose report disagrees with its request is not stored at all.

**A constant is fitted only when the data determines it** (:mod:`llamafit.bench.calibrate`).
Four measurements pinned four constants on the reference machine, and two would not have
pinned three. A fit with fewer points than parameters, or with columns that vary together,
is refused by name and with a reason -- and so is one whose answer comes out impossible,
because the parameters of a least-squares solution are chosen against each other and cannot
be kept one at a time.

**A configuration that pages is caught** (:mod:`llamafit.bench.paging`). It does not fail.
It runs at half speed with a clean log, and section 16.4's two-part signature is the only
thing that tells it apart from a machine that is merely slower than predicted.

**The estimate is shown beside the measurement** (:mod:`llamafit.bench.compare`), from
before the run, so the gap stays visible. Replacing a bad prediction with a fresh
measurement and saying nothing would leave a reader better informed about one model and no
better informed about the next.
"""

from llamafit.bench.calibrate import calibrate
from llamafit.bench.compare import compare, metric_label, within_tolerance
from llamafit.bench.paging import detect_paging, reason_text
from llamafit.bench.parse import (
    device_compute_bytes,
    parse_buffer_sizes,
    parse_llama_bench_json,
    parse_llama_bench_markdown,
    parse_used_vram,
)
from llamafit.bench.run import (
    BenchInputs,
    BenchOptions,
    FakeBenchHttp,
    FakeServerLauncher,
    FakeVramSampler,
    HttpxBenchClient,
    NvidiaSmiSampler,
    SubprocessServerLauncher,
    llama_bench_argv,
    run_benchmark,
    store_report,
    traffic_of,
)
from llamafit.bench.store import BenchStore, database_path, open_store
from llamafit.bench.types import (
    BenchKind,
    BenchReport,
    BenchRun,
    Calibration,
    ComparisonRow,
    PagingCheck,
    Refusal,
    RunConditions,
    RunTraffic,
)

__all__ = [
    "BenchInputs",
    "BenchKind",
    "BenchOptions",
    "BenchReport",
    "BenchRun",
    "BenchStore",
    "Calibration",
    "ComparisonRow",
    "FakeBenchHttp",
    "FakeServerLauncher",
    "FakeVramSampler",
    "HttpxBenchClient",
    "NvidiaSmiSampler",
    "PagingCheck",
    "Refusal",
    "RunConditions",
    "RunTraffic",
    "SubprocessServerLauncher",
    "calibrate",
    "compare",
    "database_path",
    "detect_paging",
    "device_compute_bytes",
    "llama_bench_argv",
    "metric_label",
    "open_store",
    "parse_buffer_sizes",
    "parse_llama_bench_json",
    "parse_llama_bench_markdown",
    "parse_used_vram",
    "reason_text",
    "run_benchmark",
    "store_report",
    "traffic_of",
    "within_tolerance",
]
