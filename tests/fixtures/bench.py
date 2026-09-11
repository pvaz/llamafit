"""Recorded llama.cpp output and builders for the benchmark tests.

The two ``llama-bench`` samples are the shapes a real binary produces: the JSON one
``-o json`` writes, and the markdown table it prints when nobody asks. The server log is
the buffer-size lines section 8.2's compute-buffer model was fitted from, in the format
``llama-server -v`` writes them.

Nothing here runs a benchmark or needs a card, which is the point.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from llamafit.bench.fingerprint import conditions_hash
from llamafit.bench.types import (
    BENCH_SCHEMA_VERSION,
    BenchKind,
    BenchRun,
    PagingCheck,
    RunConditions,
    RunTraffic,
)

MIB = 1024**2
FINGERPRINT = "0123456789abcdef"

LLAMA_BENCH_JSON = json.dumps(
    [
        {
            "build_commit": "f3f1a8f27",
            "build_number": 10867,
            "cpu_info": "Intel(R) Core(TM) i9-14900KF",
            "gpu_info": "NVIDIA GeForce RTX 4060",
            "backends": "CUDA",
            "model_filename": "Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
            "model_type": "qwen3moe 80B.A3B Q4_K - Medium",
            "model_size": 49600000000,
            "model_n_params": 80000000000,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_threads": 16,
            "type_k": "f16",
            "type_v": "f16",
            "n_gpu_layers": 99,
            "n_cpu_moe": 48,
            "flash_attn": 1,
            "n_prompt": 2048,
            "n_gen": 0,
            "avg_ts": 194.25,
            "stddev_ts": 1.42,
        },
        {
            "build_commit": "f3f1a8f27",
            "build_number": 10867,
            "model_filename": "Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
            "model_size": 49600000000,
            "n_batch": 4096,
            "n_ubatch": 1024,
            "n_threads": 16,
            "type_k": "f16",
            "type_v": "f16",
            "n_gpu_layers": 99,
            "n_cpu_moe": 48,
            "flash_attn": 1,
            "n_prompt": 0,
            "n_gen": 128,
            "avg_ts": 24.7,
            "stddev_ts": 0.31,
        },
    ]
)


def llama_bench_json(
    *,
    ngl: int,
    micro_batch: int,
    batch: int,
    threads: int,
    n_cpu_moe: int | None = None,
    kv_type: str = "f16",
    pp_tps: float = 194.25,
    gen_tps: float = 24.7,
    build: int = 10867,
) -> str:
    """``llama-bench -o json`` output for a run that really used these settings.

    Built from the placement rather than pinned, because the store refuses a result whose
    reported settings disagree with the command line it was given -- which is the whole
    point of that check, and would otherwise make every test here fight it.
    """
    shared: dict[str, Any] = {
        "build_commit": "f3f1a8f27",
        "build_number": build,
        "model_filename": "model.gguf",
        "model_size": 49600000000,
        "n_batch": batch,
        "n_ubatch": micro_batch,
        "n_threads": threads,
        "type_k": kv_type,
        "type_v": kv_type,
        "n_gpu_layers": ngl,
        "flash_attn": 1,
    }
    if n_cpu_moe is not None:
        shared["n_cpu_moe"] = n_cpu_moe
    return json.dumps(
        [
            {**shared, "n_prompt": 2048, "n_gen": 0, "avg_ts": pp_tps, "stddev_ts": 1.42},
            {**shared, "n_prompt": 0, "n_gen": 128, "avg_ts": gen_tps, "stddev_ts": 0.31},
        ]
    )


LLAMA_BENCH_MARKDOWN = """\
| model                          |       size |     params | backend    | ngl |   n_ubatch | \
         test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | \
------------: | -------------------: |
| qwen3moe 80B.A3B Q4_K - Medium |  46.20 GiB |    80.00 B | CUDA       |  99 |       1024 | \
        pp2048 |        194.25 ± 1.42 |
| qwen3moe 80B.A3B Q4_K - Medium |  46.20 GiB |    80.00 B | CUDA       |  99 |       1024 | \
         tg128 |         24.70 ± 0.31 |

build: f3f1a8f27 (10867)
"""

SERVER_LOG = """\
llama_model_loader: loaded meta data with 42 key-value pairs
load_tensors:        CUDA0 model buffer size =  4367.00 MiB
load_tensors:   CPU_Mapped model buffer size = 45000.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =  1056.00 MiB
llama_context:      CUDA0 compute buffer size =  1337.02 MiB
llama_context:  CUDA_Host compute buffer size =    24.01 MiB
main: server is listening on http://127.0.0.1:8080 - starting the main loop
"""


def completion_response(
    *, gen_tps: float, pp_tps: float, prompt_ms: float, tokens_evaluated: int, tokens_predicted: int
) -> dict[str, Any]:
    """A ``/completion`` answer shaped the way llama-server writes one."""
    return {
        "content": "...",
        "tokens_evaluated": tokens_evaluated,
        "tokens_predicted": tokens_predicted,
        "timings": {
            "prompt_ms": prompt_ms,
            "prompt_per_second": pp_tps,
            "predicted_per_second": gen_tps,
        },
    }


def tool_call_response(*, well_formed: bool = True) -> dict[str, Any]:
    """A chat answer with a tool call in it, well-formed or not."""
    if well_formed:
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Lisbon"}'},
                }
            ],
        }
    else:
        message = {"role": "assistant", "content": 'get_weather({"city": "Lisbon"})'}
    return {"choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}]}


def conditions(
    *,
    fingerprint: str = FINGERPRINT,
    model_id: str = "qwen3-coder-next",
    quant: str = "UD-Q4_K_XL",
    settings: dict[str, str] | None = None,
    argv: tuple[str, ...] = ("llama-bench", "-m", "model.gguf"),
    context: int | None = None,
    micro_batch: int | None = 1024,
    n_prompt: int | None = None,
    n_gen: int | None = None,
    build: int | None = 10867,
) -> RunConditions:
    """Conditions a test can vary one field of at a time."""
    chosen = dict(settings if settings is not None else {"ngl": "99", "ub": "1024"})
    if context:
        chosen.setdefault("c", str(context))
    return RunConditions(
        host_fingerprint=fingerprint,
        host_summary="a machine",
        llama_cpp_build=build,
        llama_cpp_commit="f3f1a8f27",
        model_id=model_id,
        quant=quant,
        model_file="/models/model.gguf",
        model_bytes=49600000000,
        argv=argv,
        settings=chosen,
        context=context,
        micro_batch=micro_batch,
        n_prompt=n_prompt,
        n_gen=n_gen,
    )


def traffic(
    *,
    device_bytes: int = 0,
    sequential_bytes: int = 0,
    scattered_bytes: int = 0,
    ram_gbps: float = 50.0,
    device_gbps: float | None = 200.0,
    streamed_expert_bytes: int = 0,
    active_params: float = 3e9,
    compute_flops: float = 15e12,
    micro_batch: int = 1024,
) -> RunTraffic:
    """A traffic record with whichever pools a test wants to put bytes in."""
    return RunTraffic(
        device_bytes=device_bytes,
        sequential_bytes=sequential_bytes,
        scattered_bytes=scattered_bytes,
        ram_gbps=ram_gbps,
        device_gbps=device_gbps,
        streamed_expert_bytes=streamed_expert_bytes,
        active_params=active_params,
        compute_flops=compute_flops,
        pcie_gbps=12.0,
        micro_batch=micro_batch,
    )


_COUNTER = {"n": 0}

TOKEN_COUNTS: dict[BenchKind, tuple[int | None, int | None]] = {
    "llama-bench-tg": (0, 128),
    "llama-bench-pp": (2048, 0),
    "server-short": (24, 128),
    "server-1k": (1056, 64),
    "server-toolcall": (None, None),
}
"""What each kind of run evaluates and generates, as the real ones report it.

A run that does not say how many tokens it moved cannot say what context its figures
belong to, and a comparison row without a context carries no ratio -- correct, and it
would otherwise make every builder here produce a run nothing can be compared against.
The tool call keeps its ``None``s: it reports no counts because it is not a speed
measurement, and a fixture that invented some would be testing a shape that never occurs.
"""


def run_of(
    *,
    kind: BenchKind = "llama-bench-tg",
    conditions_: RunConditions | None = None,
    gen_tps: float | None = None,
    pp_tps: float | None = None,
    traffic_: RunTraffic | None = None,
    paging: PagingCheck | None = None,
    buffer_bytes: dict[str, int] | None = None,
    peak_vram_bytes: int | None = None,
    vram_total_bytes: int | None = None,
    estimated_gen_tps: float | None = None,
    estimated_pp_tps: float | None = None,
) -> BenchRun:
    """A stored run built by hand, with a unique identifier and an increasing timestamp.

    Conditions the caller does not supply carry the token counts and, for the server kinds,
    the server context that the real run of that kind would report.
    """
    _COUNTER["n"] += 1
    prompt_tokens, generated = TOKEN_COUNTS[kind]
    settings = conditions_ or conditions(
        n_prompt=prompt_tokens,
        n_gen=generated,
        context=32768 if kind.startswith("server") else None,
    )
    return BenchRun(
        id=f"run{_COUNTER['n']:04d}",
        schema_version=BENCH_SCHEMA_VERSION,
        recorded_at=datetime(2026, 9, 9, tzinfo=timezone.utc) + timedelta(minutes=_COUNTER["n"]),
        kind=kind,
        conditions=settings,
        conditions_hash=conditions_hash(kind, settings),
        gen_tps=gen_tps,
        pp_tps=pp_tps,
        traffic=traffic_,
        paging=paging,
        buffer_bytes=buffer_bytes or {},
        peak_vram_bytes=peak_vram_bytes,
        vram_total_bytes=vram_total_bytes,
        estimated_gen_tps=estimated_gen_tps,
        estimated_pp_tps=estimated_pp_tps,
        llamafit_version="0.0.0-test",
    )


# The four measurements docs/specs section 10.1 identifies the estimator's constants from,
# with the traffic column exactly as the specification states it: bytes per token, the
# token embedding table left out.
REFERENCE_MEASUREMENTS = [
    # (device GB, sequential GB, scattered GB, tokens per second)
    (0.0, 0.47, 0.0, 78.0),
    (0.47, 0.0, 0.0, 279.5),
    (2.36, 0.0, 0.916, 23.0),
    (4.83, 0.0, 1.504, 13.9),
]
REFERENCE_RAM_GBPS = 57.0
REFERENCE_DEVICE_GBPS = 272.0


def reference_generation_runs(limit: int | None = None) -> list[BenchRun]:
    """The reference machine's own four generation measurements, as stored runs."""
    runs = []
    for index, (device, sequential, scattered, tps) in enumerate(
        REFERENCE_MEASUREMENTS[: limit if limit is not None else len(REFERENCE_MEASUREMENTS)]
    ):
        runs.append(
            run_of(
                kind="llama-bench-tg",
                conditions_=conditions(settings={"ngl": str(index), "ub": "512"}),
                gen_tps=tps,
                traffic_=traffic(
                    device_bytes=round(device * 1e9),
                    sequential_bytes=round(sequential * 1e9),
                    scattered_bytes=round(scattered * 1e9),
                    ram_gbps=REFERENCE_RAM_GBPS,
                    device_gbps=REFERENCE_DEVICE_GBPS,
                ),
            )
        )
    return runs
