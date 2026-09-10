# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Collect what a unified-memory Mac needs to become a second reference machine.

LlamaFit's speed estimator assumes two memory pools with a link between them, which is
what a desktop with a discrete graphics card has. Apple silicon has one pool, so the
estimator would read one piece of silicon's bandwidth as two independent figures and
charge a transfer that never happens. Nothing in the code can be fixed honestly without
numbers from such a machine, and this script collects them.

Run it on the Mac, with llama.cpp built and at least one small model to hand:

    python scripts/record_mac_reference.py
        --llama-bench ~/llama.cpp/build/bin/llama-bench
        --model ~/models/Qwen3-0.6B-Q8_0.gguf

It writes one JSON file and prints where. Nothing is uploaded and nothing is changed.

Why these particular runs. A small dense model is the only way to isolate a memory pool:
its traffic per token is exactly its block weights, nothing else competes, and the answer
does not depend on any constant we are trying to measure. On a machine with two pools the
same model is run twice, once in each, and the difference identifies both efficiencies.
On a machine with one pool there is nothing to separate, and that is itself the finding:
the two runs should agree, and if they do not, the assumption that unified memory is one
pool is wrong and we want to know that before writing code around it.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

TG_PATTERN = re.compile(r"\btg(\d+)\s*\|\s*([\d.]+)")
PP_PATTERN = re.compile(r"\bpp(\d+)\s*\|\s*([\d.]+)")


def run(command: list[str]) -> str:
    """Run a command and return its output, or an empty string if it could not run."""
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout + done.stderr


def machine_facts() -> dict[str, Any]:
    """What this Mac is, from its own tools rather than from guesswork."""
    facts: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "recorded_on": date.today().isoformat(),
    }
    for key, argv in {
        "chip": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "performance_cores": ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
        "efficiency_cores": ["sysctl", "-n", "hw.perflevel1.physicalcpu"],
        "memory_bytes": ["sysctl", "-n", "hw.memsize"],
        "page_size": ["sysctl", "-n", "hw.pagesize"],
    }.items():
        value = run(argv).strip()
        facts[key] = value or None
    # The GPU core count and the memory bandwidth are not in sysctl; they come from the
    # system profiler, which is slow but authoritative about Apple's own silicon.
    facts["system_profiler_display"] = run(["system_profiler", "SPDisplaysDataType"])[:4000]
    return facts


def bench(binary: str, model: str, layers: int, threads: int | None) -> dict[str, Any]:
    """One llama-bench run, returning its throughput figures and the raw output."""
    argv = [binary, "-m", model, "-ngl", str(layers), "-p", "512", "-n", "128", "-r", "3"]
    if threads is not None:
        argv += ["-t", str(threads)]
    output = run(argv)
    generation = TG_PATTERN.search(output)
    prompt = PP_PATTERN.search(output)
    return {
        "flags": argv[1:],
        "gen_tps": float(generation.group(2)) if generation else None,
        "pp_tps": float(prompt.group(2)) if prompt else None,
        "raw": output[-4000:],
    }


def main(argv: list[str]) -> int:
    """Collect the runs and write them beside this script's output path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bench", required=True, help="path to the llama-bench binary")
    parser.add_argument("--model", required=True, help="path to a small dense GGUF, 1B or less")
    parser.add_argument("--threads", type=int, default=None, help="passed through as -t")
    parser.add_argument("--out", default="mac-reference.json", help="where to write the result")
    args = parser.parse_args(argv)

    if platform.system() != "Darwin":
        print("This collects numbers about Apple silicon; run it on the Mac.", file=sys.stderr)
        return 2

    result: dict[str, Any] = {"machine": machine_facts(), "runs": {}}

    # Everything on the GPU, then everything on the CPU. On a machine with one pool these
    # two read the same memory at the same speed, and the gap between them is the compute
    # and the overheads rather than a second pool.
    result["runs"]["gpu"] = bench(args.llama_bench, args.model, 99, args.threads)
    result["runs"]["cpu"] = bench(args.llama_bench, args.model, 0, args.threads)

    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    gpu = result["runs"]["gpu"]["gen_tps"]
    cpu = result["runs"]["cpu"]["gen_tps"]
    print(f"wrote {out.resolve()}")
    if gpu and cpu:
        print(f"generation: {gpu:.1f} tok/s on the GPU, {cpu:.1f} on the CPU")
        print(f"ratio {gpu / cpu:.2f} — one pool would put this near the compute ratio, not a")
        print("bandwidth ratio, which is the thing the estimator has to be told.")
    else:
        print("llama-bench produced no throughput figures; the raw output is in the file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
