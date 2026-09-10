# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Reading what llama.cpp says about a run it has just done.

Three sources, in decreasing order of how much they can be trusted to describe the run
rather than the request.

``llama-bench -o json`` is the good one: every row carries the settings that row ran at,
the build number the binary was compiled as, and the throughput with its spread. That is
what a record wants, and it is what :func:`parse_llama_bench_json` returns.

The markdown table ``llama-bench`` prints by default carries less -- the layer count, the
test name and the throughput, and the build on a line of its own underneath -- but it is
what a person has in their scrollback and in the issue they are quoting from, so
:func:`parse_llama_bench_markdown` reads it too. A record made from it says less about
itself, which is honest: the settings it does not name are absent rather than assumed.

The server's verbose log is the third. It has no structure to speak of, but it is the only
place the buffer sizes appear, and those are the one memory figure section 8 models rather
than reads. :func:`parse_buffer_sizes` pulls them out.

Nothing here raises on a line it does not understand. A benchmark that produced three good
rows and one line of driver chatter is three good rows.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llamafit.bench.types import BenchKind
from llamafit.i18n import _

_MIB = 1024**2

_TEST_NAME = re.compile(r"^(pp|tg)(\d+)$", re.IGNORECASE)
_BUILD_LINE = re.compile(r"^build:\s*([0-9a-f]+)\s*\((\d+)\)", re.IGNORECASE | re.MULTILINE)
_BUFFER_LINE = re.compile(
    # `llama_context:` and friends are a prefix and not part of the name; the name is
    # everything between it and the words "buffer size", which is where llama.cpp puts
    # both the device (`CUDA0`) and the role (`KV`, `compute`, `model`). Both halves are
    # needed: a total that mixed the card's compute buffer with the host's would be a
    # memory figure of nothing in particular.
    r"^\s*(?:[A-Za-z0-9_]+:)?\s*(?P<name>[A-Za-z0-9_.()\- ]+?)\s+buffer size"
    r"\s*=\s*(?P<size>[\d.]+)\s*MiB",
    re.MULTILINE,
)


class BenchRow:
    """One row of a ``llama-bench`` result: what ran, and how fast.

    Attributes:
        kind: Whether this row measures prompt processing or generation.
        tokens_per_second: The throughput the row reports.
        stddev: The spread across repetitions, when the source gives one.
        n_prompt: Prompt tokens the row used.
        n_gen: Generated tokens the row used.
        settings: The run as the tool reported it, keyed the way
            :attr:`llamafit.bench.types.RunConditions.settings` is keyed.
        build: The build number the binary reported, when it reported one.
        commit: The commit it reported, when it reported one.
        model_file: The file the row was run against, when the source names it.
        model_bytes: That file's size, when the source gives it.
    """

    __slots__ = (
        "build",
        "commit",
        "kind",
        "model_bytes",
        "model_file",
        "n_gen",
        "n_prompt",
        "settings",
        "stddev",
        "tokens_per_second",
    )

    def __init__(
        self,
        *,
        kind: BenchKind,
        tokens_per_second: float,
        n_prompt: int | None = None,
        n_gen: int | None = None,
        stddev: float | None = None,
        settings: dict[str, str] | None = None,
        build: int | None = None,
        commit: str | None = None,
        model_file: str | None = None,
        model_bytes: int | None = None,
    ) -> None:
        """Store one row exactly as its source reported it."""
        self.kind = kind
        self.tokens_per_second = tokens_per_second
        self.n_prompt = n_prompt
        self.n_gen = n_gen
        self.stddev = stddev
        self.settings = settings or {}
        self.build = build
        self.commit = commit
        self.model_file = model_file
        self.model_bytes = model_bytes

    def __repr__(self) -> str:
        """A short form naming the kind and the figure, for a failing assertion."""
        return f"BenchRow({self.kind}, {self.tokens_per_second:.2f} t/s)"


def _as_str(value: object) -> str | None:
    """A JSON scalar as the string a settings map holds, or ``None`` for anything else."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _settings_from_json(entry: dict[str, Any]) -> dict[str, str]:
    """The settings a ``llama-bench`` JSON row says it ran at.

    Every key is taken from the row itself. A field the build does not emit is left out
    rather than filled from the request, because the whole point of reading the tool's own
    report is to learn what it did rather than what it was told.
    """
    mapping = {
        "n_gpu_layers": "ngl",
        "n_ubatch": "ub",
        "n_batch": "b",
        "n_threads": "t",
        "type_k": "ctk",
        "type_v": "ctv",
        "flash_attn": "fa",
        "n_cpu_moe": "n-cpu-moe",
    }
    settings: dict[str, str] = {}
    for source, key in mapping.items():
        text = _as_str(entry.get(source))
        if text is not None:
            settings[key] = text
    return settings


def parse_llama_bench_json(text: str) -> list[BenchRow]:
    """Read ``llama-bench -o json`` output.

    Args:
        text: The tool's stdout.

    Returns:
        One row per result, in the order the tool produced them.

    Raises:
        ValueError: If the text is not JSON, or is JSON of a shape no row can be read
            from. A benchmark whose output could not be read is a benchmark that did not
            happen, and quietly returning nothing would let the caller store a run with no
            numbers in it.
    """
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(_("llama-bench did not produce JSON")) from exc
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError(_("llama-bench JSON is not a list of results"))
    rows: list[BenchRow] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        row = _row_from_json(entry)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError(_("llama-bench JSON held no readable results"))
    return rows


def _row_from_json(entry: dict[str, Any]) -> BenchRow | None:
    """One JSON result as a row, or ``None`` when it carries no throughput."""
    speed = entry.get("avg_ts")
    if not isinstance(speed, (int, float)) or speed <= 0:
        return None
    n_prompt = entry.get("n_prompt")
    n_gen = entry.get("n_gen")
    prompt = int(n_prompt) if isinstance(n_prompt, (int, float)) else None
    gen = int(n_gen) if isinstance(n_gen, (int, float)) else None
    kind: BenchKind = "llama-bench-tg" if gen else "llama-bench-pp"
    stddev = entry.get("stddev_ts")
    build = entry.get("build_number")
    size = entry.get("model_size")
    return BenchRow(
        kind=kind,
        tokens_per_second=float(speed),
        n_prompt=prompt,
        n_gen=gen,
        stddev=float(stddev) if isinstance(stddev, (int, float)) else None,
        settings=_settings_from_json(entry),
        build=int(build) if isinstance(build, (int, float)) else None,
        commit=_as_str(entry.get("build_commit")),
        model_file=_as_str(entry.get("model_filename")),
        model_bytes=int(size) if isinstance(size, (int, float)) and size > 0 else None,
    )


def parse_llama_bench_markdown(text: str) -> list[BenchRow]:
    """Read the markdown table ``llama-bench`` prints when nobody asks for JSON.

    Args:
        text: The tool's stdout, table and trailing build line included.

    Returns:
        One row per result line, in order. A row here carries only what the table has --
        the layer count, the test and the throughput -- and says nothing about the
        micro-batch or the thread count unless the table happens to have a column for it.

    Raises:
        ValueError: If no result row could be read at all.

    The columns are read by their headers rather than by position, because ``llama-bench``
    adds a column whenever a sweep varies something, and a parser counting from the left
    would silently start reading the throughput out of the wrong one.
    """
    build, commit = _build_from_markdown(text)
    header: list[str] | None = None
    rows: list[BenchRow] = []
    for line in text.splitlines():
        cells = _markdown_cells(line)
        if cells is None:
            continue
        if header is None:
            header = [cell.strip().lower() for cell in cells]
            continue
        if all(set(cell.strip()) <= set("-: ") for cell in cells):
            continue
        row = _row_from_markdown(dict(zip(header, cells, strict=False)), build, commit)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError(_("no llama-bench result rows found"))
    return rows


def _markdown_cells(line: str) -> list[str] | None:
    """The cells of one table row, or ``None`` when the line is not one."""
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _build_from_markdown(text: str) -> tuple[int | None, str | None]:
    """The build number and commit from the ``build: <hash> (<number>)`` line."""
    found = _BUILD_LINE.search(text)
    if not found:
        return None, None
    return int(found.group(2)), found.group(1)


def _row_from_markdown(
    cells: dict[str, str], build: int | None, commit: str | None
) -> BenchRow | None:
    """One table row as a :class:`BenchRow`, or ``None`` when it is not a result."""
    test = cells.get("test", "").strip()
    match = _TEST_NAME.match(test)
    speed_text = cells.get("t/s", "").strip()
    if match is None or not speed_text:
        return None
    speed, stddev = _split_measurement(speed_text)
    if speed is None:
        return None
    count = int(match.group(2))
    generation = match.group(1).lower() == "tg"
    settings: dict[str, str] = {}
    for column, key in (("ngl", "ngl"), ("n_ubatch", "ub"), ("threads", "t"), ("fa", "fa")):
        value = cells.get(column, "").strip()
        if value:
            settings[key] = value
    return BenchRow(
        kind="llama-bench-tg" if generation else "llama-bench-pp",
        tokens_per_second=speed,
        n_prompt=None if generation else count,
        n_gen=count if generation else None,
        stddev=stddev,
        settings=settings,
        build=build,
        commit=commit,
    )


def _split_measurement(text: str) -> tuple[float | None, float | None]:
    """``"24.70 ± 0.31"`` as a value and a spread; either may be missing."""
    parts = re.split(r"±|\+/-", text)
    try:
        value = float(parts[0].strip())
    except ValueError:
        return None, None
    spread: float | None = None
    if len(parts) > 1:
        try:
            spread = float(parts[1].strip())
        except ValueError:
            spread = None
    return value, spread


def parse_buffer_sizes(log: str) -> dict[str, int]:
    """The buffer sizes a ``llama-server -v`` log reports, in bytes, keyed by buffer.

    Args:
        log: Everything the server wrote to its log.

    Returns:
        A mapping of buffer name to bytes. Repeated names are summed, because llama.cpp
        prints one line per device and the budget this is checked against is a total.

    These are the lines section 8.2's compute-buffer model was fitted from, and they are
    the only place the allocation appears at all: ``nvidia-smi`` sees one number for the
    whole process. A benchmark that captures them is a benchmark that can tell a memory
    prediction from a memory measurement.
    """
    sizes: dict[str, int] = {}
    for found in _BUFFER_LINE.finditer(log):
        name = " ".join(found.group("name").split()).strip(": ").lower()
        if not name:
            continue
        try:
            mib = float(found.group("size"))
        except ValueError:  # a line that looked like a size and was not
            continue
        sizes[name] = sizes.get(name, 0) + int(mib * _MIB)
    return sizes


def device_compute_bytes(buffers: dict[str, int]) -> int | None:
    """The compute buffer the graphics card holds, out of every buffer the log named.

    Args:
        buffers: What :func:`parse_buffer_sizes` returned.

    Returns:
        The total of every compute buffer on a device, or ``None`` when the log named
        none. Host-side compute buffers are excluded: section 8.2 models the allocation on
        the card, which is the one that decides whether a configuration pages, and the
        calibration record's own table is explicitly "all in MiB on the GPU".
    """
    total = 0
    found = False
    for name, size in buffers.items():
        if not name.endswith("compute"):
            continue
        if "host" in name or "cpu" in name:
            continue
        total += size
        found = True
    return total if found else None


def parse_used_vram(text: str) -> int | None:
    """The VRAM in use, in bytes, from ``nvidia-smi --query-gpu=memory.used``.

    Args:
        text: One line of the tool's CSV output, in MiB with no units.

    Returns:
        The reading, or ``None`` when the tool reported one it does not support. A card
        that reports ``[N/A]`` has not reported zero, and treating it as zero would clear
        a configuration of paging on the strength of a number that was never taken.
    """
    for line in text.splitlines():
        first = line.split(",")[0].strip()
        try:
            return int(float(first)) * _MIB
        except ValueError:
            continue
    return None
