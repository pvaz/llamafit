"""Reading what llama.cpp says about a run, out of the three forms it says it in."""

from __future__ import annotations

import pytest

from llamafit.bench.parse import (
    device_compute_bytes,
    parse_buffer_sizes,
    parse_llama_bench_json,
    parse_llama_bench_markdown,
    parse_used_vram,
)
from tests.fixtures.bench import LLAMA_BENCH_JSON, LLAMA_BENCH_MARKDOWN, MIB, SERVER_LOG


def test_json_rows_carry_the_settings_the_run_actually_used() -> None:
    rows = parse_llama_bench_json(LLAMA_BENCH_JSON)
    assert [row.kind for row in rows] == ["llama-bench-pp", "llama-bench-tg"]
    prompt, generation = rows
    assert prompt.tokens_per_second == pytest.approx(194.25)
    assert generation.tokens_per_second == pytest.approx(24.7)
    # Not the request: the row's own report of what it ran at.
    assert prompt.settings == {
        "ngl": "99",
        "ub": "1024",
        "b": "4096",
        "t": "16",
        "ctk": "f16",
        "ctv": "f16",
        "fa": "1",
        "n-cpu-moe": "48",
    }
    assert prompt.build == 10867
    assert prompt.commit == "f3f1a8f27"
    assert prompt.model_bytes == 49600000000


def test_a_prompt_row_and_a_generation_row_are_told_apart_by_which_count_is_set() -> None:
    rows = parse_llama_bench_json(LLAMA_BENCH_JSON)
    assert rows[0].n_prompt == 2048 and rows[0].n_gen == 0
    assert rows[1].n_gen == 128


def test_json_that_is_not_json_is_refused_rather_than_returning_nothing() -> None:
    with pytest.raises(ValueError, match="JSON"):
        parse_llama_bench_json("| model | t/s |")


def test_json_with_no_readable_row_is_refused() -> None:
    with pytest.raises(ValueError, match="readable"):
        parse_llama_bench_json('[{"n_prompt": 2048}]')


def test_the_markdown_table_is_read_by_its_headers_not_by_position() -> None:
    rows = parse_llama_bench_markdown(LLAMA_BENCH_MARKDOWN)
    assert [row.kind for row in rows] == ["llama-bench-pp", "llama-bench-tg"]
    assert rows[0].tokens_per_second == pytest.approx(194.25)
    assert rows[0].stddev == pytest.approx(1.42)
    assert rows[0].settings["ub"] == "1024"
    assert rows[0].settings["ngl"] == "99"
    assert rows[1].n_gen == 128


def test_the_build_line_under_the_table_reaches_every_row() -> None:
    rows = parse_llama_bench_markdown(LLAMA_BENCH_MARKDOWN)
    assert all(row.build == 10867 for row in rows)
    assert all(row.commit == "f3f1a8f27" for row in rows)


def test_a_markdown_row_says_nothing_about_settings_its_table_has_no_column_for() -> None:
    """A record made from the table is honest about what it does not know.

    The default table has no thread column, so the row does not claim one. That is the
    difference between a record read from JSON and one read from a table, and it is why
    JSON is asked for first.
    """
    rows = parse_llama_bench_markdown(LLAMA_BENCH_MARKDOWN)
    assert "t" not in rows[0].settings


def test_output_with_no_table_at_all_is_refused() -> None:
    with pytest.raises(ValueError, match="no llama-bench result"):
        parse_llama_bench_markdown("error while loading model\n")


def test_the_server_log_gives_up_its_buffer_sizes() -> None:
    buffers = parse_buffer_sizes(SERVER_LOG)
    assert buffers["cuda0 compute"] == int(1337.02 * MIB)
    assert buffers["cuda0 kv"] == 1056 * MIB
    assert buffers["cuda0 model"] == 4367 * MIB
    assert buffers["cuda_host compute"] == int(24.01 * MIB)


def test_only_the_card_s_compute_buffer_counts_as_the_compute_buffer() -> None:
    """Section 8.2 models the allocation on the card; the host-side one is not it."""
    buffers = parse_buffer_sizes(SERVER_LOG)
    assert device_compute_bytes(buffers) == int(1337.02 * MIB)


def test_a_log_with_no_buffer_lines_has_no_compute_buffer_rather_than_a_zero_one() -> None:
    assert device_compute_bytes(parse_buffer_sizes("nothing to see here")) is None


def test_a_vram_reading_the_tool_does_not_support_is_not_read_as_zero() -> None:
    assert parse_used_vram("7770\n") == 7770 * MIB
    assert parse_used_vram("[N/A]\n") is None
    assert parse_used_vram("") is None
