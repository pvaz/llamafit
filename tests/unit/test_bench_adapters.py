"""The real adapters, and the edges of the readers, without running anything real.

``run.py`` ships a fake beside every real adapter so a test can replay a recorded machine.
That arrangement is only worth anything if the real ones are exercised too, or the fakes
become a description of code nobody has run. Here the real classes meet a subprocess that
does not exist, an HTTP client whose transport always fails, and a command runner that
answers from a recording.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from llamafit.bench.calibrate import refusal_text
from llamafit.bench.fingerprint import conditions_conflicts, flags_from_argv
from llamafit.bench.parse import parse_buffer_sizes, parse_llama_bench_json
from llamafit.bench.run import (
    FakeVramSampler,
    HttpxBenchClient,
    NvidiaSmiSampler,
    SubprocessServerLauncher,
    _SubprocessHandle,
    _tool_call_is_well_formed,
)
from llamafit.bench.types import Refusal
from llamafit.errors import ProbeError
from llamafit.hardware.runner import CommandResult, FakeRunner
from tests.fixtures.bench import conditions, tool_call_response

MIB = 1024**2


def test_starting_a_program_that_is_not_there_is_reported_not_raised_raw(tmp_path: Path) -> None:
    launcher = SubprocessServerLauncher(log_path=tmp_path / "server.log")
    with pytest.raises(ProbeError, match="could not start"):
        launcher.start(["llamafit-no-such-program-exists"])


def test_a_log_that_was_never_written_reads_as_empty(tmp_path: Path) -> None:
    handle = _SubprocessHandle(process=None, log_path=tmp_path / "missing.log")  # type: ignore[arg-type]
    assert handle.log() == ""


def test_a_server_log_that_is_there_is_read_back(tmp_path: Path) -> None:
    path = tmp_path / "server.log"
    path.write_text("llama_context:      CUDA0 compute buffer size =  100.00 MiB\n", "utf-8")
    handle = _SubprocessHandle(process=None, log_path=path)  # type: ignore[arg-type]
    assert parse_buffer_sizes(handle.log())["cuda0 compute"] == 100 * MIB


def test_the_http_client_turns_every_failure_into_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    client = HttpxBenchClient()

    def refuse(*_args: object, **_kwargs: object) -> object:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setattr(httpx, "post", refuse)
    assert client.get_json("http://127.0.0.1:1/health") is None
    assert client.post_json("http://127.0.0.1:1/completion", {}) is None


class _Response:
    """The two attributes the client reads off a response."""

    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        """The parsed body."""
        return self._payload


def test_the_http_client_reads_a_good_answer_and_ignores_a_bad_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "get", lambda *_a, **_k: _Response(200, {"status": "ok"}))
    monkeypatch.setattr(httpx, "post", lambda *_a, **_k: _Response(503, None))
    client = HttpxBenchClient()
    assert client.get_json("http://x/health") == {"status": "ok"}
    assert client.post_json("http://x/completion", {}) is None


def test_the_vram_sampler_reads_the_vendor_tool_and_survives_its_absence() -> None:
    present = NvidiaSmiSampler(runner=FakeRunner({"nvidia-smi": "7770\n"}))
    assert present.sample() == 7770 * MIB
    absent = NvidiaSmiSampler(runner=FakeRunner({}))
    assert absent.sample() is None
    broken = NvidiaSmiSampler(
        runner=FakeRunner({"nvidia-smi": CommandResult(["nvidia-smi"], 9, "", "boom", 1)})
    )
    assert broken.sample() is None


def test_the_fake_sampler_repeats_its_last_reading_rather_than_running_out() -> None:
    sampler = FakeVramSampler(readings=[1, 2])
    assert [sampler.sample() for _ in range(4)] == [1, 2, 2, 2]
    assert FakeVramSampler().sample() is None


def test_a_tool_call_has_to_be_a_tool_call_all_the_way_down() -> None:
    assert _tool_call_is_well_formed(tool_call_response()) is True
    assert _tool_call_is_well_formed(tool_call_response(well_formed=False)) is False
    assert _tool_call_is_well_formed(None) is False
    assert _tool_call_is_well_formed({"choices": []}) is False
    assert _tool_call_is_well_formed({"choices": [{"message": None}]}) is False
    assert _tool_call_is_well_formed({"choices": [{"message": {"tool_calls": []}}]}) is False
    assert (
        _tool_call_is_well_formed(
            {"choices": [{"message": {"tool_calls": [{"function": {"name": ""}}]}}]}
        )
        is False
    )
    # A call whose arguments are not JSON is not a call anybody can act on.
    assert (
        _tool_call_is_well_formed(
            {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "f", "arguments": "{"}}]}}
                ]
            }
        )
        is False
    )
    # Some builds hand back an object rather than a string, and that is fine.
    assert (
        _tool_call_is_well_formed(
            {
                "choices": [
                    {"message": {"tool_calls": [{"function": {"name": "f", "arguments": {}}}]}}
                ]
            }
        )
        is True
    )
    assert (
        _tool_call_is_well_formed(
            {"choices": [{"message": {"tool_calls": [{"function": {"name": "f"}}]}}]}
        )
        is False
    )


def test_the_long_and_short_spelling_of_a_flag_are_the_same_flag() -> None:
    assert flags_from_argv(["llama-server", "--n-gpu-layers", "99"]) == {"ngl": "99"}
    assert flags_from_argv(["llama-server", "--ubatch-size", "512"]) == {"ub": "512"}
    assert flags_from_argv(["llama-server", "-ncmoe", "48"]) == {"n-cpu-moe": "48"}


def test_repeated_tensor_overrides_are_kept_and_a_repeated_flag_takes_its_last_value() -> None:
    found = flags_from_argv(["x", "-ot", "a=CPU", "-ot", "b=CPU", "-ngl", "0", "-ngl", "99"])
    assert found["ot"] == "a=CPU,b=CPU"
    assert found["ngl"] == "99"


def test_a_flag_at_the_end_with_no_value_is_not_read_as_one() -> None:
    assert flags_from_argv(["llama-server", "-ngl"]) == {}


def test_the_same_answer_spelled_differently_is_not_a_disagreement() -> None:
    """``-fa on`` against a reported ``1``, and ``99`` against ``99.0``."""
    assert (
        conditions_conflicts(
            conditions(
                argv=("llama-server", "-c", "32768", "-t", "16"),
                settings={"c": "32768.0", "t": " 16 "},
            )
        )
        == ()
    )


def test_a_tensor_override_is_recorded_and_never_compared() -> None:
    """The tool reports which tensors moved, not which patterns it was handed."""
    assert (
        conditions_conflicts(
            conditions(argv=("llama-server", "-ot", "ffn=CPU"), settings={"ot": "something else"})
        )
        == ()
    )


def test_a_setting_the_tool_says_nothing_about_cannot_disagree() -> None:
    assert conditions_conflicts(conditions(argv=("x", "-c", "4096"), settings={})) == ()


def test_a_disagreement_names_the_flag_and_both_values() -> None:
    problems = conditions_conflicts(
        conditions(argv=("x", "--n-cpu-moe", "48"), settings={"n-cpu-moe": "24"})
    )
    assert len(problems) == 1
    assert "--n-cpu-moe" in problems[0]
    assert "48" in problems[0] and "24" in problems[0]


def test_a_json_row_with_no_throughput_is_skipped_not_stored_as_zero() -> None:
    rows = parse_llama_bench_json(
        '[{"avg_ts": 0}, {"avg_ts": 12.5, "n_gen": 128, "flash_attn": true, "model_size": 0}]'
    )
    assert len(rows) == 1
    assert rows[0].kind == "llama-bench-tg"
    assert rows[0].settings["fa"] == "1"
    assert rows[0].model_bytes is None


def test_a_single_json_object_is_read_as_one_row() -> None:
    rows = parse_llama_bench_json('{"avg_ts": 12.5, "n_prompt": 512}')
    assert rows[0].kind == "llama-bench-pp"
    assert repr(rows[0]).startswith("BenchRow(")


def test_json_of_the_wrong_shape_altogether_is_refused() -> None:
    with pytest.raises(ValueError, match="not a list"):
        parse_llama_bench_json("42")


def test_a_buffer_line_whose_size_is_not_a_number_is_skipped() -> None:
    assert parse_buffer_sizes("llama_context: CUDA0 compute buffer size = .. MiB") == {}


def test_buffers_of_the_same_name_on_several_devices_are_summed() -> None:
    log = (
        "llama_context:      CUDA0 compute buffer size =  100.00 MiB\n"
        "llama_context:      CUDA0 compute buffer size =   50.00 MiB\n"
    )
    assert parse_buffer_sizes(log)["cuda0 compute"] == 150 * MIB


def test_every_refusal_reason_has_words_of_its_own() -> None:
    seen = {
        refusal_text(Refusal(parameter="x", reason=reason, measurements=2, parameters=3, value=1.6))
        for reason in (
            "no-data",
            "too-few-measurements",
            "not-identifiable",
            "unphysical",
            "discarded-with-the-fit",
            "no-term-in-the-model",
        )
    }
    assert len(seen) == 6


def test_a_refusal_with_no_value_still_reads_as_a_sentence() -> None:
    assert refusal_text(Refusal(parameter="x", reason="unphysical")).strip()
