"""Running a benchmark with every external thing replaced by a recording.

Nothing here starts a process, opens a socket or needs a card. The four things ``bench``
reaches outside itself -- a command runner, a server launcher, an HTTP client and a VRAM
sampler -- are protocols with fakes shipped beside them, which is the same arrangement
``hardware/runner.py`` and ``llamacpp/server.py`` already use.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path

import pytest

from llamafit.bench.run import (
    BenchInputs,
    BenchOptions,
    FakeBenchHttp,
    FakeServerLauncher,
    FakeVramSampler,
    llama_bench_argv,
    long_prompt,
    run_benchmark,
    store_report,
    traffic_of,
    wait_for_health,
)
from llamafit.bench.store import BenchStore
from llamafit.errors import ProbeError
from llamafit.hardware.runner import CommandResult, FakeRunner
from llamafit.models.plan import Needs
from llamafit.services.plan import PlanReport, plan_report
from tests.fixtures.bench import (
    LLAMA_BENCH_MARKDOWN,
    SERVER_LOG,
    completion_response,
    llama_bench_json,
    tool_call_response,
)
from tests.fixtures.board import model_and_quant
from tests.fixtures.budget_hosts import reference_host

BASE_URL = "http://127.0.0.1:8080"
CARD = 8188 * 1024**2


def bench_runner(plan: PlanReport) -> FakeRunner:
    """A ``llama-bench`` that reports having run exactly the placement it was given."""
    placement = plan.placement
    return FakeRunner(
        {
            "llama-bench": llama_bench_json(
                ngl=placement.gpu_layers,
                micro_batch=placement.micro_batch,
                batch=placement.batch,
                threads=placement.threads,
                n_cpu_moe=placement.cpu_moe_layers,
                kv_type=placement.kv_type,
            )
        }
    )


def plan_for(model_id: str = "qwen3-coder-next") -> tuple[PlanReport, object, object]:
    """A real plan for a real bundled model on the recorded reference machine."""
    model, quant = model_and_quant(model_id)
    report = plan_report(
        model,
        quant,
        reference_host(),
        needs=Needs(use_case=model.use_cases[0]),
    )
    return report, model, quant


def inputs_for(plan: PlanReport, model: object, quant: object) -> BenchInputs:
    """The inputs ``run_benchmark`` takes, wired to the fixtures above."""
    return BenchInputs(
        plan=plan,
        host=reference_host(),
        facts=quant.gguf_facts,  # type: ignore[attr-defined]
        active_params=model.params.active_b * 1e9,  # type: ignore[attr-defined]
        bench_executable="llama-bench",
        server_argv=plan.command,
        base_url=BASE_URL,
        llama_cpp_build=10867,
        llama_cpp_commit="f3f1a8f27",
    )


def http_for(plan: PlanReport, *, well_formed_tool_call: bool = True) -> FakeBenchHttp:
    """A recorded server that is healthy and answers the three fixed requests."""
    return FakeBenchHttp(
        gets={
            f"{BASE_URL}/health": {"status": "ok"},
            f"{BASE_URL}/props": {
                "default_generation_settings": {"n_ctx": plan.placement.context},
                "build_info": "b10867",
            },
        },
        posts={
            f"{BASE_URL}/completion": [
                completion_response(
                    gen_tps=8.3,
                    pp_tps=12.0,
                    prompt_ms=2000.0,
                    tokens_evaluated=24,
                    tokens_predicted=128,
                ),
                completion_response(
                    gen_tps=13.7,
                    pp_tps=51.5,
                    prompt_ms=20.5,
                    tokens_evaluated=1056,
                    tokens_predicted=64,
                ),
            ],
            f"{BASE_URL}/v1/chat/completions": [
                tool_call_response(well_formed=well_formed_tool_call)
            ],
        },
    )


def test_the_bench_command_line_names_every_setting_the_placement_carries() -> None:
    plan, _model, _quant = plan_for("qwen3.8-flash-next")
    argv = llama_bench_argv(
        plan.placement, executable="llama-bench", model_path="m.gguf", options=BenchOptions()
    )
    line = " ".join(argv)
    assert "-ngl" in line
    assert "--n-cpu-moe" in line
    assert "-ot ffn_.*_shexp=CPU" in line
    assert f"-ub {plan.placement.micro_batch}" in line
    assert f"-t {plan.placement.threads}" in line
    assert "-o json" in line
    assert "-r 3" in line


def test_a_sweep_asks_for_every_micro_batch_on_one_command_line() -> None:
    plan, _model, _quant = plan_for()
    argv = llama_bench_argv(
        plan.placement,
        executable="llama-bench",
        model_path="m.gguf",
        options=BenchOptions(micro_batches=(512, 2048)),
    )
    index = argv.index("-ub")
    assert set(argv[index + 1].split(",")) == {str(plan.placement.micro_batch), "512", "2048"}


def test_a_whole_benchmark_produces_five_results_from_two_tools() -> None:
    plan, model, quant = plan_for()
    launcher = FakeServerLauncher(log_text=SERVER_LOG)
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=launcher,
        http=http_for(plan),
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        sleep=lambda _seconds: None,
    )
    assert [run.kind for run in result.runs] == [
        "llama-bench-pp",
        "llama-bench-tg",
        "server-short",
        "server-1k",
        "server-toolcall",
    ]
    assert launcher.started == [list(plan.command)]
    by_kind = {run.kind: run for run in result.runs}
    assert by_kind["llama-bench-tg"].gen_tps == pytest.approx(24.7)
    assert by_kind["llama-bench-pp"].pp_tps == pytest.approx(194.25)
    assert by_kind["server-short"].gen_tps == pytest.approx(8.3)
    assert by_kind["server-1k"].gen_tps == pytest.approx(13.7)
    assert by_kind["server-1k"].ttft_ms == pytest.approx(20.5)
    assert by_kind["server-toolcall"].tool_call_ok is True


def test_the_estimate_kept_beside_a_measurement_is_the_one_made_before_it() -> None:
    plan, model, quant = plan_for()
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=http_for(plan),
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        sleep=lambda _seconds: None,
    )
    assert plan.speed is not None
    generation = next(run for run in result.runs if run.kind == "llama-bench-tg")
    assert generation.estimated_gen_tps == pytest.approx(plan.speed.gen_tps)
    assert generation.estimated_gen_tps != generation.gen_tps


def test_the_server_s_own_buffer_sizes_are_kept_with_the_run() -> None:
    plan, model, quant = plan_for()
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=http_for(plan),
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        sleep=lambda _seconds: None,
    )
    warm = next(run for run in result.runs if run.kind == "server-1k")
    assert warm.buffer_bytes["cuda0 compute"] > 0
    assert warm.buffer_bytes["cuda0 kv"] > 0


def test_a_run_that_filled_the_card_and_crawled_is_reported_as_paging() -> None:
    plan, model, quant = plan_for()
    slow = http_for(plan)
    slow.posts[f"{BASE_URL}/completion"][1] = completion_response(
        gen_tps=3.0, pp_tps=10.0, prompt_ms=200.0, tokens_evaluated=1056, tokens_predicted=64
    )
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=slow,
        sampler=FakeVramSampler(readings=[CARD]),
        sleep=lambda _seconds: None,
    )
    assert result.paging is not None
    assert result.paging.paged is True


def test_no_vram_reading_leaves_the_paging_question_open_rather_than_answered() -> None:
    plan, model, quant = plan_for()
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=http_for(plan),
        sampler=FakeVramSampler(readings=[]),
        sleep=lambda _seconds: None,
    )
    assert result.paging is not None
    assert result.paging.paged is None
    assert result.paging.reason == "no-vram-reading"


def test_a_tool_call_written_as_prose_is_not_a_tool_call() -> None:
    plan, model, quant = plan_for()
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=http_for(plan, well_formed_tool_call=False),
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        sleep=lambda _seconds: None,
    )
    call = next(run for run in result.runs if run.kind == "server-toolcall")
    assert call.tool_call_ok is False


def test_without_a_server_there_are_two_results_and_no_paging_verdict() -> None:
    plan, model, quant = plan_for()
    launcher = FakeServerLauncher(log_text=SERVER_LOG)
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=launcher,
        http=http_for(plan),
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        options=BenchOptions(with_server=False),
        sleep=lambda _seconds: None,
    )
    assert len(result.runs) == 2
    assert result.paging is None
    assert launcher.started == []
    assert result.server_command == []


def test_a_build_too_old_for_json_still_gets_its_table_read() -> None:
    plan, model, quant = plan_for()
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=FakeRunner({"llama-bench": LLAMA_BENCH_MARKDOWN}),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=http_for(plan),
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        options=BenchOptions(with_server=False),
        sleep=lambda _seconds: None,
    )
    assert [run.kind for run in result.runs] == ["llama-bench-pp", "llama-bench-tg"]


def test_a_benchmark_that_did_not_run_is_an_error_and_not_an_empty_result() -> None:
    plan, model, quant = plan_for()
    with pytest.raises(ProbeError):
        run_benchmark(
            inputs_for(plan, model, quant),
            runner=FakeRunner(
                {"llama-bench": CommandResult(["llama-bench"], 1, "", "out of memory", 10)}
            ),
            launcher=FakeServerLauncher(),
            http=FakeBenchHttp(),
            sampler=FakeVramSampler(),
            sleep=lambda _seconds: None,
        )


def test_a_server_that_never_answers_is_stopped_and_reported() -> None:
    plan, model, quant = plan_for()
    launcher = FakeServerLauncher(log_text="failed to allocate\n", alive=False)
    with pytest.raises(ProbeError, match="healthy"):
        run_benchmark(
            inputs_for(plan, model, quant),
            runner=bench_runner(plan),
            launcher=launcher,
            http=FakeBenchHttp(),
            sampler=FakeVramSampler(),
            sleep=lambda _seconds: None,
        )
    assert launcher.started


def test_storing_a_report_gives_every_result_an_identity(tmp_path: Path) -> None:
    plan, model, quant = plan_for()
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=http_for(plan),
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        sleep=lambda _seconds: None,
    )
    with closing(BenchStore(tmp_path / "benchmarks.sqlite")) as store:
        stored = store_report(store, result)
        assert stored.stored is True
        assert all(run.id != "pending" for run in stored.runs)
        assert len(store.runs()) == len(result.runs)


def test_a_server_running_at_a_context_it_was_not_asked_for_is_refused(tmp_path: Path) -> None:
    """llama.cpp clamps a context it cannot honour, and a record must not claim otherwise."""
    plan, model, quant = plan_for()
    lying = http_for(plan)
    lying.gets[f"{BASE_URL}/props"] = {"default_generation_settings": {"n_ctx": 4096}}
    result = run_benchmark(
        inputs_for(plan, model, quant),
        runner=bench_runner(plan),
        launcher=FakeServerLauncher(log_text=SERVER_LOG),
        http=lying,
        sampler=FakeVramSampler(readings=[7 * 1024**3]),
        sleep=lambda _seconds: None,
    )
    with (
        closing(BenchStore(tmp_path / "benchmarks.sqlite")) as store,
        pytest.raises(ProbeError, match="did not use the settings"),
    ):
        store_report(store, result)


def test_the_traffic_recorded_with_a_run_is_the_traffic_the_estimate_was_made_from() -> None:
    plan, model, quant = plan_for("qwen3.8-flash-next")
    recorded = traffic_of(
        plan.placement,
        quant.gguf_facts,  # type: ignore[union-attr]
        reference_host(),
        active_params=model.params.active_b * 1e9,  # type: ignore[attr-defined]
    )
    assert recorded.scattered_bytes > 0
    assert recorded.device_bytes > 0
    assert recorded.ram_gbps > 0
    assert recorded.micro_batch == plan.placement.micro_batch


def test_the_long_prompt_is_the_same_prompt_every_time() -> None:
    assert long_prompt() == long_prompt()
    assert 900 <= len(long_prompt().split()) <= 1100


def test_waiting_gives_up_when_the_process_has_already_died() -> None:
    launcher = FakeServerLauncher(alive=False)
    handle = launcher.start(["llama-server"])
    ticks = iter([0.0, 1.0, 2.0])
    assert not wait_for_health(
        FakeBenchHttp(),
        BASE_URL,
        timeout=5.0,
        sleep=lambda _seconds: None,
        clock=lambda: next(ticks),
        handle=handle,
    )


def test_waiting_gives_up_when_the_time_runs_out() -> None:
    ticks = iter([0.0, 10.0])
    assert not wait_for_health(
        FakeBenchHttp(),
        BASE_URL,
        timeout=1.0,
        sleep=lambda _seconds: None,
        clock=lambda: next(ticks),
    )
