"""``llamafit bench`` through Typer's runner, with nothing real ever executed.

The orchestration itself is tested in ``test_bench_run.py`` against recorded output. What
is tested here is the command: what it prints, what it refuses to do, and that the estimate
survives into the table beside the measurement instead of being quietly replaced by it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llamafit.bench.fingerprint import host_fingerprint
from llamafit.bench.types import BenchReport, PagingCheck
from llamafit.cli.app import app
from llamafit.cli.bench_cmd import base_url_of
from llamafit.errors import NotInstalledError, ProbeError
from llamafit.models.host import Simulation
from llamafit.models.llamacpp import LlamaCpp, LocalModel
from llamafit.models.report import SystemReport
from tests.fixtures.bench import conditions, run_of
from tests.fixtures.board import model_and_quant
from tests.fixtures.budget_hosts import reference_host

runner = CliRunner()


def flat(output: str) -> str:
    """The output as one line, so an assertion is not defeated by where Rich wrapped."""
    return " ".join(output.split())


def system_report(tmp_path: Path, *, with_bench: bool = True, simulated: bool = False) -> object:
    """A scan of the reference machine with a llama.cpp installation that really exists."""
    binaries = tmp_path / "llama.cpp"
    binaries.mkdir(exist_ok=True)
    (binaries / "llama-server.exe").write_text("", encoding="utf-8")
    if with_bench:
        (binaries / "llama-bench.exe").write_text("", encoding="utf-8")
    _model, quant = model_and_quant("qwen3-coder-next")
    local = tmp_path / Path(quant.files[0]).name
    local.write_bytes(b"gguf")
    host = reference_host()
    if simulated:
        host = host.model_copy(update={"simulation": Simulation(profile="somebody-else")})
    return SystemReport(
        host=host,
        llamacpp=LlamaCpp(
            installed=True,
            path=str(binaries),
            build=10867,
            commit="f3f1a8f27",
            local_models=[LocalModel(path=str(local), bytes=4)],
        ),
        version="0.0.0-test",
    )


@pytest.fixture
def machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Every command sees the reference machine and a private data directory."""
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    # A table folded at eighty columns would make these assertions about where Rich broke
    # a word rather than about what the command said.
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr("llamafit.cli.bench_cmd.scan", lambda **_kwargs: system_report(tmp_path))
    yield tmp_path


def test_a_dry_run_prints_both_command_lines_and_runs_nothing(machine: Path) -> None:
    result = runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "llama-bench" in result.output
    assert "llama-server" in result.output
    assert "-o json" in flat(result.output)


def test_without_a_server_only_one_command_line_is_offered(machine: Path) -> None:
    result = runner.invoke(
        app,
        ["--language", "en", "bench", "qwen3-coder-next", "--dry-run", "--no-server"],
    )
    assert "llama-bench" in result.output
    assert "llama-server" not in result.output


def test_a_sweep_reaches_the_command_line(machine: Path) -> None:
    result = runner.invoke(
        app, ["--language", "en", "bench", "qwen3-coder-next", "--dry-run", "--sweep"]
    )
    assert "512" in flat(result.output)
    assert "2048" in flat(result.output)


def test_asking_for_nothing_in_particular_says_what_to_ask_for(machine: Path) -> None:
    result = runner.invoke(app, ["bench"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProbeError)
    assert "needs a model" in result.exception.render()


def test_a_simulated_machine_cannot_be_measured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "llamafit.cli.bench_cmd.scan",
        lambda **_kwargs: system_report(tmp_path, simulated=True),
    )
    result = runner.invoke(app, ["bench", "qwen3-coder-next"])
    assert isinstance(result.exception, ProbeError)
    assert "simulated" in result.exception.render()


def test_an_installation_without_llama_bench_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        "llamafit.cli.bench_cmd.scan",
        lambda **_kwargs: system_report(tmp_path, with_bench=False),
    )
    result = runner.invoke(app, ["bench", "qwen3-coder-next"])
    assert isinstance(result.exception, NotInstalledError)
    assert "llama-bench" in result.exception.render()


def test_a_model_that_is_not_on_the_disk_is_not_benchmarked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    report = system_report(tmp_path)
    report.llamacpp.local_models = []  # type: ignore[attr-defined]
    monkeypatch.setattr("llamafit.cli.bench_cmd.scan", lambda **_kwargs: report)
    result = runner.invoke(app, ["bench", "qwen3-coder-next"])
    assert isinstance(result.exception, ProbeError)
    assert "not on this machine" in result.exception.render()


def canned_report(fingerprint: str) -> BenchReport:
    """What a benchmark of the reference machine would have produced."""
    settings = {"ngl": "99", "n-cpu-moe": "47", "ub": "2048"}
    micro_batch = 2048
    return BenchReport(
        model_id="qwen3-coder-next",
        quant="UD-Q4_K_XL",
        host_fingerprint=fingerprint,
        command=["llama-bench", "-m", "model.gguf"],
        server_command=["llama-server", "-m", "model.gguf"],
        runs=[
            run_of(
                kind="llama-bench-tg",
                conditions_=conditions(
                    fingerprint=fingerprint,
                    settings=settings,
                    micro_batch=micro_batch,
                    n_prompt=0,
                    n_gen=128,
                ),
                gen_tps=24.7,
                estimated_gen_tps=19.0,
            ),
            run_of(
                kind="llama-bench-pp",
                conditions_=conditions(
                    fingerprint=fingerprint,
                    settings=settings,
                    micro_batch=micro_batch,
                    n_prompt=2048,
                    n_gen=0,
                ),
                pp_tps=323.0,
                estimated_pp_tps=300.0,
            ),
        ],
        paging=PagingCheck(paged=False, vram_ratio=0.5, reason="card-not-full"),
        planned_context=32768,
        planned_gen_tps=17.5,
    )


@pytest.fixture
def measured(machine: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A benchmark that produces recorded results instead of running anything."""
    fingerprint = host_fingerprint(reference_host())
    monkeypatch.setattr(
        "llamafit.cli.bench_cmd.run_benchmark",
        lambda *_args, **_kwargs: canned_report(fingerprint),
    )
    return machine


def test_the_table_shows_the_estimate_the_measurement_and_the_gap(measured: Path) -> None:
    result = runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next"])
    assert result.exit_code == 0, result.output
    line = flat(result.output)
    assert "Estimated against measured" in line
    assert "24.70" in line
    assert "19.00" in line
    assert "1.30" in line  # 24.7 / 19.0, outside the 0.8 to 1.25 band
    assert "outside the 0.8 to 1.25 band" in line


def test_the_table_says_what_context_and_micro_batch_each_row_is_for(measured: Path) -> None:
    """A ratio is only an error when both its halves answer the same question."""
    result = runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next"])
    line = flat(result.output)
    assert "context" in line
    assert "micro-batch" in line
    assert "128" in line  # the tg128 row's own cache depth, not the planned 32,768
    assert "2,048" in line


def test_the_planned_estimate_is_quoted_and_never_given_a_ratio(measured: Path) -> None:
    """The figure `plan` prints is at a context no row reaches, so it is not compared."""
    result = runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next"])
    line = flat(result.output)
    assert "sizes this configuration for 32,768 tokens" in line
    assert "17.50" in line
    assert "reported and not compared" in line


def test_the_paging_verdict_is_printed_with_the_reason_it_reached(measured: Path) -> None:
    result = runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next"])
    assert "was not paging" in flat(result.output)
    assert "nothing for the driver to page" in flat(result.output)


def verdict_line(monkeypatch: pytest.MonkeyPatch, check: PagingCheck) -> str:
    """What the command prints for one paging verdict, as one line."""
    fingerprint = host_fingerprint(reference_host())
    report = canned_report(fingerprint).model_copy(update={"paging": check})
    monkeypatch.setattr("llamafit.cli.bench_cmd.run_benchmark", lambda *_a, **_k: report)
    return flat(runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next"]).output)


def test_a_card_that_stayed_clear_gets_a_sentence_and_not_a_gap(
    machine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The line that shipped read "generation was not compared percent of the estimate".

    It was printed whenever the card stayed clear of full, which is the ordinary outcome
    rather than the rare one: the detector clears such a run on the memory signal alone
    and never reaches the speed half, so there is no ratio to substitute. Nothing tested
    the words this command produces, so a sentence with a hole in it shipped.
    """
    line = verdict_line(
        monkeypatch, PagingCheck(paged=False, vram_ratio=0.5, reason="card-not-full")
    )
    assert "not compared percent" not in line
    assert "Peak VRAM was 50 percent of the card." in line
    assert "Generation was not compared with the estimate." in line


def test_a_card_that_filled_up_quotes_both_figures_in_one_sentence(
    machine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    line = verdict_line(
        monkeypatch,
        PagingCheck(paged=False, vram_ratio=0.98, speed_ratio=0.92, reason="speed-as-expected"),
    )
    assert "Peak VRAM was 98 percent of the card, and generation was 92 percent" in line


def test_results_are_stored_and_come_back_from_show(measured: Path) -> None:
    assert runner.invoke(app, ["bench", "qwen3-coder-next"]).exit_code == 0
    listed = runner.invoke(app, ["--language", "en", "bench", "--show"])
    assert listed.exit_code == 0, listed.output
    assert "qwen3-coder-next" in listed.output
    assert "llama-bench-tg" in flat(listed.output)


def test_nothing_is_stored_when_the_caller_says_not_to(measured: Path) -> None:
    result = runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next", "--no-store"])
    assert "Nothing was written to the database" in flat(result.output)
    listed = runner.invoke(app, ["--language", "en", "bench", "--show"])
    assert "Nothing has been measured" in listed.output


def test_the_json_is_the_report_the_table_was_drawn_from(measured: Path) -> None:
    result = runner.invoke(app, ["--json", "bench", "qwen3-coder-next"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["model_id"] == "qwen3-coder-next"
    assert data["stored"] is True
    rows = {row["metric"]: row for row in data["comparison"]}
    assert rows["generation-bench"]["ratio"] == pytest.approx(24.7 / 19.0)
    assert rows["generation-bench"]["context"] == 128
    assert rows["generation-bench"]["micro_batch"] == 2048
    assert rows["prompt-bench"]["context"] == 2048
    assert rows["prompt-bench"]["micro_batch"] == 2048
    assert data["planned_context"] == 32768
    assert data["planned_gen_tps"] == pytest.approx(17.5)
    assert data["paging"]["paged"] is False


def test_calibrating_after_a_benchmark_names_what_it_could_not_fit(measured: Path) -> None:
    result = runner.invoke(app, ["--language", "en", "bench", "qwen3-coder-next", "--calibrate"])
    assert result.exit_code == 0, result.output
    line = flat(result.output)
    assert "Calibration" in line
    assert "was not fitted" in line
    assert "layer_overhead_ms" in line


def test_calibrating_on_its_own_works_from_what_is_already_stored(machine: Path) -> None:
    result = runner.invoke(app, ["--language", "en", "bench", "--calibrate"])
    assert result.exit_code == 0, result.output
    assert "Nothing was fitted" in flat(result.output)


def test_show_on_a_machine_that_has_measured_nothing_says_so(machine: Path) -> None:
    result = runner.invoke(app, ["--language", "en", "bench", "--show"])
    assert result.exit_code == 0, result.output
    assert "Nothing has been measured" in result.output


def test_show_as_json_is_a_list_even_when_it_is_empty(machine: Path) -> None:
    result = runner.invoke(app, ["--json", "bench", "--show"])
    assert json.loads(result.output) == []


def test_calibrating_as_json_carries_the_refusals(machine: Path) -> None:
    result = runner.invoke(app, ["--json", "bench", "--calibrate"])
    data = json.loads(result.output)
    assert {item["parameter"] for item in data["refusals"]} >= {"eff_vram", "layer_overhead_ms"}


def test_the_server_the_benchmark_talks_to_is_the_one_the_plan_renders() -> None:
    assert base_url_of(["llama-server", "--host", "127.0.0.1", "--port", "8123"]) == (
        "http://127.0.0.1:8123"
    )
    assert base_url_of(["llama-server"]) == "http://127.0.0.1:8080"


def test_the_command_is_documented() -> None:
    root = Path(__file__).resolve().parents[2]
    assert "`llamafit bench`" in (root / "docs" / "cli.md").read_text(encoding="utf-8")
