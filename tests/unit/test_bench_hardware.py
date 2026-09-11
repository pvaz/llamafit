"""``llamafit bench`` against the card in this machine, rather than against a recording.

Everything else about ``bench`` is tested from replayed output, which is the right way to
test orchestration and the wrong way to test a comparison: a recording of ``llama-bench``
agrees with whatever the estimate was compared against, because both were written down by
whoever wrote the fixture. The one claim that needs a card is the one the command exists to
make -- that the estimate and the measurement beside it are the same quantity.

They were not. The published sample had ``qwen3-0.6b`` generating at 279.8 tokens per
second against an estimate of 41.3 and called the quotient a 6.77 error, while
``docs/calibration/`` recorded the same machine at 279.5 -- the measurement was right to a
tenth of a percent and the comparison was wrong. The estimate had been made at the planned
32,768 tokens of key-value cache and the measurement at the 128 ``tg128`` fills. This file
is what would have caught that, and it needs the card because nothing short of the card can
produce the 279.

``--no-server``: the server half of the benchmark binds the plan's port, and a test that
takes port 8080 off whoever is sitting at the machine is not a test worth having. What it
would add here is the paging verdict, whose prose is pinned in ``test_cli_bench.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app

runner = CliRunner()

BIN = Path("D:/llama.cpp/bin")
MODEL = Path("D:/llama.cpp/models/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf")
TG_TOKENS = 128
"""What ``llama-bench``'s ``tg128`` generates, and so the cache it has filled by the end."""


def skip_unless_installed() -> None:
    """Skip when this machine has no llama.cpp or no copy of the model to measure."""
    if not any(BIN.glob("llama-bench*")):
        pytest.skip(f"{BIN} has no llama-bench")
    if not MODEL.exists():
        pytest.skip(f"{MODEL} is not on this machine")


@pytest.mark.hardware
def test_a_real_measurement_lands_beside_the_estimate_for_its_own_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The finding, in the one form that cannot be faked by a fixture.

    ``qwen3-0.6b`` is the model ``docs/calibration/`` used, so both halves of this row are
    known quantities on this machine. Anything far outside the band means the two columns
    have drifted apart again, and the direction says how: a ratio near 6.8 is the whole
    planned context against an empty one.
    """
    skip_unless_installed()
    monkeypatch.setenv("LLAMA_CPP_PATH", str(BIN))
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["--json", "bench", "qwen3-0.6b", "--no-server", "--no-store"])
    assert result.exit_code == 0, result.output

    report = json.loads(result.output)
    rows = {row["metric"]: row for row in report["comparison"]}
    generation = rows["generation-bench"]
    assert generation["context"] == TG_TOKENS
    assert report["planned_context"] > TG_TOKENS
    assert generation["measured"] > 100.0, "this is meant to be the card, not the processor"
    assert 0.8 <= generation["ratio"] <= 1.25, (
        f"generation came out at {generation['ratio']:.2f} of the estimate for"
        f" {generation['context']} tokens of cache; the planned context is"
        f" {report['planned_context']}, and a ratio near 6.8 is the two being compared again"
    )


@pytest.mark.hardware
def test_the_planned_figure_is_carried_without_being_compared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The estimate `plan` prints is at a context no row here reaches, and says so."""
    skip_unless_installed()
    monkeypatch.setenv("LLAMA_CPP_PATH", str(BIN))
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["--json", "bench", "qwen3-0.6b", "--no-server", "--no-store"])
    assert result.exit_code == 0, result.output

    report = json.loads(result.output)
    assert report["planned_gen_tps"] is not None
    contexts = {row["context"] for row in report["comparison"] if row["context"] is not None}
    assert report["planned_context"] not in contexts
