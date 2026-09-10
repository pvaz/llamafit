"""The two lines that are always on screen: which machine, and what a speed is.

The second matters more than it looks. Nothing has been benchmarked on any machine yet, so
every figure on the board came out of a formula, and the sentence saying so is the whole
of what stops a reader treating a prediction as a measurement.
"""

from __future__ import annotations

from llamafit.cli.render_board import confidence_sentence
from llamafit.models.host import Gpu
from llamafit.models.llamacpp import LlamaCpp
from llamafit.tui import summary
from tests.fixtures.budget_hosts import machine
from tests.fixtures.dashboard import ready

GIB = 1024**3


def test_before_a_scan_the_line_says_it_is_scanning_rather_than_showing_an_empty_machine() -> None:
    line = summary.machine_line(None, None)
    assert line
    assert "0" not in line


def test_the_line_names_the_card_the_memory_the_processor_and_llama_cpp() -> None:
    host = machine(vram_total=8 * GIB, vram_used=GIB, ram_total=64 * GIB)
    line = summary.machine_line(host, LlamaCpp(installed=True, build=10867))
    assert "Test GPU" in line
    assert "Test CPU" in line
    assert "10867" in line


def test_a_machine_with_no_card_says_so_rather_than_showing_nothing() -> None:
    line = summary.machine_line(machine(vram_total=None), None)
    assert summary.gpu_phrase(machine(vram_total=None)) in line
    assert "GPU" in line


def test_a_card_whose_size_nobody_could_read_is_not_a_card_of_zero_bytes() -> None:
    host = machine()
    host.gpus = [Gpu(index=0, vendor="nvidia", name="Mystery", backend_hint="cuda")]
    phrase = summary.gpu_phrase(host)
    assert "Mystery" in phrase
    assert "0" not in phrase


def test_llama_cpp_says_which_of_its_three_states_it_is_in() -> None:
    absent = summary.llamacpp_phrase(LlamaCpp(installed=False))
    unversioned = summary.llamacpp_phrase(LlamaCpp(installed=True))
    built = summary.llamacpp_phrase(LlamaCpp(installed=True, build=10867))
    assert len({absent, unversioned, built}) == 3


def test_a_scanned_machine_carries_no_badge_and_a_substituted_one_carries_the_word() -> None:
    board = ready()
    assert summary.simulated_badge(board.host) == ""
    from llamafit.tui.state import Substitution

    board.substitute(Substitution(ram=256 * GIB))
    assert summary.simulated_badge(board.host) == "SIMULATED"
    assert summary.simulated_badge(None) == ""


def test_the_band_says_today_every_speed_is_a_formula_and_nothing_is_a_measurement() -> None:
    band = summary.speed_band(ready().rows)
    assert band == confidence_sentence("estimated")
    assert "no figure here is a measurement" in band


def test_rows_that_disagree_get_a_band_pointing_at_the_column_instead() -> None:
    from llamafit.models.plan import Candidate, SpeedEstimate
    from llamafit.services.recommend import BoardRow

    def row(model_id: str, confidence: str) -> BoardRow:
        return BoardRow(
            rank=1,
            model_id=model_id,
            name=model_id,
            quant="Q4_K_M",
            candidate=Candidate(
                model_id=model_id,
                quant="Q4_K_M",
                speed=SpeedEstimate(gen_tps=1.0, pp_tps=1.0, confidence=confidence),  # type: ignore[arg-type]
            ),
        )

    band = summary.speed_band([row("a", "estimated"), row("b", "measured")])
    assert band != confidence_sentence("estimated")
    assert band != confidence_sentence("measured")


def test_a_board_with_nothing_on_it_makes_no_claim_about_speeds() -> None:
    assert summary.speed_band([]) == ""


def test_the_separator_is_a_translatable_entry_rather_than_a_join_in_python() -> None:
    assert summary.separator().strip()
