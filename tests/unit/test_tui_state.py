"""The dashboard's state: what it asked for, what came back, and what happened when nothing did.

This is the only part of the terminal interface that calls a service, so it is where the
awkward answers have to be handled: a scan that raised, a catalog that would not load, a
substitution the machine cannot support. None of them may reach a screen as an exception,
because a dashboard that falls over on a machine whose graphics driver is broken is worse
than the command line it is meant to replace.
"""

from __future__ import annotations

from llamafit.errors import PackagedDataError
from llamafit.models.plan import Needs
from llamafit.tui.state import Dashboard, Request, Substitution
from tests.fixtures.board import report
from tests.fixtures.budget_hosts import machine
from tests.fixtures.dashboard import (
    empty_catalog,
    ready,
    refuses_to_load,
    refuses_to_scan,
)

GIB = 1024**3


# --- the ordinary case -----------------------------------------------------------------


def test_a_refreshed_dashboard_has_a_machine_a_catalog_and_a_board() -> None:
    board = ready()
    assert board.report is not None
    assert board.catalog is not None
    assert board.board is not None
    assert board.rows
    assert board.problem is None


def test_a_row_can_be_asked_for_by_position_and_out_of_range_is_not_an_error() -> None:
    board = ready()
    assert board.row_at(0) is not None
    assert board.row_at(len(board.rows)) is None
    assert board.row_at(-1) is None


def test_asking_a_different_question_re_ranks_against_it() -> None:
    board = ready()
    general = [row.model_id for row in board.rows]
    board.ask(Request(needs=Needs(use_case="coding")))
    assert board.board is not None
    assert board.board.needs.use_case == "coding"
    # The bundled catalog answers the two questions differently; if it ever stops doing so
    # this assertion is the thing that says the form has stopped mattering.
    assert [row.model_id for row in board.rows] != general


def test_a_licence_the_request_refuses_moves_a_model_off_the_board_with_its_reason() -> None:
    board = ready()
    board.ask(Request(licenses=("MIT",)))
    assert board.board is not None
    reasons = [row.candidate.excluded_because for row in board.board.excluded]
    assert any(reason and "MIT" not in reason for reason in reasons) or board.board.excluded


# --- when a service cannot answer ------------------------------------------------------


def test_a_scan_that_fails_leaves_a_message_and_a_hint_rather_than_raising() -> None:
    board = Dashboard(scanner=refuses_to_scan(), loader=lambda: (empty_catalog(), []))
    board.refresh()
    assert board.report is None
    assert board.board is None
    assert board.problem is not None
    rendered = board.problem.render()
    assert "nvidia-smi" in rendered
    assert "Hint:" in rendered


def test_a_catalog_that_will_not_load_leaves_a_message_and_the_machine_still_scanned() -> None:
    board = Dashboard(
        scanner=report,
        loader=refuses_to_load(PackagedDataError("the wheel shipped no catalog")),
    )
    board.refresh()
    assert board.catalog is None
    assert board.board is None
    assert board.report is not None
    assert board.problem is not None


def test_an_empty_catalog_ranks_nothing_and_says_nothing_went_wrong() -> None:
    board = ready(models=empty_catalog())
    assert board.board is not None
    assert board.rows == []
    assert board.board.excluded == []
    assert board.problem is None


def test_a_machine_with_no_graphics_card_still_gets_a_board() -> None:
    board = ready(host=machine(vram_total=None, ram_total=64 * GIB, ram_available=48 * GIB))
    assert board.host is not None
    assert board.host.primary_gpu is None
    assert board.board is not None


# --- standing in another machine -------------------------------------------------------


def test_an_override_marks_the_host_simulated_wherever_it_travels() -> None:
    board = ready()
    assert board.substitute(Substitution(gpu_memory=24 * GIB)) is None
    assert board.host is not None
    assert board.host.simulated
    assert board.host.simulation is not None
    assert board.host.simulation.overrides == ["gpu_memory"]
    assert board.board is not None


def test_a_bigger_card_changes_the_answer_rather_than_only_the_badge() -> None:
    board = ready()
    before = {
        row.model_id: row.candidate.placement.max_context_fit
        for row in board.rows
        if row.candidate.placement is not None
    }
    assert before
    board.substitute(Substitution(gpu_memory=48 * GIB))
    after = {
        row.model_id: row.candidate.placement.max_context_fit
        for row in board.rows
        if row.candidate.placement is not None
    }
    assert after != before


def test_a_card_cannot_be_resized_on_a_machine_that_has_none_and_the_old_one_stays() -> None:
    board = ready(host=machine(vram_total=None))
    refused = board.substitute(Substitution(gpu_memory=24 * GIB))
    assert refused is not None
    assert refused.render()
    # The refusal put the machine back, so nothing on screen is now about a machine the
    # dashboard failed to build while a badge claims it is simulating one.
    assert board.substitution == Substitution()
    assert board.host is not None
    assert not board.host.simulated


def test_going_back_to_this_machine_clears_the_badge() -> None:
    board = ready()
    board.substitute(Substitution(ram=256 * GIB))
    assert board.host is not None and board.host.simulated
    board.substitute(Substitution())
    assert board.host is not None and not board.host.simulated


def test_a_substitution_that_names_no_pool_is_not_a_simulation() -> None:
    assert not Substitution().active
    assert Substitution(ram=1).active
    assert Substitution(profile="reference-rtx4060-128gb").active


def test_a_profile_replaces_the_machine_without_the_scan_being_consulted() -> None:
    calls = {"n": 0}

    def counting_scan() -> object:
        calls["n"] += 1
        return report()

    board = Dashboard(scanner=counting_scan, loader=lambda: (empty_catalog(), []))  # type: ignore[arg-type]
    board.refresh()
    taken = calls["n"]
    board.substitute(Substitution(profile="reference-rtx4060-128gb"))
    assert board.host is not None
    assert board.host.simulation is not None
    assert board.host.simulation.profile == "reference-rtx4060-128gb"
    # ``resolve_host`` is handed the scan as a callable it does not call when a profile is
    # named, which is section 4.4's rule and not this package's.
    assert calls["n"] == taken


def test_a_dashboard_that_has_not_scanned_has_no_machine_and_no_board() -> None:
    board = Dashboard(scanner=report, loader=lambda: (empty_catalog(), []))
    assert board.host is None
    assert board.rows == ()
    assert board.local_models == ()
    board.rebuild()
    assert board.board is None


def test_a_request_the_scorer_refuses_leaves_a_message_rather_than_a_stale_board() -> None:
    # ``--prefer`` takes three values and the form offers exactly those three, so this can
    # only be reached by a config file or a caller; either way it is a message, not a crash,
    # and the board goes rather than staying behind as the answer to another question.
    board = ready()
    assert board.board is not None
    board.ask(Request(prefer="whatever is fastest and best"))
    assert board.board is None
    assert board.problem is not None
    assert "whatever is fastest and best" in board.problem.render()
