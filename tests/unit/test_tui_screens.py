"""The corners of each screen: the second button, the empty field, the answer that failed.

The paths a reader meets when something is not as expected. A plan for a quantisation
nobody has read the header of, a copy with nothing to copy, a form field left blank, a
profile directory that did not ship. None of them may raise past the screen, and each has
to say which of them happened rather than leaving the pane where a figure would be.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console, RenderableType
from textual.widgets import Button, Input, Static

from llamafit.errors import BudgetError, PackagedDataError
from llamafit.services.recommend import BoardRow
from llamafit.tui import explain
from llamafit.tui import format as fmt
from llamafit.tui.app import LlamaFitApp
from llamafit.tui.screens import plan as plan_screen
from llamafit.tui.screens import simulate as simulate_screen
from llamafit.tui.screens.board import BoardPane
from llamafit.tui.screens.needs import NeedsPane
from llamafit.tui.screens.plan import PlanPane
from llamafit.tui.screens.simulate import SimulatePane
from llamafit.tui.widgets import RichPane
from llamafit.units import format_bytes
from tests.fixtures.dashboard import empty_catalog, ready

GIB = 1024**3


def drawn(renderable: RenderableType, width: int = 120) -> str:
    console = Console(width=width, no_color=True, highlight=False, record=True, file=io.StringIO())
    console.print(renderable)
    return " ".join(console.export_text().split())


def pane_text(app: LlamaFitApp, selector: str, width: int = 120) -> str:
    return drawn(app.query_one(selector, RichPane).renderable, width)


# --- the plan ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_command_line_goes_to_the_clipboard_whole() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.press("p")
        plan = app.query_one("#plan", PlanPane)
        await pilot.press("y")
        assert app.clipboard == " ".join(plan.command)
        assert "llama-server" in app.clipboard


@pytest.mark.asyncio
async def test_copying_before_there_is_anything_to_copy_says_so() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        app.query_one("#plan", PlanPane).action_copy()
        await pilot.pause()
        assert app.clipboard == ""


@pytest.mark.asyncio
async def test_a_row_whose_model_has_left_the_catalog_says_that_rather_than_raising() -> None:
    dashboard = ready()
    row = dashboard.rows[0]
    app = LlamaFitApp(dashboard)
    async with app.run_test(size=(120, 44)) as pilot:
        gone = BoardRow(
            rank=1,
            model_id="a-model-nobody-has",
            name="Gone",
            quant=row.quant,
            candidate=row.candidate,
        )
        app.query_one("#plan", PlanPane).show_row(gone)
        await pilot.pause()
        assert "a-model-nobody-has" in pane_text(app, "#plan-body")


@pytest.mark.asyncio
async def test_a_quantisation_nobody_has_read_the_header_of_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> object:
        raise BudgetError(
            "nobody has read this file's header", hint="Run `llamafit catalog refresh`."
        )

    dashboard = ready()
    app = LlamaFitApp(dashboard)
    async with app.run_test(size=(120, 44)) as pilot:
        monkeypatch.setattr(plan_screen, "plan_report", refuse)
        app.query_one("#plan", PlanPane).show_row(dashboard.rows[0])
        await pilot.pause()
        text = pane_text(app, "#plan-body")
        assert "nobody has read this file's header" in text
        assert "Hint:" in text
        assert app.query_one("#plan", PlanPane).command == []


@pytest.mark.asyncio
async def test_a_quantisation_the_catalog_no_longer_publishes_is_named() -> None:
    dashboard = ready()
    row = dashboard.rows[0]
    app = LlamaFitApp(dashboard)
    async with app.run_test(size=(120, 44)) as pilot:
        missing = BoardRow(
            rank=1,
            model_id=row.model_id,
            name=row.name,
            quant="Q0_NOT_REAL",
            candidate=row.candidate,
        )
        app.query_one("#plan", PlanPane).show_row(missing)
        await pilot.pause()
        assert row.model_id in pane_text(app, "#plan-body")


@pytest.mark.asyncio
async def test_the_context_keys_do_nothing_before_a_row_has_been_chosen() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        plan = app.query_one("#plan", PlanPane)
        plan.action_longer()
        plan.action_shorter()
        await pilot.pause()
        assert plan.context is None


@pytest.mark.asyncio
async def test_the_ladder_says_so_when_there_is_no_rung_below() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.press("p")
        plan = app.query_one("#plan", PlanPane)
        for _ in range(len(plan.tiers) + 2):
            await pilot.press("minus")
        assert pane_text(app, "#plan-notice")


# --- the board -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enter_opens_the_explanation_when_it_had_been_closed() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.press("x")
        assert app.query_one("#board-why").has_class("hidden")
        await pilot.press("enter")
        assert not app.query_one("#board-why").has_class("hidden")


@pytest.mark.asyncio
async def test_asking_for_a_plan_with_no_row_to_plan_does_nothing() -> None:
    app = LlamaFitApp(ready(models=empty_catalog()))
    async with app.run_test(size=(120, 44)) as pilot:
        assert app.query_one("#board", BoardPane).selected_row is None
        await pilot.press("p")
        await pilot.pause()
        assert app.query_one("#tabs").active == "tab-board"


# --- the form ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_reset_button_does_what_the_reset_key_does() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        app.query_one("#tabs").active = "tab-needs"
        await pilot.pause()
        app.query_one("#needs-min-context", Input).value = "65536"
        app.query_one("#needs-apply", Button).press()
        await pilot.pause()
        assert app.query_one("#needs", NeedsPane).dashboard.request.needs.min_context == 65536
        app.query_one("#tabs").active = "tab-needs"
        await pilot.pause()
        app.query_one("#needs-reset", Button).press()
        await pilot.pause()
        assert app.query_one("#needs", NeedsPane).dashboard.request.needs.min_context == 0


@pytest.mark.asyncio
async def test_a_negative_minimum_context_is_refused_by_name() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        needs = app.query_one("#needs", NeedsPane)
        app.query_one("#needs-min-context", Input).value = "-1"
        needs.action_apply()
        await pilot.pause()
        assert str(app.query_one("#needs-error", Static).render())


@pytest.mark.asyncio
async def test_blank_fields_mean_no_limit_rather_than_a_limit_of_nothing() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        needs = app.query_one("#needs", NeedsPane)
        app.query_one("#needs-min-context", Input).value = ""
        app.query_one("#needs-max-download", Input).value = ""
        needs.action_apply()
        await pilot.pause()
        asked = needs.dashboard.request.needs
        assert asked.min_context == 0
        assert asked.max_download_bytes is None


# --- simulating ------------------------------------------------------------------------------


def test_a_profile_directory_that_did_not_ship_is_an_answer_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**_kwargs: object) -> object:
        raise PackagedDataError("this wheel shipped no hardware profiles")

    monkeypatch.setattr(simulate_screen, "load_profiles", refuse)
    profiles, problem = simulate_screen.available_profiles()
    assert profiles == []
    assert problem is not None


@pytest.mark.asyncio
async def test_the_simulate_screen_says_so_when_it_has_no_profiles_to_offer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(**_kwargs: object) -> object:
        raise PackagedDataError("this wheel shipped no hardware profiles")

    monkeypatch.setattr(simulate_screen, "load_profiles", refuse)
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)):
        assert "no hardware profiles" in str(app.query_one("#simulate-error", Static).render())


@pytest.mark.asyncio
async def test_the_apply_button_does_what_the_apply_key_does() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        app.query_one("#tabs").active = "tab-simulate"
        await pilot.pause()
        app.query_one("#simulate-ram", Input).value = "256G"
        app.query_one("#simulate-apply", Button).press()
        await pilot.pause()
        host = app.query_one("#simulate", SimulatePane).dashboard.host
        assert host is not None and host.simulated


@pytest.mark.asyncio
async def test_a_pretend_card_size_that_is_not_a_size_says_what_one_looks_like() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        app.query_one("#simulate-vram", Input).value = "as much as you like"
        app.query_one("#simulate", SimulatePane).action_apply()
        await pilot.pause()
        assert "7.5GiB" in str(app.query_one("#simulate-error", Static).render())
        assert not app.query_one("#simulate", SimulatePane).dashboard.substitution.active


@pytest.mark.asyncio
async def test_the_back_button_does_what_the_back_key_does() -> None:
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)) as pilot:
        app.query_one("#simulate-ram", Input).value = "256G"
        app.query_one("#simulate", SimulatePane).action_apply()
        await pilot.pause()
        simulated = app.query_one("#simulate", SimulatePane).dashboard.host
        assert simulated is not None and simulated.simulated
        app.query_one("#tabs").active = "tab-simulate"
        await pilot.pause()
        app.query_one("#simulate-reset", Button).press()
        await pilot.pause()
        host = app.query_one("#simulate", SimulatePane).dashboard.host
        assert host is not None and not host.simulated


# --- the small pieces --------------------------------------------------------------------------


def test_a_share_nobody_could_compute_is_a_word_and_not_zero_per_cent() -> None:
    assert fmt.percent(None) != fmt.percent(0.0)
    assert "50" in fmt.percent(0.5)


def test_a_context_that_is_not_a_round_number_of_kibitokens_is_written_out() -> None:
    assert "K" in fmt.context(32768)
    assert "K" not in fmt.context(1000)


def test_a_size_nobody_filled_in_is_the_word_for_unknown() -> None:
    assert fmt.size(None) == format_bytes(None)
    assert "GiB" in fmt.size(4 * GIB)


def test_a_row_can_be_named_without_a_catalog_to_look_it_up_in() -> None:
    dashboard = ready()
    row = dashboard.rows[0]
    without = explain.identity_line(row, None)
    assert row.quant in without
    assert "licensed" not in without


def test_the_not_ranked_pane_answers_before_there_is_a_board_to_answer_from() -> None:
    assert drawn(explain.not_ranked(None))
    board = ready(models=empty_catalog()).board
    assert drawn(explain.not_ranked(board))


@pytest.mark.asyncio
async def test_a_pane_can_be_given_plain_text_that_is_never_read_as_markup() -> None:
    # A model id, a path or a flag may hold square brackets, and a pane that parsed them as
    # a style tag would silently drop the part of the line that identifies the thing.
    app = LlamaFitApp(ready())
    async with app.run_test(size=(120, 44)):
        pane = app.query_one("#plan-notice", RichPane)
        pane.show_text("[not a tag] --verbose")
        assert "[not a tag] --verbose" in drawn(pane.renderable)
