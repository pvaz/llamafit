"""The dashboard driven by Textual's own harness: navigation, state, and what is on screen.

Nothing here re-tests a number. The budget, the speed, the score and the ranking are
tested underneath, against recorded machines, and a dashboard that recomputed any of them
would be the bug. What these tests are about is the layer this package adds: which screen
a key takes you to, what a filter hides and whether it says so, whether the explanation
follows the cursor, and whether a command line comes out at the end of it.

A pane's contents are checked by drawing the Rich renderable it holds through a console of
a known width -- the same thing the command line's own renderer tests do. A picture of a
terminal would be a screenshot nobody can check.
"""

from __future__ import annotations

import io
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from rich.console import Console, RenderableType
from textual.widgets import Button, Checkbox, DataTable, Input, Select, Static

from llamafit.tui.app import LlamaFitApp
from llamafit.tui.screens.board import BoardPane
from llamafit.tui.screens.help import KeyMapScreen
from llamafit.tui.screens.needs import NeedsPane
from llamafit.tui.screens.plan import PlanPane
from llamafit.tui.screens.simulate import SimulatePane
from llamafit.tui.state import Dashboard
from llamafit.tui.widgets import RichPane
from tests.fixtures.board import report
from tests.fixtures.budget_hosts import machine
from tests.fixtures.dashboard import empty_catalog, ready, unscanned

GIB = 1024**3
SIZE = (120, 44)


def drawn(renderable: RenderableType, width: int = 100) -> str:
    """What a pane's renderable puts on a console, as one flat string."""
    console = Console(width=width, no_color=True, highlight=False, record=True, file=io.StringIO())
    console.print(renderable)
    return " ".join(console.export_text().split())


def pane_text(app: LlamaFitApp, selector: str, width: int = 100) -> str:
    """What one Rich pane currently says."""
    return drawn(app.query_one(selector, RichPane).renderable, width)


def label(app: LlamaFitApp, selector: str) -> str:
    """What one plain line currently says."""
    return str(app.query_one(selector, Static).render())


@asynccontextmanager
async def running(
    dashboard: Dashboard | None = None, size: tuple[int, int] = SIZE
) -> AsyncIterator:
    """A mounted dashboard over a machine and a catalog that were handed to it."""
    app = LlamaFitApp(dashboard if dashboard is not None else ready())
    async with app.run_test(size=size) as pilot:
        yield pilot


def table_of(app: LlamaFitApp) -> DataTable[object]:
    return app.query_one("#board-table", DataTable)


def headings(app: LlamaFitApp) -> list[str]:
    return [str(column.label) for column in table_of(app).columns.values()]


def column_of(app: LlamaFitApp, heading: str) -> list[str]:
    index = headings(app).index(heading)
    table = table_of(app)
    return [str(table.get_row_at(row)[index]) for row in range(table.row_count)]


# --- the first five seconds ------------------------------------------------------------


@pytest.mark.asyncio
async def test_it_opens_on_the_board_with_rows_and_a_line_saying_what_to_do() -> None:
    async with running() as pilot:
        app = pilot.app
        assert app.query_one("#tabs").active == "tab-board"
        assert table_of(app).row_count > 0
        hint = label(app, "#board-hint")
        # A newcomer must be able to act without reading a flag, so the two keys that do
        # something are named on the screen rather than behind a question mark.
        assert "Enter" in hint
        assert "p " in hint or hint.endswith("p")


@pytest.mark.asyncio
async def test_the_band_says_every_speed_is_a_formula_before_a_reader_reads_one() -> None:
    async with running() as pilot:
        band = label(pilot.app, "#board-band")
        assert "no figure here is a measurement" in band
        assert "benchmarked on this machine" in band


@pytest.mark.asyncio
async def test_the_header_names_the_machine_every_figure_is_about() -> None:
    async with running() as pilot:
        line = label(pilot.app, "#machine-line")
        assert "RTX 4060" in line
        assert "llama.cpp" in line


@pytest.mark.asyncio
async def test_the_explanation_is_open_before_anybody_asks_for_it() -> None:
    async with running() as pilot:
        text = pane_text(pilot.app, "#board-why-pane")
        # Section 12.3's promise: the score with its parts, the weights, and the budget.
        assert "Score" in text
        assert "quality" in text and "Weight" in text
        assert "Component" in text
        assert "Context tiers" in text


@pytest.mark.asyncio
async def test_the_explanation_follows_the_cursor() -> None:
    async with running() as pilot:
        app = pilot.app
        first = pane_text(app, "#board-why-pane")
        await pilot.press("down")
        assert pane_text(app, "#board-why-pane") != first


@pytest.mark.asyncio
async def test_a_row_expands_into_the_licence_and_the_capabilities_the_board_has_no_room_for() -> (
    None
):
    async with running() as pilot:
        text = pane_text(pilot.app, "#board-why-pane")
        assert "licensed" in text
        assert "it can:" in text


# --- getting a command line --------------------------------------------------------------


@pytest.mark.asyncio
async def test_pressing_p_gives_the_command_line_that_runs_the_selected_row() -> None:
    async with running() as pilot:
        app = pilot.app
        chosen = app.query_one("#board", BoardPane).selected_row
        assert chosen is not None
        await pilot.press("p")
        assert app.query_one("#tabs").active == "tab-plan"
        text = pane_text(app, "#plan-body", width=200)
        assert "Command line" in text
        assert "llama-server" in text
        assert chosen.model_id in text or chosen.name in text


@pytest.mark.asyncio
async def test_the_plan_shows_the_budget_the_ladder_and_where_a_token_s_time_goes() -> None:
    async with running() as pilot:
        await pilot.press("p")
        text = pane_text(pilot.app, "#plan-body", width=200)
        assert "Memory budget" in text
        assert "Context tiers" in text
        assert "A token's time" in text


@pytest.mark.asyncio
async def test_the_context_keys_move_along_the_ladder_the_planner_produced() -> None:
    async with running() as pilot:
        app = pilot.app
        await pilot.press("p")
        plan = app.query_one("#plan", PlanPane)
        rungs = plan.tiers
        assert rungs
        await pilot.press("plus")
        assert plan.context in rungs
        moved = plan.context
        await pilot.press("minus")
        assert plan.context != moved


@pytest.mark.asyncio
async def test_the_ladder_says_so_rather_than_inventing_a_rung_it_has_not_got() -> None:
    async with running() as pilot:
        app = pilot.app
        await pilot.press("p")
        plan = app.query_one("#plan", PlanPane)
        for _ in range(len(plan.tiers) + 2):
            await pilot.press("plus")
        assert plan.context in plan.tiers
        assert pane_text(app, "#plan-notice")


@pytest.mark.asyncio
async def test_turning_the_vision_projector_off_replans_rather_than_relabelling() -> None:
    async with running() as pilot:
        app = pilot.app
        await pilot.press("p")
        plan = app.query_one("#plan", PlanPane)
        before = pane_text(app, "#plan-body", width=200)
        await pilot.press("v")
        assert plan.vision is False
        assert pane_text(app, "#plan-body", width=200) != before


@pytest.mark.asyncio
async def test_the_plan_screen_says_what_to_do_before_a_row_has_been_chosen() -> None:
    async with running() as pilot:
        text = pane_text(pilot.app, "#plan-body")
        assert "Board" in text and "p" in text


# --- looking at the board differently ----------------------------------------------------


@pytest.mark.asyncio
async def test_sorting_reorders_the_screen_and_leaves_the_ranking_alone() -> None:
    async with running() as pilot:
        app = pilot.app
        before = column_of(app, "#")
        await pilot.press("s")
        after = column_of(app, "#")
        assert sorted(after) == sorted(before)
        assert app.query_one("#board", BoardPane).view.sort == "speed"
        assert "speed" in label(app, "#board-state")


@pytest.mark.asyncio
async def test_the_fit_filter_cycles_and_the_state_line_says_which_one_is_on() -> None:
    async with running() as pilot:
        app = pilot.app
        board = app.query_one("#board", BoardPane)
        every = table_of(app).row_count
        await pilot.press("f")
        assert board.view.filters.min_fit == "tight"
        assert table_of(app).row_count <= every
        await pilot.press("f")
        assert board.view.filters.min_fit == "fits"
        await pilot.press("f")
        assert board.view.filters.min_fit == "comfortable"
        from llamafit.cli.render_board import filter_label

        assert filter_label("comfortable") in label(app, "#board-state")
        await pilot.press("f")
        assert board.view.filters.min_fit is None


@pytest.mark.asyncio
async def test_showing_only_what_is_on_disk_empties_the_table_and_says_why() -> None:
    async with running() as pilot:
        app = pilot.app
        await pilot.press("a")
        assert table_of(app).row_count == 0
        state = label(app, "#board-state")
        assert "0" in state
        board = app.query_one("#board", BoardPane).dashboard.board
        assert board is not None
        assert str(len(board.rows) + len(board.excluded)) in state


@pytest.mark.asyncio
async def test_searching_narrows_the_table_and_puts_the_cursor_back_on_it() -> None:
    async with running() as pilot:
        app = pilot.app
        wanted = app.query_one("#board", BoardPane).dashboard.rows[0].model_id
        await pilot.press("slash")
        box = app.query_one("#board-search", Input)
        assert box.has_focus
        box.value = wanted
        await pilot.press("enter")
        assert table_of(app).row_count == 1
        assert column_of(app, "Model") == [wanted]
        assert table_of(app).has_focus


@pytest.mark.asyncio
async def test_all_quantisations_shows_more_rows_than_the_best_one_per_model() -> None:
    async with running() as pilot:
        app = pilot.app
        before = table_of(app).row_count
        await pilot.press("A")
        assert app.query_one("#board", BoardPane).dashboard.request.all_quants
        assert table_of(app).row_count >= before


@pytest.mark.asyncio
async def test_the_explanation_can_be_closed_to_give_the_table_the_screen() -> None:
    async with running() as pilot:
        app = pilot.app
        await pilot.press("x")
        assert app.query_one("#board-why").has_class("hidden")
        await pilot.press("x")
        assert not app.query_one("#board-why").has_class("hidden")


@pytest.mark.asyncio
async def test_the_candidates_that_did_not_qualify_are_one_key_away_with_their_reasons() -> None:
    async with running() as pilot:
        app = pilot.app
        await pilot.press("n")
        await pilot.pause()
        board = app.query_one("#board", BoardPane)
        row = board.selected_row
        # The cursor lands on the first unranked row, which is on the table with the
        # ranked ones, and the pane under it says why that one did not qualify.
        assert row is not None and row.rank is None
        assert row.candidate.excluded_because
        text = pane_text(app, "#board-why-pane", width=200)
        assert " ".join(row.candidate.excluded_because.split())[:40] in text


# --- asking a different question ---------------------------------------------------------


@pytest.mark.asyncio
async def test_the_needs_form_re_ranks_the_board_and_goes_back_to_it() -> None:
    async with running() as pilot:
        app = pilot.app
        before = column_of(app, "Model")
        app.query_one("#tabs").active = "tab-needs"
        await pilot.pause()
        app.query_one("#needs-use-case", Select).value = "coding"
        app.query_one("#needs", NeedsPane).action_apply()
        await pilot.pause()
        assert app.query_one("#tabs").active == "tab-board"
        board = app.query_one("#board", BoardPane).dashboard.board
        assert board is not None and board.needs.use_case == "coding"
        assert column_of(app, "Model") != before


@pytest.mark.asyncio
async def test_the_form_offers_the_licences_the_catalog_actually_carries() -> None:
    async with running() as pilot:
        app = pilot.app
        catalog = app.query_one("#needs", NeedsPane).dashboard.catalog
        assert catalog is not None
        offered = {
            str(option.value)
            for option in app.query_one("#needs-licenses").options  # type: ignore[attr-defined]
        }
        assert offered == {model.license.spdx for model in catalog.models}


@pytest.mark.asyncio
async def test_a_field_that_is_not_a_number_says_so_and_changes_nothing() -> None:
    async with running() as pilot:
        app = pilot.app
        needs = app.query_one("#needs", NeedsPane)
        before = needs.dashboard.request
        app.query_one("#needs-min-context", Input).value = "lots"
        needs.action_apply()
        await pilot.pause()
        assert label(app, "#needs-error")
        assert needs.dashboard.request == before


@pytest.mark.asyncio
async def test_a_download_ceiling_that_is_not_a_size_says_what_a_size_looks_like() -> None:
    async with running() as pilot:
        app = pilot.app
        needs = app.query_one("#needs", NeedsPane)
        app.query_one("#needs-max-download", Input).value = "quite big"
        needs.action_apply()
        await pilot.pause()
        assert "7.5GiB" in label(app, "#needs-error")


@pytest.mark.asyncio
async def test_reset_puts_every_field_back_and_asks_the_question_again() -> None:
    async with running() as pilot:
        app = pilot.app
        needs = app.query_one("#needs", NeedsPane)
        app.query_one("#needs-use-case", Select).value = "coding"
        app.query_one("#needs-min-context", Input).value = "65536"
        needs.action_apply()
        await pilot.pause()
        needs.action_reset()
        await pilot.pause()
        assert app.query_one("#needs-use-case", Select).value == "general"
        assert needs.dashboard.request.needs.use_case == "general"
        assert needs.dashboard.request.needs.min_context == 0


@pytest.mark.asyncio
async def test_the_two_buttons_do_what_the_two_keys_do() -> None:
    async with running() as pilot:
        app = pilot.app
        app.query_one("#tabs").active = "tab-needs"
        await pilot.pause()
        app.query_one("#needs-vision", Checkbox).value = False
        app.query_one("#needs-apply", Button).press()
        await pilot.pause()
        assert app.query_one("#needs", NeedsPane).dashboard.request.vision is False


# --- standing in another machine ---------------------------------------------------------


@pytest.mark.asyncio
async def test_simulating_a_bigger_card_puts_the_badge_up_and_re_ranks() -> None:
    async with running() as pilot:
        app = pilot.app
        before = column_of(app, "Ctx")
        app.query_one("#tabs").active = "tab-simulate"
        await pilot.pause()
        app.query_one("#simulate-vram", Input).value = "48G"
        app.query_one("#simulate", SimulatePane).action_apply()
        await pilot.pause()
        assert label(app, "#machine-badge") == "SIMULATED"
        assert app.query_one("#tabs").active == "tab-board"
        assert column_of(app, "Ctx") != before


@pytest.mark.asyncio
async def test_going_back_to_this_machine_takes_the_badge_down() -> None:
    async with running() as pilot:
        app = pilot.app
        simulate = app.query_one("#simulate", SimulatePane)
        app.query_one("#simulate-ram", Input).value = "256G"
        simulate.action_apply()
        await pilot.pause()
        assert label(app, "#machine-badge") == "SIMULATED"
        simulate.action_reset()
        await pilot.pause()
        assert label(app, "#machine-badge") == ""


@pytest.mark.asyncio
async def test_a_substitution_the_machine_cannot_support_is_refused_on_the_form() -> None:
    async with running(ready(host=machine(vram_total=None))) as pilot:
        app = pilot.app
        simulate = app.query_one("#simulate", SimulatePane)
        app.query_one("#simulate-vram", Input).value = "24G"
        simulate.action_apply()
        await pilot.pause()
        assert label(app, "#simulate-error")
        assert label(app, "#machine-badge") == ""


@pytest.mark.asyncio
async def test_a_bundled_profile_can_be_stood_in_for_this_machine() -> None:
    async with running(ready(host=machine(vram_total=None))) as pilot:
        app = pilot.app
        simulate = app.query_one("#simulate", SimulatePane)
        app.query_one("#simulate-profile", Select).value = "reference-rtx4060-128gb"
        simulate.action_apply()
        await pilot.pause()
        assert label(app, "#machine-badge") == "SIMULATED"
        assert "RTX 4060" in label(app, "#machine-line")


@pytest.mark.asyncio
async def test_a_size_that_is_not_a_size_is_refused_before_the_machine_changes() -> None:
    async with running() as pilot:
        app = pilot.app
        app.query_one("#simulate-cores", Input).value = "many"
        app.query_one("#simulate", SimulatePane).action_apply()
        await pilot.pause()
        assert label(app, "#simulate-error")
        assert app.query_one("#simulate", SimulatePane).dashboard.substitution.cpu_cores is None


# --- the machine screen ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_host_screen_is_system_and_doctor_on_one_page() -> None:
    async with running() as pilot:
        app = pilot.app
        assert "RTX 4060" in pane_text(app, "#host-scan", width=200)
        assert "llama.cpp" in pane_text(app, "#host-llamacpp", width=200)
        assert "Probes" in pane_text(app, "#host-probes", width=200)
        assert "Findings" in pane_text(app, "#host-findings", width=200)


@pytest.mark.asyncio
async def test_rescanning_asks_the_machine_again() -> None:
    calls = {"n": 0}

    def counting() -> object:
        calls["n"] += 1
        return report()

    dashboard = Dashboard(scanner=counting, loader=lambda: (empty_catalog(), []))  # type: ignore[arg-type]
    dashboard.refresh()
    taken = calls["n"]
    async with running(dashboard) as pilot:
        pilot.app.query_one("#host").action_rescan()
        await pilot.pause()
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
    assert calls["n"] > taken


@pytest.mark.asyncio
async def test_a_dashboard_that_has_not_scanned_scans_when_it_is_opened() -> None:
    dashboard = unscanned()
    async with running(dashboard) as pilot:
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()
        assert dashboard.report is not None
        assert table_of(pilot.app).row_count > 0


# --- the keys ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_key_bar_names_the_keys_of_the_screen_in_front_of_you() -> None:
    async with running() as pilot:
        app = pilot.app
        board_bar = label(app, "#key-bar")
        assert "Enter" in board_bar
        app.query_one("#tabs").active = "tab-plan"
        await pilot.pause()
        plan_bar = label(app, "#key-bar")
        assert plan_bar != board_bar
        assert "+" in plan_bar
        assert "?" in plan_bar


@pytest.mark.asyncio
async def test_the_key_map_opens_and_closes() -> None:
    async with running() as pilot:
        app = pilot.app
        await pilot.press("question_mark")
        assert isinstance(app.screen, KeyMapScreen)
        text = drawn(app.screen.query_one("#key-map", RichPane).renderable, 120)
        assert "Board" in text and "Simulate" in text
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, KeyMapScreen)


@pytest.mark.asyncio
async def test_the_theme_key_changes_the_theme() -> None:
    async with running() as pilot:
        before = pilot.app.theme
        await pilot.press("t")
        assert pilot.app.theme != before


# --- the keys that make the terminal as rich as the page ---------------------------------


@pytest.mark.asyncio
async def test_the_sort_can_be_walked_back_and_turned_round() -> None:
    from llamafit.cli.render_board import SORT_KEYS

    async with running() as pilot:
        app = pilot.app
        board = app.query_one("#board", BoardPane)
        await pilot.press("S")
        assert board.view.sort == SORT_KEYS[-1]
        await pilot.press("s")
        assert board.view.sort == "score"
        largest_first = column_of(app, "#")[:3]
        await pilot.press("o")
        assert board.view.descending is False
        assert "reverse" in label(app, "#board-state")
        # The ranked rows now run smallest score first -- two rows with one score keep
        # the ranking's order between them -- and the unranked ones, which have no score,
        # stay after them whichever way the key runs.
        ranked = [rank for rank in column_of(app, "#") if rank]
        assert int(ranked[0]) > int(largest_first[0])
        scores = [float(score) for score in column_of(app, "Score")[: len(ranked)]]
        assert scores == sorted(scores)
        assert all(not rank for rank in column_of(app, "#")[len(ranked) :])


@pytest.mark.asyncio
async def test_the_unranked_rows_are_on_the_table_dimmed_and_e_takes_them_off() -> None:
    async with running() as pilot:
        app = pilot.app
        board = app.query_one("#board", BoardPane)
        ranked = len(board.dashboard.rows)
        assert table_of(app).row_count > ranked
        # Every column filled: the score column carries the word for why, not a blank.
        scores = column_of(app, "Score")
        assert "too slow" in scores or "unsupported" in scores
        await pilot.press("e")
        assert table_of(app).row_count == ranked
        await pilot.press("e")
        assert table_of(app).row_count > ranked


@pytest.mark.asyncio
async def test_c_draws_every_column_and_the_table_scrolls_sideways() -> None:
    from llamafit.cli.render_board import BOARD_ORDER

    async with running(size=(80, 44)) as pilot:
        app = pilot.app
        narrow = len(headings(app))
        await pilot.press("c")
        assert len(headings(app)) == len(BOARD_ORDER)
        table = table_of(app)
        assert table.virtual_size.width > table.size.width
        await pilot.press("c")
        assert len(headings(app)) == narrow


@pytest.mark.asyncio
async def test_the_filter_box_takes_the_page_s_terms_and_names_them_on_the_state_line() -> None:
    async with running() as pilot:
        app = pilot.app
        board = app.query_one("#board", BoardPane)
        await pilot.press("slash")
        box = app.query_one("#board-search", Input)
        box.value = "speed>=20 fit>=fits"
        await pilot.press("enter")
        assert board.view.filters.min_speed == 20.0
        assert board.view.filters.min_fit == "fits"
        for row in board.shown:
            speed = row.candidate.speed
            assert speed is None or speed.gen_tps >= 20.0
        state = label(app, "#board-state")
        assert "20" in state and "fit" in state
        # The keys and the box share one model: f moves on from the box's own verdict,
        # and opening the box again shows what is in force.
        await pilot.press("f")
        assert board.view.filters.min_fit == "comfortable"
        await pilot.press("slash")
        assert "fit>=comfortable" in app.query_one("#board-search", Input).value
        assert "speed>=20" in app.query_one("#board-search", Input).value


@pytest.mark.asyncio
async def test_a_term_the_box_cannot_read_is_refused_by_name_and_hides_nothing() -> None:
    async with running() as pilot:
        app = pilot.app
        board = app.query_one("#board", BoardPane)
        before = table_of(app).row_count
        await pilot.press("slash")
        app.query_one("#board-search", Input).value = "fit>=snug"
        await pilot.press("enter")
        assert "comfortable" in label(app, "#board-error")
        assert table_of(app).row_count == before
        assert not board.view.filters.active
        # An empty box on Enter clears the filters a key set.
        await pilot.press("escape")
        app.query_one("#board-search", Input).value = ""
        await pilot.press("enter")
        assert label(app, "#board-error") == ""


@pytest.mark.asyncio
async def test_the_needs_form_takes_the_speed_floor_the_command_line_has() -> None:
    async with running() as pilot:
        app = pilot.app
        needs = app.query_one("#needs", NeedsPane)
        app.query_one("#needs-min-tps", Input).value = "12"
        needs.action_apply()
        await pilot.pause()
        assert needs.dashboard.request.needs.min_tps == 12.0
        app.query_one("#needs-min-tps", Input).value = "0"
        needs.action_apply()
        await pilot.pause()
        assert needs.dashboard.request.needs.min_tps == 0.0
        app.query_one("#needs-min-tps", Input).value = "fast"
        needs.action_apply()
        await pilot.pause()
        assert "tokens per second" in label(app, "#needs-error")
        assert needs.dashboard.request.needs.min_tps == 0.0
        needs.action_reset()
        await pilot.pause()
        assert needs.dashboard.request.needs.min_tps is None
