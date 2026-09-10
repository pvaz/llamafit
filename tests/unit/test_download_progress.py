"""The progress display: what it shows, and what it drops when the window is narrow."""

from __future__ import annotations

import io

from rich.console import Console

from llamafit.download.progress import (
    MEDIUM_CONSOLE_CELLS,
    NARROW_CONSOLE_CELLS,
    NullProgress,
    RichProgress,
)
from llamafit.i18n import set_language, translator


def _console(width: int, *, terminal: bool = True) -> Console:
    return Console(
        file=io.StringIO(),
        width=width,
        force_terminal=terminal,
        no_color=True,
        highlight=False,
        legacy_windows=False,
    )


def test_the_null_reporter_accepts_everything_and_draws_nothing() -> None:
    reporter = NullProgress()
    reporter.file_started("a.gguf", 10, 0)
    reporter.bytes_received("a.gguf", 5)
    reporter.file_verifying("a.gguf", 10)
    reporter.file_finished("a.gguf")
    reporter.note("something")


def test_a_wide_window_gets_every_column() -> None:
    progress = RichProgress(_console(120))
    assert len(progress._columns(_console(120))) == 6


def test_a_narrow_window_drops_the_name_the_rate_and_the_time_left() -> None:
    progress = RichProgress(_console(NARROW_CONSOLE_CELLS + 1))
    assert len(progress._columns(_console(NARROW_CONSOLE_CELLS + 1))) == 3


def test_a_very_narrow_window_drops_the_bar_too() -> None:
    # A bar that wraps leaves a torn copy of itself on every refresh, and the number the
    # reader wanted is the one that fell off the end.
    progress = RichProgress(_console(NARROW_CONSOLE_CELLS - 20))
    assert len(progress._columns(_console(NARROW_CONSOLE_CELLS - 20))) == 2


def test_a_window_at_the_medium_threshold_keeps_the_wide_set() -> None:
    progress = RichProgress(_console(MEDIUM_CONSOLE_CELLS))
    assert len(progress._columns(_console(MEDIUM_CONSOLE_CELLS))) == 6


def test_a_live_display_draws_a_bar_and_leaves_it_behind() -> None:
    console = _console(120)
    with RichProgress(console) as progress:
        progress.file_started("model-Q4_K_M.gguf", 1000, 0)
        progress.bytes_received("model-Q4_K_M.gguf", 500)
        progress.file_verifying("model-Q4_K_M.gguf", 1000)
        progress.bytes_received("model-Q4_K_M.gguf", 1000)
        progress.file_finished("model-Q4_K_M.gguf")
    written = console.file.getvalue()  # type: ignore[attr-defined]
    assert "model-Q4_K_M.gguf" in written


def test_a_pipe_gets_lines_rather_than_a_bar_that_cannot_be_redrawn() -> None:
    console = _console(120, terminal=False)
    progress = RichProgress(console)
    assert progress.live is False
    with progress:
        progress.file_started("model-Q4_K_M.gguf", 4096, 0)
        progress.bytes_received("model-Q4_K_M.gguf", 4096)
        progress.file_verifying("model-Q4_K_M.gguf", 4096)
        progress.file_finished("model-Q4_K_M.gguf")
    written = console.file.getvalue()  # type: ignore[attr-defined]
    assert "Fetching model-Q4_K_M.gguf" in written
    assert "4.0 KiB" in written
    assert "Checking model-Q4_K_M.gguf" in written
    assert "Done: model-Q4_K_M.gguf" in written


def test_a_note_reaches_the_console() -> None:
    console = _console(120, terminal=False)
    RichProgress(console).note("the server asked for fewer connections")
    assert "fewer connections" in console.file.getvalue()  # type: ignore[attr-defined]


def test_bytes_for_a_file_nobody_started_are_ignored_rather_than_a_crash() -> None:
    progress = RichProgress(_console(120))
    progress.bytes_received("never-started.gguf", 10)
    progress.file_verifying("never-started.gguf", 10)
    progress.file_finished("never-started.gguf")


def test_the_size_columns_are_written_in_the_reader_s_own_punctuation() -> None:
    # Portuguese writes 4,7 KiB where English writes 4.7 KiB, and the download display
    # goes through the same units module every other figure in the tool does.
    console = _console(120, terminal=False)
    try:
        set_language("pt_PT")
        RichProgress(console).file_started("model.gguf", 4813, 0)
    finally:
        translator.reset()
    written = console.file.getvalue()  # type: ignore[attr-defined]
    assert "4,7 KiB" in written
