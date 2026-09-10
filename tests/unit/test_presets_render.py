"""Files on disk: their line endings, their stamps, and the edits they must not lose.

The rule this module is mostly about is the one a person only notices when it is broken.
These files exist to be tuned by hand -- that is the whole reason for writing a file rather
than printing a command -- so a second ``llamafit preset`` must not quietly discard an
evening's work.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from llamafit.presets import PresetFile, is_unedited, render_files, stamp_of, write_files
from llamafit.presets.render import PLACEHOLDER_STAMP, report_of
from tests.fixtures.presets import cpu_spec, posix_spec, windows_spec


def test_a_windows_preset_is_a_batch_file_a_router_section_and_a_readme() -> None:
    names = [file.name for file in render_files(windows_spec())]
    assert names == [
        "start-qwen3-coder-next.cmd",
        "models-qwen3-coder-next.ini",
        "README-qwen3-coder-next.md",
    ]


def test_a_posix_preset_carries_a_shell_script_rather_than_a_batch_file() -> None:
    names = [file.name for file in render_files(posix_spec())]
    assert names[0] == "start-qwen3-coder-next.sh"
    assert not any(name.endswith(".cmd") for name in names)


def test_a_batch_file_is_written_with_carriage_returns_and_no_byte_order_mark(
    tmp_path: Path,
) -> None:
    write_files(render_files(windows_spec()), tmp_path)
    raw = (tmp_path / "start-qwen3-coder-next.cmd").read_bytes()
    assert raw.startswith(b"@echo off\r\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_a_shell_script_is_written_with_bare_newlines_even_on_windows(
    tmp_path: Path,
) -> None:
    """``#!/bin/sh\\r`` is an interpreter that does not exist, and the kernel says so."""
    write_files(render_files(posix_spec()), tmp_path)
    raw = (tmp_path / "start-qwen3-coder-next.sh").read_bytes()
    assert raw.startswith(b"#!/bin/sh\n")
    assert b"\r" not in raw


def test_a_shell_script_is_made_executable_where_there_is_such_a_bit(
    tmp_path: Path,
) -> None:
    write_files(render_files(posix_spec()), tmp_path)
    mode = (tmp_path / "start-qwen3-coder-next.sh").stat().st_mode
    if os.name != "nt":  # Windows has no execute bit to set
        assert mode & stat.S_IXUSR


def test_a_fresh_directory_is_created_and_every_file_reported_as_written(
    tmp_path: Path,
) -> None:
    results = write_files(render_files(windows_spec()), tmp_path / "presets")
    assert [result.outcome for result in results] == ["created", "created", "created"]


def test_writing_the_same_plan_twice_changes_nothing_and_says_so(tmp_path: Path) -> None:
    files = render_files(windows_spec())
    write_files(files, tmp_path)
    again = write_files(files, tmp_path)
    assert [result.outcome for result in again] == ["unchanged", "unchanged", "unchanged"]


def test_a_new_plan_over_an_untouched_file_is_an_update(tmp_path: Path) -> None:
    write_files(render_files(windows_spec()), tmp_path)
    results = write_files(render_files(windows_spec(port=9001)), tmp_path)
    assert results[0].outcome == "updated"
    assert "9001" in (tmp_path / "start-qwen3-coder-next.cmd").read_text(encoding="utf-8")


def test_a_file_the_user_edited_is_kept_and_named(tmp_path: Path) -> None:
    write_files(render_files(windows_spec()), tmp_path)
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_text(script.read_text(encoding="utf-8") + "\nrem my own note\n", encoding="utf-8")
    results = write_files(render_files(windows_spec(port=9001)), tmp_path)
    assert results[0].outcome == "kept-your-edits"
    assert results[0].kept
    assert "my own note" in script.read_text(encoding="utf-8")
    assert "9001" not in script.read_text(encoding="utf-8")


def test_only_the_file_that_was_edited_is_kept(tmp_path: Path) -> None:
    write_files(render_files(windows_spec()), tmp_path)
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_text(script.read_text(encoding="utf-8") + "\nrem mine\n", encoding="utf-8")
    results = write_files(render_files(windows_spec(port=9001)), tmp_path)
    assert [result.kept for result in results] == [True, False, False]


def test_force_overwrites_an_edited_file_and_reports_that_it_did(tmp_path: Path) -> None:
    write_files(render_files(windows_spec()), tmp_path)
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_text(script.read_text(encoding="utf-8") + "\nrem mine\n", encoding="utf-8")
    results = write_files(render_files(windows_spec()), tmp_path, force=True)
    assert results[0].outcome == "overwritten"
    assert "rem mine" not in script.read_text(encoding="utf-8")


def test_a_file_with_no_stamp_at_all_counts_as_edited(tmp_path: Path) -> None:
    """A file that lost its stamp was hand-written or hand-cut; both are somebody's work."""
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_text("@echo off\nrem someone wrote this by hand\n", encoding="utf-8")
    results = write_files(render_files(windows_spec()), tmp_path)
    assert results[0].outcome == "kept-your-edits"


def test_a_file_that_is_not_even_text_is_kept_rather_than_crashed_on(tmp_path: Path) -> None:
    script = tmp_path / "start-qwen3-coder-next.cmd"
    script.write_bytes(b"\xff\xfe\x00binary rubbish")
    results = write_files(render_files(windows_spec()), tmp_path)
    assert results[0].outcome == "kept-your-edits"


def test_a_stamp_covers_the_whole_file_but_not_itself() -> None:
    script = render_files(windows_spec())[0]
    assert is_unedited(script.text)
    blanked = script.text.replace(_stamp_in(script.text), PLACEHOLDER_STAMP)
    assert stamp_of(blanked) == _stamp_in(script.text)


def test_a_changed_stamp_alone_makes_a_file_read_as_edited() -> None:
    script = render_files(windows_spec())[0]
    tampered = script.text.replace(_stamp_in(script.text), "a" * 64)
    assert not is_unedited(tampered)


def test_every_file_of_a_preset_carries_its_own_stamp() -> None:
    for file in render_files(windows_spec()):
        assert is_unedited(file.text), file.name


def test_the_line_endings_a_file_gets_do_not_change_its_stamp(tmp_path: Path) -> None:
    """The stamp is computed on the text; the disk gets whichever endings the file needs."""
    write_files(render_files(windows_spec()), tmp_path)
    script = tmp_path / "start-qwen3-coder-next.cmd"
    assert b"\r\n" in script.read_bytes()
    assert is_unedited(script.read_text(encoding="utf-8"))


def test_the_report_names_every_file_and_the_ladder_the_script_will_walk(
    tmp_path: Path,
) -> None:
    spec = windows_spec()
    results = write_files(render_files(spec), tmp_path)
    report = report_of(spec, tmp_path, results)
    assert report.model_id == "qwen3-coder-next"
    assert report.endpoint == "http://127.0.0.1:8080"
    assert [rung.tokens for rung in report.rungs] == [r.tokens for r in spec.rungs]
    assert len(report.files) == 3
    assert report.probe_free_memory is True


def test_a_plan_with_no_ladder_says_so_in_the_report(tmp_path: Path) -> None:
    spec = cpu_spec()
    report = report_of(spec, tmp_path, write_files(render_files(spec), tmp_path))
    assert report.probe_free_memory is False


def test_the_readme_shows_the_free_memory_each_rung_asks_for() -> None:
    spec = windows_spec()
    readme = next(f for f in render_files(spec) if f.name.startswith("README-")).text
    top = spec.rungs[0]
    assert f"| {top.tokens} | {top.vram_required_mib} MiB |" in readme
    assert f"{top.vram_required_mib + spec.reserve_mib} MiB |" in readme
    assert "http://127.0.0.1:8080/v1/chat/completions" in readme


def test_the_readme_says_what_happens_when_the_machine_changes() -> None:
    readme = next(f for f in render_files(windows_spec()) if f.name.startswith("README-")).text
    assert "The model file has moved" in readme
    assert "A different card" in readme
    assert "exits 3" in readme


def test_a_file_is_written_where_it_was_asked_for(tmp_path: Path) -> None:
    file = PresetFile(name="x.txt", text="hello\n", newline="\n", executable=False)
    results = write_files([file], tmp_path / "deep" / "deeper")
    assert results[0].path == tmp_path / "deep" / "deeper" / "x.txt"
    assert results[0].path.read_text(encoding="utf-8") == "hello\n"


def _stamp_in(text: str) -> str:
    """The stamp a rendered file carries."""
    marker = "llamafit-preset-stamp: "
    start = text.index(marker) + len(marker)
    return text[start : start + 64]
