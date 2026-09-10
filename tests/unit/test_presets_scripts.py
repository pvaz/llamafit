"""The two dialects: quoting, the ladder as written down, and the whole file byte for byte.

Three kinds of test here and they are deliberately different in kind.

The **golden files** under ``tests/golden/presets`` are the whole rendering, compared as
text. A generated script is text and text is exactly testable, so anything that changes in
one is visible in a diff rather than argued about.

The **quoting tests** take a path holding every character that breaks a careless generator
-- a space, a per cent sign, an ampersand, an exclamation mark, an apostrophe -- and check
that it survives each dialect's own rules.

The **ladder tests** read the thresholds back out of each rendered script and compare the
choice it would make, at a range of free-memory figures, with what
:func:`~llamafit.presets.spec.choose_context` says. That is the join the whole package
turns on: the rule is tested once as arithmetic, and each script is tested against the rule
rather than against a second copy of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from llamafit.presets import (
    PresetSpec,
    choose_context,
    cmd_argument,
    cmd_assignment,
    cmd_echo,
    render_cmd,
    render_files,
    render_ini_section,
    render_sh,
    sh_quote,
)
from tests.fixtures.presets import (
    AWKWARD_POSIX_PATH,
    AWKWARD_WINDOWS_PATH,
    cpu_spec,
    posix_spec,
    vision_spec,
    windows_spec,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden" / "presets"
STAMP = "0" * 64

_CMD_RUNG = re.compile(r'if %USABLE_MIB% GEQ (\d+) if not defined CTX set "CTX=(\d+)"')
_SH_RUNG = re.compile(r'\[ "\$USABLE_MIB" -ge (\d+) \]; then CTX=(\d+)')


def golden(name: str) -> str:
    """One golden file, read as text so line endings never enter the comparison."""
    return (GOLDEN / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# The whole file
# --------------------------------------------------------------------------------------


def test_the_windows_script_is_what_it_was() -> None:
    assert render_cmd(windows_spec(), stamp=STAMP) == golden("start-qwen3-coder-next.cmd")


def test_the_posix_script_is_what_it_was() -> None:
    assert render_sh(posix_spec(), stamp=STAMP) == golden("start-qwen3-coder-next.sh")


def test_the_router_section_is_what_it_was() -> None:
    assert render_ini_section(windows_spec(), stamp=STAMP) == golden("models-qwen3-coder-next.ini")


def test_the_readme_is_what_it_was() -> None:
    files = {file.name: file for file in render_files(windows_spec())}
    text = files["README-qwen3-coder-next.md"].text
    expected = golden("README-qwen3-coder-next.md")
    assert _blank_stamp(text) == _blank_stamp(expected)


def test_a_machine_with_nothing_on_the_card_says_so_instead_of_probing() -> None:
    text = render_cmd(cpu_spec(), stamp=STAMP)
    assert "nvidia-smi" not in text
    assert "WHY THIS SCRIPT DOES NOT CHOOSE A CONTEXT WHEN IT RUNS" in text
    assert 'set "CTX=32768"' in text


def test_a_card_with_no_tool_to_ask_says_which_of_the_two_reasons_it_is() -> None:
    spec = posix_spec(gpu_vendor="intel", probe_free_memory=False, mode="gpu")
    text = render_sh(spec, stamp=STAMP)
    assert "no vendor tool a script can call" in text
    assert "LLAMAFIT_CONTEXT" in text


# --------------------------------------------------------------------------------------
# Quoting
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("--jinja", "--jinja"),
        ("ffn_.*_shexp=CPU", "ffn_.*_shexp=CPU"),
        ("C:\\Models\\Qwen 3.gguf", '"C:\\Models\\Qwen 3.gguf"'),
        ("100% Mine", '"100%% Mine"'),
        ("a&b", '"a&b"'),
        ('{"enable_thinking":false}', '"{\\"enable_thinking\\":false}"'),
        ("C:\\my dir\\", '"C:\\my dir\\\\"'),
    ],
)
def test_a_batch_argument_survives_cmd_and_the_c_runtime(word: str, expected: str) -> None:
    assert cmd_argument(word) == expected


def test_a_batch_assignment_quotes_the_whole_pair_and_doubles_a_per_cent() -> None:
    assert cmd_assignment("MODEL", AWKWARD_WINDOWS_PATH) == (
        'set "MODEL=D:\\Models\\100%% Mine & Yours!\\Qwen3 (v2).gguf"'
    )


def test_an_echo_escapes_what_cmd_would_read_as_syntax() -> None:
    assert cmd_echo("a & b (c) ^ d") == "echo a ^& b ^(c^) ^^ d"
    assert cmd_echo("free: %FREE_MIB% MiB") == "echo free: %FREE_MIB% MiB"
    assert cmd_echo("") == "echo."


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("--jinja", "--jinja"),
        ("ffn_.*_shexp=CPU", "'ffn_.*_shexp=CPU'"),
        ("/home/pv/a b.gguf", "'/home/pv/a b.gguf'"),
        ("it's", "'it'\\''s'"),
        ("$HOME", "'$HOME'"),
    ],
)
def test_a_shell_argument_is_single_quoted_unless_it_is_plainly_safe(
    word: str, expected: str
) -> None:
    assert sh_quote(word) == expected


def test_a_windows_path_that_would_break_a_careless_generator_is_written_correctly() -> None:
    spec = windows_spec(model_path=AWKWARD_WINDOWS_PATH)
    text = render_cmd(_repath(spec, AWKWARD_WINDOWS_PATH), stamp=STAMP)
    assert 'set "MODEL=D:\\Models\\100%% Mine & Yours!\\Qwen3 (v2).gguf"' in text
    # The path itself is never repeated on the command line: the variable is.
    assert AWKWARD_WINDOWS_PATH not in text.split('set "MODEL=', 1)[1].split("\n", 1)[1]
    assert '-m "%MODEL%"' in text


def test_a_posix_path_that_would_break_a_careless_generator_is_written_correctly() -> None:
    spec = posix_spec(model_path=AWKWARD_POSIX_PATH)
    text = render_sh(_repath(spec, AWKWARD_POSIX_PATH), stamp=STAMP)
    assert "MODEL='/home/pv/models/it'\\''s 100% mine & more/Qwen3 $HOME (v2).gguf'" in text
    assert '-m "$MODEL"' in text


def test_a_path_that_is_not_ascii_makes_the_batch_file_set_its_code_page() -> None:
    spec = windows_spec(model_path="D:\\Modelos\\Ficheiro Português.gguf")
    text = render_cmd(_repath(spec, "D:\\Modelos\\Ficheiro Português.gguf"), stamp=STAMP)
    assert "chcp 65001" in text


def test_an_ascii_plan_needs_no_code_page_line() -> None:
    assert "chcp" not in render_cmd(windows_spec(), stamp=STAMP)


def test_the_batch_file_never_uses_delayed_expansion() -> None:
    """``!`` is legal in a Windows path, and delayed expansion would eat it."""
    assert "enabledelayedexpansion" not in render_cmd(windows_spec(), stamp=STAMP).lower()


def test_the_shell_script_has_no_carriage_returns_and_starts_with_its_shebang() -> None:
    text = render_sh(posix_spec(), stamp=STAMP)
    assert text.startswith("#!/bin/sh\n")
    assert "\r" not in text


# --------------------------------------------------------------------------------------
# The ladder, as each script actually spells it
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("free_mib", [0, 200, 4000, 5900, 6000, 6300, 6600, 7000, 20000])
def test_the_batch_ladder_chooses_what_the_rule_chooses(free_mib: int) -> None:
    spec = windows_spec()
    rungs = _CMD_RUNG.findall(render_cmd(spec, stamp=STAMP))
    assert rungs, "the batch file has no ladder"
    assert _simulate(rungs, free_mib, spec.reserve_mib) == choose_context(
        spec.rungs, free_mib, reserve_mib=spec.reserve_mib
    )


@pytest.mark.parametrize("free_mib", [0, 200, 4000, 5900, 6000, 6300, 6600, 7000, 20000])
def test_the_shell_ladder_chooses_what_the_rule_chooses(free_mib: int) -> None:
    spec = posix_spec()
    rungs = _SH_RUNG.findall(render_sh(spec, stamp=STAMP))
    assert rungs, "the shell script has no ladder"
    assert _simulate(rungs, free_mib, spec.reserve_mib) == choose_context(
        spec.rungs, free_mib, reserve_mib=spec.reserve_mib
    )


def test_both_scripts_write_the_same_thresholds_in_the_same_order() -> None:
    windows = _CMD_RUNG.findall(render_cmd(windows_spec(), stamp=STAMP))
    posix = _SH_RUNG.findall(render_sh(posix_spec(), stamp=STAMP))
    assert windows == posix


def test_each_script_stops_rather_than_starting_at_a_rung_that_does_not_fit() -> None:
    assert "exit /b 4" in render_cmd(windows_spec(), stamp=STAMP)
    assert "exit 4" in render_sh(posix_spec(), stamp=STAMP)


def test_each_script_refuses_to_start_when_the_model_has_moved() -> None:
    assert "exit /b 3" in render_cmd(windows_spec(), stamp=STAMP)
    assert "exit 3" in render_sh(posix_spec(), stamp=STAMP)


def test_each_script_falls_back_to_the_planned_context_when_it_cannot_measure() -> None:
    windows = render_cmd(windows_spec(), stamp=STAMP)
    assert ":no_probe" in windows
    assert "Could not read how much card memory is free" in windows
    posix = render_sh(posix_spec(), stamp=STAMP)
    assert "Could not read how much card memory is free" in posix


def test_a_swapped_card_is_reported_but_does_not_stop_the_launch() -> None:
    spec = windows_spec()
    text = render_cmd(spec, stamp=STAMP)
    card = text.split("is this still the same card", 1)[1].split(":card_checked", 1)[0]
    assert f'if "%TOTAL_MIB%"=="{spec.gpu_total_mib}" goto card_checked' in card
    assert "exit /b" not in card


def test_an_amd_card_on_linux_is_read_from_sysfs_rather_than_from_a_tool() -> None:
    text = render_sh(posix_spec(gpu_vendor="amd"), stamp=STAMP)
    assert "mem_info_vram_total" in text
    assert "rocm-smi -" not in text  # named in a comment, never called


def test_apple_silicon_counts_the_pages_the_kernel_would_hand_over() -> None:
    spec = posix_spec(gpu_vendor="apple", os_name="macos", unified_memory=True)
    text = render_sh(spec, stamp=STAMP)
    assert "vm_stat" in text
    assert "Pages inactive" in text


def _simulate(rungs: list[tuple[str, str]], free_mib: int, reserve_mib: int) -> int | None:
    """Walk the ladder exactly as the rendered script walks it: first match, largest first."""
    usable = free_mib - reserve_mib
    for threshold, tokens in rungs:
        if usable >= int(threshold):
            return int(tokens)
    return None


def _repath(spec: PresetSpec, path: str) -> PresetSpec:
    """A copy whose flags name the same model path its fields do."""
    flags = tuple(path if word.endswith(".gguf") else word for word in spec.flags)
    return spec.model_copy(update={"flags": flags})


def _blank_stamp(text: str) -> str:
    """The text with any stamp replaced, so a golden README is not a hash of itself."""
    return re.sub(r"llamafit-preset-stamp: [0-9a-f]{64}", "llamafit-preset-stamp: STAMP", text)


def test_a_vision_model_names_its_projector_through_a_variable_of_its_own() -> None:
    spec = vision_spec()
    assert spec.projector_path is not None
    windows = render_cmd(spec, stamp=STAMP)
    assert 'set "MMPROJ=' in windows
    assert '--mmproj "%MMPROJ%"' in windows
    posix = render_sh(spec.model_copy(update={"os_name": "linux"}), stamp=STAMP)
    assert "MMPROJ=" in posix
    assert '--mmproj "$MMPROJ"' in posix


def test_the_router_section_gives_the_projector_a_key_rather_than_an_argument() -> None:
    spec = vision_spec()
    section = render_ini_section(spec, stamp=STAMP)
    assert f"mmproj = {spec.projector_path}" in section
    assert "--mmproj" not in section.split("args = ", 1)[1]
    assert "-m " not in section.split("args = ", 1)[1]
