"""The one decision a launch script makes, tested as a decision rather than as text.

Given a free-memory figure, which rung. Everything about batch files and shell quoting is
in the next test module; this one is arithmetic, because the arithmetic is the thing the
whole package exists to get right and it must be checkable without reading a script.
"""

from __future__ import annotations

from datetime import date

import pytest

from llamafit.constants import MIB, VRAM_RESERVE_BYTES
from llamafit.models.host import Host
from llamafit.models.plan import Budget, ContextTier, Placement
from llamafit.placement import LaunchOptions, render_flags
from llamafit.presets import (
    CONTEXT_TOKEN,
    PresetSpec,
    Rung,
    build_spec,
    can_probe_free_memory,
    choose_context,
    group_flags,
    mib_ceiling,
)
from llamafit.presets.spec import comment_block
from tests.fixtures.board import model_and_quant, report
from tests.fixtures.budget_hosts import machine
from tests.fixtures.presets import windows_spec

RESERVE_MIB = VRAM_RESERVE_BYTES // MIB

LADDER = (
    Rung(tokens=32768, vram_required=6000 * MIB),
    Rung(tokens=24576, vram_required=5000 * MIB),
    Rung(tokens=16384, vram_required=4000 * MIB),
)


@pytest.mark.parametrize(
    ("free_mib", "expected"),
    [
        (8000, 32768),
        (6256, 32768),
        (6255, 24576),
        (5256, 24576),
        (5255, 16384),
        (4256, 16384),
        (4255, None),
        (0, None),
    ],
)
def test_the_largest_rung_that_fits_under_the_free_memory_is_chosen(
    free_mib: int, expected: int | None
) -> None:
    assert choose_context(LADDER, free_mib, reserve_mib=RESERVE_MIB) == expected


def test_nothing_fitting_is_an_answer_rather_than_the_smallest_rung() -> None:
    """A rung that does not fit must never be taken: that is what pages, silently."""
    assert choose_context(LADDER, 100, reserve_mib=RESERVE_MIB) is None


def test_a_free_figure_below_the_reserve_does_not_wrap_into_a_large_number() -> None:
    assert choose_context(LADDER, 10, reserve_mib=RESERVE_MIB) is None


def test_a_requirement_is_rounded_up_to_the_next_mebibyte_never_down() -> None:
    assert mib_ceiling(MIB) == 1
    assert mib_ceiling(MIB + 1) == 2
    assert mib_ceiling(0) == 0


def test_the_ladder_stops_at_the_context_the_planner_chose() -> None:
    spec = windows_spec()
    assert spec.rungs[0].tokens == spec.planned_context
    assert all(rung.tokens <= spec.planned_context for rung in spec.rungs)


def test_the_ladder_descends_so_the_first_rung_that_fits_is_the_largest() -> None:
    spec = windows_spec()
    tokens = [rung.tokens for rung in spec.rungs]
    assert tokens == sorted(tokens, reverse=True)


def test_a_rung_the_machine_could_not_hold_is_not_offered() -> None:
    """A rung with ``fits`` false is off the ladder, whichever pool it overflowed.

    The script reads free card memory and nothing else, so it could not tell a rung that
    overflowed the card from one that overflowed system memory. Leaving both off costs
    nothing: the planner had already said no to them.
    """
    tiers = (
        _tier(16384, 1),
        ContextTier(tokens=24576, vram_required=99 * 1024**3, fits=False, verdict="too-tight"),
        ContextTier(tokens=30000, vram_required=2 * 1024**3, fits=False, verdict="does-not-fit"),
    )
    spec = _spec_for(context=32768, tiers=tiers)
    assert [rung.tokens for rung in spec.rungs] == [32768, 16384]


def test_the_top_rung_is_the_planned_configuration_even_off_the_tier_grid() -> None:
    """``plan --context 50000`` is a context no ladder contains; the preset still runs it."""
    spec = _spec_for(context=50_000, tiers=(_tier(16384, 1), _tier(49152, 2)))
    assert spec.planned_context == 50_000
    assert spec.rungs[0].tokens == 50_000
    assert [rung.tokens for rung in spec.rungs] == [50_000, 49152, 16384]


def test_the_context_is_replaced_by_a_marker_so_a_script_substitutes_one_value() -> None:
    spec = windows_spec()
    assert CONTEXT_TOKEN in spec.flags
    assert spec.flags[spec.flags.index("-c") + 1] == CONTEXT_TOKEN
    assert str(spec.planned_context) not in spec.flags


@pytest.mark.parametrize(
    ("vendor", "os_name", "expected"),
    [
        ("nvidia", "windows", True),
        ("nvidia", "linux", True),
        ("amd", "linux", True),
        ("amd", "windows", False),
        ("apple", "macos", True),
        ("intel", "linux", False),
        (None, "linux", False),
    ],
)
def test_only_a_card_with_a_tool_that_ships_with_its_driver_can_be_asked(
    vendor: str | None, os_name: str, expected: bool
) -> None:
    assert can_probe_free_memory(vendor, os_name) is expected  # type: ignore[arg-type]


def test_a_machine_with_no_card_gets_no_probe_and_no_ladder() -> None:
    spec = _spec_for(host=machine(vram_total=None, ram_total=64 * 1024**3))
    assert spec.probe_free_memory is False
    assert spec.gpu_vendor is None


def test_each_flag_is_paired_with_the_value_that_belongs_to_it() -> None:
    assert group_flags(["-m", "/x.gguf", "--jinja", "-c", "4096"]) == [
        ["-m", "/x.gguf"],
        ["--jinja"],
        ["-c", "4096"],
    ]


def test_a_lone_value_at_the_front_is_not_attached_to_nothing() -> None:
    assert group_flags(["llama-server", "-c", "8"]) == [["llama-server"], ["-c", "8"]]


def test_a_comment_block_never_breaks_a_command_across_two_lines() -> None:
    lines = comment_block("# ", ["Run `llamafit preset qwen3-coder-next --force` to redo it."])
    assert any("llamafit" in line for line in lines)
    assert all(line.startswith("# ") for line in lines)
    assert "qwen3-coder-next" in " ".join(lines)


def test_paragraphs_are_separated_by_a_bare_comment_line() -> None:
    lines = comment_block("# ", ["one", "two"])
    assert lines == ["# one", "#", "# two"]


def _tier(tokens: int, gib: int) -> ContextTier:
    return ContextTier(tokens=tokens, vram_required=gib * 1024**3, fits=True, verdict="fits")


def _spec_for(
    *,
    context: int = 32768,
    tiers: tuple[ContextTier, ...] = (),
    host: Host | None = None,
) -> PresetSpec:
    """A specification built from a placement made up for one question."""
    model, quant = model_and_quant("qwen3-coder-next")
    system = report()
    used_host = host if host is not None else system.host
    budget = Budget(
        lines=(),
        vram_required=3 * 1024**3,
        ram_required=1024**3,
        vram_available=7 * 1024**3,
        ram_available=64 * 1024**3,
        vram_utilisation=0.4,
        ram_utilisation=0.1,
        verdict="fits",
    )
    placement = Placement(
        mode="gpu",
        context=context,
        micro_batch=2048,
        batch=4096,
        kv_type="f16",
        gpu_layers=99,
        threads=16,
        budget=budget,
        max_context_fit=context,
        tiers=tiers,
    )
    options = LaunchOptions(model_path="/m.gguf", port=8080)
    return build_spec(
        model=model,
        quant=quant.name,
        placement=placement,
        flags=render_flags(placement, model, options),
        model_path="/m.gguf",
        projector_path=None,
        host=used_host,
        server_path=None,
        llamacpp_build=1,
        port=8080,
        generated=date(2026, 9, 10),
        version="1.2.3",
    )
