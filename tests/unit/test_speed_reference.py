"""The estimator against the two configurations that were actually measured.

Everything here is checked against ``docs/calibration/2026-09-09-reference-machine.md``
and the ``measured`` blocks the catalog carries for Qwen3-Coder-Next and
Qwen3.8-Flash-Next. Two models is not a validation set, but it is what exists, and a
speed estimator that has never been held against a real number is a random number
generator with good manners.

**The tolerance.** For a tool whose job is to tell a person which model to run, 15 percent
is the error that does not change the advice: the reference machine's own generation
figure moves from 22.8 to 24.7 tokens per second just by changing the thread count, and
candidates a person is choosing between differ by much more than that. So 15 percent is
what these tests would like to assert.

The formula does not reach it. With section 10.1's constants exactly as written it lands
37 to 42 percent below both measurements, for a reason recorded in
:data:`llamafit.constants.EFF_RAM_SCATTERED`: the 0.36 was derived by dividing the expert
traffic by the *whole* token time, so using it alongside the graphics-card term and the
per-layer overheads counts a token's time about 1.6 times over. Rather than widen the
tolerance until it passes, the tests below assert the gap that is actually there, so it
stays visible and so they fail the day the constants are reconciled -- which is exactly
when somebody should look at this file again.
"""

from typing import NamedTuple

import pytest

from llamafit.speed import Formula, estimate_speed, formula_estimate, resolve_bandwidths
from tests.fixtures.speed import catalog_facts, catalog_model, placement, reference_host

QUANT = "UD-Q4_K_XL"

# The tolerance a speed advisor should meet, and does not yet. See the module docstring.
USEFUL_TOLERANCE = 0.15


class Case(NamedTuple):
    model_id: str
    active_params: float
    context: int
    micro_batch: int
    working_context: int
    gen_tps: float
    pp_tps: float


# The reference machine's expert-offload configuration for each model, as recorded in the
# calibration document: the working context the run actually used, the micro-batch, and
# the figures it produced.
CODER = Case("qwen3-coder-next", 3e9, 262144, 2048, 128, 24.7, 323.0)
FLASH = Case("qwen3.8-flash-next", 6e9, 40960, 1024, 1056, 13.9, 49.4)


def formula_for(case: Case) -> Formula:
    """Run the formula alone, with no benchmark allowed to correct it."""
    return formula_estimate(
        placement(micro_batch=case.micro_batch, context=case.context),
        catalog_facts(case.model_id, QUANT),
        resolve_bandwidths(reference_host()),
        working_context=case.working_context,
        micro_batch=case.micro_batch,
        active_params=case.active_params,
    )


@pytest.mark.parametrize(("case", "low", "high"), [(CODER, 0.30, 0.45), (FLASH, 0.30, 0.45)])
def test_the_formula_lands_well_below_both_reference_measurements(
    case: Case, low: float, high: float
) -> None:
    """The specification's constants under-predict generation by 37 to 42 percent.

    This asserts the gap rather than a tolerance, deliberately. The estimator is not
    within the 15 percent that would make it useful on its own, and pretending otherwise
    by loosening a bound would hide the one thing about section 10.1 that needs fixing.
    """
    predicted = formula_for(case).gen_tps
    measured = case.gen_tps
    shortfall = 1 - predicted / measured
    assert low < shortfall < high, f"predicted {predicted:.2f} against a measured {measured}"
    assert shortfall > USEFUL_TOLERANCE, "the formula now meets the tolerance; revisit this file"


def test_the_formula_gets_the_ratio_between_the_two_models_nearly_right() -> None:
    """What is wrong with the formula is its level, not its shape.

    The two models' expert traffic differs by 64 percent and their measured speeds by 78
    percent. The formula reproduces that spread to within nine percent of the measured
    ratio and within two percent of the ratio between the figures section 10.1 itself
    quotes, which is why the constant that sets the level is worth fixing rather than the
    model being worth replacing.
    """
    predicted = formula_for(CODER).gen_tps / formula_for(FLASH).gen_tps
    assert predicted == pytest.approx(24.7 / 13.9, rel=0.10)
    assert predicted == pytest.approx(23.0 / 13.9, rel=0.03)


@pytest.mark.parametrize("case", [CODER, FLASH])
def test_the_expert_read_dominates_both_reference_predictions(case: Case) -> None:
    """Which term dominates, which is what a person who doubts the number needs to see."""
    formula = formula_for(case)
    total = formula.vram_seconds + formula.ram_seconds + formula.overhead_seconds
    assert formula.ram_seconds / total > 0.6
    assert formula.ram_seconds > formula.vram_seconds


@pytest.mark.parametrize("case", [CODER, FLASH])
def test_the_measured_rung_reproduces_the_benchmark_exactly(case: Case) -> None:
    """With the machine's own benchmarks in hand, the tool shows them, dated."""
    estimate = estimate_speed(
        placement(micro_batch=case.micro_batch, context=case.context),
        catalog_facts(case.model_id, QUANT),
        reference_host(),
        working_context=case.working_context,
        active_params=case.active_params,
        quant=QUANT,
        measurements=catalog_model(case.model_id).measured,
    )
    assert estimate.confidence == "measured"
    assert estimate.measured_on is not None
    assert estimate.measured_on.isoformat() == "2026-09-09"
    assert estimate.gen_tps == pytest.approx(case.gen_tps)
    assert estimate.pp_tps == pytest.approx(case.pp_tps)
    shares = (
        estimate.vram_seconds_per_token
        + estimate.ram_seconds_per_token
        + estimate.overhead_seconds_per_token
    )
    assert shares == pytest.approx(1 / estimate.gen_tps)
    assert estimate.ram_seconds_per_token > estimate.vram_seconds_per_token


def test_prompt_processing_matches_the_model_its_constant_was_fitted_on() -> None:
    """Section 10.2's 4 GB/s came from Flash-Next, and Flash-Next it predicts."""
    predicted = formula_for(FLASH).pp_tps
    assert predicted == pytest.approx(49.4, rel=USEFUL_TOLERANCE)


def test_prompt_processing_is_half_the_truth_for_a_model_that_fits_in_memory() -> None:
    """And Coder-Next, whose expert set stays in the page cache, it under-predicts twofold.

    Section 10.2's 4 GB/s is not a property of the link. It is what Qwen3.8-Flash-Next
    reaches because its 111 GB of weights plus a 29 GB streamed table do not fit in the
    machine's 128 GB, so part of every micro-batch's expert read comes off the disk.
    Qwen3-Coder-Next's 50 GB expert set stays cached and streams at about 12.8 GB/s, which
    is the whole of the gap asserted here.
    """
    predicted = formula_for(CODER).pp_tps
    assert 0.40 < predicted / 323.0 < 0.55


def test_generation_barely_moves_with_the_micro_batch() -> None:
    """Which is why a generation benchmark is matched on the offload and not on ``-ub``.

    The reference machine measured Flash-Next at 14.5 tokens per second with ``-ub 2048``
    and 13.9 with ``-ub 1024``. Taking the first as a benchmark of the second's
    configuration costs four percent, and refusing to would cost the label entirely.
    """
    flash = catalog_model("qwen3.8-flash-next")
    anchor = [m for m in flash.measured if m.context == 16384]
    estimate = estimate_speed(
        placement(micro_batch=1024, context=40960),
        catalog_facts("qwen3.8-flash-next", QUANT),
        reference_host(),
        working_context=1056,
        active_params=6e9,
        quant=QUANT,
        measurements=anchor,
    )
    assert estimate.gen_tps == pytest.approx(13.9, rel=0.05)


def test_a_prompt_benchmark_from_another_micro_batch_is_only_as_good_as_its_anchor() -> None:
    """Calibration inherits whatever is wrong with the run it calibrates on.

    The catalog's ``-ub 2048`` prompt figure for Flash-Next, 50 tokens per second, is a
    real 1,056-token request: only one micro-batch of 1,056 tokens was ever formed, so the
    figure belongs to a micro-batch half the size its flags claim. Calibrating the
    ``-ub 1024`` estimate on it therefore drags the prediction well below the 49.4 that
    was measured there. Recorded rather than worked around: the fix belongs in the
    catalog, which should say what micro-batch a prompt figure really used.
    """
    flash = catalog_model("qwen3.8-flash-next")
    anchor = [m for m in flash.measured if m.context == 16384]
    estimate = estimate_speed(
        placement(micro_batch=1024, context=40960),
        catalog_facts("qwen3.8-flash-next", QUANT),
        reference_host(),
        working_context=1056,
        active_params=6e9,
        quant=QUANT,
        measurements=anchor,
    )
    assert estimate.confidence == "calibrated"
    assert estimate.pp_tps < 49.4 * (1 - USEFUL_TOLERANCE)
