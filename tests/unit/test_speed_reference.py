"""The estimator against the four configurations that were actually measured.

Everything here is checked against the ``measured`` blocks the catalog carries for
Qwen3-0.6B, Qwen3-Coder-Next and Qwen3.8-Flash-Next, all taken on the reference machine.

**Why four and not two.** Two runs cannot identify three efficiencies and an overhead, and
an earlier revision of section 10.1 tried: it derived the scattered-read efficiency by
dividing the expert traffic by the whole token time, which already contains the card
traffic and the overhead the formula then adds again, and the estimator built on it came
out 42 percent low. The two Qwen3-0.6B runs exist to break that degeneracy. The same small
dense model held entirely in system memory and then entirely on the card puts all of the
traffic in one pool at a time with nothing competing, so each efficiency is read off
directly rather than fitted.

**The tolerance.** 15 percent is the error that does not change the advice: the reference
machine's own generation figure for Coder-Next moves from 22.8 to 24.7 tokens per second
just by changing the thread count, and models a person is choosing between differ by far
more than that. Generation now meets it on all four runs with room to spare -- the worst is
4 percent. Prompt processing does not, and the tests at the bottom of this file assert the
gap that is there rather than a tolerance wide enough to hide it.
"""

from typing import NamedTuple

import pytest

from llamafit.models.plan import Placement, Pool, RunMode
from llamafit.speed import (
    Formula,
    estimate_speed,
    formula_estimate,
    per_token_traffic,
    resolve_bandwidths,
)
from tests.fixtures.speed import catalog_facts, catalog_model, placement, reference_host

# The tolerance a speed advisor has to meet. See the module docstring.
USEFUL_TOLERANCE = 0.15


class Run(NamedTuple):
    label: str
    model_id: str
    quant: str
    active_params: float
    mode: RunMode
    gpu_layers: int
    cpu_moe_layers: int | None
    context: int
    micro_batch: int
    working_context: int
    gen_tps: float
    pp_tps: float
    # Where the always-on shared experts and the vision projector went. Neither enters
    # the traffic model, and both are how the estimator tells the reference machine's two
    # Flash-Next runs apart: one put them on the card, the winning one did not.
    shared_experts_pool: Pool | None = None
    projector_pool: Pool | None = None


DENSE_IN_RAM = Run(
    "Qwen3-0.6B -ngl 0", "qwen3-0.6b", "Q8_0", 0.6e9, "cpu", 0, None, 4096, 512, 128, 78.0, 2924.0
)
DENSE_ON_CARD = Run(
    "Qwen3-0.6B -ngl 99",
    "qwen3-0.6b",
    "Q8_0",
    0.6e9,
    "gpu",
    99,
    None,
    4096,
    512,
    128,
    279.5,
    21734.0,
)
CODER = Run(
    "Qwen3-Coder-Next",
    "qwen3-coder-next",
    "UD-Q4_K_XL",
    3e9,
    "moe-offload",
    99,
    48,
    262144,
    2048,
    128,
    24.7,
    323.0,
)
FLASH = Run(
    "Qwen3.8-Flash-Next",
    "qwen3.8-flash-next",
    "UD-Q4_K_XL",
    6e9,
    "moe-offload",
    99,
    48,
    40960,
    1024,
    1056,
    13.9,
    49.4,
    shared_experts_pool="ram",
    projector_pool="ram",
)
FLASH_EXPERTS_ON_CARD = FLASH._replace(
    label="Qwen3.8-Flash-Next, shared experts and vision on the card",
    context=16384,
    micro_batch=2048,
    gen_tps=14.5,
    pp_tps=50.0,
    shared_experts_pool=None,
    projector_pool="vram",
)
ALL_FOUR = [DENSE_IN_RAM, DENSE_ON_CARD, CODER, FLASH]
EXPERT_MODELS = [CODER, FLASH]


def placement_for(run: Run) -> Placement:
    return placement(
        mode=run.mode,
        gpu_layers=run.gpu_layers,
        cpu_moe_layers=run.cpu_moe_layers,
        context=run.context,
        micro_batch=run.micro_batch,
        shared_experts_pool=run.shared_experts_pool,
        projector_pool=run.projector_pool,
    )


def formula_for(run: Run) -> Formula:
    return formula_estimate(
        placement_for(run),
        catalog_facts(run.model_id, run.quant),
        resolve_bandwidths(reference_host()),
        working_context=run.working_context,
        micro_batch=run.micro_batch,
        active_params=run.active_params,
    )


@pytest.mark.parametrize("run", ALL_FOUR, ids=lambda run: run.label)
def test_the_formula_reproduces_every_reference_measurement(run: Run) -> None:
    predicted = formula_for(run).gen_tps
    assert predicted == pytest.approx(run.gen_tps, rel=USEFUL_TOLERANCE)
    assert predicted == pytest.approx(run.gen_tps, rel=0.08), (
        f"{run.label}: predicted {predicted:.2f} against a measured {run.gen_tps}"
    )


def test_the_formula_gets_the_ratio_between_the_two_expert_models_right() -> None:
    predicted = formula_for(CODER).gen_tps / formula_for(FLASH).gen_tps
    assert predicted == pytest.approx(24.7 / 13.9, rel=0.05)


def test_the_formula_spans_the_whole_range_from_the_fastest_run_to_the_slowest() -> None:
    """A model that only works at one speed is a constant fitted to one machine state."""
    fastest = formula_for(DENSE_ON_CARD).gen_tps
    slowest = formula_for(FLASH).gen_tps
    assert fastest / slowest == pytest.approx(279.5 / 13.9, rel=0.08)


@pytest.mark.parametrize("run", EXPERT_MODELS, ids=lambda run: run.label)
def test_the_expert_read_dominates_both_expert_predictions(run: Run) -> None:
    """Which term dominates, which is what a person who doubts the number needs to see."""
    formula = formula_for(run)
    total = formula.vram_seconds + formula.ram_seconds + formula.overhead_seconds
    assert formula.ram_seconds / total > 0.6
    assert formula.ram_seconds > formula.vram_seconds


def test_the_card_dominates_a_model_that_fits_on_it() -> None:
    formula = formula_for(DENSE_ON_CARD)
    total = formula.vram_seconds + formula.ram_seconds + formula.overhead_seconds
    assert formula.ram_seconds == 0.0
    assert formula.vram_seconds / total > 0.7


def test_the_overhead_is_one_fixed_term_and_does_not_grow_with_depth() -> None:
    """A per-layer overhead exceeded the whole measured token of a 28-layer model.

    Section 10.1 carried 0.20 ms per transformer block, which for Qwen3-0.6B is 5.6 ms
    against a token measured at 3.58 ms. The term was impossible, not merely large, and it
    capped every 28-layer model near 105 tokens per second however small.
    """
    shallow = formula_for(DENSE_ON_CARD)
    deep = formula_for(CODER)
    assert shallow.overhead_seconds == deep.overhead_seconds == 0.001
    assert 1 / shallow.gen_tps < 28 * 0.0002
    assert shallow.gen_tps > 200


def test_the_token_embedding_table_is_not_read_per_token() -> None:
    """A lookup reads one row, and counting the table breaks the memory bus.

    Qwen3-0.6B keeps 165 MB of token embeddings against 468 MB of block weights, a quarter
    of the file. Charging a token for the whole table puts its apparent read rate above
    anything the hardware can deliver, which is how the earlier error was caught.
    """
    facts = catalog_facts("qwen3-0.6b", "Q8_0")
    assert facts.bytes_token_embd == 165_306_368
    traffic = per_token_traffic(placement_for(DENSE_ON_CARD), facts, working_context=0)
    assert traffic.total_bytes == facts.bytes_dense_block_weights
    assert all("embd" not in line.component for line in traffic.lines)


@pytest.mark.parametrize("run", ALL_FOUR, ids=lambda run: run.label)
def test_the_measured_rung_reproduces_the_benchmark_exactly(run: Run) -> None:
    """With the machine's own benchmarks in hand, the tool shows them, dated."""
    estimate = estimate_speed(
        placement_for(run),
        catalog_facts(run.model_id, run.quant),
        reference_host(),
        working_context=run.working_context,
        active_params=run.active_params,
        quant=run.quant,
        measurements=catalog_model(run.model_id).measured,
    )
    assert estimate.confidence == "measured"
    assert estimate.measured_on is not None
    assert estimate.gen_tps == pytest.approx(run.gen_tps)
    assert estimate.pp_tps == pytest.approx(run.pp_tps)
    shares = (
        estimate.vram_seconds_per_token
        + estimate.ram_seconds_per_token
        + estimate.overhead_seconds_per_token
    )
    assert shares == pytest.approx(1 / estimate.gen_tps)


def test_generation_barely_moves_with_the_micro_batch() -> None:
    """Which is why a generation benchmark is matched on the offload and not on ``-ub``.

    The reference machine measured Flash-Next at 14.5 tokens per second with ``-ub 2048``
    and 13.9 with ``-ub 1024``. Asked about that same placement at 1024, the ``-ub 2048``
    run is still a benchmark of it: the micro-batch moves no byte between pools, and
    refusing it would cost the label to buy four percent.
    """
    anchor = [m for m in catalog_model("qwen3.8-flash-next").measured if m.context == 16384]
    at_1024 = placement_for(FLASH_EXPERTS_ON_CARD._replace(micro_batch=1024))
    estimate = estimate_speed(
        at_1024,
        catalog_facts(FLASH.model_id, FLASH.quant),
        reference_host(),
        working_context=FLASH.working_context,
        active_params=FLASH.active_params,
        quant=FLASH.quant,
        measurements=anchor,
    )
    assert estimate.gen_tps == pytest.approx(14.5)
    assert "Generation is a benchmark of this configuration" in " ".join(estimate.notes)


def test_the_two_flash_next_runs_are_not_benchmarks_of_each_other() -> None:
    """The winning configuration moved the shared experts and the projector off the card.

    Both runs are Flash-Next at UD-Q4_K_XL on the same machine on the same day, and one of
    them is 862 MB of projector and 239 MB of shared experts lighter on an eight-gigabyte
    card. A benchmark of the first is not a benchmark of the second, and the label has to
    say so: 14.5 tokens per second reported as a measurement of the configuration that ran
    at 13.9 is the kind of false ``measured`` that is worse than no label at all.
    """
    runs = catalog_model("qwen3.8-flash-next").measured
    estimate = estimate_speed(
        placement_for(FLASH),
        catalog_facts(FLASH.model_id, FLASH.quant),
        reference_host(),
        working_context=FLASH.working_context,
        active_params=FLASH.active_params,
        quant=FLASH.quant,
        measurements=[m for m in runs if m.context == 16384],
    )
    assert estimate.confidence == "calibrated"
    assert estimate.gen_tps != pytest.approx(14.5)


# --- Prompt processing, which section 10.2 has not had the same treatment ------------


def test_prompt_processing_matches_the_model_its_constant_was_fitted_on() -> None:
    """Section 10.2's 4 GB/s came from Flash-Next, and Flash-Next it predicts."""
    assert formula_for(FLASH).pp_tps == pytest.approx(49.4, rel=USEFUL_TOLERANCE)


def test_prompt_processing_is_half_the_truth_for_a_model_that_fits_in_memory() -> None:
    """And Coder-Next, whose expert set stays in the page cache, it under-predicts twofold.

    Section 10.2's 4 GB/s is not a property of the link. It is what Qwen3.8-Flash-Next
    reaches because its 111 GB of weights plus a 29 GB streamed table do not fit in the
    machine's 128 GB, so part of every micro-batch's expert read comes off the disk.
    Qwen3-Coder-Next's 50 GB expert set stays cached and streams at about 12.8 GB/s, which
    is the whole of the gap asserted here.
    """
    assert 0.40 < formula_for(CODER).pp_tps / CODER.pp_tps < 0.55


def test_prompt_processing_is_an_order_out_on_a_small_dense_model() -> None:
    """The compute term's peak figure is wrong, and the small dense runs make it visible.

    Nothing streams for a dense model held on the card, so the whole prompt formula is
    ``2 x active_params x ub / (tflops_fp16 x eff_pp)`` and a measurement reads the product
    of those two constants off directly. Qwen3-0.6B managed 21,734 prompt tokens per
    second, which is 26 TFLOP/s -- more than the 15 the bundled table gives this card, so
    no efficiency below one can reach it. The table's figure is the RTX 4060's fp32 shader
    throughput rather than what its tensor cores do with fp16, and ``EFF_PP`` has been
    absorbing the difference on the models where the streaming term hid it.

    Section 10.1 was settled by four runs chosen to isolate its pools; section 10.2 has had
    no such treatment, and this is what that costs. Recorded rather than refitted to a
    single point, which is the mistake that produced the 0.36.
    """
    predicted = formula_for(DENSE_ON_CARD).pp_tps
    assert predicted < DENSE_ON_CARD.pp_tps * 0.25
    assert 2 * 0.6e9 * DENSE_ON_CARD.pp_tps / 1e12 > 15.0


def test_a_prompt_benchmark_from_another_micro_batch_is_only_as_good_as_its_anchor() -> None:
    """Calibration inherits whatever is wrong with the run it calibrates on.

    The catalog's ``-ub 2048`` prompt figure for Flash-Next, 50 tokens per second, is a
    real 1,056-token request: only one micro-batch of 1,056 tokens was ever formed, so the
    figure belongs to a micro-batch half the size its flags claim. Calibrating the
    ``-ub 1024`` estimate on it drags the prediction well below the 49.4 that was measured
    there. Recorded rather than worked around: the fix belongs in the catalog, which should
    say what micro-batch a prompt figure really used.
    """
    anchor = [m for m in catalog_model("qwen3.8-flash-next").measured if m.context == 16384]
    estimate = estimate_speed(
        placement_for(FLASH),
        catalog_facts(FLASH.model_id, FLASH.quant),
        reference_host(),
        working_context=FLASH.working_context,
        active_params=FLASH.active_params,
        quant=FLASH.quant,
        measurements=anchor,
    )
    assert estimate.confidence == "calibrated"
    assert estimate.pp_tps < 49.4 * (1 - USEFUL_TOLERANCE)
