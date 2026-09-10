"""The calibration arithmetic, against figures that can be checked on paper.

Two kinds of test here. The synthetic ones build measurements *from* known constants and
assert the fit gives those constants back, so the arithmetic is checkable end to end. The
others are about refusal, which is the part that matters more: what the fit does when the
data does not determine the answer, and what it does when the answer it gets is impossible.
"""

from __future__ import annotations

import pytest

from llamafit.bench.calibrate import calibrate, refusal_text
from llamafit.bench.types import BenchRun, Calibration, PagingCheck
from tests.fixtures.bench import (
    FINGERPRINT,
    MIB,
    conditions,
    reference_generation_runs,
    run_of,
    traffic,
)

# The constants the synthetic measurements below are built from. Nothing in the fit knows
# these; they are what it has to arrive at.
EFF_VRAM = 0.5
EFF_SEQUENTIAL = 0.8
EFF_SCATTERED = 0.4
OVERHEAD_S = 0.002
RAM_GBPS = 50.0
DEVICE_GBPS = 200.0


def generation_run(tag: str, *, device_gb: float, seq_gb: float, sca_gb: float) -> BenchRun:
    """One measurement a machine with the constants above would have produced."""
    seconds = (
        device_gb * 1e9 / (DEVICE_GBPS * 1e9 * EFF_VRAM)
        + seq_gb * 1e9 / (RAM_GBPS * 1e9 * EFF_SEQUENTIAL)
        + sca_gb * 1e9 / (RAM_GBPS * 1e9 * EFF_SCATTERED)
        + OVERHEAD_S
    )
    return run_of(
        kind="llama-bench-tg",
        conditions_=conditions(settings={"ngl": tag, "ub": "1024"}),
        gen_tps=1.0 / seconds,
        traffic_=traffic(
            device_bytes=round(device_gb * 1e9),
            sequential_bytes=round(seq_gb * 1e9),
            scattered_bytes=round(sca_gb * 1e9),
            ram_gbps=RAM_GBPS,
            device_gbps=DEVICE_GBPS,
        ),
    )


def fit(runs: list[BenchRun]) -> Calibration:
    """Calibrate the reference fingerprint from these runs."""
    return calibrate(runs, host_fingerprint=FINGERPRINT)


def refusal(result: Calibration, name: str) -> object:
    """The refusal for one constant, or ``None`` when it was not refused."""
    return next((item for item in result.refusals if item.parameter == name), None)


def test_the_four_constants_come_back_when_the_pools_were_isolated() -> None:
    """Five configurations, four unknowns, and the fit lands on the numbers it was built from."""
    result = fit(
        [
            generation_run("a", device_gb=1.0, seq_gb=0.0, sca_gb=0.0),
            generation_run("b", device_gb=0.0, seq_gb=1.0, sca_gb=0.0),
            generation_run("c", device_gb=0.0, seq_gb=0.0, sca_gb=1.0),
            generation_run("d", device_gb=2.0, seq_gb=1.0, sca_gb=2.0),
            generation_run("e", device_gb=1.0, seq_gb=2.0, sca_gb=1.0),
        ]
    )
    assert result.eff_vram == pytest.approx(EFF_VRAM, rel=1e-6)
    assert result.eff_ram_sequential == pytest.approx(EFF_SEQUENTIAL, rel=1e-6)
    assert result.eff_ram_scattered == pytest.approx(EFF_SCATTERED, rel=1e-6)
    assert result.fixed_overhead_s == pytest.approx(OVERHEAD_S, rel=1e-6)
    assert result.generation_runs == 5
    assert result.generation_residual == pytest.approx(0.0, abs=1e-9)


def test_two_measurements_do_not_pin_three_parameters() -> None:
    """The specification's own point: the reference set has four runs because two would not do.

    A run entirely in system memory and a run entirely on the card put a non-zero in two
    columns, and the constant column is always there. Three free parameters, two equations.
    """
    result = fit(
        [
            generation_run("a", device_gb=1.0, seq_gb=0.0, sca_gb=0.0),
            generation_run("b", device_gb=0.0, seq_gb=1.0, sca_gb=0.0),
        ]
    )
    assert result.eff_vram is None
    assert result.eff_ram_sequential is None
    assert result.fixed_overhead_s is None
    refused = refusal(result, "eff_vram")
    assert refused is not None
    assert refused.reason == "too-few-measurements"  # type: ignore[attr-defined]
    assert refused.measurements == 2  # type: ignore[attr-defined]
    assert refused.parameters == 3  # type: ignore[attr-defined]


def test_a_pool_nothing_touched_is_refused_for_having_no_equation_to_appear_in() -> None:
    result = fit(
        [
            generation_run("a", device_gb=1.0, seq_gb=0.0, sca_gb=0.0),
            generation_run("b", device_gb=0.0, seq_gb=1.0, sca_gb=0.0),
        ]
    )
    refused = refusal(result, "eff_ram_scattered")
    assert refused is not None
    assert refused.reason == "no-data"  # type: ignore[attr-defined]


def test_two_points_on_one_pool_give_its_efficiency_and_the_overhead_exactly() -> None:
    """Arithmetic anybody can do on paper, which is why it is here.

    One gigabyte of contiguous weights takes 27 ms a token and two take 52. The slope is
    25 ms per 20 ms of raw read, which is 1.25 = 1/0.8, and the intercept is 2 ms.
    """
    result = fit(
        [
            generation_run("a", device_gb=0.0, seq_gb=1.0, sca_gb=0.0),
            generation_run("b", device_gb=0.0, seq_gb=2.0, sca_gb=0.0),
        ]
    )
    assert result.eff_ram_sequential == pytest.approx(0.8, rel=1e-9)
    assert result.fixed_overhead_s == pytest.approx(0.002, rel=1e-9)
    # Two rows and two free parameters: the line goes through both points, so there is no
    # residual and nothing corroborating either number.
    assert result.generation_residual is None


def test_the_reference_machines_own_four_runs_do_not_solve_to_a_possible_efficiency() -> None:
    """The published constants were chosen by judgement, and this is why.

    Section 10.1's four measurements are four equations in four unknowns, so they have a
    unique solution -- and that solution puts the fraction of the card's bandwidth
    generation reaches above 1.6, which is not a fraction. Small disagreements between the
    runs have nowhere to go in an exactly determined system, so they come out as an
    impossible parameter instead of as a residual.
    """
    result = fit(reference_generation_runs())
    assert result.eff_vram is None
    assert result.eff_ram_scattered is None
    refused = refusal(result, "eff_vram")
    assert refused is not None
    assert refused.reason == "unphysical"  # type: ignore[attr-defined]
    assert refused.value > 1.0  # type: ignore[attr-defined]
    # The others were plausible on their own and go with it anyway, because least squares
    # chose them against the impossible one.
    scattered = refusal(result, "eff_ram_scattered")
    assert scattered is not None
    assert scattered.reason == "discarded-with-the-fit"  # type: ignore[attr-defined]


def test_configurations_that_vary_together_cannot_separate_the_terms_they_share() -> None:
    """Enough measurements, and still not enough information.

    Three runs where the expert set is always exactly twice the card's traffic vary in one
    direction, not two: nothing in them says which of the two pools the time went to. The
    count check passes and the rank check is what catches it, which is why both exist.
    """
    result = fit(
        [
            generation_run(str(size), device_gb=size, seq_gb=0.0, sca_gb=2.0 * size)
            for size in (1.0, 2.0, 3.0)
        ]
    )
    assert result.eff_vram is None
    assert result.eff_ram_scattered is None
    refused = refusal(result, "eff_vram")
    assert refused is not None
    assert refused.reason == "not-identifiable"  # type: ignore[attr-defined]
    assert refused.measurements == 3  # type: ignore[attr-defined]
    assert refused.parameters == 3  # type: ignore[attr-defined]


def test_a_run_the_paging_detector_caught_is_never_fitted_against() -> None:
    """A paged run measured the driver moving pages, not the configuration."""
    good = [
        generation_run("a", device_gb=1.0, seq_gb=0.0, sca_gb=0.0),
        generation_run("b", device_gb=0.0, seq_gb=1.0, sca_gb=0.0),
        generation_run("c", device_gb=0.0, seq_gb=0.0, sca_gb=1.0),
        generation_run("d", device_gb=2.0, seq_gb=1.0, sca_gb=2.0),
    ]
    paged = run_of(
        kind="llama-bench-tg",
        conditions_=conditions(settings={"ngl": "paged", "ub": "1024"}),
        gen_tps=1.0,  # a tenth of what the configuration would run at
        traffic_=traffic(
            device_bytes=int(3e9),
            sequential_bytes=0,
            scattered_bytes=int(1e9),
            ram_gbps=RAM_GBPS,
            device_gbps=DEVICE_GBPS,
        ),
        paging=PagingCheck(paged=True, vram_ratio=0.99, speed_ratio=0.2, reason="paging"),
    )
    result = fit([*good, paged])
    assert result.generation_runs == 4
    assert result.eff_vram == pytest.approx(EFF_VRAM, rel=1e-6)


def test_the_first_request_after_a_cold_start_is_not_fitted_against() -> None:
    """It measures a disk waking up, and a memory constant fitted to it would be wrong."""
    cold = run_of(
        kind="server-short",
        conditions_=conditions(settings={"ngl": "cold", "ub": "1024"}, context=32768),
        gen_tps=1.0,
        traffic_=traffic(device_bytes=int(1e9), ram_gbps=RAM_GBPS, device_gbps=DEVICE_GBPS),
    )
    runs = [
        generation_run("a", device_gb=1.0, seq_gb=0.0, sca_gb=0.0),
        generation_run("b", device_gb=0.0, seq_gb=1.0, sca_gb=0.0),
        generation_run("c", device_gb=0.0, seq_gb=0.0, sca_gb=1.0),
        generation_run("d", device_gb=2.0, seq_gb=1.0, sca_gb=2.0),
        cold,
    ]
    result = fit(runs)
    assert result.generation_runs == 4
    assert result.eff_ram_scattered == pytest.approx(EFF_SCATTERED, rel=1e-6)


def prompt_run(micro_batch: int, *, eff_pp: float = 0.30, pcie_gbps: float = 12.8) -> BenchRun:
    """A prompt measurement a machine with those two constants would have produced."""
    active, flops, streamed = 3e9, 15e12, int(50e9)
    seconds = 2 * active * micro_batch / (flops * eff_pp) + streamed / (pcie_gbps * 1e9)
    return run_of(
        kind="llama-bench-pp",
        conditions_=conditions(
            settings={"ngl": "99", "ub": str(micro_batch)}, micro_batch=micro_batch
        ),
        pp_tps=micro_batch / seconds,
        traffic_=traffic(
            streamed_expert_bytes=streamed,
            active_params=active,
            compute_flops=flops,
            micro_batch=micro_batch,
        ),
    )


def test_prompt_processing_gives_back_the_compute_efficiency_and_the_effective_link() -> None:
    result = fit([prompt_run(512), prompt_run(1024), prompt_run(2048)])
    assert result.eff_pp == pytest.approx(0.30, rel=1e-6)
    assert result.pcie_effective_gbps == pytest.approx(12.8, rel=1e-6)
    assert result.prompt_runs == 3


def test_one_micro_batch_is_one_equation_and_two_parameters_need_two() -> None:
    result = fit([prompt_run(1024)])
    assert result.eff_pp is None
    assert result.pcie_effective_gbps is None
    refused = refusal(result, "pcie_effective_gbps")
    assert refused is not None
    assert refused.reason == "too-few-measurements"  # type: ignore[attr-defined]


def test_a_link_nothing_streamed_across_is_refused_rather_than_invented() -> None:
    """With every expert on the card there is no streaming term to measure."""
    runs = [
        run_of(
            kind="llama-bench-pp",
            conditions_=conditions(settings={"ub": str(size)}, micro_batch=size),
            pp_tps=float(size),
            traffic_=traffic(streamed_expert_bytes=0, micro_batch=size),
        )
        for size in (512, 1024)
    ]
    result = fit(runs)
    refused = refusal(result, "pcie_effective_gbps")
    assert refused is not None
    assert refused.reason == "no-data"  # type: ignore[attr-defined]


def buffer_run(context: int, mib: float, micro_batch: int = 1024) -> BenchRun:
    """A server run whose log reported a compute buffer of ``mib`` at ``context``."""
    return run_of(
        kind="server-1k",
        conditions_=conditions(
            settings={"ngl": "99", "ub": str(micro_batch)},
            context=context,
            micro_batch=micro_batch,
        ),
        traffic_=traffic(micro_batch=micro_batch),
        buffer_bytes={"cuda0 compute": int(mib * MIB)},
    )


def test_the_compute_buffer_is_fitted_from_two_contexts_and_matches_the_record() -> None:
    """1,337 MiB at 32K and 1,433 at 64K is 3 MiB per 1K on a base of 1,241.

    Section 8.2's own constants for this micro-batch are 1,240 MiB and 3 MiB per 1K, from
    the same two rows of the calibration record.
    """
    result = fit([buffer_run(32768, 1337.0), buffer_run(65536, 1433.0)])
    assert len(result.compute_buffer) == 1
    point = result.compute_buffer[0]
    assert point.micro_batch == 1024
    assert point.per_1k_context_bytes == pytest.approx(3 * MIB, rel=1e-3)
    assert point.base_bytes == pytest.approx(1241 * MIB, rel=1e-3)


def test_one_context_determines_no_slope_and_is_refused() -> None:
    result = fit([buffer_run(32768, 1337.0)])
    assert result.compute_buffer == ()
    refused = refusal(result, "compute_buffer[1024]")
    assert refused is not None
    assert refused.reason == "too-few-measurements"  # type: ignore[attr-defined]


def test_the_per_layer_overhead_is_never_fitted_because_there_is_no_such_term() -> None:
    """Section 10.1 removed it. A number fitted for it would be read by nothing."""
    result = fit(reference_generation_runs())
    refused = refusal(result, "layer_overhead_ms")
    assert refused is not None
    assert refused.reason == "no-term-in-the-model"  # type: ignore[attr-defined]
    assert "no such term" in refusal_text(refused)  # type: ignore[arg-type]


def test_a_calibration_from_nothing_refuses_every_constant_by_name() -> None:
    result = fit([])
    assert not result.fitted_anything
    names = {item.parameter for item in result.refusals}
    assert {
        "eff_vram",
        "eff_ram_sequential",
        "eff_ram_scattered",
        "fixed_overhead_s",
        "eff_pp",
        "pcie_effective_gbps",
        "layer_overhead_ms",
    } <= names


def test_runs_from_another_machine_are_not_fitted_to_this_one() -> None:
    elsewhere = [
        run_of(
            kind="llama-bench-tg",
            conditions_=conditions(fingerprint="ffffffffffffffff", settings={"ngl": str(index)}),
            gen_tps=20.0,
            traffic_=traffic(device_bytes=int(1e9)),
        )
        for index in range(5)
    ]
    assert calibrate(elsewhere, host_fingerprint=FINGERPRINT).generation_runs == 0


def test_every_refusal_can_be_put_into_words() -> None:
    result = fit(reference_generation_runs(limit=2))
    assert result.refusals
    for item in result.refusals:
        assert refusal_text(item).strip()
