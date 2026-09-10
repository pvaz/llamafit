from datetime import date

import pytest

from llamafit.constants import EFF_RAM_SCATTERED, EFF_RAM_SEQUENTIAL, EFF_VRAM
from llamafit.models.catalog import Measured
from llamafit.models.gguf import GgufFacts
from llamafit.speed import (
    active_expert_bytes,
    estimate_speed,
    formula_estimate,
    kv_bytes_per_token,
    per_token_traffic,
    resolve_bandwidths,
    streamed_expert_fraction,
)
from tests.fixtures.speed import placement, reference_host

GB = 1_000_000_000


def moe_facts(**overrides: object) -> GgufFacts:
    values: dict[str, object] = {
        "arch": "qwen3next",
        "n_layer": 48,
        "n_embd": 2048,
        "n_vocab": 151936,
        "n_head": 16,
        "n_head_kv": 2,
        "head_dim": 256,
        "value_head_dim": 256,
        "attention_layers": 12,
        "attention_layers_source": "tensors",
        "n_expert": 512,
        "n_expert_used": 10,
        "has_shared_experts": True,
        "bytes_expert_weights": 48 * GB,
        "bytes_dense_block_weights": 2 * GB,
        "bytes_output_head": 1 * GB,
        "bytes_token_embd": 0,
        "bytes_lazy_tables": 0,
        "bytes_global_weights": 0,
        "bytes_total": 51 * GB,
        "kv_bytes_per_token_f16": 24576,
    }
    values.update(overrides)
    values["bytes_total"] = (
        int(values["bytes_expert_weights"])  # type: ignore[arg-type]
        + int(values["bytes_dense_block_weights"])  # type: ignore[arg-type]
        + int(values["bytes_output_head"])  # type: ignore[arg-type]
        + int(values["bytes_token_embd"])  # type: ignore[arg-type]
        + int(values["bytes_lazy_tables"])  # type: ignore[arg-type]
        + int(values["bytes_global_weights"])  # type: ignore[arg-type]
    )
    return GgufFacts(**values)  # type: ignore[arg-type]


def dense_facts() -> GgufFacts:
    return moe_facts(
        arch="qwen3",
        n_expert=None,
        n_expert_used=None,
        has_shared_experts=False,
        bytes_expert_weights=0,
        bytes_dense_block_weights=12 * GB,
        bytes_output_head=1 * GB,
    )


def test_moe_offload_reads_the_active_experts_from_ram_and_the_rest_from_the_card() -> None:
    traffic = per_token_traffic(placement(), moe_facts(), working_context=8192)
    by_component = {(line.component, line.pool): line for line in traffic.lines}

    assert by_component[("routed-experts", "ram")].access == "scattered"
    assert by_component[("routed-experts", "ram")].bytes_ == 48 * GB * 10 // 512
    assert by_component[("block-weights", "vram")].access == "device"
    assert ("routed-experts", "vram") not in by_component
    assert traffic.scattered_bytes == active_expert_bytes(moe_facts())
    assert traffic.sequential_bytes == 0
    assert traffic.total_bytes == traffic.device_bytes + traffic.scattered_bytes


def test_a_dense_model_reads_nothing_scattered() -> None:
    traffic = per_token_traffic(
        placement(mode="gpu", cpu_moe_layers=None), dense_facts(), working_context=8192
    )
    assert traffic.scattered_bytes == 0
    assert traffic.device_bytes == 13 * GB + 24576 * 8192


def test_cpu_mode_reads_everything_from_system_memory() -> None:
    traffic = per_token_traffic(
        placement(mode="cpu", gpu_layers=0, cpu_moe_layers=None), moe_facts(), working_context=4096
    )
    assert traffic.device_bytes == 0
    assert traffic.scattered_bytes == active_expert_bytes(moe_facts())
    assert traffic.sequential_bytes == 2 * GB + 1 * GB + 24576 * 4096


def test_half_the_layers_on_the_card_splits_the_weights_in_half() -> None:
    facts = moe_facts()
    traffic = per_token_traffic(
        placement(mode="hybrid", gpu_layers=24, cpu_moe_layers=None), facts, working_context=0
    )
    by_component = {(line.component, line.pool): line.bytes_ for line in traffic.lines}
    assert by_component[("block-weights", "vram")] == 1 * GB
    assert by_component[("block-weights", "ram")] == 1 * GB
    # Half the layers are on the card, so half the experts a token wants are already there
    # and only the other half is a scattered read from system memory.
    assert traffic.scattered_bytes == round(active_expert_bytes(facts) / 2)
    assert by_component[("routed-experts", "vram")] == round(active_expert_bytes(facts) / 2)
    # The output head stays off the card until every layer is on it.
    assert by_component[("output-head", "ram")] == 1 * GB


def test_a_quantised_cache_costs_less_per_token() -> None:
    facts = moe_facts()
    assert kv_bytes_per_token(facts, "f16") == 24576
    assert kv_bytes_per_token(facts, "q8_0") == round(24576 * 8.5 / 16)
    assert kv_bytes_per_token(facts, "Q4_0") == round(24576 * 4.5 / 16)


def test_an_unknown_cache_type_is_treated_as_f16() -> None:
    assert kv_bytes_per_token(moe_facts(), "something-new") == 24576


def test_a_file_that_reports_no_cache_size_contributes_no_cache_traffic() -> None:
    facts = moe_facts(kv_bytes_per_token_f16=None)
    assert kv_bytes_per_token(facts, "f16") == 0
    traffic = per_token_traffic(placement(), facts, working_context=131072)
    assert all(line.component != "kv-cache" for line in traffic.lines)


def test_facts_with_no_layer_count_put_everything_in_system_memory() -> None:
    traffic = per_token_traffic(placement(), moe_facts(n_layer=None), working_context=0)
    assert traffic.device_bytes == 0
    assert traffic.sequential_bytes > 0


def test_a_micro_batch_of_256_touches_nearly_every_expert() -> None:
    facts = moe_facts()
    assert streamed_expert_fraction(facts, 256) > 0.99
    assert streamed_expert_fraction(facts, 2048) > 0.999
    assert streamed_expert_fraction(facts, 16) < 0.3
    assert streamed_expert_fraction(facts, 0) == 0.0
    assert streamed_expert_fraction(dense_facts(), 512) == 0.0


def test_active_expert_bytes_is_zero_without_routed_experts() -> None:
    assert active_expert_bytes(dense_facts()) == 0


def test_system_memory_has_two_bandwidths_and_the_scattered_one_is_about_half() -> None:
    bandwidths = resolve_bandwidths(reference_host())
    assert bandwidths.ram_gbps == 57.0
    assert bandwidths.sequential == pytest.approx(57.0 * EFF_RAM_SEQUENTIAL * 1e9)
    assert bandwidths.scattered == pytest.approx(57.0 * EFF_RAM_SCATTERED * 1e9)
    assert bandwidths.scattered / bandwidths.sequential == pytest.approx(0.514, abs=0.01)
    assert bandwidths.device == pytest.approx(272.0 * EFF_VRAM * 1e9)
    assert not bandwidths.assumed


def test_an_unmeasured_machine_falls_back_by_cpu_architecture() -> None:
    bandwidths = resolve_bandwidths(reference_host(ram_gbps=None, bandwidth_source="unknown"))
    assert bandwidths.ram_gbps == 60.0
    assert bandwidths.assumed
    assert "60 GB/s assumed for x86_64" in " ".join(bandwidths.notes)


def test_an_estimated_memory_figure_is_used_but_still_counts_as_assumed() -> None:
    bandwidths = resolve_bandwidths(reference_host(ram_gbps=19.0, bandwidth_source="estimated"))
    assert bandwidths.ram_gbps == 19.0
    assert bandwidths.assumed


def test_an_assumed_memory_figure_is_used_but_still_counts_as_assumed() -> None:
    bandwidths = resolve_bandwidths(reference_host(ram_gbps=40.0, bandwidth_source="assumed"))
    assert bandwidths.ram_gbps == 40.0
    assert bandwidths.assumed


def test_a_card_the_table_does_not_know_falls_back_by_backend() -> None:
    host = reference_host()
    host.gpus[0].name = "Some Unreleased Accelerator"
    host.gpus[0].bandwidth_gbps = None
    host.gpus[0].compute_tflops_fp16 = None
    bandwidths = resolve_bandwidths(host)
    assert bandwidths.device_gbps == 250.0
    assert bandwidths.assumed
    assert "cuda fallback" in " ".join(bandwidths.notes)


def test_a_machine_with_no_card_estimates_prompt_compute_on_the_cpu() -> None:
    bandwidths = resolve_bandwidths(reference_host(with_gpu=False))
    assert bandwidths.device is None
    assert bandwidths.compute_flops == pytest.approx(8 * 0.05 * 1e12)
    assert bandwidths.assumed


def test_the_card_bandwidth_comes_from_the_bundled_table_when_the_probe_missed_it() -> None:
    host = reference_host()
    host.gpus[0].bandwidth_gbps = None
    host.gpus[0].compute_tflops_fp16 = None
    bandwidths = resolve_bandwidths(host)
    assert bandwidths.device_gbps == 272.0
    assert bandwidths.compute_flops == pytest.approx(15e12)
    assert not bandwidths.assumed


def test_an_unsupported_placement_produces_no_number() -> None:
    estimate = estimate_speed(
        placement(mode="unsupported"), moe_facts(), reference_host(), active_params=3e9
    )
    assert estimate.gen_tps == 0.0
    assert estimate.pp_tps == 0.0
    assert estimate.confidence == "unsupported"
    assert estimate.measured_on is None


def test_the_three_shares_add_up_to_the_figure_shown() -> None:
    estimate = estimate_speed(placement(), moe_facts(), reference_host(), active_params=3e9)
    total = (
        estimate.vram_seconds_per_token
        + estimate.ram_seconds_per_token
        + estimate.overhead_seconds_per_token
    )
    assert total == pytest.approx(1 / estimate.gen_tps)


def test_the_expert_read_dominates_an_expert_offload_token() -> None:
    estimate = estimate_speed(placement(), moe_facts(), reference_host(), active_params=3e9)
    assert estimate.ram_seconds_per_token > estimate.vram_seconds_per_token
    assert estimate.ram_seconds_per_token > estimate.overhead_seconds_per_token


def test_the_notes_name_the_scattered_efficiency_and_say_why() -> None:
    estimate = estimate_speed(placement(), moe_facts(), reference_host(), active_params=3e9)
    joined = " ".join(estimate.notes)
    assert "0.36" in joined
    assert "scatter of small blocks" in joined
    assert "57 GB/s" in joined


def test_a_dense_model_on_the_cpu_names_the_contiguous_efficiency() -> None:
    estimate = estimate_speed(
        placement(mode="cpu", gpu_layers=0, cpu_moe_layers=None),
        dense_facts(),
        reference_host(),
        active_params=8e9,
    )
    joined = " ".join(estimate.notes)
    assert "0.70" in joined
    assert "contiguous weights" in joined
    assert "scatter of small blocks" not in joined


def test_an_assumed_bandwidth_is_never_labelled_better_than_estimated() -> None:
    measurement = Measured(
        profile="llama-bench tg128",
        quant="UD-Q4_K_XL",
        gen_tps=24.7,
        pp_tps=323.0,
        flags="-ngl 99 --n-cpu-moe 48 -ub 1024",
        date=date(2026, 9, 9),
    )
    known = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert known.confidence == "measured"

    estimate = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(ram_gbps=None, bandwidth_source="unknown"),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.gen_tps == 24.7
    assert estimate.confidence == "estimated"
    assert estimate.measured_on is None


def test_a_measured_figure_carries_the_date_it_was_taken() -> None:
    measurement = Measured(
        profile="llama-bench tg128",
        quant="UD-Q4_K_XL",
        gen_tps=24.7,
        pp_tps=323.0,
        flags="-ngl 99 --n-cpu-moe 48 -ub 1024",
        date=date(2026, 9, 9),
    )
    estimate = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "measured"
    assert estimate.measured_on == date(2026, 9, 9)
    assert estimate.gen_tps == 24.7
    assert estimate.pp_tps == 323.0


def test_a_benchmark_of_another_quant_is_not_borrowed() -> None:
    measurement = Measured(
        profile="llama-bench tg128", quant="Q8_0", gen_tps=99.0, flags="-ngl 99 --n-cpu-moe 48"
    )
    estimate = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "estimated"
    assert estimate.gen_tps != 99.0


def test_a_benchmark_of_another_offload_calibrates_rather_than_replaces() -> None:
    measurement = Measured(
        profile="llama-bench tg128",
        quant="UD-Q4_K_XL",
        gen_tps=24.7,
        pp_tps=323.0,
        flags="-ngl 99 --n-cpu-moe 48",
        date=date(2026, 9, 9),
    )
    facts = moe_facts()
    host = reference_host()
    target = placement(cpu_moe_layers=24)
    plain = estimate_speed(target, facts, host, active_params=3e9)
    calibrated = estimate_speed(
        target,
        facts,
        host,
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )

    anchor = formula_estimate(
        placement(),
        facts,
        resolve_bandwidths(host),
        working_context=8192,
        micro_batch=1024,
        active_params=3e9,
    )
    assert calibrated.confidence == "calibrated"
    assert calibrated.measured_on is None
    assert calibrated.gen_tps == pytest.approx(plain.gen_tps * 24.7 / anchor.gen_tps)
    assert "corrected by" in " ".join(calibrated.notes)


def test_a_benchmark_with_no_date_still_calibrates() -> None:
    measurement = Measured(
        profile="a run somebody forgot to date",
        quant="UD-Q4_K_XL",
        gen_tps=24.7,
        pp_tps=323.0,
        flags="-ngl 99 --n-cpu-moe 48",
    )
    estimate = estimate_speed(
        placement(cpu_moe_layers=24),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "calibrated"


def test_a_prompt_benchmark_at_another_micro_batch_calibrates_rather_than_replaces() -> None:
    measurement = Measured(
        profile="llama-bench pp2048",
        quant="UD-Q4_K_XL",
        gen_tps=24.7,
        pp_tps=323.0,
        flags="-ngl 99 --n-cpu-moe 48 -ub 2048",
        date=date(2026, 9, 9),
    )
    estimate = estimate_speed(
        placement(micro_batch=512),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "calibrated"
    assert "Prompt processing is the formula corrected" in " ".join(estimate.notes)


def test_a_benchmark_that_pins_the_experts_elsewhere_is_not_this_configuration() -> None:
    measurement = Measured(
        profile="all experts on the card",
        quant="UD-Q4_K_XL",
        gen_tps=90.0,
        pp_tps=900.0,
        flags="-ngl 99 --n-cpu-moe 0",
        date=date(2026, 9, 9),
    )
    estimate = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "calibrated"


def test_a_benchmark_that_offloaded_nothing_is_not_a_full_offload() -> None:
    measurement = Measured(
        profile="cpu only",
        quant="UD-Q4_K_XL",
        gen_tps=4.0,
        pp_tps=40.0,
        flags="-ngl 0",
        date=date(2026, 9, 9),
    )
    estimate = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "calibrated"


def test_a_benchmark_naming_no_flags_at_all_matches_any_configuration() -> None:
    measurement = Measured(
        profile="somebody's number",
        quant="UD-Q4_K_XL",
        gen_tps=20.0,
        pp_tps=200.0,
        date=date(2026, 9, 8),
    )
    estimate = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "measured"
    assert estimate.gen_tps == 20.0


def test_an_expert_offload_benchmark_needs_the_placement_to_offload_experts_too() -> None:
    measurement = Measured(
        profile="llama-bench tg128",
        quant="UD-Q4_K_XL",
        gen_tps=24.7,
        pp_tps=323.0,
        flags="-ngl 99 --n-cpu-moe 48",
        date=date(2026, 9, 9),
    )
    estimate = estimate_speed(
        placement(mode="gpu", cpu_moe_layers=None),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[measurement],
    )
    assert estimate.confidence == "calibrated"


def test_a_placement_that_would_page_says_the_figure_assumes_it_does_not() -> None:
    estimate = estimate_speed(
        placement(verdict="too-tight"), moe_facts(), reference_host(), active_params=3e9
    )
    assert "assumes nothing pages" in " ".join(estimate.notes)


def test_an_unsupplied_parameter_count_is_inferred_and_said_so() -> None:
    with_count = estimate_speed(placement(), moe_facts(), reference_host(), active_params=3e9)
    without = estimate_speed(placement(), moe_facts(), reference_host())
    assert without.gen_tps == pytest.approx(with_count.gen_tps)
    assert without.pp_tps != pytest.approx(with_count.pp_tps)
    assert "inferred from the weight bytes" in " ".join(without.notes)


def test_the_working_context_never_exceeds_the_placement_context() -> None:
    small = placement(context=4096)
    at_default = estimate_speed(small, moe_facts(), reference_host(), active_params=3e9)
    asked_for_more = estimate_speed(
        small, moe_facts(), reference_host(), working_context=131072, active_params=3e9
    )
    assert at_default.gen_tps == pytest.approx(asked_for_more.gen_tps)


def test_a_longer_context_costs_generation_speed() -> None:
    long_placement = placement(context=131072)
    short = estimate_speed(
        long_placement, moe_facts(), reference_host(), working_context=2048, active_params=3e9
    )
    long = estimate_speed(
        long_placement, moe_facts(), reference_host(), working_context=131072, active_params=3e9
    )
    assert long.gen_tps < short.gen_tps


def test_prompt_processing_falls_back_to_system_memory_without_a_card() -> None:
    host = reference_host(with_gpu=False)
    estimate = estimate_speed(
        placement(mode="cpu", gpu_layers=0, cpu_moe_layers=None),
        moe_facts(),
        host,
        active_params=3e9,
    )
    assert estimate.pp_tps > 0
    assert estimate.confidence == "estimated"


def test_the_weaker_of_the_two_labels_stands_for_both_numbers() -> None:
    generation_only = Measured(
        profile="llama-bench tg128",
        quant="UD-Q4_K_XL",
        gen_tps=24.7,
        flags="-ngl 99 --n-cpu-moe 48",
        date=date(2026, 9, 9),
    )
    estimate = estimate_speed(
        placement(),
        moe_facts(),
        reference_host(),
        active_params=3e9,
        quant="UD-Q4_K_XL",
        measurements=[generation_only],
    )
    assert estimate.gen_tps == 24.7
    assert estimate.confidence == "estimated"
    assert estimate.measured_on is None
    assert "Generation is a benchmark of this configuration" in " ".join(estimate.notes)
