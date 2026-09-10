"""The run modes and the ladders the placement search walks (design section 9.1 and 9.2)."""

from __future__ import annotations

import pytest

from llamafit.constants import ALL_GPU_LAYERS, MIN_CONTEXT_TOKENS
from llamafit.models.host import Cpu
from llamafit.models.plan import Budget, RunMode
from llamafit.placement.modes import (
    MODE_ORDER,
    PlacementSettings,
    available_modes,
    batch_for,
    context_ladder,
    has_margin,
    initial_settings,
    is_acceptable,
    is_moe,
    kv_ladder,
    layer_count,
    layer_ladder,
    native_context,
    projector_ladder,
    projector_of,
    rank,
    shared_expert_ladder,
    thread_count,
)
from tests.fixtures.placement import GIB, cpu_only_host, make_facts, make_model, reference_host


def budget(verdict: str = "fits") -> Budget:
    return Budget(
        lines=(),
        vram_required=1,
        ram_required=1,
        vram_available=2,
        ram_available=2,
        vram_utilisation=0.5,
        ram_utilisation=0.5,
        verdict=verdict,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("micro_batch", "expected"), [(2048, 4096), (1024, 2048), (512, 2048), (256, 2048)]
)
def test_the_batch_is_twice_the_micro_batch_with_a_floor(micro_batch: int, expected: int) -> None:
    assert batch_for(micro_batch) == expected


def test_threads_are_the_threads_the_performance_cores_provide_on_a_hybrid_part() -> None:
    # Section 9.4: 8 performance cores with simultaneous multithreading provide 16 of this
    # part's 32 threads, and 16 is the measured optimum -- four percent faster than 8.
    cpu = Cpu(model="i9-14900KF", physical_cores=24, logical_cores=32, performance_cores=8)
    assert thread_count(cpu) == 16


def test_threads_are_every_thread_when_no_core_is_an_efficiency_core() -> None:
    cpu = Cpu(model="Ryzen 9", physical_cores=16, logical_cores=32, performance_cores=None)
    assert thread_count(cpu) == 32


def test_threads_are_the_cores_on_a_part_without_multithreading() -> None:
    cpu = Cpu(model="Apple M3 Max", physical_cores=12, logical_cores=12, performance_cores=8)
    assert thread_count(cpu) == 8


def test_threads_are_never_zero() -> None:
    cpu = Cpu(model="mystery", physical_cores=0, logical_cores=0, performance_cores=0)
    assert thread_count(cpu) == 1


def test_the_context_ladder_steps_down_the_tier_ladder_not_by_halving() -> None:
    # Section 9.2: "Candidate contexts come from that ladder, not from halving."
    assert context_ladder(32768, 0) == (32768, 24576, 16384)
    assert context_ladder(40960, 0) == (40960, 32768, 24576, 16384)


def test_the_users_minimum_raises_the_floor() -> None:
    assert context_ladder(262144, 65536) == (262144, 196608, 131072, 98304, 65536)


def test_an_explicit_request_below_the_floor_is_honoured() -> None:
    # A user who asks for 8,192 tokens means it; the floor is only for a user who said
    # nothing, and it must not silently size them up.
    assert context_ladder(8192, 0) == (8192,)


def test_the_ladder_is_never_empty() -> None:
    assert context_ladder(MIN_CONTEXT_TOKENS, MIN_CONTEXT_TOKENS) == (MIN_CONTEXT_TOKENS,)


def test_the_kv_ladder_offers_the_quantised_cache_only_where_it_is_allowed() -> None:
    both, _ = make_model(kv_types=["f16", "q8_0"])
    f16_only, _ = make_model(kv_types=["f16"])
    unstated, _ = make_model()
    assert kv_ladder(both, allow_kv_quant=True) == ("f16", "q8_0")
    assert kv_ladder(f16_only, allow_kv_quant=True) == ("f16",)
    assert kv_ladder(both, allow_kv_quant=False) == ("f16",)
    # Nothing recorded is not the same as nothing works.
    assert kv_ladder(unstated, allow_kv_quant=True) == ("f16", "q8_0")


def test_the_layer_ladder_says_what_each_mode_varies() -> None:
    assert layer_ladder("gpu", 48) == (ALL_GPU_LAYERS,)
    assert layer_ladder("cpu", 48) == (0,)
    assert layer_ladder("moe-offload", 4) == (4, 3, 2, 1)
    assert layer_ladder("hybrid", 4) == (3, 2, 1)


def test_without_a_layer_count_moe_offload_keeps_its_one_configuration_and_hybrid_none() -> None:
    assert layer_ladder("moe-offload", None) == (ALL_GPU_LAYERS,)
    assert layer_ladder("hybrid", None) == ()


def test_a_model_is_moe_when_the_file_says_so_or_the_curator_does() -> None:
    dense, _ = make_model()
    labelled, _ = make_model(moe=True)
    assert is_moe(dense, make_facts(experts=128)) is True
    assert is_moe(labelled, None) is True
    assert is_moe(dense, make_facts()) is False


def test_layer_count_is_none_without_facts() -> None:
    assert layer_count(None) is None
    assert layer_count(make_facts(layers=12)) == 12


def test_the_planner_sizes_for_the_native_context_not_the_extended_one() -> None:
    model, _ = make_model(native=262144)
    extended = model.model_copy(
        update={"context": model.context.model_copy(update={"extended": 1048576})}
    )
    assert native_context(extended) == 262144


def test_every_mode_is_available_on_a_machine_that_can_run_them() -> None:
    model, quant = make_model(moe=True, facts=make_facts(layers=8, experts=128))
    assert available_modes(model, quant, reference_host()) == MODE_ORDER


def test_without_a_card_only_the_processor_is_left() -> None:
    model, quant = make_model(moe=True, facts=make_facts(experts=128))
    assert available_modes(model, quant, cpu_only_host()) == ("cpu",)


def test_a_dense_model_has_no_experts_to_offload() -> None:
    model, quant = make_model(facts=make_facts())
    assert available_modes(model, quant, reference_host()) == ("gpu", "hybrid", "cpu")


def test_hybrid_needs_a_layer_count_read_from_the_file() -> None:
    model, quant = make_model(moe=True)
    assert available_modes(model, quant, reference_host()) == ("gpu", "moe-offload", "cpu")


def test_the_projector_ladder_is_empty_without_a_projector_or_without_vision() -> None:
    plain, plain_quant = make_model()
    seeing, seeing_quant = make_model(projector=True)
    assert projector_ladder(plain, plain_quant, vision=True) == (None,)
    assert projector_ladder(seeing, seeing_quant, vision=False) == (None,)
    assert projector_ladder(seeing, seeing_quant, vision=True) == ("vram", "ram", None)


def test_the_projector_comes_from_the_source_that_publishes_the_quant() -> None:
    model, quant = make_model(projector=True)
    extra = projector_of(model, quant)
    assert extra is not None
    assert extra.file == "mmproj-F16.gguf"
    other = model.sources[0].quants[0].model_copy(update={"name": "Q2_K"})
    assert projector_of(model, other) is not None  # falls back to the whole entry


def test_verdicts_that_may_be_recommended_and_verdicts_with_room_to_spare() -> None:
    assert [is_acceptable(v) for v in ("comfortable", "fits", "tight")] == [True, True, True]
    assert [is_acceptable(v) for v in ("too-tight", "does-not-fit")] == [False, False]
    assert [has_margin(v) for v in ("comfortable", "fits")] == [True, True]
    assert has_margin("tight") is False


def settings_for(mode: RunMode, context: int, **changes: object) -> PlacementSettings:
    base = initial_settings(
        mode, context=context, micro_batch=2048, kv_type="f16", projector_pool=None
    )
    for name, value in changes.items():
        base = base.__class__(**{**base.__dict__, name: value})
    return base


def test_the_mode_outranks_the_context() -> None:
    # The regression this ordering exists for: a 125-billion-parameter model whose whole
    # context fits in system memory must not be recommended on the processor alone.
    on_the_card = settings_for("gpu", 16384)
    on_the_processor = settings_for("cpu", 262144)
    assert rank(on_the_card, budget(), requested_context=262144) > rank(
        on_the_processor, budget("comfortable"), requested_context=262144
    )


def test_within_a_mode_the_longer_context_wins() -> None:
    long, short = settings_for("gpu", 32768), settings_for("gpu", 16384)
    assert rank(long, budget(), requested_context=32768) > rank(
        short, budget("comfortable"), requested_context=32768
    )


def test_context_beyond_the_request_does_not_earn_anything_extra() -> None:
    asked = settings_for("gpu", 32768)
    more = settings_for("gpu", 65536)
    assert (
        rank(more, budget(), requested_context=32768)[:2]
        == (rank(asked, budget(), requested_context=32768)[:2])
    )


def test_an_unquantised_cache_and_a_larger_micro_batch_win_their_ties() -> None:
    plain = settings_for("gpu", 32768)
    quantised = settings_for("gpu", 32768, kv_type="q8_0")
    smaller = plain.with_micro_batch(512)
    assert rank(plain, budget(), requested_context=32768) > rank(
        quantised, budget(), requested_context=32768
    )
    assert rank(plain, budget(), requested_context=32768) > rank(
        smaller, budget(), requested_context=32768
    )


def test_fewer_experts_in_system_memory_wins() -> None:
    base = initial_settings(
        "moe-offload", context=32768, micro_batch=2048, kv_type="f16", projector_pool=None
    )
    assert rank(base.with_layer_choice(10), budget(), requested_context=32768) > rank(
        base.with_layer_choice(40), budget(), requested_context=32768
    )


def test_the_layer_choice_lands_in_the_field_its_mode_uses() -> None:
    moe = initial_settings(
        "moe-offload", context=1024, micro_batch=512, kv_type="f16", projector_pool=None
    ).with_layer_choice(12)
    hybrid = initial_settings(
        "hybrid", context=1024, micro_batch=512, kv_type="f16", projector_pool=None
    ).with_layer_choice(12)
    gpu = initial_settings(
        "gpu", context=1024, micro_batch=512, kv_type="f16", projector_pool=None
    ).with_layer_choice(12)
    assert (moe.gpu_layers, moe.cpu_moe_layers) == (ALL_GPU_LAYERS, 12)
    assert (hybrid.gpu_layers, hybrid.cpu_moe_layers) == (12, None)
    assert (gpu.gpu_layers, gpu.cpu_moe_layers) == (ALL_GPU_LAYERS, None)


def test_changing_the_micro_batch_keeps_the_batch_consistent() -> None:
    settings = initial_settings(
        "gpu", context=1024, micro_batch=2048, kv_type="f16", projector_pool=None
    )
    assert (settings.micro_batch, settings.batch) == (2048, 4096)
    smaller = settings.with_micro_batch(512)
    assert (smaller.micro_batch, smaller.batch) == (512, 2048)


def test_a_settings_value_becomes_the_placement_everything_else_reads() -> None:
    settings = initial_settings(
        "gpu", context=8192, micro_batch=1024, kv_type="f16", projector_pool="vram"
    ).with_context(4096)
    placement = settings.to_placement(budget(), threads=8, max_context_fit=4096, notes=("a note",))
    assert placement.mode == "gpu"
    assert placement.context == 4096
    assert placement.threads == 8
    assert placement.projector_pool == "vram"
    assert placement.notes == ("a note",)
    assert placement.budget.verdict == "fits"


def test_the_projector_can_be_moved() -> None:
    settings = initial_settings(
        "gpu", context=8192, micro_batch=1024, kv_type="f16", projector_pool="vram"
    )
    assert settings.with_projector("ram").projector_pool == "ram"
    assert settings.with_projector(None).projector_pool is None


def test_a_card_with_no_free_memory_is_not_a_card_to_place_anything_on() -> None:
    host = reference_host()
    full = host.gpus[0].model_copy(update={"vram_used_bytes": host.gpus[0].vram_total_bytes})
    host = host.model_copy(update={"gpus": [full]})
    model, quant = make_model(facts=make_facts())
    assert available_modes(model, quant, host) == ("cpu",)


def test_the_fixture_host_is_the_machine_the_measurements_came_from() -> None:
    host = reference_host()
    assert host.cpu.performance_cores == 8
    assert host.memory.total_bytes == 128 * GIB
    assert host.primary_gpu is not None
    assert host.primary_gpu.name.endswith("RTX 4060")


def test_only_the_offload_mode_asks_where_the_shared_experts_go() -> None:
    facts = make_facts(layers=48, experts=512, shared_expert_bytes=GIB // 4)
    assert shared_expert_ladder("moe-offload", facts) == (None, "ram")
    for mode in ("gpu", "hybrid", "cpu"):
        assert shared_expert_ladder(mode, facts) == (None,)  # type: ignore[arg-type]


def test_a_model_with_no_shared_experts_has_nothing_to_move() -> None:
    assert shared_expert_ladder("moe-offload", make_facts(layers=48, experts=512)) == (None,)
    assert shared_expert_ladder("moe-offload", None) == (None,)


def test_keeping_the_shared_experts_on_the_card_outranks_moving_them() -> None:
    base = initial_settings(
        "moe-offload", context=32768, micro_batch=1024, kv_type="f16", projector_pool=None
    )
    kept = rank(base, budget("fits"), requested_context=32768)
    moved = rank(base.with_shared_experts("ram"), budget("fits"), requested_context=32768)
    assert kept > moved, "they run for every token; moving them is a cost, never a preference"
