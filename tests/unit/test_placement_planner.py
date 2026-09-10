"""The placement search (design section 9.2 and 9.4).

The search order is the thing worth testing here. Getting it wrong does not produce an
error: it produces a configuration that runs, which is why it would never be reported.
"""

from __future__ import annotations

import pytest

from llamafit.constants import ALL_GPU_LAYERS, MIN_CONTEXT_TOKENS
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.host import Host
from llamafit.models.plan import Needs, Verdict
from llamafit.placement.modes import (
    BudgetFn,
    PlacementSettings,
    available_modes,
    context_ladder,
    has_margin,
    initial_settings,
    is_acceptable,
    kv_ladder,
    layer_count,
    layer_ladder,
    projector_ladder,
    rank,
)
from llamafit.placement.planner import plan_placement
from tests.fixtures.placement import (
    GIB,
    MIB,
    ScriptedBudget,
    SectionEightBudget,
    cpu_only_host,
    make_facts,
    make_model,
    reference_host,
)

MICRO_BATCHES = (2048, 1024, 512, 256)


def moe_model(layers: int = 48) -> tuple[CatalogModel, Quant]:
    return make_model(
        moe=True,
        facts=make_facts(layers=layers, experts=128, dense_bytes=4 * GIB, expert_bytes=40 * GIB),
    )


def dense_model(layers: int = 32) -> tuple[CatalogModel, Quant]:
    return make_model(facts=make_facts(layers=layers, dense_bytes=6 * GIB))


def everything(_settings: PlacementSettings) -> Verdict:
    return "comfortable"


def nothing(_settings: PlacementSettings) -> Verdict:
    return "does-not-fit"


def test_the_best_case_is_everything_on_the_card_at_the_requested_context() -> None:
    model, quant = moe_model()
    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(everything),
        needs=Needs(requested_context=65536),
    )
    assert placement.mode == "gpu"
    assert placement.context == 65536
    assert placement.micro_batch == 2048
    assert placement.batch == 4096
    assert placement.kv_type == "f16"
    assert placement.gpu_layers == ALL_GPU_LAYERS
    assert placement.cpu_moe_layers is None
    assert placement.threads == 8


def test_a_mode_that_works_stops_the_search_before_the_slower_ones() -> None:
    model, quant = moe_model()
    scripted = ScriptedBudget(everything)
    plan_placement(model, quant, reference_host(), budget_for=scripted, needs=Needs())
    assert {call.mode for call in scripted.calls} == {"gpu"}


def test_the_search_keeps_looking_after_its_first_acceptable_answer() -> None:
    """The whole point of section 9.2: MoE-offload starts at its worst configuration.

    Every routed expert in system memory is the first thing tried and it fits, so a
    planner that stopped there would recommend it. The best configuration is the one with
    the *fewest* experts in system memory that VRAM still allows, and it is found by
    continuing past that first acceptable answer.
    """
    model, quant = moe_model(layers=48)

    def only_with_most_experts_off_the_card(settings: PlacementSettings) -> Verdict:
        if settings.mode != "moe-offload":
            return "does-not-fit"
        return "fits" if (settings.cpu_moe_layers or 0) >= 30 else "too-tight"

    scripted = ScriptedBudget(only_with_most_experts_off_the_card)
    placement = plan_placement(
        model, quant, reference_host(), budget_for=scripted, needs=Needs(requested_context=32768)
    )
    assert placement.mode == "moe-offload"
    assert placement.cpu_moe_layers == 30
    assert placement.gpu_layers == ALL_GPU_LAYERS
    # It stopped where VRAM ran out rather than walking the whole ladder to one.
    assert min(call.cpu_moe_layers or 99 for call in scripted.calls) == 29


def test_moving_experts_back_to_the_card_is_how_a_system_memory_shortage_is_fixed() -> None:
    model, quant = moe_model(layers=48)

    def needs_experts_on_the_card(settings: PlacementSettings) -> Verdict:
        """Too many experts in system memory exhausts it; too few exhaust the card."""
        if settings.mode != "moe-offload":
            return "does-not-fit"
        on_the_processor = settings.cpu_moe_layers or 0
        if on_the_processor > 20:
            return "does-not-fit"
        return "fits" if on_the_processor >= 15 else "too-tight"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(needs_experts_on_the_card),
        needs=Needs(requested_context=32768),
    )
    # The first rung, every expert in system memory, does not fit at all; the ladder is
    # walked down past it and stops where the card runs out.
    assert placement.cpu_moe_layers == 15


def test_hybrid_takes_the_largest_number_of_layers_that_fits() -> None:
    model, quant = dense_model(layers=32)

    def only_hybrid_below_twenty(settings: PlacementSettings) -> Verdict:
        if settings.mode != "hybrid":
            return "does-not-fit"
        return "fits" if settings.gpu_layers <= 20 else "too-tight"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(only_hybrid_below_twenty),
        needs=Needs(requested_context=32768),
    )
    assert placement.mode == "hybrid"
    assert placement.gpu_layers == 20


def test_a_slower_mode_never_wins_on_context_alone() -> None:
    """The regression that made this ordering explicit.

    Everything the processor does fits, because system memory is enormous; the card only
    reaches a quarter of the requested context. Ranking context first recommended a
    125-billion-parameter model on the processor at about a token a second.
    """
    model, quant = moe_model()

    def card_is_short_and_the_processor_is_not(settings: PlacementSettings) -> Verdict:
        if settings.mode == "cpu":
            return "comfortable"
        if settings.mode == "gpu" and settings.context <= 65536:
            return "fits"
        return "does-not-fit"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(card_is_short_and_the_processor_is_not),
        needs=Needs(requested_context=262144),
    )
    assert placement.mode == "gpu"
    assert placement.context == 65536


def test_the_context_is_halved_before_a_mode_is_given_up() -> None:
    model, quant = moe_model()

    def only_short(settings: PlacementSettings) -> Verdict:
        return "fits" if settings.mode == "gpu" and settings.context == 16384 else "does-not-fit"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(only_short),
        needs=Needs(requested_context=131072),
    )
    assert (placement.mode, placement.context) == ("gpu", 16384)
    assert any("16,384" in note for note in placement.notes)


def test_the_cache_is_quantised_only_when_it_buys_something() -> None:
    model, quant = make_model(
        kv_types=["f16", "q8_0"], facts=make_facts(layers=32, dense_bytes=6 * GIB)
    )

    def only_quantised(settings: PlacementSettings) -> Verdict:
        return "fits" if settings.kv_type == "q8_0" else "does-not-fit"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(only_quantised),
        needs=Needs(requested_context=32768),
    )
    assert placement.kv_type == "q8_0"
    assert any("q8_0" in note for note in placement.notes)


def test_a_user_who_refuses_a_quantised_cache_is_not_given_one() -> None:
    model, quant = make_model(
        kv_types=["f16", "q8_0"], facts=make_facts(layers=32, dense_bytes=6 * GIB)
    )

    def only_quantised(settings: PlacementSettings) -> Verdict:
        return "fits" if settings.kv_type == "q8_0" else "does-not-fit"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(only_quantised),
        needs=Needs(requested_context=32768, allow_kv_quant=False),
    )
    assert placement.mode == "unsupported"


def test_the_micro_batch_drops_when_the_compute_buffer_breaks_the_fit() -> None:
    model, quant = dense_model()

    def only_small_micro_batches(settings: PlacementSettings) -> Verdict:
        return "fits" if settings.micro_batch <= 1024 else "too-tight"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(only_small_micro_batches),
        needs=Needs(requested_context=32768),
    )
    assert placement.micro_batch == 1024
    assert placement.batch == 2048


def test_the_projector_takes_the_card_only_with_margin() -> None:
    model, quant = make_model(projector=True, facts=make_facts(layers=32))

    def tight_on_the_card(settings: PlacementSettings) -> Verdict:
        if settings.projector_pool == "vram":
            return "tight"
        return "comfortable"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(tight_on_the_card),
        needs=Needs(requested_context=32768),
    )
    assert placement.projector_pool == "ram"
    assert any("--no-mmproj-offload" in note for note in placement.notes)


def test_the_projector_keeps_the_card_when_there_is_room() -> None:
    model, quant = make_model(projector=True, facts=make_facts(layers=32))
    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(everything),
        needs=Needs(requested_context=32768),
    )
    assert placement.projector_pool == "vram"


def test_vision_is_turned_off_and_said_out_loud_when_nothing_else_works() -> None:
    model, quant = make_model(projector=True, facts=make_facts(layers=32))

    def no_projector_anywhere(settings: PlacementSettings) -> Verdict:
        return "does-not-fit" if settings.projector_pool is not None else "fits"

    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(no_projector_anywhere),
        needs=Needs(requested_context=32768),
    )
    assert placement.projector_pool is None
    assert any("Vision is off" in note for note in placement.notes)


def test_a_user_who_does_not_want_vision_is_never_offered_it() -> None:
    model, quant = make_model(projector=True, facts=make_facts(layers=32))
    scripted = ScriptedBudget(everything)
    placement = plan_placement(
        model, quant, reference_host(), budget_for=scripted, vision=False, needs=Needs()
    )
    assert placement.projector_pool is None
    assert all(call.projector_pool is None for call in scripted.calls)


def test_nothing_fitting_is_reported_with_the_budget_of_the_smallest_thing_tried() -> None:
    model, quant = dense_model()
    placement = plan_placement(
        model, quant, reference_host(), budget_for=ScriptedBudget(nothing), needs=Needs()
    )
    assert placement.mode == "unsupported"
    assert placement.context == MIN_CONTEXT_TOKENS
    assert placement.max_context_fit == 0
    assert placement.tiers == ()
    assert placement.budget.vram_required > 0
    assert placement.notes == ("No configuration of this model fits this machine.",)


def test_a_model_shorter_than_the_minimum_asked_for_is_reported_as_such() -> None:
    model, quant = make_model(native=32768, facts=make_facts())
    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(everything),
        needs=Needs(min_context=131072),
    )
    assert placement.mode == "unsupported"
    assert "32,768" in placement.notes[0]
    assert "131,072" in placement.notes[0]


def test_a_minimum_above_the_request_is_still_a_minimum() -> None:
    model, quant = dense_model()
    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=ScriptedBudget(everything),
        needs=Needs(requested_context=16384, min_context=65536),
    )
    assert placement.context == 65536


def test_a_machine_with_no_card_gets_a_processor_placement_and_is_told_so() -> None:
    model, quant = dense_model()
    placement = plan_placement(
        model, quant, cpu_only_host(), budget_for=ScriptedBudget(everything), needs=Needs()
    )
    assert placement.mode == "cpu"
    assert placement.gpu_layers == 0
    assert any("graphics card" in note for note in placement.notes)


def test_a_hybrid_placement_says_how_many_layers_are_on_the_card() -> None:
    model, quant = dense_model(layers=32)

    def only_hybrid(settings: PlacementSettings) -> Verdict:
        return "fits" if settings.mode == "hybrid" else "does-not-fit"

    placement = plan_placement(
        model, quant, reference_host(), budget_for=ScriptedBudget(only_hybrid), needs=Needs()
    )
    assert any("31 layers" in note for note in placement.notes)


def test_the_thread_count_explains_itself_on_a_hybrid_processor() -> None:
    model, quant = dense_model()
    placement = plan_placement(
        model, quant, reference_host(), budget_for=ScriptedBudget(everything), needs=Needs()
    )
    assert any("efficiency cores" in note for note in placement.notes)


def test_a_tight_verdict_says_what_would_push_it_over() -> None:
    model, quant = dense_model()

    def barely(_settings: PlacementSettings) -> Verdict:
        return "tight"

    placement = plan_placement(
        model, quant, reference_host(), budget_for=ScriptedBudget(barely), needs=Needs()
    )
    assert any("Tight" in note for note in placement.notes)


def test_a_prose_quirk_reaches_the_reader_as_a_note() -> None:
    model, quant = make_model(
        facts=make_facts(),
        quirks=["--lazy-mode on streams the n-gram table from disk", "--no-context-shift"],
    )
    placement = plan_placement(
        model, quant, reference_host(), budget_for=ScriptedBudget(everything), needs=Needs()
    )
    assert "--lazy-mode on streams the n-gram table from disk" in placement.notes
    assert "--no-context-shift" not in placement.notes


def test_the_planner_reports_a_context_ladder_and_the_largest_context_it_holds() -> None:
    model, quant = dense_model()
    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=SectionEightBudget(),
        needs=Needs(requested_context=16384),
    )
    assert placement.tiers
    assert placement.max_context_fit >= placement.context


def exhaustive_best(
    model: CatalogModel, quant: Quant, host: Host, budget_for: BudgetFn, needs: Needs
) -> tuple[int, ...] | None:
    """The same search with no pruning at all, for the pruned one to be held against."""
    requested = min(max(needs.requested_context or 32768, needs.min_context), model.context.native)
    best: tuple[int, ...] | None = None
    for mode in available_modes(model, quant, host):
        for context in context_ladder(requested, needs.min_context):
            for kv_type in kv_ladder(model, allow_kv_quant=needs.allow_kv_quant):
                for micro_batch in MICRO_BATCHES:
                    for projector in projector_ladder(model, quant, vision=True):
                        base = initial_settings(
                            mode,
                            context=context,
                            micro_batch=micro_batch,
                            kv_type=kv_type,
                            projector_pool=projector,
                        )
                        gate = has_margin if projector == "vram" else is_acceptable
                        for value in layer_ladder(mode, layer_count(quant.gguf_facts)):
                            settings = base.with_layer_choice(value)
                            budget = budget_for(model, quant, host, settings)
                            if not gate(budget.verdict):
                                continue
                            key = rank(settings, budget, requested_context=requested)
                            if best is None or key > best:
                                best = key
    return best


@pytest.mark.parametrize("free_vram", [2 * GIB, 5 * GIB, 7 * GIB, 12 * GIB, 24 * GIB])
@pytest.mark.parametrize("requested", [16384, 65536, 262144])
def test_the_pruned_search_finds_what_an_exhaustive_one_would(
    free_vram: int, requested: int
) -> None:
    """Every prune in the planner claims a placement it skipped could not have won.

    The claim is checked rather than argued: the same space is walked with no pruning at
    all and the two are compared by the rank itself, across enough machines and requests
    that each mode and each ladder is the deciding one somewhere.
    """
    model, quant = make_model(
        moe=True,
        projector=True,
        kv_types=["f16", "q8_0"],
        facts=make_facts(layers=6, experts=64, dense_bytes=3 * GIB, expert_bytes=20 * GIB),
    )
    host = reference_host()
    needs = Needs(requested_context=requested)
    pruned = plan_placement(
        model,
        quant,
        host,
        budget_for=SectionEightBudget(vram_available_override=free_vram),
        needs=needs,
    )
    expected = exhaustive_best(
        model, quant, host, SectionEightBudget(vram_available_override=free_vram), needs
    )
    if expected is None:
        assert pruned.mode == "unsupported"
        return
    settings = PlacementSettings(
        mode=pruned.mode,
        context=pruned.context,
        micro_batch=pruned.micro_batch,
        batch=pruned.batch,
        kv_type=pruned.kv_type,
        gpu_layers=pruned.gpu_layers,
        cpu_moe_layers=pruned.cpu_moe_layers,
        projector_pool=pruned.projector_pool,
    )
    assert rank(settings, pruned.budget, requested_context=min(requested, 262144)) == expected


def test_the_reference_machine_gets_its_experts_into_system_memory() -> None:
    """Qwen3.8-Flash-Next on the machine it was measured on: 125B on an 8 GB card.

    The recorded winning configuration holds attention and the KV cache on the card and
    the routed experts in system memory, which is what the planner has to arrive at.
    """
    model, quant = make_model(
        moe=True,
        kv_types=["f16"],
        facts=make_facts(
            layers=48,
            experts=512,
            dense_bytes=4147719680,
            expert_bytes=77017907200,
            lazy_bytes=28800138240,
            kv_per_token=24576,
            recurrent_bytes=117669888,
        ),
    )
    placement = plan_placement(
        model,
        quant,
        reference_host(),
        budget_for=SectionEightBudget(),
        needs=Needs(requested_context=262144),
    )
    assert placement.mode == "moe-offload"
    assert placement.cpu_moe_layers == 48
    assert placement.kv_type == "f16"
    assert placement.threads == 8
    assert 16384 <= placement.context <= 65536
    assert placement.budget.ram_required > 60 * GIB
    assert placement.budget.vram_required < 8188 * MIB
