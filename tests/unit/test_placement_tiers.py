"""The context tier table and the largest context a placement holds (design section 9.3)."""

from __future__ import annotations

from llamafit.constants import CONTEXT_TIERS
from llamafit.models.plan import Verdict
from llamafit.placement.context_tiers import context_tiers, max_context_fit
from llamafit.placement.modes import PlacementSettings, initial_settings
from tests.fixtures.placement import (
    GIB,
    MIB,
    ScriptedBudget,
    SectionEightBudget,
    make_facts,
    make_model,
    reference_host,
)


def settings(context: int = 16384, micro_batch: int = 1024) -> PlacementSettings:
    return initial_settings(
        "gpu", context=context, micro_batch=micro_batch, kv_type="f16", projector_pool=None
    )


def test_every_rung_the_model_supports_is_costed() -> None:
    model, quant = make_model(native=262144, facts=make_facts())
    tiers = context_tiers(
        model, quant, reference_host(), settings(), budget_for=SectionEightBudget()
    )
    assert tuple(tier.tokens for tier in tiers) == CONTEXT_TIERS


def test_a_context_the_model_cannot_reach_is_not_offered_at_all() -> None:
    """A launch script compares free memory against a rung, not the model's data sheet.

    A rung the model would refuse has to be absent rather than merely marked, because the
    script never sees the mark: it sees a memory figure it can satisfy.
    """
    model, quant = make_model(native=40960, facts=make_facts())
    tiers = context_tiers(
        model, quant, reference_host(), settings(), budget_for=SectionEightBudget()
    )
    assert tuple(tier.tokens for tier in tiers) == (16384, 24576, 32768, 40960)


def test_the_memory_a_rung_needs_grows_with_the_context() -> None:
    model, quant = make_model(facts=make_facts())
    tiers = context_tiers(
        model, quant, reference_host(), settings(), budget_for=SectionEightBudget()
    )
    required = [tier.vram_required for tier in tiers]
    assert required == sorted(required)
    assert all(tier.vram_required > 0 for tier in tiers)


def test_fits_describes_the_machine_as_it_was_scanned() -> None:
    model, quant = make_model(facts=make_facts(dense_bytes=2 * GIB, kv_per_token=16 * 1024))
    tiers = context_tiers(
        model,
        quant,
        reference_host(),
        settings(),
        budget_for=SectionEightBudget(vram_available_override=6 * GIB),
    )
    fitting = [tier.tokens for tier in tiers if tier.fits]
    assert fitting
    assert max(fitting) < CONTEXT_TIERS[-1]
    # Once a rung stops fitting, nothing above it starts fitting again.
    seen_failure = False
    for tier in tiers:
        seen_failure = seen_failure or not tier.fits
        assert not (seen_failure and tier.fits)


def test_the_largest_context_is_found_by_bisection_on_a_thousand_token_grid() -> None:
    model, quant = make_model(native=262144, facts=make_facts())

    def below_a_hundred_thousand(configuration: PlacementSettings) -> Verdict:
        return "fits" if configuration.context <= 100_000 else "too-tight"

    found = max_context_fit(
        model,
        quant,
        reference_host(),
        settings(16384),
        budget_for=ScriptedBudget(below_a_hundred_thousand),
    )
    assert found == 16384 + 81 * 1024  # 99,328: the last rung of the grid that still fits
    assert found <= 100_000


def test_the_models_own_limit_is_tried_on_its_own_account() -> None:
    """The grid rarely lands on the native context, and that is the number users name."""
    model, quant = make_model(native=40000, facts=make_facts())
    found = max_context_fit(
        model,
        quant,
        reference_host(),
        settings(16384),
        budget_for=ScriptedBudget(lambda _s: "fits"),
    )
    assert found == 40000


def test_a_placement_already_at_the_models_limit_is_left_alone() -> None:
    model, quant = make_model(native=16384, facts=make_facts())
    scripted = ScriptedBudget(lambda _s: "fits")
    assert (
        max_context_fit(model, quant, reference_host(), settings(16384), budget_for=scripted)
        == 16384
    )
    assert scripted.calls == []


def test_nothing_above_the_chosen_context_fitting_leaves_it_where_it_is() -> None:
    model, quant = make_model(native=262144, facts=make_facts())
    found = max_context_fit(
        model,
        quant,
        reference_host(),
        settings(16384),
        budget_for=ScriptedBudget(
            lambda configuration: "fits" if configuration.context <= 16384 else "does-not-fit"
        ),
    )
    assert found == 16384


def test_the_tier_table_is_costed_for_the_placement_that_was_chosen() -> None:
    """Only the context varies: a rung has to be the plan with one thing changed."""
    model, quant = make_model(facts=make_facts())
    scripted = ScriptedBudget(lambda _s: "fits")
    chosen = settings(32768, micro_batch=512).with_projector(None)
    context_tiers(model, quant, reference_host(), chosen, budget_for=scripted)
    assert {call.micro_batch for call in scripted.calls} == {512}
    assert {call.kv_type for call in scripted.calls} == {"f16"}
    assert {call.mode for call in scripted.calls} == {"gpu"}


def test_the_reference_machine_reproduces_its_recorded_ladder_shape() -> None:
    """Flash-Next's recorded rungs: 49,152 needs 7.4 GB free and 65,536 needs 8.2 GB.

    The fake budget is not the real one and the figures are not expected to match to the
    megabyte -- it does not model the second KV cache the hybrid architecture carries, and
    runs some 400 MiB light because of it. What has to hold is the shape, which is what a
    launch script acts on: about a gigabyte between those two rungs, and the boundary
    between what an 8 GB card holds and what it pages falling in the same place the
    calibration record puts it.
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
    chosen = initial_settings(
        "moe-offload", context=32768, micro_batch=1024, kv_type="f16", projector_pool=None
    ).with_layer_choice(48)
    tiers = {
        tier.tokens: tier
        for tier in context_tiers(
            model, quant, reference_host(), chosen, budget_for=SectionEightBudget()
        )
    }
    step = tiers[65536].vram_required - tiers[32768].vram_required
    assert 700 * MIB < step < 1300 * MIB
    assert [tokens for tokens, tier in tiers.items() if tier.fits][-1] == 49152
    assert not tiers[65536].fits
