"""``plan_report``: the whole answer ``llamafit plan`` prints, on the recorded machine.

The reference machine's own winning configuration is recorded in the catalog, so this is
where the planner is held against a human's hand-tuned flags rather than against another
piece of this program.
"""

from __future__ import annotations

from llamafit.errors import LlamaFitError
from llamafit.models.catalog import Quant
from llamafit.models.host import Host
from llamafit.models.plan import Needs
from llamafit.placement.modes import batch_for, is_acceptable
from llamafit.services.plan import (
    Counterfactuals,
    QuantSwap,
    budget_for,
    counterfactuals,
    launch_paths,
    next_quant_down,
    plan_model,
    plan_report,
    replan,
    settings_of,
)
from llamafit.services.recommend import quant_entries
from tests.fixtures.board import catalog, model_and_quant
from tests.fixtures.budget_hosts import card_with_free, machine, reference_host

GIB = 1024**3
MIB = 1024**2


def test_the_plan_reproduces_the_flags_the_reference_machine_was_measured_with() -> None:
    """The recorded winner is `--n-cpu-moe 48 -ot ffn_.*_shexp=CPU -ub 1024 -t 16`."""
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host())

    assert report.placement.mode == "moe-offload"
    assert report.placement.cpu_moe_layers == 48
    assert report.placement.shared_experts_pool == "ram"
    assert report.placement.micro_batch == 1024
    assert report.placement.threads == 16
    args = report.command
    assert args[0] == "llama-server"
    assert "ffn_.*_shexp=CPU" in args
    assert args[args.index("-ub") + 1] == "1024"
    assert args[args.index("-t") + 1] == "16"
    assert args[args.index("-ngl") + 1] == "99"


def test_the_catalogs_recorded_runs_travel_with_the_plan_but_never_become_the_estimate() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host())
    assert [run.profile for run in report.measurements]
    assert report.speed is not None
    assert report.speed.confidence == "estimated"


def test_a_context_that_does_not_fit_is_costed_and_shown_rather_than_only_retreated_from() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(
        model, quant, reference_host(), needs=Needs(use_case="coding", requested_context=262144)
    )
    assert report.requested_context == 262144
    assert report.placement.context < 262144
    assert report.requested_budget is not None
    # Section 8.4: over the card is not a refusal, it is a silent collapse in speed.
    assert report.requested_budget.verdict in ("too-tight", "does-not-fit")


def test_a_context_that_fits_leaves_no_second_budget_to_explain() -> None:
    model, quant = model_and_quant("qwen3-0.6b")
    report = plan_report(
        model, quant, reference_host(), needs=Needs(use_case="coding", requested_context=16384)
    )
    assert report.placement.context == 16384
    assert report.requested_budget is None


def test_a_hand_set_micro_batch_rebuilds_the_budget_rather_than_relabelling_it() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    host = reference_host()
    planned = plan_report(model, quant, host)
    forced = plan_report(model, quant, host, micro_batch=512)

    assert forced.placement.micro_batch == 512
    assert forced.placement.batch == batch_for(512)
    assert forced.flags[forced.flags.index("-ub") + 1] == "512"
    # A smaller micro-batch is a smaller compute buffer, so the card total has to move.
    assert forced.placement.budget.vram_required != planned.placement.budget.vram_required


def test_every_tier_of_the_ladder_carries_its_own_verdict() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host())
    tiers = report.placement.tiers
    assert tiers
    assert all(tier.fits == (tier.verdict in ("comfortable", "fits", "tight")) for tier in tiers)
    # The ladder only ever gets more expensive, so its verdicts may not improve going up.
    order = ["comfortable", "fits", "tight", "too-tight", "does-not-fit"]
    ranks = [order.index(tier.verdict) for tier in tiers]
    assert ranks == sorted(ranks)


def test_a_target_that_is_met_says_so() -> None:
    model, quant = model_and_quant("qwen3-0.6b")
    report = plan_report(model, quant, reference_host(), target_tps=5.0)
    assert report.target is not None
    assert report.target.reached


def test_a_target_nothing_reaches_still_reports_the_fastest_figure_found() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    report = plan_report(model, quant, reference_host(), target_tps=500.0)
    assert report.target is not None
    assert not report.target.reached
    assert report.target.best_context is None
    assert report.target.best_tps > 0


def test_the_command_line_names_a_local_file_when_the_machine_has_one() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    local = f"/models/{quant.files[0].rsplit('/', 1)[-1]}"
    report = plan_report(model, quant, reference_host(), local_files=[local])
    assert report.model_present
    assert report.model_path == local
    assert local in report.command


def test_the_command_line_names_the_download_directory_when_it_does_not() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    report = plan_report(model, quant, reference_host(), local_files=[])
    assert not report.model_present
    assert report.model_path.endswith(".gguf")


def test_launch_paths_matches_a_sharded_quant_by_its_bare_file_name() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    bare = quant.files[0].rsplit("/", 1)[-1]
    path, present, projector = launch_paths(model, quant, None, [f"/elsewhere/{bare}"])
    assert present
    assert path == f"/elsewhere/{bare}"
    assert projector is None


def test_settings_round_trip_through_the_placement_they_produced() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    host = reference_host()
    report = plan_report(model, quant, host)
    again = replan(
        model, quant, host, settings_of(report.placement), requested_context=32768, vision=True
    )
    assert again.budget.vram_required == report.placement.budget.vram_required
    assert again.mode == report.placement.mode


def test_a_machine_with_no_room_still_produces_a_plan_that_says_why() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    tiny = machine(vram_total=None, ram_total=4 * GIB, ram_available=2 * GIB)
    report = plan_report(model, quant, tiny)
    assert report.placement.mode == "unsupported"
    assert report.placement.notes
    assert report.requested_budget is None


def test_a_roomier_card_keeps_the_shared_experts_where_they_are() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    roomy = card_with_free(24 * GIB, ram_available=160 * GIB)
    report = plan_report(model, quant, roomy)
    assert report.placement.shared_experts_pool is None
    assert "-ot" not in report.command


# --- counterfactuals: what would move a candidate (section 12.3) -----------------------


def _counterfactuals(model_id: str, host: Host, **kwargs: object) -> Counterfactuals:
    """Plan one bundled model on one machine and ask what would change it."""
    model, quant = model_and_quant(model_id)
    quants = quant_entries(model)
    quant = next(q for q in quants if q.name == quant.name)
    placement = plan_model(model, quant, host)
    return counterfactuals(model, quant, host, placement, quants=quants, **kwargs)  # type: ignore[arg-type]


def test_the_context_rung_offered_is_one_the_machine_cannot_take_yet() -> None:
    found = _counterfactuals("qwen3-coder-next", reference_host())
    assert found.context is not None
    assert found.context.tokens > 0
    assert found.context.verdict in ("too-tight", "does-not-fit")


def test_the_card_figure_is_what_the_rung_costs_and_not_a_scaling_of_the_one_below() -> None:
    """Every figure in a counterfactual comes from a budget computed for that rung."""
    model, quant = model_and_quant("qwen3-coder-next")
    host = reference_host()
    placement = plan_model(model, quant, host)
    found = counterfactuals(model, quant, host, placement, quants=quant_entries(model))
    assert found.context is not None
    rung = next(t for t in placement.tiers if t.tokens == found.context.tokens)
    assert found.context.vram_required == rung.vram_required
    assert found.context.vram_available == placement.budget.vram_available


def test_freeing_what_the_rung_asks_for_is_enough_to_reach_it() -> None:
    """The promise is checkable, so this checks it: free that much and plan again."""
    model, quant = model_and_quant("qwen3-coder-next")
    host = reference_host()
    placement = plan_model(model, quant, host)
    found = counterfactuals(model, quant, host, placement, quants=quant_entries(model))
    assert found.context is not None and found.context.verdict == "too-tight"
    assert found.context.vram_to_free > 0

    # Give the card exactly what the sentence asks for, and cost that rung again. Nothing
    # else moves: the same card by name, so it keeps its measured backend overhead, the
    # same settings, the same context and the same system memory.
    card = host.gpus[0]
    roomier = host.model_copy(
        update={
            "gpus": [
                card.model_copy(
                    update={
                        "vram_total_bytes": (card.vram_total_bytes or 0)
                        + found.context.vram_to_free
                    }
                )
            ]
        }
    )
    again = budget_for(
        model, quant, roomier, settings_of(placement).with_context(found.context.tokens)
    )
    assert is_acceptable(again.verdict), again.verdict


def test_a_rung_nothing_could_absorb_is_not_offered_as_a_card_problem() -> None:
    """Free 1.2 GB is false when the overflow has nowhere to go, so the figure is zero."""
    host = machine(vram_total=6 * GIB, ram_total=10 * GIB, ram_available=8 * GIB)
    found = _counterfactuals("granite-4.2-8b", host)
    assert found.context is not None
    assert found.context.verdict == "does-not-fit"
    assert found.context.vram_to_free == 0


def test_a_figure_to_free_is_offered_exactly_where_freeing_the_card_is_the_answer() -> None:
    """The invariant behind both sentences, held over every entry the catalog ships."""
    host = machine(vram_total=6 * GIB, ram_total=10 * GIB, ram_available=8 * GIB)
    seen = set()
    for model in catalog().models:
        for quant in quant_entries(model):
            try:
                placement = plan_model(model, quant, host)
            except LlamaFitError:
                continue
            if placement.mode == "unsupported":
                continue
            found = counterfactuals(model, quant, host, placement, quants=[quant])
            if found.context is None:
                continue
            seen.add(found.context.verdict)
            assert (found.context.vram_to_free > 0) == (found.context.verdict == "too-tight")
    assert {"too-tight", "does-not-fit"} <= seen, "this test needs both failures to occur"


def test_a_machine_that_holds_every_rung_says_so_rather_than_saying_nothing() -> None:
    host = machine(vram_total=80 * GIB, ram_total=256 * GIB, ram_available=200 * GIB)
    found = _counterfactuals("qwen3.5-4b", host)
    assert found.context is None
    assert found.every_context_fits


def test_a_candidate_that_fits_nowhere_is_never_told_every_context_fits() -> None:
    """An empty ladder is not a ladder every rung of which fits."""
    tiny = machine(vram_total=None, ram_total=2 * GIB, ram_available=1 * GIB)
    model, quant = model_and_quant("gpt-oss-120b")
    placement = plan_model(model, quant, tiny)
    assert placement.mode == "unsupported" and placement.tiers == ()
    found = counterfactuals(model, quant, tiny, placement, quants=[quant])
    assert found.context is None
    assert not found.every_context_fits


def test_the_quantisation_offered_is_the_next_one_down_and_was_actually_planned() -> None:
    found = _counterfactuals("gpt-oss-120b", reference_host())
    assert found.quant is not None
    assert found.quant.mode is not None, "an offer with no placement behind it is a guess"
    assert found.quant.verdict is not None
    assert found.quant.max_context_fit > 0


def test_a_model_publishing_nothing_smaller_is_offered_no_quantisation_at_all() -> None:
    """The rule the whole feature turns on: no alternative, no sentence."""
    found = _counterfactuals("qwen3-coder-next", reference_host())
    assert found.quant is None


def test_next_quant_down_takes_one_step_and_takes_it_downwards() -> None:
    model, _quant = model_and_quant("gpt-oss-120b")
    quants = quant_entries(model)
    by_name = {q.name: q for q in quants}
    largest = max(quants, key=lambda q: q.bytes_ or 0)
    step = next_quant_down(quants, largest)
    assert step is not None
    assert (step.bytes_ or 0) < (largest.bytes_ or 0)
    # One step, not the smallest there is: the smallest is still below the one chosen.
    smallest = min(quants, key=lambda q: q.bytes_ or 0)
    if smallest.name != step.name:
        assert (smallest.bytes_ or 0) < (step.bytes_ or 0)
    assert next_quant_down(quants, smallest) is None
    assert by_name[step.name] is step


def test_two_quantisations_with_no_size_between_them_are_not_compared() -> None:
    """Bytes against bits per weight is not an ordering, and a guess here is a false promise."""
    sized = Quant(name="Q4_K_M", files=["a.gguf"], bytes=8_000_000_000, bpw=None)
    weighed = Quant(name="Q3_K_M", files=["b.gguf"], bytes=None, bpw=3.4)
    assert next_quant_down([sized, weighed], sized) is None


def test_two_quantisations_with_no_sizes_are_ordered_by_their_widths() -> None:
    """A catalog nobody has refreshed still knows which of two quantisations is smaller."""
    wide = Quant(name="Q8_0", files=["a.gguf"], bytes=None, bpw=8.5)
    narrow = Quant(name="Q4_K_M", files=["b.gguf"], bytes=None, bpw=4.5)
    assert next_quant_down([wide, narrow], wide) is narrow
    assert next_quant_down([wide, narrow], narrow) is None


def test_an_alternative_nobody_could_size_carries_the_reason_rather_than_a_placement() -> None:
    """The step down is offered from a placement or not at all, and the reason travels."""
    model, quant = model_and_quant("gpt-oss-120b")
    host = reference_host()
    placement = plan_model(model, quant, host)
    unreadable = Quant(name="Q2_K", files=["nobody-has-read-this.gguf"], bytes=1, bpw=2.0)
    found = counterfactuals(model, quant, host, placement, quants=[quant, unreadable], gen_tps=10.0)
    assert found.quant is not None
    assert found.quant.quant == "Q2_K"
    assert found.quant.unplaceable_because
    assert found.quant.mode is None and found.quant.verdict is None
    assert not found.quant.improves


def test_a_faster_alternative_has_to_be_faster_by_more_than_the_estimator_claims() -> None:
    """A two per cent gain is the width of the error bar, not a reason to fetch a file."""
    model, quant = model_and_quant("gpt-oss-120b")
    host = reference_host()
    quants = quant_entries(model)
    placement = plan_model(model, quant, host)
    found = counterfactuals(
        model, quant, host, placement, quants=quants, gen_tps=0.001, working_context=8192
    )
    assert found.quant is not None and found.quant.gen_tps is not None
    assert found.quant.faster, "anything beats a thousandth of a token per second"

    unbeatable = counterfactuals(
        model, quant, host, placement, quants=quants, gen_tps=1_000.0, working_context=8192
    )
    assert unbeatable.quant is not None
    assert not unbeatable.quant.faster


def test_improves_is_the_three_comparisons_and_cannot_drift_from_them() -> None:
    swap = QuantSwap(quant="Q3_K_M")
    assert not swap.improves
    assert swap.model_copy(update={"more_context": True}).improves
    assert swap.model_copy(update={"faster": True}).improves
    assert swap.model_copy(update={"better_verdict": True}).improves
    assert "improves" in swap.model_dump()
