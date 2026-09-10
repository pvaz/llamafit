"""``plan_report``: the whole answer ``llamafit plan`` prints, on the recorded machine.

The reference machine's own winning configuration is recorded in the catalog, so this is
where the planner is held against a human's hand-tuned flags rather than against another
piece of this program.
"""

from __future__ import annotations

from llamafit.models.plan import Needs
from llamafit.placement.modes import batch_for
from llamafit.services.plan import launch_paths, plan_report, replan, settings_of
from tests.fixtures.board import model_and_quant
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
