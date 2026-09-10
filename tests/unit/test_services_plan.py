"""The join between section 8's budget and section 9's search, on real models.

Everywhere else the planner is exercised against a fake budget, which is the right way to
test a search. This is the other half: the real budget, the real catalog and the recorded
reference machine, held against the configuration that machine was actually measured
running. A search that is correct against a fake and wrong against the real thing would
pass every other test in the suite.
"""

from __future__ import annotations

from llamafit.catalog import load_catalog
from llamafit.models.catalog import CatalogModel, Quant
from llamafit.models.plan import Needs
from llamafit.placement import LaunchOptions, command_line
from llamafit.services.plan import budget_for, plan_model
from tests.fixtures.budget_hosts import first_quant, machine, reference_host

GIB = 1024**3
MIB = 1024**2


def model_and_quant(model_id: str) -> tuple[CatalogModel, Quant]:
    catalog, problems = load_catalog()
    assert problems == []
    model = catalog.by_id[model_id]
    return model, first_quant(model)


def test_the_planner_reaches_the_reference_machines_own_winning_configuration() -> None:
    """`-ub 1024 -ot ffn_.*_shexp=CPU` at 32,768: the flags the record calls the winner."""
    model, quant = model_and_quant("qwen3.8-flash-next")
    placement = plan_model(model, quant, reference_host())

    assert placement.mode == "moe-offload"
    assert placement.context == 32768
    assert placement.micro_batch == 1024
    assert placement.kv_type == "f16"
    assert placement.gpu_layers == 99
    assert placement.cpu_moe_layers == 48
    assert placement.shared_experts_pool == "ram"
    assert placement.threads == 8
    assert placement.budget.verdict == "tight"

    args = command_line(placement, model, LaunchOptions(model_path="M.gguf"))
    for flag in ("-ot", "ffn_.*_shexp=CPU", "--n-cpu-moe", "-ub"):
        assert flag in args
    assert args[args.index("-ub") + 1] == "1024"
    assert args[args.index("-c") + 1] == "32768"


def test_the_override_is_a_step_of_the_search_and_not_a_habit() -> None:
    """A model that fits without moving the shared experts keeps them on the card."""
    model, quant = model_and_quant("qwen3-coder-next")
    placement = plan_model(model, quant, reference_host())
    assert placement.mode == "moe-offload"
    assert placement.shared_experts_pool is None
    assert "-ot" not in command_line(placement, model, LaunchOptions(model_path="M.gguf"))


def test_the_note_says_where_the_shared_experts_went() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    notes = " ".join(plan_model(model, quant, reference_host()).notes)
    assert "shared experts" in notes and "ffn_.*_shexp=CPU" in notes


def test_a_roomier_card_does_not_need_the_override_at_all() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    roomy = machine(vram_total=24 * GIB, ram_total=192 * GIB, ram_available=160 * GIB)
    placement = plan_model(model, quant, roomy)
    assert placement.shared_experts_pool is None
    assert placement.budget.verdict in ("comfortable", "fits", "tight")


def test_the_adapter_costs_what_the_planner_chose() -> None:
    """The budget carried by a placement is the one its own settings produce."""
    model, quant = model_and_quant("qwen3.8-flash-next")
    host = reference_host()
    placement = plan_model(model, quant, host, needs=Needs(requested_context=16384))
    from llamafit.placement import PlacementSettings

    again = budget_for(
        model,
        quant,
        host,
        PlacementSettings(
            mode=placement.mode,
            context=placement.context,
            micro_batch=placement.micro_batch,
            batch=placement.batch,
            kv_type=placement.kv_type,
            gpu_layers=placement.gpu_layers,
            cpu_moe_layers=placement.cpu_moe_layers,
            projector_pool=placement.projector_pool,
            shared_experts_pool=placement.shared_experts_pool,
        ),
    )
    assert again.vram_required == placement.budget.vram_required
    assert again.verdict == placement.budget.verdict
