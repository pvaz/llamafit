"""The whole budget: what it adds up to, and what it says about whether that fits.

Two of these are proofs rather than assertions, because they are the two claims the rest of
LlamaFit rests on. A configuration one byte past what the card has free is reported as
paging, not as fitting -- the host in that test is built backwards from the budget so the
byte is real rather than arranged. And the byte figures for a real model add back up to the
size of the file they came from, for every placement, so no placement loses or invents
weights.
"""

import pytest

from llamafit.budget import compute
from llamafit.budget.budget import pool_verdict, utilisation
from llamafit.catalog import load_catalog
from llamafit.constants import GIB, MIB, VRAM_RESERVE_BYTES
from llamafit.errors import BudgetError
from llamafit.models.catalog import CatalogModel, Extra, Quant
from llamafit.models.plan import Budget
from tests.fixtures.budget_hosts import card_with_free, first_quant, machine, reference_host

WEIGHT_COMPONENTS = frozenset(
    {
        "dense-weights",
        "expert-weights",
        "token-embedding",
        "output-head",
        "global-weights",
        "lazy-tables",
    }
)


def model_and_quant(model_id: str) -> tuple[CatalogModel, Quant]:
    catalog, problems = load_catalog()
    assert problems == []
    model = catalog.by_id[model_id]
    return model, first_quant(model)


def weight_bytes(budget: Budget) -> int:
    return sum(line.bytes_ for line in budget.lines if line.component in WEIGHT_COMPONENTS)


@pytest.mark.parametrize(
    "model_id",
    [
        "qwen3-0.6b",
        "qwen3.8-flash-next",
        "qwen3-coder-next",
        "llama-3.1-8b-instruct",
        "gemma-3-27b-it",
    ],
)
def test_a_real_models_weight_lines_add_up_to_its_file_in_every_mode(model_id: str) -> None:
    model, quant = model_and_quant(model_id)
    assert quant.gguf_facts is not None
    host = machine(ram_available=200 * GIB)
    for mode, gpu_layers in (("gpu", None), ("moe-offload", None), ("cpu", None), ("hybrid", 12)):
        budget = compute(
            model,
            quant,
            host,
            context=8192,
            mode=mode,
            gpu_layers=gpu_layers,  # type: ignore[arg-type]
        )
        assert weight_bytes(budget) == quant.gguf_facts.bytes_total, mode


def test_one_byte_over_the_card_is_paging_and_not_a_fit() -> None:
    """Built backwards from the budget itself, so the byte is real and not arranged."""
    model, quant = model_and_quant("qwen3.8-flash-next")
    roomy = machine(ram_total=128 * GIB, ram_available=100 * GIB)
    probe = compute(model, quant, roomy, context=32768, mode="moe-offload", micro_batch=1024)
    needed = probe.vram_required

    exactly = compute(
        model,
        quant,
        card_with_free(needed + VRAM_RESERVE_BYTES),
        context=32768,
        mode="moe-offload",
        micro_batch=1024,
    )
    assert exactly.vram_required == needed
    assert exactly.vram_utilisation == 1.0

    over = compute(
        model,
        quant,
        card_with_free(needed + VRAM_RESERVE_BYTES - 1),
        context=32768,
        mode="moe-offload",
        micro_batch=1024,
    )
    assert over.vram_required == over.vram_available + 1
    assert over.verdict == "too-tight", "the driver pages; it does not refuse"
    assert over.verdict not in ("comfortable", "fits", "tight")


def test_a_card_that_cannot_page_anywhere_does_not_fit_at_all() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    host = machine(vram_total=2 * GIB, ram_total=4 * GIB, ram_available=1 * GIB)
    budget = compute(model, quant, host, context=131072, mode="gpu")
    assert budget.verdict == "does-not-fit"


def test_the_reference_machine_sizes_flash_next_at_32k() -> None:
    """The configuration the calibration record recommends, against what it measured."""
    model, quant = model_and_quant("qwen3.8-flash-next")
    budget = compute(
        model,
        quant,
        reference_host(),
        context=32768,
        mode="moe-offload",
        micro_batch=1024,
    )
    lines = {(line.component, line.pool): line for line in budget.lines}

    # llama-server printed 4,606 MiB of model, 113 of recurrent state and 1,337 of compute
    # buffer for exactly this configuration.
    on_card = sum(line.bytes_ for line in budget.lines if line.pool == "vram")
    model_buffer = sum(
        line.bytes_
        for (component, pool), line in lines.items()
        if pool == "vram" and component in WEIGHT_COMPONENTS
    )
    assert abs(model_buffer - 4606 * MIB) < MIB
    assert abs(lines[("recurrent-state", "vram")].bytes_ - 113 * MIB) < MIB
    assert abs(lines[("compute-buffer", "vram")].bytes_ - 1337 * MIB) <= MIB
    assert on_card == budget.vram_required

    assert budget.vram_available == (8188 - 550) * MIB - VRAM_RESERVE_BYTES
    assert budget.ram_available == 100 * GIB
    assert budget.verdict == "too-tight", "7.0 GB of a 7.4 GB budget, with the projector off"


def test_the_lines_come_in_the_order_a_reader_should_meet_them() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    budget = compute(model, quant, reference_host(), context=16384, mode="moe-offload")
    components = [line.component for line in budget.lines]
    assert components[0] == "dense-weights"
    assert components[-1] == "process-overhead"
    assert components.index("kv-cache") < components.index("compute-buffer")
    assert components.index("expert-weights") < components.index("kv-cache")


def test_a_budget_says_of_every_line_whether_it_was_measured_or_modelled() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    budget = compute(
        model,
        quant,
        reference_host(),
        context=16384,
        mode="moe-offload",
        projector=Extra(role="mmproj", file="mmproj-F16.gguf", bytes=862 * MIB),
        projector_pool="vram",
    )
    exact = {line.component for line in budget.lines if line.exact}
    modelled = {line.component for line in budget.lines if not line.exact}
    assert exact | {"lazy-tables"} >= WEIGHT_COMPONENTS
    assert "kv-cache" in exact
    assert "vision-projector" in exact
    assert modelled == {
        "recurrent-state",
        "compute-buffer",
        "output-buffer",
        "vision-projector-compute",
        "cuda-context",
        "process-overhead",
    }


def test_a_mixture_of_experts_placement_is_not_penalised_for_being_one() -> None:
    """Experts in memory with attention and the cache on the card can be comfortable."""
    model, quant = model_and_quant("qwen3-coder-next")
    host = machine(vram_total=24 * GIB, ram_total=128 * GIB, ram_available=100 * GIB)
    budget = compute(model, quant, host, context=32768, mode="moe-offload", micro_batch=1024)
    assert budget.verdict == "comfortable"


def test_a_processor_only_placement_never_rates_above_fits() -> None:
    model, quant = model_and_quant("qwen3-0.6b")
    host = machine(vram_total=None, ram_total=128 * GIB, ram_available=100 * GIB)
    on_cpu = compute(model, quant, host, context=4096, mode="cpu")
    assert on_cpu.ram_utilisation < 0.65
    assert on_cpu.verdict == "fits", "measured generation on a processor is rarely comfortable"


def test_a_machine_with_no_card_has_no_card_utilisation() -> None:
    model, quant = model_and_quant("qwen3-0.6b")
    budget = compute(model, quant, machine(vram_total=None), context=4096, mode="cpu")
    assert budget.vram_required == 0
    assert budget.vram_utilisation is None


def test_unified_memory_is_one_pool_with_two_names() -> None:
    model, quant = model_and_quant("gemma-3-27b-it")
    apple = machine(vram_total=None, ram_total=64 * GIB, ram_available=48 * GIB, unified=True)
    budget = compute(model, quant, apple, context=8192, mode="gpu")
    assert budget.vram_required == 0
    assert budget.vram_utilisation is None
    assert all(line.pool != "vram" for line in budget.lines)
    assert not [line for line in budget.lines if line.component == "cuda-context"]


def test_a_hybrid_placement_puts_the_layers_it_says_on_the_card() -> None:
    model, quant = model_and_quant("gemma-3-27b-it")
    host = machine(vram_total=24 * GIB, ram_available=64 * GIB)
    whole = compute(model, quant, host, context=8192, mode="gpu")
    half = compute(model, quant, host, context=8192, mode="hybrid", gpu_layers=31)
    assert half.vram_required < whole.vram_required
    assert half.ram_required > whole.ram_required
    assert weight_bytes(half) == weight_bytes(whole)


def test_a_hybrid_placement_that_does_not_say_how_many_layers_is_refused() -> None:
    model, quant = model_and_quant("gemma-3-27b-it")
    with pytest.raises(BudgetError, match="how many layers"):
        compute(model, quant, machine(), context=8192, mode="hybrid")


def test_a_quant_nobody_has_read_the_header_of_cannot_be_sized() -> None:
    model, quant = model_and_quant("qwen3-0.6b")
    with pytest.raises(BudgetError, match="has not read"):
        compute(model, quant.model_copy(update={"gguf_facts": None}), machine(), context=4096)


def test_a_bigger_context_costs_more_and_a_quantised_cache_costs_less() -> None:
    model, quant = model_and_quant("qwen3-coder-next")
    host = machine(vram_total=24 * GIB)
    small = compute(model, quant, host, context=16384, mode="moe-offload")
    large = compute(model, quant, host, context=65536, mode="moe-offload")
    cheap = compute(model, quant, host, context=65536, mode="moe-offload", kv_type="q8_0")
    assert small.vram_required < large.vram_required
    assert cheap.vram_required < large.vram_required


def test_turning_vision_off_is_worth_about_two_gigabytes_on_a_small_card() -> None:
    model, quant = model_and_quant("qwen3.8-flash-next")
    projector = Extra(role="mmproj", file="mmproj-F16.gguf", bytes=862 * MIB)
    host = reference_host()
    off = compute(model, quant, host, context=32768, mode="moe-offload", micro_batch=512)
    on = compute(
        model,
        quant,
        host,
        context=32768,
        mode="moe-offload",
        micro_batch=512,
        projector=projector,
        projector_pool="vram",
    )
    cost = on.vram_required - off.vram_required
    assert 1.7 * GIB < cost < 2.1 * GIB, "the calibration record measured about 1.9 GB"

    on_cpu = compute(
        model,
        quant,
        host,
        context=32768,
        mode="moe-offload",
        micro_batch=512,
        projector=projector,
        projector_pool="ram",
    )
    assert on_cpu.vram_required == off.vram_required
    assert on_cpu.ram_required == off.ram_required + 862 * MIB


@pytest.mark.parametrize(
    ("share", "expected"),
    [
        (0.0, "comfortable"),
        (0.65, "comfortable"),
        (0.66, "fits"),
        (0.85, "fits"),
        (0.86, "tight"),
        (0.95, "tight"),
        (0.951, "too-tight"),
        (5.0, "too-tight"),
    ],
)
def test_the_thresholds_are_the_ones_the_specification_gives(share: float, expected: str) -> None:
    assert pool_verdict(share, "vram") == expected


def test_system_memory_past_the_line_does_not_fit_because_there_is_nowhere_to_page() -> None:
    assert pool_verdict(0.96, "ram") == "does-not-fit"
    assert pool_verdict(0.96, "disk") == "does-not-fit"


def test_an_empty_pool_is_at_zero_and_a_pool_with_no_room_is_infinitely_over() -> None:
    assert utilisation(0, 0) == 0.0
    assert utilisation(1, 0) == float("inf")
    assert utilisation(1, 2) == 0.5
