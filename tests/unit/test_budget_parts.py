"""The four halves of a budget on their own: weights, caches, buffers and the projector.

Where a number can be checked against something real it is: the compute-buffer model is
checked against the buffer sizes ``llama-server -v`` printed on the reference machine, and
the weight split is checked against the file's own tensor table.
"""

import pytest

from llamafit.budget.compute_buffer import (
    batch_for,
    buffer_lines,
    compute_buffer_bytes,
    fit_for,
    output_buffer_bytes,
)
from llamafit.budget.kv import (
    KV_TYPES,
    cache_lines,
    derived_kv_cache_bytes,
    kv_cache_bytes,
    unaccounted_kv_cache_bytes,
)
from llamafit.budget.projector import projector_lines
from llamafit.budget.weights import split_by_layers, weight_lines
from llamafit.catalog import load_catalog
from llamafit.constants import MIB, PROJECTOR_COMPUTE_BYTES
from llamafit.errors import BudgetError
from llamafit.models.catalog import Extra
from llamafit.models.gguf import GgufFacts
from tests.fixtures.budget_hosts import first_quant


def facts_for(model_id: str) -> GgufFacts:
    catalog, problems = load_catalog()
    assert problems == []
    quant = first_quant(catalog.by_id[model_id])
    assert quant.gguf_facts is not None
    return quant.gguf_facts


def test_a_split_always_adds_back_up_to_what_was_split() -> None:
    for layers in range(0, 49):
        on_gpu, in_ram = split_by_layers(111_323_630_080, layers, 48)
        assert on_gpu + in_ram == 111_323_630_080


def test_a_model_with_no_layer_count_goes_wherever_the_offload_flag_points() -> None:
    assert split_by_layers(100, 0, 0) == (0, 100)
    assert split_by_layers(100, 99, 0) == (100, 0)


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
def test_the_weight_lines_of_a_real_model_add_up_to_the_file(model_id: str) -> None:
    """Every byte of the file is in exactly one line, whatever the placement."""
    facts = facts_for(model_id)
    for gpu_layers, cpu_moe in ((99, 0), (0, 99), (12, 6), (24, 0)):
        lines = weight_lines(facts, gpu_layers=gpu_layers, cpu_moe_layers=cpu_moe)
        assert sum(line.bytes_ for line in lines) == facts.bytes_total
        assert all(line.exact for line in lines), "a weight line is a sum of tensor sizes"


def test_the_flash_next_model_buffer_matches_what_llama_server_reported() -> None:
    """4,606 MiB on the card with the shared experts, 4,367 without: both measured."""
    facts = facts_for("qwen3.8-flash-next")
    with_shared = weight_lines(facts, gpu_layers=99, cpu_moe_layers=48)
    assert abs(sum(line.bytes_ for line in with_shared if line.pool == "vram") - 4606 * MIB) < MIB

    without = weight_lines(facts, gpu_layers=99, cpu_moe_layers=48, shared_experts_pool="ram")
    assert abs(sum(line.bytes_ for line in without if line.pool == "vram") - 4367 * MIB) < MIB


def test_the_embedding_table_stays_in_memory_but_a_tied_one_follows_the_output() -> None:
    separate = {
        line.component: line.pool
        for line in weight_lines(facts_for("llama-3.1-8b-instruct"), gpu_layers=99)
    }
    assert separate["token-embedding"] == "ram" and separate["output-head"] == "vram"

    gemma = weight_lines(facts_for("gemma-3-27b-it"), gpu_layers=99)
    tied = {line.component: line.pool for line in gemma}
    assert "output-head" not in tied, "gemma 3 ties its output projection to its embedding"
    assert tied["token-embedding"] == "vram"


def test_a_tied_embedding_stays_in_memory_when_the_layers_do() -> None:
    gemma = weight_lines(facts_for("gemma-3-27b-it"), gpu_layers=0)
    tied = {line.component: line.pool for line in gemma}
    assert tied["token-embedding"] == "ram"


def test_experts_move_to_memory_one_layer_at_a_time() -> None:
    facts = facts_for("qwen3.8-flash-next")
    previous = 0
    for cpu_moe in range(0, 49, 8):
        lines = weight_lines(facts, gpu_layers=99, cpu_moe_layers=cpu_moe)
        in_ram = sum(
            line.bytes_
            for line in lines
            if line.component == "expert-weights" and line.pool == "ram"
        )
        assert in_ram >= previous
        previous = in_ram
    assert previous == facts.bytes_expert_weights


def test_a_streamed_table_is_on_disk_and_a_held_one_is_in_memory() -> None:
    facts = facts_for("qwen3.8-flash-next")
    streamed = next(
        line for line in weight_lines(facts, gpu_layers=99) if line.component == "lazy-tables"
    )
    assert streamed.pool == "disk" and streamed.bytes_ == facts.bytes_lazy_tables

    held = next(
        line
        for line in weight_lines(facts, gpu_layers=99, stream_lazy_tables=False)
        if line.component == "lazy-tables"
    )
    assert held.pool == "ram"


def test_a_model_with_no_streamable_table_has_no_such_line() -> None:
    facts = facts_for("llama-3.1-8b-instruct")
    assert not [
        line for line in weight_lines(facts, gpu_layers=99) if line.component == "lazy-tables"
    ]


def test_shared_experts_are_a_bucket_of_their_own_and_a_line_of_their_own() -> None:
    """The file's `_shexp` tensors come to 239.5 MiB; the record measured 239 MiB freed."""
    facts = facts_for("qwen3.8-flash-next")
    assert facts.has_shared_experts is True
    assert abs(facts.bytes_shared_expert_weights - 239 * MIB) < MIB

    lines = {
        (line.component, line.pool): line.bytes_ for line in weight_lines(facts, gpu_layers=99)
    }
    assert lines[("shared-expert-weights", "vram")] == facts.bytes_shared_expert_weights
    assert lines[("dense-weights", "vram")] == facts.bytes_dense_block_weights, (
        "the bucket is carved out of the dense one, so nothing is subtracted here"
    )


def test_a_model_with_no_shared_experts_gets_no_such_line() -> None:
    facts = facts_for("llama-3.1-8b-instruct")
    assert facts.bytes_shared_expert_weights == 0
    assert not [
        line
        for line in weight_lines(facts, gpu_layers=99)
        if line.component == "shared-expert-weights"
    ]


def test_the_shared_expert_line_is_the_only_thing_an_override_moves() -> None:
    """`-ot ffn_.*_shexp=CPU`, now that the bytes are separable."""
    facts = facts_for("qwen3.8-flash-next")
    shared = facts.bytes_shared_expert_weights

    with_their_layers = {
        (line.component, line.pool): line.bytes_ for line in weight_lines(facts, gpu_layers=99)
    }
    moved = {
        (line.component, line.pool): line.bytes_
        for line in weight_lines(facts, gpu_layers=99, shared_experts_pool="ram")
    }
    assert with_their_layers[("shared-expert-weights", "vram")] == shared
    assert moved[("shared-expert-weights", "ram")] == shared
    assert ("shared-expert-weights", "vram") not in moved
    assert {k: v for k, v in moved.items() if k[0] != "shared-expert-weights"} == {
        k: v for k, v in with_their_layers.items() if k[0] != "shared-expert-weights"
    }


@pytest.mark.parametrize("kv_type", KV_TYPES)
def test_the_cache_grows_in_proportion_to_the_context(kv_type: str) -> None:
    facts = facts_for("qwen3.8-flash-next")
    at_16k = kv_cache_bytes(facts, context=16384, kv_type=kv_type)
    assert kv_cache_bytes(facts, context=32768, kv_type=kv_type) == 2 * at_16k
    assert kv_cache_bytes(facts, context=0, kv_type=kv_type) == 0


def test_quantising_the_cache_makes_it_smaller() -> None:
    facts = facts_for("qwen3-coder-next")
    f16 = kv_cache_bytes(facts, context=65536, kv_type="f16")
    assert kv_cache_bytes(facts, context=65536, kv_type="q8_0") < f16
    assert kv_cache_bytes(facts, context=65536, kv_type="q4_0") < f16


def test_a_cache_type_llamafit_cannot_size_is_refused_rather_than_guessed() -> None:
    with pytest.raises(BudgetError, match="q2_k"):
        kv_cache_bytes(facts_for("qwen3-0.6b"), context=4096, kv_type="q2_k")


def test_a_file_that_never_said_its_attention_shape_is_refused_too() -> None:
    facts = facts_for("qwen3-0.6b").model_copy(update={"n_head_kv": None})
    with pytest.raises(BudgetError, match="per token"):
        kv_cache_bytes(facts, context=4096, kv_type="f16")


def test_the_cache_and_the_state_follow_their_layers() -> None:
    facts = facts_for("qwen3.8-flash-next")
    on_card = {
        line.component: line.pool
        for line in cache_lines(facts, context=32768, kv_type="f16", gpu_layers=99)
    }
    assert on_card == {
        "kv-cache": "vram",
        "kv-cache-unaccounted": "vram",
        "recurrent-state": "vram",
    }

    in_memory = {
        line.component: line.pool
        for line in cache_lines(facts, context=32768, kv_type="f16", gpu_layers=0)
    }
    assert in_memory == {
        "kv-cache": "ram",
        "kv-cache-unaccounted": "ram",
        "recurrent-state": "ram",
    }


def test_the_recurrent_state_is_modelled_and_a_derivable_cache_is_not() -> None:
    lines = {
        line.component: line
        for line in cache_lines(
            facts_for("qwen3-coder-next"), context=32768, kv_type="f16", gpu_layers=99
        )
    }
    assert lines["kv-cache"].exact is True
    assert "kv-cache-unaccounted" not in lines, "nothing was measured beyond this file's shape"
    assert lines["recurrent-state"].exact is False


def test_the_flash_next_recurrent_state_matches_what_llama_server_reported() -> None:
    lines = {
        line.component: line
        for line in cache_lines(
            facts_for("qwen3.8-flash-next"), context=32768, kv_type="f16", gpu_layers=99
        )
    }
    assert abs(lines["recurrent-state"].bytes_ - 113 * MIB) < MIB, "llama.cpp reported 113 MiB"


def test_a_cache_the_header_cannot_account_for_is_named_rather_than_dropped() -> None:
    """llama-server allocated 1,056 MiB where the shape accounts for 768."""
    facts = facts_for("qwen3.8-flash-next")
    lines = {
        line.component: line
        for line in cache_lines(facts, context=32768, kv_type="f16", gpu_layers=99)
    }
    derived, extra = lines["kv-cache"], lines["kv-cache-unaccounted"]
    assert derived.bytes_ == 768 * MIB
    assert extra.bytes_ == 288 * MIB
    assert derived.bytes_ + extra.bytes_ == 1056 * MIB

    assert derived.exact is False, "a figure known to be short is not an exact figure"
    assert extra.exact is False
    assert "qwen4exp" in (derived.note or "") and "qwen4exp" in (extra.note or "")


def test_the_unaccounted_cache_grows_with_the_context_and_is_zero_elsewhere() -> None:
    flash = facts_for("qwen3.8-flash-next")
    assert unaccounted_kv_cache_bytes(flash, context=1024) == 9 * MIB
    assert unaccounted_kv_cache_bytes(flash, context=131072) == 9 * 128 * MIB
    assert unaccounted_kv_cache_bytes(flash, context=0) == 0
    assert unaccounted_kv_cache_bytes(facts_for("llama-3.1-8b-instruct"), context=32768) == 0


def test_the_cache_a_caller_asks_for_is_all_of_it() -> None:
    facts = facts_for("qwen3.8-flash-next")
    derived = derived_kv_cache_bytes(facts, context=32768, kv_type="f16")
    extra = unaccounted_kv_cache_bytes(facts, context=32768)
    assert kv_cache_bytes(facts, context=32768, kv_type="f16") == derived + extra
    assert extra > 0, "this is the architecture that has one"


def test_an_assumed_attention_layer_count_makes_the_cache_line_inexact() -> None:
    facts = facts_for("qwen3-0.6b").model_copy(update={"attention_layers_source": "all-layers"})
    line = next(
        line
        for line in cache_lines(facts, context=4096, kv_type="f16", gpu_layers=99)
        if line.component == "kv-cache"
    )
    assert line.exact is False and line.note is not None


def test_a_model_with_no_recurrent_state_has_no_such_line() -> None:
    facts = facts_for("llama-3.1-8b-instruct")
    lines = cache_lines(facts, context=4096, kv_type="f16", gpu_layers=99)
    assert [line.component for line in lines] == ["kv-cache"]


# Every row of the calibration record that reports a main compute buffer with vision off,
# as (micro-batch, context, MiB llama-server printed).
CALIBRATION_ROWS = [
    (512, 65536, 1142),
    (512, 49152, 1118),
    (512, 32768, 1094),
    (1024, 32768, 1337),
    (1024, 40960, 1361),
    (1024, 65536, 1433),
    (2048, 32768, 2330),
    (2048, 65536, 2586),
    (2048, 131072, 4168),
]


@pytest.mark.parametrize(("micro_batch", "context", "measured_mib"), CALIBRATION_ROWS)
def test_the_compute_buffer_model_reproduces_the_measurements_it_was_fitted_to(
    micro_batch: int, context: int, measured_mib: int
) -> None:
    """Within 5 percent of every buffer size llama.cpp printed on the reference machine."""
    predicted = compute_buffer_bytes(micro_batch=micro_batch, context=context)
    assert abs(predicted - measured_mib * MIB) <= 0.05 * measured_mib * MIB


def test_the_compute_buffer_never_shrinks_as_the_context_grows() -> None:
    sizes = [compute_buffer_bytes(micro_batch=2048, context=c) for c in range(0, 262144, 8192)]
    assert sizes == sorted(sizes)


def test_a_micro_batch_the_fit_never_saw_is_served_by_the_next_one_up() -> None:
    assert fit_for(256) == fit_for(512)
    assert fit_for(1536) == fit_for(2048)
    assert fit_for(4096) == fit_for(2048), "nothing larger was ever measured"


def test_the_batch_follows_the_micro_batch_but_never_drops_below_2048() -> None:
    assert batch_for(512) == 2048
    assert batch_for(1024) == 2048
    assert batch_for(2048) == 4096


def test_the_output_buffer_is_one_float_per_vocabulary_entry_per_batch_token() -> None:
    assert output_buffer_bytes(n_vocab=1000, batch=2048) == 1000 * 4 * 2048
    assert output_buffer_bytes(n_vocab=0, batch=2048) == 0


def test_the_two_buffer_lines_are_both_modelled() -> None:
    lines = buffer_lines(micro_batch=1024, context=32768, n_vocab=151936)
    assert [line.component for line in lines] == ["compute-buffer", "output-buffer"]
    assert [line.pool for line in lines] == ["vram", "ram"]
    assert not any(line.exact for line in lines)


def test_a_model_with_no_vocabulary_gets_no_output_buffer_line() -> None:
    lines = buffer_lines(micro_batch=1024, context=4096, n_vocab=0)
    assert [line.component for line in lines] == ["compute-buffer"]


def test_an_offloaded_projector_puts_a_floor_under_the_compute_buffer() -> None:
    """At -ub 512 the vision graph is the larger of the two; at -ub 2048 it is not."""
    small = compute_buffer_bytes(micro_batch=512, context=65536, projector_on_gpu=True)
    assert abs(small - 1922 * MIB) < MIB, "llama-server reported 1,922 MiB"

    large_without = compute_buffer_bytes(micro_batch=2048, context=65536)
    large_with = compute_buffer_bytes(micro_batch=2048, context=65536, projector_on_gpu=True)
    assert large_with == large_without


PROJECTOR = Extra(role="mmproj", file="mmproj-F16.gguf", bytes=862 * MIB)


def test_an_offloaded_projector_costs_its_file_and_its_own_compute_space() -> None:
    lines = projector_lines(PROJECTOR, pool="vram")
    assert [line.component for line in lines] == ["vision-projector", "vision-projector-compute"]
    assert [line.exact for line in lines] == [True, False]
    assert sum(line.bytes_ for line in lines) == 862 * MIB + PROJECTOR_COMPUTE_BYTES


def test_a_projector_on_the_processor_costs_only_its_file() -> None:
    lines = projector_lines(PROJECTOR, pool="ram")
    assert [(line.component, line.pool, line.bytes_) for line in lines] == [
        ("vision-projector", "ram", 862 * MIB)
    ]


@pytest.mark.parametrize(
    ("projector", "pool"),
    [(None, "vram"), (PROJECTOR, None), (Extra(role="mmproj", file="m.gguf"), "vram")],
)
def test_no_projector_no_lines(projector: Extra | None, pool: str | None) -> None:
    assert projector_lines(projector, pool=pool) == ()  # type: ignore[arg-type]
