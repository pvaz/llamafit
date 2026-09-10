"""The ``llama-server`` command line (design section 9.5).

The order is the specification's order, and the tests hold it against the flags the
reference machine's own measured configurations recorded, because those are the closest
thing to ground truth this repository has.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llamafit.catalog import load_catalog
from llamafit.i18n import LANGUAGE_ENV_VAR, translator
from llamafit.models.catalog import CatalogModel, Extra
from llamafit.models.plan import Budget, Placement, Pool
from llamafit.placement.flags import (
    LaunchOptions,
    command_line,
    prose_quirks,
    quirk_arguments,
    render_flags,
)
from tests.fixtures.placement import make_facts, make_model

SPEC_ORDER = [
    "-m",
    "--mmproj",
    "--no-mmproj-offload",
    "--alias",
    "--host",
    "--port",
    "-c",
    "-fa",
    "-ctk",
    "-ctv",
    "-ngl",
    "--n-cpu-moe",
    "-ot",
    "--fit",
    "--lazy-mode",
    "-t",
    "-tb",
    "-b",
    "-ub",
    "--temp",
    "--top-p",
    "--top-k",
    "--min-p",
    "--presence-penalty",
    "--repeat-penalty",
    "--jinja",
    "--reasoning-format",
    "--chat-template-kwargs",
    "-np",
    "--cache-ram",
]
"""Section 9.5's list, in its own words and its own order."""

MEASURED_CODER_FLAGS = "-ngl 99 --n-cpu-moe 48 --fit off -fa on -t 16 -tb 16 -b 4096 -ub 2048"
"""What Qwen3-Coder-Next was actually measured with on the reference machine."""


def budget() -> Budget:
    return Budget(
        lines=(),
        vram_required=1,
        ram_required=1,
        vram_available=2,
        ram_available=2,
        vram_utilisation=0.5,
        ram_utilisation=0.5,
        verdict="fits",
    )


def placement(
    *,
    mode: str = "moe-offload",
    context: int = 40960,
    micro_batch: int = 1024,
    kv_type: str = "f16",
    gpu_layers: int = 99,
    cpu_moe_layers: int | None = 48,
    projector_pool: Pool | None = None,
    threads: int = 8,
) -> Placement:
    return Placement(
        mode=mode,  # type: ignore[arg-type]
        context=context,
        micro_batch=micro_batch,
        batch=max(2 * micro_batch, 2048),
        kv_type=kv_type,
        gpu_layers=gpu_layers,
        cpu_moe_layers=cpu_moe_layers,
        projector_pool=projector_pool,
        threads=threads,
        budget=budget(),
        max_context_fit=context,
    )


def options(**changes: object) -> LaunchOptions:
    base = {"model_path": "D:/models/test.gguf"}
    base.update(changes)
    return LaunchOptions(**base)  # type: ignore[arg-type]


def flags_of(model: CatalogModel, **changes: object) -> list[str]:
    return render_flags(placement(), model, options(**changes))


def positions(rendered: list[str], names: list[str]) -> list[int]:
    return [rendered.index(name) for name in names if name in rendered]


def test_every_flag_the_specification_lists_comes_out_in_its_order() -> None:
    model, _quant = make_model(
        projector=True,
        temp=0.7,
        reasoning_format="deepseek",
        thinking_toggle="enable_thinking",
        requires={"lazy_mode": "recommended"},
        facts=make_facts(),
    )
    rendered = render_flags(
        placement(kv_type="q8_0", projector_pool="ram"),
        model,
        options(
            projector_path="D:/models/mmproj.gguf",
            tensor_overrides=("ffn_.*_shexp=CPU",),
            thinking=True,
            cache_ram_mib=4096,
        ),
    )
    found = [name for name in SPEC_ORDER if name in rendered]
    assert positions(rendered, found) == sorted(positions(rendered, found))
    # Nothing in the list is missing from a configuration that exercises all of it.
    assert set(SPEC_ORDER) - set(rendered) == {
        "--top-p",
        "--top-k",
        "--min-p",
        "--presence-penalty",
        "--repeat-penalty",
    }


def test_the_model_path_comes_first_and_the_program_name_before_that() -> None:
    model, _quant = make_model(facts=make_facts())
    assert flags_of(model)[:2] == ["-m", "D:/models/test.gguf"]
    assert command_line(placement(), model, options())[0] == "llama-server"
    assert command_line(placement(), model, options(), executable="/opt/llama-server")[0] == (
        "/opt/llama-server"
    )


def test_the_alias_is_the_catalog_id_unless_the_caller_says_otherwise() -> None:
    model, _quant = make_model(model_id="qwen3-coder-next", facts=make_facts())
    rendered = flags_of(model)
    assert rendered[rendered.index("--alias") + 1] == "qwen3-coder-next"
    named = flags_of(model, alias="coder")
    assert named[named.index("--alias") + 1] == "coder"


def test_the_endpoint_is_loopback_and_llama_cpps_own_port_by_default() -> None:
    model, _quant = make_model(facts=make_facts())
    rendered = flags_of(model)
    assert rendered[rendered.index("--host") + 1] == "127.0.0.1"
    assert rendered[rendered.index("--port") + 1] == "8080"


def test_the_default_cache_type_is_not_named_and_a_quantised_one_is() -> None:
    model, _quant = make_model(facts=make_facts())
    assert "-ctk" not in render_flags(placement(), model, options())
    quantised = render_flags(placement(kv_type="q8_0"), model, options())
    assert quantised[quantised.index("-ctk") + 1] == "q8_0"
    assert quantised[quantised.index("-ctv") + 1] == "q8_0"


def test_flash_attention_and_the_refusal_of_automatic_placement_are_always_there() -> None:
    model, _quant = make_model(facts=make_facts())
    rendered = flags_of(model)
    assert rendered[rendered.index("-fa") + 1] == "on"
    assert rendered[rendered.index("--fit") + 1] == "off"


def test_the_expert_offload_flag_appears_only_when_experts_are_offloaded() -> None:
    model, _quant = make_model(facts=make_facts())
    assert "--n-cpu-moe" in render_flags(placement(), model, options())
    assert "--n-cpu-moe" not in render_flags(
        placement(mode="gpu", cpu_moe_layers=None), model, options()
    )


@pytest.mark.parametrize(
    ("pool", "expected"),
    [
        ("vram", ["--mmproj", "D:/p.gguf"]),
        ("ram", ["--mmproj", "D:/p.gguf", "--no-mmproj-offload"]),
        (None, ["--no-mmproj"]),
    ],
)
def test_the_projector_is_named_wherever_it_ends_up_and_when_it_is_left_out(
    pool: Pool | None, expected: list[str]
) -> None:
    model, _quant = make_model(projector=True, facts=make_facts())
    rendered = render_flags(
        placement(projector_pool=pool), model, options(projector_path="D:/p.gguf")
    )
    assert rendered[2 : 2 + len(expected)] == expected


def test_a_model_with_no_projector_says_nothing_about_one() -> None:
    model, _quant = make_model(facts=make_facts())
    rendered = flags_of(model)
    assert "--mmproj" not in rendered
    assert "--no-mmproj" not in rendered


def test_lazy_mode_is_rendered_when_the_catalog_recommends_it() -> None:
    recommends, _ = make_model(requires={"lazy_mode": "recommended"}, facts=make_facts())
    silent, _ = make_model(facts=make_facts())
    optional, _ = make_model(requires={"lazy_mode": "optional"}, facts=make_facts())
    assert "--lazy-mode" in flags_of(recommends)
    assert "--lazy-mode" not in flags_of(silent)
    assert "--lazy-mode" not in flags_of(optional)


def test_the_caller_can_overrule_the_catalog_about_lazy_mode() -> None:
    recommends, _ = make_model(requires={"lazy_mode": "recommended"}, facts=make_facts())
    silent, _ = make_model(facts=make_facts())
    assert "--lazy-mode" not in flags_of(recommends, lazy_mode=False)
    assert "--lazy-mode" in flags_of(silent, lazy_mode=True)


def test_only_the_sampling_the_curator_recorded_is_rendered() -> None:
    model, _quant = make_model(temp=0.6, facts=make_facts())
    rendered = flags_of(model)
    assert rendered[rendered.index("--temp") + 1] == "0.6"
    assert "--top-p" not in rendered
    bare, _ = make_model(facts=make_facts())
    assert "--temp" not in flags_of(bare)


def test_the_thinking_toggle_is_only_rendered_when_somebody_asks_for_it() -> None:
    model, _quant = make_model(
        reasoning_format="deepseek", thinking_toggle="enable_thinking", facts=make_facts()
    )
    plain = flags_of(model)
    assert plain[plain.index("--reasoning-format") + 1] == "deepseek"
    assert "--chat-template-kwargs" not in plain
    thinking = flags_of(model, thinking=True)
    assert thinking[thinking.index("--chat-template-kwargs") + 1] == '{"enable_thinking":true}'
    off = flags_of(model, thinking=False)
    assert off[off.index("--chat-template-kwargs") + 1] == '{"enable_thinking":false}'


def test_the_prompt_cache_limit_is_left_to_llama_cpp_unless_a_number_is_chosen() -> None:
    model, _quant = make_model(facts=make_facts())
    assert "--cache-ram" not in flags_of(model)
    rendered = flags_of(model, cache_ram_mib=0)
    assert rendered[rendered.index("--cache-ram") + 1] == "0"


def test_the_caller_always_has_the_last_word() -> None:
    model, _quant = make_model(facts=make_facts())
    assert flags_of(model, extra_args=("--verbose",))[-1] == "--verbose"


def test_a_quirk_that_is_a_sentence_is_not_pasted_onto_a_command_line() -> None:
    """Section 6.2's own example of a quirk is prose that happens to open with a flag."""
    prose = "--lazy-mode on streams the n-gram table from disk"
    assert quirk_arguments([prose]) == []
    model, _quant = make_model(quirks=[prose], facts=make_facts())
    assert prose_quirks(model) == (prose,)
    assert "streams" not in flags_of(model)


@pytest.mark.parametrize(
    "quirk",
    [
        "--no-context-shift",
        "-ot ffn_.*_shexp=CPU",
        "--n-cpu-moe 48",
        "--lazy-mode on",
        "--override-tensor exps=CUDA0",
        "--rope-scaling yarn",
        "--chat-template-file D:/templates/t.jinja",
    ],
)
def test_a_quirk_that_is_arguments_is_rendered_as_arguments(quirk: str) -> None:
    assert quirk_arguments([quirk]) == quirk.split()
    model, _quant = make_model(quirks=[quirk], facts=make_facts())
    assert prose_quirks(model) == ()
    assert flags_of(model)[-len(quirk.split()) :] == quirk.split()


@pytest.mark.parametrize(
    "quirk",
    ["needs a recent build", "", "the n-gram table streams from disk", "use --jinja for tools"],
)
def test_anything_that_does_not_open_with_a_flag_is_prose(quirk: str) -> None:
    assert quirk_arguments([quirk]) == []


def test_numbers_on_a_command_line_are_never_localised() -> None:
    """A Portuguese reader gets ``0,95`` everywhere in the interface and never here."""
    model, _quant = make_model(temp=0.95, facts=make_facts())
    import os

    previous = os.environ.get(LANGUAGE_ENV_VAR)
    os.environ[LANGUAGE_ENV_VAR] = "pt_PT"
    translator.reset()
    try:
        rendered = flags_of(model)
    finally:
        if previous is None:
            del os.environ[LANGUAGE_ENV_VAR]
        else:
            os.environ[LANGUAGE_ENV_VAR] = previous
        translator.reset()
    assert rendered[rendered.index("--temp") + 1] == "0.95"
    assert rendered[rendered.index("-c") + 1] == "40960"


def reference_model(model_id: str) -> CatalogModel:
    catalog, _problems = load_catalog(custom_path=Path("no-such-custom-models.yaml"))
    return catalog.by_id[model_id]


def test_the_renderer_reproduces_the_reference_machines_measured_flags() -> None:
    """Qwen3-Coder-Next's recorded configuration, flag for flag on the part they share.

    ``-ngl 99 --n-cpu-moe 48 --fit off -fa on -t 16 -tb 16 -b 4096 -ub 2048`` is what the
    machine was actually measured with. Everything in it is reproduced except the thread
    count, which was tuned by hand: the recorded 16 is the hyper-threaded pair count of
    the eight performance cores, and section 9.4 asks for the cores themselves.
    """
    model = reference_model("qwen3-coder-next")
    rendered = render_flags(
        placement(micro_batch=2048, threads=16), model, options(model_path="D:/coder.gguf")
    )
    measured = MEASURED_CODER_FLAGS.split()
    for flag, value in zip(measured[::2], measured[1::2], strict=True):
        assert rendered[rendered.index(flag) + 1] == value
    assert positions(rendered, ["-ngl", "--n-cpu-moe", "--fit", "-t", "-tb", "-b", "-ub"]) == (
        sorted(positions(rendered, ["-ngl", "--n-cpu-moe", "--fit", "-t", "-tb", "-b", "-ub"]))
    )


def test_the_reference_machines_winning_configuration_renders() -> None:
    """Flash-Next's winner: ``-ub 1024 --no-mmproj-offload -ot ffn_.*_shexp=CPU``."""
    model = reference_model("qwen3.8-flash-next")
    projector = Extra(role="mmproj", file="mmproj-F16.gguf")
    with_projector = model.model_copy(
        update={"sources": [model.sources[0].model_copy(update={"extras": [projector]})]}
    )
    rendered = render_flags(
        placement(micro_batch=1024, projector_pool="ram", threads=16),
        with_projector,
        options(
            model_path="D:/flash-00001-of-00004.gguf",
            projector_path="D:/mmproj-F16.gguf",
            tensor_overrides=("ffn_.*_shexp=CPU",),
        ),
    )
    assert rendered[rendered.index("-ub") + 1] == "1024"
    assert "--no-mmproj-offload" in rendered
    assert rendered[rendered.index("-ot") + 1] == "ffn_.*_shexp=CPU"
    assert rendered[rendered.index("-c") + 1] == "40960"
