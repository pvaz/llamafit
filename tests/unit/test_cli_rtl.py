"""What the renderer emits in Arabic, and what it must never emit anywhere else.

Three things are asserted here and nothing else is claimed:

* the marks the renderer puts into a table are the ones UAX #9 names, in the places the
  standard puts them;
* a left-to-right language gets output with not one of those characters in it;
* ``--json`` gets none either, in any language, because a program parsing LlamaFit must
  not have to strip direction marks out of a field.

How any of it is drawn depends on the terminal, which no test can see. ``docs/translations.md``
says what is known about that and what is not.
"""

import json
from collections.abc import Iterator
from datetime import date

import pytest
from typer.testing import CliRunner

from llamafit.cli.app import app
from llamafit.cli.render import (
    render_catalog_list,
    render_findings,
    render_host,
    render_llamacpp,
    render_model_facts,
    render_probes,
    render_quants,
)
from llamafit.i18n import set_language
from llamafit.i18n.bidi import FIRST_STRONG_ISOLATE as FSI
from llamafit.i18n.bidi import POP_DIRECTIONAL_ISOLATE as PDI
from llamafit.i18n.bidi import RIGHT_TO_LEFT_MARK as RLM
from llamafit.models import LlamaCpp, Probe
from llamafit.models.catalog import (
    Architecture,
    Benchmark,
    CatalogModel,
    Context,
    License,
    ModelSource,
    Params,
    Quality,
)
from llamafit.services.catalog import ModelSummary, QuantDetail
from llamafit.services.doctor import Finding, diagnose
from tests.unit.test_cli import fake_report

runner = CliRunner()

MARKS = {FSI: "<FSI>", PDI: "<PDI>", RLM: "<RLM>", "\u2066": "<LRI>", "\u2067": "<RLI>"}
EVERY_MARK = "\u2066\u2067\u2068\u2069\u2069\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u061c"


def spell(text: str) -> str:
    """The same text with every invisible direction character written out by name."""
    for char, name in MARKS.items():
        text = text.replace(char, name)
    return text


def marks_in(text: str) -> set[str]:
    """Every direction-control character the text holds, named."""
    return {MARKS.get(c, hex(ord(c))) for c in text if c in EVERY_MARK}


@pytest.fixture
def arabic() -> Iterator[None]:
    """Speak the shipped Arabic catalog for the duration of one test."""
    set_language("ar")
    yield


def render(*tables: object, width: int = 200) -> str:
    """Print tables to a wide console and give back the plain text."""
    from rich.console import Console

    console = Console(width=width, no_color=True)
    with console.capture() as capture:
        for table in tables:
            console.print(table)
    return capture.get()


def sample_model() -> CatalogModel:
    return CatalogModel(
        id="qwen3-coder-next",
        name="Qwen3 Coder Next",
        vendor="Alibaba",
        family="qwen3",
        release_date=date(2026, 2, 1),
        license=License(spdx="Apache-2.0", url="https://spdx.org/licenses/Apache-2.0.html"),
        params=Params(total_b=27, active_b=3),
        context=Context(native=262144, extended=1048576, extended_method="yarn"),
        architecture=Architecture(class_="moe", gguf_arch="qwen3moe"),
        capabilities=["coding", "tools"],
        use_cases=["coding"],
        quality=Quality(
            baseline=74,
            benchmarks=[Benchmark(name="MMLU-Pro", score=71.2, source="vendor")],
        ),
        sources=[ModelSource(repo="unsloth/Qwen3-Coder-Next-GGUF", trust="community")],
    )


def sample_summary() -> ModelSummary:
    return ModelSummary(
        id="qwen3-coder-next",
        name="Qwen3 Coder Next",
        vendor="Alibaba",
        params_total_b=27,
        params_active_b=3,
        capabilities=["coding", "tools"],
        context_native=262144,
        license_spdx="Apache-2.0",
        quant_names=["Q4_K_M"],
        largest_quant_bytes=None,
        quality_baseline=74,
    )


def every_table() -> list[object]:
    """One of each table this module knows how to draw, with content in every cell."""
    report = fake_report()
    return [
        render_host(report.host),
        render_llamacpp(
            LlamaCpp(
                installed=True,
                path="D:/llama.cpp/bin",
                build=10867,
                commit="f3f1a8f27",
                backends=["cuda", "cpu"],
                problems=["one probe did not run"],
            )
        ),
        render_probes(
            [
                Probe(name="llama-server --version", ok=False, duration_ms=4, error="not found"),
                Probe(name="nvidia-smi", ok=True, duration_ms=12),
                Probe(name="server:8080", ok=False, duration_ms=1, error="no server"),
            ]
        ),
        render_findings(diagnose(report).findings),
        render_catalog_list([sample_summary()], console_width=100),
        render_model_facts(sample_model()),
        render_quants([QuantDetail(name="Q4_K_M", bytes=None, bpw=4.5, facts=None, files=[])]),
    ]


# --- nothing changes for a left-to-right language ----------------------------------


def test_english_output_carries_no_direction_marks_at_all() -> None:
    output = render(*every_table())
    assert marks_in(output) == set()


def test_portuguese_output_carries_no_direction_marks_either() -> None:
    set_language("pt_PT")
    output = render(*every_table())
    assert marks_in(output) == set()


def test_english_columns_are_still_in_the_order_they_were_written() -> None:
    table = render_catalog_list([sample_summary()], console_width=100)
    assert next(str(column.header) for column in table.columns) == "ID"


# --- the machine-readable output is untouched --------------------------------------


@pytest.fixture(autouse=True)
def patch_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **kwargs: fake_report())
    monkeypatch.setattr("llamafit.cli.doctor_cmd.scan_system", lambda **kwargs: fake_report())


@pytest.mark.parametrize(
    "command",
    [
        ["--language", "ar", "--json", "system"],
        ["--language", "ar", "--json", "doctor"],
        ["--language", "he", "--json", "list"],
        ["--language", "ur", "--json", "info", "qwen3-coder-next"],
    ],
)
def test_json_output_never_holds_a_direction_mark(command: list[str]) -> None:
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert marks_in(result.output) == set()
    json.loads(result.output)  # still parses, and parses as what it was


def test_the_service_layer_that_feeds_json_is_where_no_mark_is_added(arabic: None) -> None:
    """A finding is built before the interface knows whether it is drawing or dumping."""
    diagnosis = diagnose(fake_report())
    assert marks_in(diagnosis.model_dump_json()) == set()


# --- Arabic: the marks, spelled out ------------------------------------------------


def test_an_interpolated_identifier_becomes_an_island(arabic: None) -> None:
    output = render(render_probes([Probe(name="llama-server --version", ok=True, duration_ms=4)]))
    assert f"{FSI}llama-server --version{PDI}" in output


def test_a_flag_a_translator_kept_inside_a_hint_becomes_an_island(arabic: None) -> None:
    """The damaging case: the flag is in the message, so only the renderer can reach it."""
    from llamafit.services.doctor import PROBE_HINTS

    hint = str(PROBE_HINTS["cpu-cores"])
    assert "`pip install --force-reinstall psutil`" in hint  # the catalog keeps it verbatim
    assert marks_in(hint) == set()  # and puts no mark of its own around it

    rendered = render(
        render_findings([Finding(level="warn", title="t", detail="d", hint=hint)]), width=300
    )
    assert f"{FSI}`pip install --force-reinstall psutil`{PDI}" in rendered


def test_a_whole_arabic_hint_before_and_after(arabic: None) -> None:
    """Exactly what changes in one shipped Arabic sentence, and exactly what does not."""
    from llamafit.i18n import for_display
    from llamafit.services.doctor import PROBE_HINTS

    before = str(PROBE_HINTS["cpu-cores"])
    after = for_display(before)

    assert after.startswith(RLM)
    assert f"{FSI}`pip install --force-reinstall psutil`{PDI}." in after
    assert after.replace(RLM, "").replace(FSI, "").replace(PDI, "") == before
    assert spell(after).count("<FSI>") == 1


def test_a_synthetic_finding_renders_to_exactly_these_characters(arabic: None) -> None:
    """A snapshot with every invisible character named, over text this test owns."""
    sentence = "\u0627\u062e\u062a\u0628\u0627\u0631 --verbose."
    rendered = render(
        render_findings([Finding(level="warn", title=sentence, detail="Q4_K_M", hint=None)]),
        width=300,
    )
    assert "<RLM>\u0627\u062e\u062a\u0628\u0627\u0631 <FSI>--verbose<PDI>." in spell(rendered)
    # The detail is a bare quant name: no Arabic in it, so it is given no mark at all.
    assert "<RLM>Q4_K_M" not in spell(rendered)
    assert "Q4_K_M" in rendered


def test_a_path_and_a_size_stay_whole_in_the_host_table(arabic: None) -> None:
    output = render(render_host(fake_report().host))
    assert f"{FSI}D:\\{PDI}" in output
    assert f"{FSI}128.0 GiB{PDI}" in output
    assert f"{FSI}NVIDIA GeForce RTX 4060{PDI}" in output


def test_a_model_id_and_its_capabilities_stay_whole_in_the_list(arabic: None) -> None:
    output = render(render_catalog_list([sample_summary()], console_width=100))
    assert f"{FSI}qwen3-coder-next{PDI}" in output
    assert f"{FSI}27/3B{PDI}" in output
    assert f"{FSI}256K{PDI}" in output
    assert f"{FSI}coding, tools{PDI}" in output


def test_the_licence_url_and_the_repository_stay_whole_in_the_facts(arabic: None) -> None:
    output = render(render_model_facts(sample_model()), width=300)
    assert f"{FSI}https://spdx.org/licenses/Apache-2.0.html{PDI}" in output
    assert f"{FSI}unsloth/Qwen3-Coder-Next-GGUF (gguf, trust=community){PDI}" in output
    assert f"{FSI}2026-02-01{PDI}" in output


# --- Arabic: the table itself ------------------------------------------------------


def test_the_first_column_is_added_last_so_it_lands_against_the_right_edge(
    arabic: None,
) -> None:
    from llamafit.cli.render import _list_headings, _quality_heading

    headings = _list_headings()  # the Arabic headings, in the order they are written
    # The quality column prints its heading followed by the footnote marker. The marker is
    # punctuation the renderer owns rather than a word inside the message, so it is not in
    # what `_list_headings` returns; `_quality_heading` is the single place that joins the
    # two, and it is what the table is given. Everything this test is about — that the
    # columns are added last-read first — is unchanged by that.
    written = [
        headings["id"],
        _quality_heading(headings),
        headings["params"],
        headings["context"],
        headings["capabilities"],
    ]

    table = render_catalog_list([sample_summary()], console_width=100)
    drawn = [str(column.header).lstrip(RLM) for column in table.columns]

    assert drawn == list(reversed(written))


def test_a_left_aligned_column_is_right_aligned_when_read_right_to_left(arabic: None) -> None:
    table = render_catalog_list([sample_summary()], console_width=100)
    justifications = [column.justify for column in table.columns]
    # id and capabilities are the two left-aligned columns in English; both are mirrored,
    # and the three numeric columns swap the other way.
    assert justifications.count("right") == 2
    assert justifications.count("left") == 3


def test_the_two_column_tables_put_the_label_on_the_right(arabic: None) -> None:
    table = render_host(fake_report().host)
    assert [str(column.header) for column in table.columns] == ["value", "key"]
    assert table.columns[-1].style == "bold"  # the key column keeps what it had
