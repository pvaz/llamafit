from __future__ import annotations

import pytest

from llamafit.models.catalog import Catalog, CatalogModel, License, ModelSource, Quant
from llamafit.models.gguf import GgufFacts
from llamafit.models.llamacpp import LocalModel
from llamafit.services.catalog import ModelFilters, describe, filter_models, summarise
from tests.unit.test_models_catalog import minimal


def _catalog(*models: CatalogModel) -> Catalog:
    return Catalog(models=list(models))


def test_use_case_filter_keeps_only_models_with_that_use_case() -> None:
    coder = minimal(id="coder-a", use_cases=["coding"])
    chatty = minimal(id="chat-a", use_cases=["chat"])
    catalog = _catalog(coder, chatty)

    result = filter_models(catalog, ModelFilters(use_case="coding"))

    assert [m.id for m in result] == ["coder-a"]


def test_capability_filter_requires_every_capability_not_any() -> None:
    both = minimal(id="both", capabilities=["coding", "tools"])
    coding_only = minimal(id="coding-only", capabilities=["coding"])
    tools_only = minimal(id="tools-only", capabilities=["tools"])
    catalog = _catalog(both, coding_only, tools_only)

    result = filter_models(catalog, ModelFilters(capabilities=["coding", "tools"]))

    assert [m.id for m in result] == ["both"]


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("id", "unusual-marker-id", "UNUSUAL-MARKER"),
        ("name", "UnusualMarkerName", "unusualmarkername"),
        ("vendor", "Unusual Marker Vendor", "MARKER VENDOR"),
        ("family", "unusualmarkerfamily", "UnusualMarkerFamily"),
    ],
)
def test_search_matches_id_name_vendor_and_family_case_insensitively(
    field: str, value: str, needle: str
) -> None:
    overrides: dict[str, object] = {field: value}
    overrides.setdefault("id", "target-model")
    target = minimal(**overrides)
    other = minimal(id="other-model")
    catalog = _catalog(target, other)

    result = filter_models(catalog, ModelFilters(search=needle))

    assert [m.id for m in result] == [target.id]


def test_license_filter_keeps_only_models_with_that_licence() -> None:
    apache = minimal(id="apache-model")
    mit = minimal(id="mit-model", license=License(spdx="MIT", url="https://example.invalid/mit"))
    catalog = _catalog(apache, mit)

    result = filter_models(catalog, ModelFilters(licenses=["MIT"]))

    assert [m.id for m in result] == ["mit-model"]


def test_every_filter_applies_as_an_and() -> None:
    keep = minimal(
        id="keep",
        vendor="Acme",
        use_cases=["coding"],
        capabilities=["coding", "tools"],
    )
    wrong_use_case = minimal(
        id="wrong-use-case",
        vendor="Acme",
        use_cases=["chat"],
        capabilities=["coding", "tools"],
    )
    wrong_capability = minimal(
        id="wrong-capability",
        vendor="Acme",
        use_cases=["coding"],
        capabilities=["coding"],
    )
    wrong_vendor = minimal(
        id="wrong-vendor",
        vendor="Other",
        use_cases=["coding"],
        capabilities=["coding", "tools"],
    )
    catalog = _catalog(keep, wrong_use_case, wrong_capability, wrong_vendor)

    result = filter_models(
        catalog,
        ModelFilters(use_case="coding", capabilities=["coding", "tools"], vendor="Acme"),
    )

    assert [m.id for m in result] == ["keep"]


def test_results_sort_by_quality_descending_then_id_for_stable_ties() -> None:
    high = minimal(id="high", quality={"baseline": 90})
    tie_b = minimal(id="tie-b", quality={"baseline": 80})
    tie_a = minimal(id="tie-a", quality={"baseline": 80})
    low = minimal(id="low", quality={"baseline": 10})
    catalog = _catalog(tie_b, low, high, tie_a)

    result = filter_models(catalog, ModelFilters())

    assert [m.id for m in result] == ["high", "tie-a", "tie-b", "low"]


def test_summarise_marks_a_model_local_when_a_quant_file_is_on_disk() -> None:
    model = minimal(
        sources=[
            ModelSource(
                repo="acme/model-gguf",
                quants=[Quant(name="Q4_K_M", files=["model-Q4_K_M.gguf"])],
            )
        ]
    )
    local = [LocalModel(path="D:/models/model-Q4_K_M.gguf", bytes=123)]

    summary = summarise(model, local)

    assert summary.is_local is True


def test_summarise_does_not_mark_a_model_local_without_a_matching_file() -> None:
    model = minimal(
        sources=[
            ModelSource(
                repo="acme/model-gguf",
                quants=[Quant(name="Q4_K_M", files=["model-Q4_K_M.gguf"])],
            )
        ]
    )
    local = [LocalModel(path="D:/models/some-other-model.gguf", bytes=123)]

    assert summarise(model, local).is_local is False
    assert summarise(model).is_local is False


def test_summarise_reports_quant_names_and_the_largest_known_size() -> None:
    model = minimal(
        sources=[
            ModelSource(
                repo="acme/model-gguf",
                quants=[
                    Quant(name="Q4_K_M", files=["m-q4.gguf"], bytes=1000),
                    Quant(name="Q8_0", files=["m-q8.gguf"], bytes=2000),
                    Quant(name="F16"),
                ],
            )
        ]
    )

    summary = summarise(model)

    assert summary.quant_names == ["Q4_K_M", "Q8_0", "F16"]
    assert summary.largest_quant_bytes == 2000


def test_summarise_reports_no_size_when_no_quant_size_is_known() -> None:
    model = minimal(sources=[ModelSource(repo="acme/model-gguf", quants=[Quant(name="Q4_K_M")])])

    assert summarise(model).largest_quant_bytes is None


def test_describe_lists_every_quant_with_its_facts() -> None:
    facts = GgufFacts(arch="qwen3next", n_layer=48)
    model = minimal(
        sources=[
            ModelSource(
                repo="acme/model-gguf",
                quants=[
                    Quant(name="Q4_K_M", files=["model-Q4_K_M.gguf"], gguf_facts=facts),
                    Quant(name="Q8_0", files=["model-Q8_0.gguf"]),
                ],
            )
        ]
    )

    detail = describe(model)

    assert [q.name for q in detail.quants] == ["Q4_K_M", "Q8_0"]
    assert detail.quants[0].facts == facts
    assert detail.quants[1].facts is None


def test_describe_lists_quants_across_every_source() -> None:
    model = minimal(
        sources=[
            ModelSource(repo="acme/model-a-gguf", quants=[Quant(name="Q4_K_M")]),
            ModelSource(repo="acme/model-b-gguf", quants=[Quant(name="Q8_0")]),
        ]
    )

    detail = describe(model)

    assert [q.name for q in detail.quants] == ["Q4_K_M", "Q8_0"]


def test_describe_reports_the_local_path_of_a_downloaded_quant() -> None:
    model = minimal(
        sources=[
            ModelSource(
                repo="acme/model-gguf",
                quants=[Quant(name="Q4_K_M", files=["model-Q4_K_M.gguf"])],
            )
        ]
    )
    local = [LocalModel(path="D:/models/model-Q4_K_M.gguf", bytes=123)]

    detail = describe(model, local)

    assert detail.quants[0].downloaded is True
    assert detail.local_paths == ["D:/models/model-Q4_K_M.gguf"]


def test_describe_marks_a_quant_not_downloaded_without_a_matching_file() -> None:
    model = minimal(
        sources=[
            ModelSource(
                repo="acme/model-gguf",
                quants=[Quant(name="Q4_K_M", files=["model-Q4_K_M.gguf"])],
            )
        ]
    )

    detail = describe(model)

    assert detail.quants[0].downloaded is False
    assert detail.local_paths == []
