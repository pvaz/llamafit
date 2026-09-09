from datetime import date

import pytest
from pydantic import ValidationError

from llamafit.models.catalog import Architecture, CatalogModel, License, ModelSource, Params, Quant


def minimal(**overrides: object) -> CatalogModel:
    base: dict[str, object] = {
        "id": "qwen3-coder-next",
        "name": "Qwen3-Coder-Next",
        "vendor": "Alibaba Qwen",
        "family": "qwen3",
        "release_date": date(2026, 2, 1),
        "license": License(spdx="Apache-2.0", url="https://example.invalid/license"),
        "params": Params(total_b=80, active_b=3),
        "architecture": Architecture(**{"class": "moe-hybrid", "gguf_arch": "qwen3next"}),
        "context": {"native": 262144},
        "capabilities": ["coding", "tools"],
        "use_cases": ["coding"],
        "quality": {"baseline": 86},
        "sources": [
            ModelSource(
                repo="unsloth/Qwen3-Coder-Next-GGUF",
                trust="unsloth",
                quants=[Quant(name="UD-Q4_K_XL")],
            )
        ],
    }
    base.update(overrides)
    return CatalogModel(**base)  # type: ignore[arg-type]


def test_a_minimal_entry_validates() -> None:
    model = minimal()
    assert model.architecture.class_ == "moe-hybrid"
    assert model.sources[0].quants[0].name == "UD-Q4_K_XL"
    assert model.sampling.temp is None


def test_class_and_bytes_use_their_yaml_names() -> None:
    model = minimal()
    dumped = model.model_dump(by_alias=True)
    assert dumped["architecture"]["class"] == "moe-hybrid"
    assert "class_" not in dumped["architecture"]
    quant = Quant(**{"name": "Q4_K_M", "bytes": 1234})
    assert quant.bytes_ == 1234
    assert quant.model_dump(by_alias=True)["bytes"] == 1234


def test_active_parameters_cannot_exceed_the_total() -> None:
    with pytest.raises(ValidationError, match="active_b"):
        minimal(params=Params(total_b=8, active_b=9))


def test_an_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        minimal(nonsense="x")


def test_the_identifier_must_be_a_slug() -> None:
    with pytest.raises(ValidationError):
        minimal(id="Qwen3 Coder Next")


def test_a_gguf_source_needs_a_repository_and_a_local_source_needs_a_path() -> None:
    with pytest.raises(ValidationError):
        ModelSource(kind="gguf", quants=[Quant(name="Q4_K_M")])
    with pytest.raises(ValidationError):
        ModelSource(kind="local", quants=[Quant(name="Q4_K_M")])
    assert ModelSource(kind="local", path="D:/models/x.gguf", quants=[Quant(name="Q4_K_M")]).path


def test_a_model_needs_at_least_one_source_and_one_use_case() -> None:
    with pytest.raises(ValidationError):
        minimal(sources=[])
    with pytest.raises(ValidationError):
        minimal(use_cases=[])
