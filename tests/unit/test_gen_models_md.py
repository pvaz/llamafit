"""Unit tests for the MODELS.md rendering logic, using synthetic catalog entries.

``tests/unit/test_docs.py`` checks that the committed page matches the real, bundled
catalog; these tests exercise rendering rules the bundled catalog doesn't happen to
trigger today, such as a model with more than one source.
"""

import datetime

from scripts.gen_models_md import render_markdown

from llamafit.models.catalog import (
    Architecture,
    Catalog,
    CatalogModel,
    Context,
    License,
    ModelSource,
    Params,
    Quality,
    Quant,
)


def _model(*, sources: list[ModelSource]) -> CatalogModel:
    return CatalogModel(
        id="fixture-model",
        name="Fixture Model",
        vendor="Fixture Vendor",
        family="fixture",
        release_date=datetime.date(2026, 1, 1),
        license=License(spdx="MIT", url="https://example.com/license"),
        params=Params(total_b=7, active_b=7),
        architecture=Architecture(class_="dense", gguf_arch="fixture"),
        context=Context(native=4096),
        capabilities=["tools"],
        use_cases=["general"],
        quality=Quality(baseline=50),
        sources=sources,
    )


def test_quants_column_is_the_union_across_a_models_sources() -> None:
    model = _model(
        sources=[
            ModelSource(
                repo="vendor/official-GGUF",
                trust="official",
                quants=[Quant(name="Q4_K_M"), Quant(name="Q8_0")],
            ),
            ModelSource(
                repo="community/community-GGUF",
                trust="community",
                quants=[Quant(name="Q8_0"), Quant(name="UD-Q4_K_XL")],
            ),
        ]
    )
    text = render_markdown(Catalog(models=[model]))
    row = next(line for line in text.splitlines() if line.startswith("| Fixture Model"))

    assert "`Q4_K_M`" in row
    assert "`Q8_0`" in row
    assert "`UD-Q4_K_XL`" in row
    assert row.count("Q8_0") == 1, "a quant named by both sources must be listed once"
