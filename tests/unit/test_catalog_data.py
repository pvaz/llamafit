from llamafit.catalog.loader import load_catalog


def test_the_bundled_catalog_loads_without_problems() -> None:
    catalog, problems = load_catalog(custom_path=None)
    assert problems == [], problems
    assert len(catalog.models) >= 5


def test_every_entry_cites_its_sources() -> None:
    catalog, _ = load_catalog(custom_path=None)
    for model in catalog.models:
        assert model.license.url.startswith("http"), model.id
        assert len(model.quality.benchmarks) >= 2, f"{model.id} needs published scores"
        for benchmark in model.quality.benchmarks:
            assert benchmark.source.startswith("http"), f"{model.id}/{benchmark.name}"


def test_every_entry_has_at_least_one_quant_from_a_named_repository() -> None:
    catalog, _ = load_catalog(custom_path=None)
    for model in catalog.models:
        assert model.sources[0].repo, model.id
        assert model.sources[0].quants, model.id


def test_the_two_next_generation_qwen_entries_carry_a_licence_identifier() -> None:
    catalog, _ = load_catalog(custom_path=None)
    for model_id in ("qwen3-coder-next", "qwen3.8-flash-next"):
        assert catalog.by_id[model_id].license.spdx, model_id


def test_the_measured_models_carry_their_reference_measurements() -> None:
    catalog, _ = load_catalog(custom_path=None)
    for model_id in ("qwen3-coder-next", "qwen3.8-flash-next"):
        measured = catalog.by_id[model_id].measured
        assert measured, f"{model_id} was measured on the reference machine"
        assert all(m.source and "calibration" in m.source for m in measured)


def test_the_schema_file_matches_the_models() -> None:
    import json
    from importlib import resources

    from llamafit.models.catalog import CatalogModel

    text = (
        resources.files("llamafit.data.schema").joinpath("catalog.schema.json").read_text("utf-8")
    )
    assert json.loads(text) == CatalogModel.model_json_schema(by_alias=True), (
        "regenerate with: python scripts/gen_schema.py"
    )
