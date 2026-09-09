"""Catalog services: filter the catalog, summarise a model, and describe one in full.

These are the read-only operations the catalog commands render: ``filter_models`` is
what backs ``llamafit list`` and ``search``, ``summarise`` is one row of that listing,
and ``describe`` is the full detail behind ``llamafit info``. As with every other
service, each function here returns a pydantic model and never prints.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePath

from pydantic import BaseModel, ConfigDict, Field

from llamafit.models.catalog import Capability, Catalog, CatalogModel, UseCase
from llamafit.models.gguf import GgufFacts
from llamafit.models.llamacpp import LocalModel


class ModelFilters(BaseModel):
    """Criteria narrowing the catalog down; every filter set here is applied as an ``and``.

    Attributes:
        use_case: Keep only models that list this use case.
        capabilities: Keep only models that have every one of these capabilities,
            not merely one of them.
        licenses: Keep only models whose licence SPDX identifier is one of these.
        vendor: Keep only models from this vendor, matched case-insensitively.
        search: Keep only models whose id, name, vendor or family contains this
            text, matched case-insensitively.
    """

    use_case: UseCase | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    licenses: list[str] = Field(default_factory=list)
    vendor: str | None = None
    search: str | None = None


def _matches(model: CatalogModel, filters: ModelFilters) -> bool:
    """Whether ``model`` satisfies every filter set in ``filters``."""
    if filters.use_case is not None and filters.use_case not in model.use_cases:
        return False
    if filters.capabilities and not set(filters.capabilities).issubset(model.capabilities):
        return False
    if filters.licenses and model.license.spdx not in filters.licenses:
        return False
    if filters.vendor is not None and model.vendor.casefold() != filters.vendor.casefold():
        return False
    if filters.search is not None:
        needle = filters.search.casefold()
        searchable = (model.id, model.name, model.vendor, model.family)
        if not any(needle in text.casefold() for text in searchable):
            return False
    return True


def filter_models(catalog: Catalog, filters: ModelFilters) -> list[CatalogModel]:
    """Every model in ``catalog`` that satisfies every filter in ``filters``.

    Results are sorted by ``quality.baseline`` descending, then by ``id`` ascending,
    so two models tied on quality always come out in the same order.
    """
    matched = [model for model in catalog.models if _matches(model, filters)]
    matched.sort(key=lambda model: (-model.quality.baseline, model.id))
    return matched


def _quant_file_names(model: CatalogModel) -> list[str]:
    """Every quant file name published by any of this model's sources."""
    return [name for source in model.sources for quant in source.quants for name in quant.files]


def _local_file_names(local_files: Sequence[LocalModel]) -> set[str]:
    """The file name, not the full path, of every local GGUF file."""
    return {PurePath(local.path).name for local in local_files}


class ModelSummary(BaseModel):
    """One row of a model listing: what a table needs, nothing more.

    Attributes:
        id: The model's identifier.
        name: The model's display name.
        vendor: Who trained the model.
        params_total_b: Total parameters, in billions.
        params_active_b: Parameters active per token, in billions.
        capabilities: What the model can do.
        context_native: Native context length, in tokens.
        license_spdx: The model's licence identifier.
        quant_names: The names of every quant any source publishes.
        largest_quant_bytes: The size of the largest quant whose size is known, or
            ``None`` when no quant's size has been filled in yet.
        is_local: Whether one of this model's quant files is already on disk.
        quality_baseline: The curator's editorial quality score, from
            ``model.quality.baseline``. This is the score before any
            quantisation penalty; phase 1C subtracts from it to reach the score
            a recommendation will actually show, so this field alone is not
            that score.
    """

    id: str
    name: str
    vendor: str
    params_total_b: float
    params_active_b: float
    capabilities: list[Capability]
    context_native: int
    license_spdx: str
    quant_names: list[str]
    largest_quant_bytes: int | None
    is_local: bool
    quality_baseline: int


def summarise(model: CatalogModel, local_files: Sequence[LocalModel] = ()) -> ModelSummary:
    """Summarise one model for a table row.

    ``is_local`` is true when one of the model's quant file names matches the file
    name of a GGUF file llama.cpp already has on disk (compared by file name, not
    full path), and false otherwise, including when ``local_files`` is empty.
    """
    quant_names = [quant.name for source in model.sources for quant in source.quants]
    sizes = [
        quant.bytes_
        for source in model.sources
        for quant in source.quants
        if quant.bytes_ is not None
    ]
    local_names = _local_file_names(local_files)
    is_local = any(name in local_names for name in _quant_file_names(model))
    return ModelSummary(
        id=model.id,
        name=model.name,
        vendor=model.vendor,
        params_total_b=model.params.total_b,
        params_active_b=model.params.active_b,
        capabilities=list(model.capabilities),
        context_native=model.context.native,
        license_spdx=model.license.spdx,
        quant_names=quant_names,
        largest_quant_bytes=max(sizes) if sizes else None,
        is_local=is_local,
        quality_baseline=model.quality.baseline,
    )


class QuantDetail(BaseModel):
    """Full facts about one quantisation of a model.

    Attributes:
        name: The quant's name.
        bytes_: Total size of its files, in bytes, when known. Serialized as
            ``bytes``, its alias, since ``bytes`` is a Python builtin, matching
            ``Quant.bytes_``, ``Extra.bytes_`` and ``TensorInfo.bytes_``.
        bpw: Bits per weight, when known.
        facts: Architecture facts read from this quant's GGUF header, filled in by
            the refresh command; ``None`` until then.
        files: The GGUF file names that make up this quant.
        downloaded: Whether one of ``files`` matches a file llama.cpp already has
            on disk.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    bytes_: int | None = Field(alias="bytes")
    bpw: float | None
    facts: GgufFacts | None
    files: list[str]
    downloaded: bool


class ModelDetail(BaseModel):
    """Everything about one model: its catalog entry plus every quant's facts and sizes.

    This reports facts and sizes only. Whether a given quant fits in the memory this
    host has is phase 1C's question, not this one: that budget belongs on
    :class:`QuantDetail`, computed against a hardware profile, once phase 1C exists
    to compute it. Nothing here estimates or guesses at it in the meantime.

    Attributes:
        model: The catalog entry itself.
        quants: Every quant across every source, each with its facts.
        local_paths: Full paths of the local files that match one of this model's
            quant file names.
    """

    model: CatalogModel
    quants: list[QuantDetail]
    local_paths: list[str]


def describe(model: CatalogModel, local_files: Sequence[LocalModel] = ()) -> ModelDetail:
    """Describe one model in full: every quant it publishes, with its facts.

    See :class:`ModelDetail` for why no memory budget appears here.
    """
    local_by_name = {PurePath(local.path).name: local.path for local in local_files}
    quants: list[QuantDetail] = []
    local_paths: list[str] = []
    for source in model.sources:
        for quant in source.quants:
            matched_paths = [local_by_name[name] for name in quant.files if name in local_by_name]
            quants.append(
                QuantDetail(
                    name=quant.name,
                    bytes_=quant.bytes_,
                    bpw=quant.bpw,
                    facts=quant.gguf_facts,
                    files=list(quant.files),
                    downloaded=bool(matched_paths),
                )
            )
            local_paths.extend(matched_paths)
    return ModelDetail(model=model, quants=quants, local_paths=local_paths)
