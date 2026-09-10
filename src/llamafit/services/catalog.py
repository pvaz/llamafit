# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
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
    quality_baseline: int


def summarise(model: CatalogModel) -> ModelSummary:
    """Summarise one model for a table row.

    There is deliberately no "already on disk" field here. That would need
    matching a quant's file names against the local files llama.cpp has, and every
    quant's ``files`` list is empty until ``catalog refresh`` has filled it in from
    Hugging Face; a field that can only ever read ``False`` today is worse than no
    field, since ``False`` reads as a fact rather than as data nobody has yet. It
    belongs here once refresh has run and a real match is possible.
    """
    quant_names = [quant.name for source in model.sources for quant in source.quants]
    sizes = [
        quant.bytes_
        for source in model.sources
        for quant in source.quants
        if quant.bytes_ is not None
    ]
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
        quality_baseline=model.quality.baseline,
    )


class QuantDetail(BaseModel):
    """Full facts about one quantisation of a model.

    There is deliberately no "downloaded" field here, for the same reason
    :func:`summarise` has no "already on disk" field: ``files`` is empty until
    ``catalog refresh`` has filled it in, so a match against the local disk would
    read ``False`` for every quant regardless of what is actually on disk.

    Attributes:
        name: The quant's name.
        bytes_: Total size of its files, in bytes, when known. Serialized as
            ``bytes``, its alias, since ``bytes`` is a Python builtin, matching
            ``Quant.bytes_``, ``Extra.bytes_`` and ``TensorInfo.bytes_``.
        bpw: Bits per weight, when known.
        facts: Architecture facts read from this quant's GGUF header, filled in by
            the refresh command; ``None`` until then.
        files: The GGUF file names that make up this quant.
    """

    model_config = ConfigDict(populate_by_name=True)

    name: str
    bytes_: int | None = Field(alias="bytes")
    bpw: float | None
    facts: GgufFacts | None
    files: list[str]


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
            # Compared as bare file names on both sides: a catalog's ``files`` entry is a
            # path inside the publishing repository and often carries a directory, while a
            # local file is wherever its owner put it.
            wanted = [PurePath(name).name for name in quant.files]
            matched_paths = [local_by_name[name] for name in wanted if name in local_by_name]
            quants.append(
                QuantDetail(
                    name=quant.name,
                    bytes_=quant.bytes_,
                    bpw=quant.bpw,
                    facts=quant.gguf_facts,
                    files=list(quant.files),
                )
            )
            local_paths.extend(matched_paths)
    return ModelDetail(model=model, quants=quants, local_paths=local_paths)
