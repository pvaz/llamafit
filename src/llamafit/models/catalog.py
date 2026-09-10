# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""The shape of a catalog entry: what LlamaFit knows about a model before it reads its files.

A :class:`CatalogModel` is everything curated by hand about one model: its identity,
licence, parameter counts, architecture, context length, capabilities, quality and the
repositories that publish its GGUF files. ``bytes_``, ``sha256`` and ``gguf_facts`` on a
:class:`Quant` are volatile: left empty here, they are filled later by reading the
Hugging Face API and the GGUF header itself, never guessed by hand.

Every model in this module forbids unknown fields, so a typo in a hand-written YAML
file is caught immediately instead of silently ignored, and the volatile fields carry
the bounds a real file has to satisfy: a size is never negative, bits per weight is a
finite number no wider than an unquantised weight, and a quant with checksums has one
per file. A curator's typo is caught when the catalog loads rather than when a memory
budget is computed from it.

The vocabularies live here too, and one relation between them: :data:`CAPABILITY_FOR_USE_CASE`
says what asking for a job asks of a model. It is beside :data:`UseCase` rather than beside
the scoring that reads it, so that whoever adds a seventh use case meets the question in the
same breath as the answer.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from llamafit.models.gguf import ByteSize, GgufFacts

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")

MAX_BPW = 32.0
"""The widest a weight could possibly be: an unquantised 32-bit float.

Anything above this is not a quantisation but a wrong number — a size that belongs to
some other file, or a parameter count that is wrong.
"""

BitsPerWeight = Annotated[float, Field(gt=0, le=MAX_BPW, allow_inf_nan=False)]
"""Bits per weight: above zero, no wider than :data:`MAX_BPW`, and finite.

``allow_inf_nan`` is set explicitly rather than left to the type: ``float`` accepts
infinities and NaN by default, and ``json.loads`` decodes ``Infinity`` and ``NaN``
without complaint, so the annotation alone would let one through.
"""

Capability = Literal[
    "coding", "thinking", "vision", "tools", "multilingual", "long-context", "embeddings", "audio"
]
UseCase = Literal["general", "coding", "reasoning", "chat", "multimodal", "embedding"]

CAPABILITY_FOR_USE_CASE: Mapping[str, Capability | None] = MappingProxyType(
    {
        "general": None,
        "coding": "coding",
        "reasoning": "thinking",
        "chat": None,
        "multimodal": "vision",
        "embedding": "embeddings",
    }
)
"""What asking for each job asks of a model, in the vocabulary of :data:`Capability`.

The two vocabularies above say different kinds of thing and this table is the bridge
between them. ``use_cases`` is what a model is *for*, which is a curator's emphasis:
writing *coding* first says the entry's baseline was set against coding work. ``capabilities``
is what a model can *do*, which is a fact about the weights. A request names a job, and the
job is only worth turning into a requirement where there is a fact behind it — so a coding
request asks for the coding capability, a reasoning one for thinking, a multimodal one for
vision, an embedding one for embeddings.

``general`` and ``chat`` require nothing, and the ``None`` is an answer rather than a gap.
They are the absence of a specialisation rather than a specialisation of their own: there
is no ability a model could lack that would make it unfit to hold a conversation, and a
model built for one job is a perfectly reasonable thing to ask about another. Anything that
runs can be talked to.

The table is exhaustive on purpose. A use case with no entry would quietly require nothing,
which is the most permissive answer available and the least likely to be the one anybody
meant, so the check below refuses to import this module until somebody has decided about
every member of :data:`UseCase` — and ``None`` counts as deciding.
"""

_UNDECIDED = set(get_args(UseCase)) ^ set(CAPABILITY_FOR_USE_CASE)
if _UNDECIDED:  # pragma: no cover - a developer error, caught the moment it is written
    raise RuntimeError(
        f"CAPABILITY_FOR_USE_CASE and UseCase disagree about {sorted(_UNDECIDED)}: "
        "every use case needs a decision about the capability it requires, and None is one."
    )

ArchClass = Literal["dense", "moe", "dense-hybrid", "moe-hybrid"]
Trust = Literal["official", "unsloth", "bartowski", "community"]
ExtraRole = Literal["mmproj", "mtp", "draft", "lora"]


class _Strict(BaseModel):
    """Base for every catalog model: unknown keys are an error, aliases are accepted."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class License(_Strict):
    """A model's licence: its identifier and where the actual text lives.

    Attributes:
        spdx: The SPDX licence identifier when the licence has one, for example
            ``Apache-2.0``; otherwise the vendor's own licence slug, for example
            ``qwen-community-1.0``. Not every open-weight licence is registered
            with SPDX, so this field is not guaranteed to resolve against that
            registry.
        url: A working URL to the licence text. Always present, regardless of
            whether ``spdx`` is a registered identifier.
    """

    spdx: str
    url: str


class Params(_Strict):
    """Parameter counts, in billions.

    Attributes:
        total_b: Total parameters, in billions, always above zero. A model with no
            parameters is not a model, and every figure derived from this one — bits
            per weight above all — is nonsense or a division by zero without it.
        active_b: Parameters active per token, in billions, always above zero. Equal to
            ``total_b`` for a dense model, smaller for a mixture-of-experts model; a
            model that activates nothing per token does not exist.
        ngram_table_b: Size of an auxiliary n-gram lookup table, in billions of entries,
            for models that stream one from disk instead of holding it in weights.
    """

    total_b: float = Field(gt=0)
    active_b: float = Field(gt=0)
    ngram_table_b: float | None = None

    @model_validator(mode="after")
    def _active_fits_in_total(self) -> Params:
        """Reject an entry whose active parameters exceed its total."""
        if self.active_b > self.total_b:
            raise ValueError("active_b must not exceed total_b")
        return self


class Architecture(_Strict):
    """The model's architecture family and how llama.cpp names it.

    Attributes:
        class_: The architecture class. Serialized as ``class``, its alias, since
            ``class`` is a Python keyword.
        gguf_arch: The ``general.architecture`` value llama.cpp expects in the GGUF
            header for this family.
        notes: Anything else worth recording about the architecture.
    """

    class_: ArchClass = Field(alias="class")
    gguf_arch: str
    notes: str | None = None


class Context(_Strict):
    """The context lengths a model supports.

    Attributes:
        native: Native context length, in tokens.
        extended: Extended context length, in tokens, when the model supports one
            beyond its native length.
        extended_method: The technique used to reach ``extended``, for example
            ``yarn`` or ``rope-scaling``.
    """

    native: int
    extended: int | None = None
    extended_method: str | None = None


class Benchmark(_Strict):
    """One published benchmark score.

    Attributes:
        name: The benchmark's name, for example ``MMLU-Pro``.
        score: The published score.
        source: A URL to where the score was published.
    """

    name: str
    score: float
    source: str


class Quality(_Strict):
    """A model's overall quality rating and the published scores behind it.

    Attributes:
        baseline: An editorial quality score from 0 to 100, used to rank models
            against each other. Not itself sourced from a single publication.
        benchmarks: Published benchmark scores, each with its own source.
    """

    baseline: int = Field(ge=0, le=100)
    benchmarks: list[Benchmark] = Field(default_factory=list)


class Sampling(_Strict):
    """Recommended sampling parameters for this model.

    Attributes:
        temp: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff.
        min_p: Minimum-probability sampling threshold.
        presence_penalty: Presence penalty.
        repeat_penalty: Repetition penalty.
    """

    temp: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None


class ChatTemplate(_Strict):
    """Quirks of a model's chat template that llama.cpp needs to know about.

    Attributes:
        reasoning_format: The chat template's reasoning/thinking format, if any.
        thinking_toggle: How thinking mode is toggled, if the template supports it.
    """

    reasoning_format: str | None = None
    thinking_toggle: str | None = None


class LlamaCppNeeds(_Strict):
    """Requirements this model places on the llama.cpp build that serves it.

    Attributes:
        min_build: The minimum llama.cpp build number known to support this model.
        kv_types_allowed: KV cache quantisation types known to work with this model.
        requires: Other requirements, keyed by name, for example a minimum CUDA
            version.
        quirks: Free-text notes about anything else llama.cpp needs to run this
            model correctly.
        lazy_tensors: Name prefixes of tensors llama.cpp can stream from disk
            rather than hold in memory, for example ``per_layer_token_embd``.
            Curated, and a property of the architecture rather than of a file:
            every quant of a model streams the same tables. The refresh passes
            these to the GGUF reader, which sums their bytes into
            :attr:`~llamafit.models.gguf.GgufFacts.bytes_lazy_tables` instead of
            counting them as resident weights. Empty for a model that streams
            nothing, which is most of them.
    """

    min_build: int | None = None
    kv_types_allowed: list[str] = Field(default_factory=list)
    requires: dict[str, str] = Field(default_factory=dict)
    quirks: list[str] = Field(default_factory=list)
    lazy_tensors: list[str] = Field(default_factory=list)


class Quant(_Strict):
    """One quantisation of a model, as a set of GGUF files.

    Attributes:
        name: The quant's name, for example ``UD-Q4_K_XL``.
        files: The GGUF file names that make up this quant, in shard order. Filled
            by the refresh command.
        bytes_: Total size of ``files``, in bytes, never negative. Serialized as
            ``bytes``, its alias, since ``bytes`` is a Python builtin. Filled by
            the refresh command.
        bpw: Bits per weight: finite, above zero and at most :data:`MAX_BPW`.
            Filled by the refresh command.
        sha256: One checksum per entry in ``files``, in the same order. Filled by
            the refresh command.
        gguf_facts: Architecture facts read from this quant's GGUF header. Filled
            by the refresh command.
    """

    name: str
    files: list[str] = Field(default_factory=list)
    bytes_: ByteSize | None = Field(default=None, alias="bytes")
    bpw: BitsPerWeight | None = None
    sha256: list[str] = Field(default_factory=list)
    gguf_facts: GgufFacts | None = None

    @model_validator(mode="after")
    def _one_checksum_per_file(self) -> Quant:
        """Reject a quant whose checksums cannot be paired with its files one to one.

        Each checksum belongs to exactly one file, in the same order, so a count
        that does not match means nobody can tell which checksum covers which file.
        Either list may be empty, since a quant is curated before it is refreshed.
        """
        if self.files and self.sha256 and len(self.files) != len(self.sha256):
            raise ValueError("sha256 needs one checksum per entry in files")
        return self


class Extra(_Strict):
    """An auxiliary file that accompanies a model, such as a vision projector.

    Attributes:
        role: What this file is for.
        file: The file name.
        bytes_: Size in bytes, never negative. Serialized as ``bytes``, its alias.
            Filled by the refresh command.
        sha256: Checksum of the file. Filled by the refresh command.
    """

    role: ExtraRole
    file: str
    bytes_: ByteSize | None = Field(default=None, alias="bytes")
    sha256: str | None = None


class ModelSource(_Strict):
    """Where a model's files come from: a Hugging Face repository or a local path.

    Attributes:
        repo: The Hugging Face repository, for a ``gguf`` source.
        kind: Whether this source is a Hugging Face repository or a local file.
        trust: Who publishes this source's files.
        path: A local file path, for a ``local`` source. It says nothing about a
            ``gguf`` source and must be left unset on one; the directory inside a
            repository is ``repo_path``, which is a different thing with a
            different name so nobody has to hold the source kind in their head to
            know which definition applies.
        repo_path: A directory inside the repository, for a ``gguf`` source, or
            ``None`` for the whole repository. When set, only files under it are
            matched against this source's quants and extras. A repository
            routinely publishes the same quant name twice, a plain build and an
            importance-matrix build side by side in two directories, and the
            matcher refuses to guess between them; naming the directory is how a
            curator says which one they meant. Left unset, every file in the
            repository is considered.
        quants: The quantisations this source publishes.
        extras: Auxiliary files this source publishes.
    """

    repo: str | None = None
    kind: Literal["gguf", "local"] = "gguf"
    trust: Trust = "community"
    path: str | None = None
    repo_path: str | None = None
    quants: list[Quant] = Field(default_factory=list)
    extras: list[Extra] = Field(default_factory=list)

    @model_validator(mode="after")
    def _repo_or_path_matches_kind(self) -> ModelSource:
        """Reject a source whose location fields do not match the kind it declares.

        A ``gguf`` source needs a repository and may narrow it with ``repo_path``; a
        ``local`` source needs a file path and can narrow nothing. Each field is
        refused on the kind it means nothing to, so a curator who reaches for the
        wrong one is told at load time rather than left wondering why it had no
        effect.
        """
        if self.kind == "gguf" and not self.repo:
            raise ValueError("a gguf source needs a repo")
        if self.kind == "gguf" and self.path:
            raise ValueError("a gguf source has no local path; use repo_path for a directory")
        if self.kind == "local" and not self.path:
            raise ValueError("a local source needs a path")
        if self.kind == "local" and self.repo_path:
            raise ValueError("a local source has no repo_path")
        return self


class Measured(_Strict):
    """One measurement of this model's speed and memory use on a real machine.

    Attributes:
        profile: A short label for what was run, for example ``llama-bench tg128``.
        quant: The quant that was measured.
        gen_tps: Generation speed, in tokens per second.
        pp_tps: Prompt processing speed, in tokens per second.
        context: Context length in use during the measurement, in tokens.
        flags: The llama.cpp flags used, quoted exactly and in full -- not the delta from
            a base line written down somewhere else. The speed estimator reads these to
            decide whether a run describes a given placement, and a run that never names
            ``-ngl`` describes none, so an abbreviated entry costs its own row the one
            label it could have earned.
        llama_cpp_build: The llama.cpp build number used.
        peak_vram_gb: Peak VRAM used, in gigabytes.
        date: The date of the measurement.
        source: Where the measurement is recorded.
    """

    profile: str
    quant: str
    gen_tps: float | None = None
    pp_tps: float | None = None
    context: int | None = None
    flags: str | None = None
    llama_cpp_build: int | None = None
    peak_vram_gb: float | None = None
    date: datetime.date | None = None
    source: str | None = None


class CatalogModel(_Strict):
    """Everything LlamaFit knows about one model before it reads its files.

    Attributes:
        id: A stable slug identifying this model, for example ``qwen3-coder-next``.
        name: The model's display name.
        vendor: Who trained the model.
        family: The model family this entry belongs to.
        release_date: When the model was released.
        license: The model's licence.
        params: Parameter counts.
        architecture: Architecture facts about the family.
        context: Supported context lengths.
        capabilities: What the model can do.
        use_cases: What the model is good for.
        quality: The model's quality rating.
        sampling: Recommended sampling parameters.
        chat_template: Chat template quirks.
        llama_cpp: Requirements this model places on llama.cpp.
        sources: Where this model's files come from.
        measured: Real measurements of this model's speed and memory use.
    """

    id: str
    name: str
    vendor: str
    family: str
    release_date: datetime.date
    license: License
    params: Params
    architecture: Architecture
    context: Context
    capabilities: list[Capability]
    use_cases: list[UseCase]
    quality: Quality
    sampling: Sampling = Field(default_factory=Sampling)
    chat_template: ChatTemplate = Field(default_factory=ChatTemplate)
    llama_cpp: LlamaCppNeeds = Field(default_factory=LlamaCppNeeds)
    sources: list[ModelSource]
    measured: list[Measured] = Field(default_factory=list)

    @model_validator(mode="after")
    def _id_is_a_slug(self) -> CatalogModel:
        """Reject an id that is not a lowercase slug."""
        if not _ID_RE.match(self.id):
            raise ValueError(f"id must match {_ID_RE.pattern!r}: {self.id!r}")
        return self

    @model_validator(mode="after")
    def _has_a_use_case_and_a_source(self) -> CatalogModel:
        """Reject an entry with no use case or no source."""
        if not self.use_cases:
            raise ValueError("use_cases must not be empty")
        if not self.sources:
            raise ValueError("sources must not be empty")
        return self


class Catalog(_Strict):
    """A collection of catalog entries.

    Attributes:
        models: Every model in the catalog.
    """

    model_config = ConfigDict(extra="forbid", ignored_types=(cached_property,))

    models: list[CatalogModel]

    @cached_property
    def by_id(self) -> dict[str, CatalogModel]:
        """Every model keyed by its identifier."""
        return {m.id: m for m in self.models}
