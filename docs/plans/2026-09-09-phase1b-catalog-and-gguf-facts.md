# LlamaFit Phase 1B — Model Catalog and GGUF Facts Implementation Plan

> Implementation plan: one task per section, each with its files, interfaces, tests, steps and commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give LlamaFit a curated catalog of open-weight models and the ability to read exact architecture facts out of GGUF files without downloading them, so that `llamafit list`, `search`, `info` and `catalog validate|refresh|show` work and phase 1C has real numbers to compute budgets from.

**Architecture:** A pydantic model tree describes a catalog entry; YAML files under `src/llamafit/data/catalog/` hold the curated data, one file per model family, and a user file can override or extend them. A byte-level GGUF header reader works over an injectable byte source, so the same parser serves a local file and a remote file read through HTTP range requests. Derived facts are computed from the parsed header and cached. A refresh command fills the volatile fields from the Hugging Face API and rewrites the YAML deterministically.

**Tech Stack:** Python 3.10+, pydantic 2, PyYAML, httpx, Typer, Rich, pytest, ruff, mypy.

**Spec:** `docs/specs/2026-09-09-llamafit-design.md`, sections 6 (model catalog) and 7 (GGUF facts); section 13.1 for the commands; section 14 for where the cache and the custom file live.

## Global Constraints

- Python floor 3.10; CI matrix Python 3.10 and 3.13 on Ubuntu, macOS and Windows; coverage at or above 85 percent, enforced by `fail_under` in `pyproject.toml`.
- `pyyaml` joins the runtime dependencies in this phase. Add it to `[project] dependencies` and to the table in `docs/development.md`, moving it out of the planned list. No other new dependency without a row in that table.
- Style: ruff (line length 100, isort rules, Google docstrings), mypy strict, type hints everywhere, no bare `except`, no `print` outside `cli/`, `tui/`, `web/`. Never add a `# noqa` for a rule that is not selected in `pyproject.toml`; `RUF100` fails on an unused one. `E402` and `I001` are selected, `BLE`, `SLF` and `S` are not.
- Every failure a user can hit raises a `LlamaFitError` subclass with `message`, `hint` and `command`; `CatalogError` and `NetworkError` already exist for this phase.
- Nothing that reads a catalog file or a GGUF header may raise on malformed input: a bad file becomes a `CatalogError` naming the file and the field, and a bad header becomes a `CatalogError` naming the URL or path.
- Every number that reaches the user carries its provenance. Facts read from a GGUF header are exact; anything derived by a family rule instead is marked so.
- No network access in any test that is not marked `hardware`. Hugging Face and GGUF reads go through injectable clients with recorded responses.
- Commit after every task with a plain conventional-commit message and **no trailer lines of any kind**. If your editor or tooling appends co-author or session trailers, strip them with `git commit --amend` before reporting.
- Work from the repository root. Run every tool through the virtual environment interpreter: `.venv\Scripts\python.exe -m pytest`, `... -m ruff`, `... -m mypy`.
- `main` is protected: work happens on a branch and lands through a pull request with the six CI jobs green.

---

## File structure for this plan

| File | Responsibility |
|---|---|
| `src/llamafit/models/catalog.py` | the pydantic tree for a catalog entry and everything under it |
| `src/llamafit/models/gguf.py` | `GgufFacts` and the raw `GgufHeader` |
| `src/llamafit/gguf/types.py` | GGUF value types, GGML tensor types and their block sizes |
| `src/llamafit/gguf/source.py` | `ByteSource` protocol, `LocalSource`, `HttpRangeSource`, `FakeSource` |
| `src/llamafit/gguf/reader.py` | parse magic, version, counts, metadata and tensor infos from a `ByteSource` |
| `src/llamafit/gguf/facts.py` | derive `GgufFacts` from a `GgufHeader` |
| `src/llamafit/gguf/cache.py` | cache parsed headers under the cache directory |
| `src/llamafit/gguf/__init__.py` | `read_header`, `read_facts` |
| `src/llamafit/catalog/loader.py` | read the bundled YAML, merge the custom file, check ids |
| `src/llamafit/catalog/validate.py` | validate files and report every problem with its location |
| `src/llamafit/catalog/hf.py` | the Hugging Face metadata client |
| `src/llamafit/catalog/refresh.py` | fill volatile fields and rewrite YAML deterministically |
| `src/llamafit/catalog/__init__.py` | `load_catalog` |
| `src/llamafit/data/catalog/*.yaml` | the seed entries |
| `src/llamafit/data/schema/catalog.schema.json` | the generated JSON schema |
| `src/llamafit/services/catalog.py` | `list_models`, `search_models`, `model_info` |
| `src/llamafit/cli/catalog_cmd.py` | `list`, `search`, `info`, `catalog validate|refresh|show` |
| `scripts/gen_models_md.py` | regenerate `MODELS.md` from the catalog |
| `tests/unit/test_*.py`, `tests/fixtures/gguf/` | tests and recorded headers |

---

### Task 1: Catalog data models

**Files:**
- Create: `src/llamafit/models/catalog.py`
- Modify: `src/llamafit/models/__init__.py`
- Test: `tests/unit/test_models_catalog.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, all pydantic `BaseModel` with `model_config = ConfigDict(extra="forbid")` so an unknown key in a hand-written YAML file is an error rather than silence:
  - `Capability = Literal["coding", "thinking", "vision", "tools", "multilingual", "long-context", "embeddings", "audio"]`
  - `UseCase = Literal["general", "coding", "reasoning", "chat", "multimodal", "embedding"]`
  - `ArchClass = Literal["dense", "moe", "dense-hybrid", "moe-hybrid"]`
  - `Trust = Literal["official", "unsloth", "bartowski", "community"]`
  - `ExtraRole = Literal["mmproj", "mtp", "draft", "lora"]`
  - `License(spdx: str, url: str)`
  - `Params(total_b: float, active_b: float, ngram_table_b: float | None = None)` with a validator that `active_b <= total_b`
  - `Architecture(class_: ArchClass, gguf_arch: str, notes: str | None = None)` — the field is named `class_` in Python and aliased to `class` in YAML with `Field(alias="class")`, and the model sets `populate_by_name=True`
  - `Context(native: int, extended: int | None = None, extended_method: str | None = None)`
  - `Benchmark(name: str, score: float, source: str)`
  - `Quality(baseline: int, benchmarks: list[Benchmark] = [])` with `0 <= baseline <= 100`
  - `Sampling(temp: float | None = None, top_p: float | None = None, top_k: int | None = None, min_p: float | None = None, presence_penalty: float | None = None, repeat_penalty: float | None = None)`
  - `ChatTemplate(reasoning_format: str | None = None, thinking_toggle: str | None = None)`
  - `LlamaCppNeeds(min_build: int | None = None, kv_types_allowed: list[str] = [], requires: dict[str, str] = {}, quirks: list[str] = [])`
  - `Quant(name: str, files: list[str] = [], bytes_: int | None = None, bpw: float | None = None, sha256: list[str] = [], gguf_facts: GgufFacts | None = None)` — `bytes_` is aliased to `bytes` in YAML for the same reason as `class_`
  - `Extra(role: ExtraRole, file: str, bytes_: int | None = None, sha256: str | None = None)`
  - `Source(repo: str | None = None, kind: Literal["gguf", "local"] = "gguf", trust: Trust = "community", path: str | None = None, quants: list[Quant] = [], extras: list[Extra] = [])` with a validator that a `gguf` source has a `repo` and a `local` source has a `path`
  - `Measured(profile: str, quant: str, gen_tps: float | None = None, pp_tps: float | None = None, context: int | None = None, flags: str | None = None, llama_cpp_build: int | None = None, peak_vram_gb: float | None = None, date: date | None = None, source: str | None = None)`
  - `CatalogModel(id: str, name: str, vendor: str, family: str, release_date: date, license: License, params: Params, architecture: Architecture, context: Context, capabilities: list[Capability], use_cases: list[UseCase], quality: Quality, sampling: Sampling = Sampling(), chat_template: ChatTemplate = ChatTemplate(), llama_cpp: LlamaCppNeeds = LlamaCppNeeds(), sources: list[Source], measured: list[Measured] = [])` with validators that `id` matches `^[a-z0-9][a-z0-9.\-]*$`, that `use_cases` is non-empty, and that `sources` is non-empty
  - `Catalog(models: list[CatalogModel])` with `by_id: dict[str, CatalogModel]` as a cached property

`GgufFacts` is defined in Task 2 and imported here; write Task 2 first if you are working straight through, or use a forward reference and `model_rebuild()`.

- [ ] **Step 1: Write the failing test**

`tests/unit/test_models_catalog.py`:

```python
from datetime import date

import pytest
from pydantic import ValidationError

from llamafit.models.catalog import Architecture, CatalogModel, License, Params, Quant, Source


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
        "sources": [Source(repo="unsloth/Qwen3-Coder-Next-GGUF", trust="unsloth",
                           quants=[Quant(name="UD-Q4_K_XL")])],
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
        Source(kind="gguf", quants=[Quant(name="Q4_K_M")])
    with pytest.raises(ValidationError):
        Source(kind="local", quants=[Quant(name="Q4_K_M")])
    assert Source(kind="local", path="D:/models/x.gguf", quants=[Quant(name="Q4_K_M")]).path


def test_a_model_needs_at_least_one_source_and_one_use_case() -> None:
    with pytest.raises(ValidationError):
        minimal(sources=[])
    with pytest.raises(ValidationError):
        minimal(use_cases=[])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_models_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamafit.models.catalog'`.

- [ ] **Step 3: Implement `models/catalog.py`**

Write the tree exactly as the Interfaces block lists it. Points that need care:

```python
"""The shape of a catalog entry: what LlamaFit knows about a model before it reads its files."""

from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from llamafit.models.gguf import GgufFacts

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")


class _Strict(BaseModel):
    """Base for every catalog model: unknown keys are an error, aliases are accepted."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
```

Every model in this file inherits `_Strict`. `Architecture` declares `class_: ArchClass = Field(alias="class")` and `Quant`/`Extra` declare `bytes_: int | None = Field(default=None, alias="bytes")`. `Params` carries

```python
    @model_validator(mode="after")
    def _active_fits_in_total(self) -> Params:
        """Reject an entry whose active parameters exceed its total."""
        if self.active_b > self.total_b:
            raise ValueError("active_b must not exceed total_b")
        return self
```

and `Source` carries the equivalent check for `repo` and `path`. `CatalogModel` validates the id with `_ID_RE` and rejects empty `use_cases` and `sources`. `Catalog` exposes

```python
    @cached_property
    def by_id(self) -> dict[str, CatalogModel]:
        """Every model keyed by its identifier."""
        return {m.id: m for m in self.models}
```

with `model_config = ConfigDict(extra="forbid", ignored_types=(cached_property,))`.

Re-export every public name from `src/llamafit/models/__init__.py`, keeping the existing `__all__` sorted.

- [ ] **Step 4: Run the tests, lint and type-check**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_models_catalog.py -v && .venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m mypy`
Expected: PASS, clean, clean.

- [ ] **Step 5: Commit**

```bash
git add src/llamafit/models tests/unit/test_models_catalog.py
git commit -m "feat: catalog data models"
```

---

### Task 2: GGUF types and the header reader

**Files:**
- Create: `src/llamafit/models/gguf.py`, `src/llamafit/gguf/__init__.py`, `src/llamafit/gguf/types.py`, `src/llamafit/gguf/source.py`, `src/llamafit/gguf/reader.py`
- Test: `tests/unit/test_gguf_reader.py`, `tests/fixtures/gguf_builder.py`

**Interfaces:**
- Produces:
  - `models/gguf.py`: `TensorInfo(name: str, dims: list[int], type: int, offset: int, bytes_: int)` (alias `bytes`), `GgufHeader(version: int, tensor_count: int, alignment: int, metadata: dict[str, object], tensors: list[TensorInfo], header_bytes: int)`, and `GgufFacts` (filled in Task 4; declare it now with every field optional so Task 1 can import it).
  - `gguf/types.py`: `class ValueType(IntEnum)` with the thirteen GGUF metadata types; `GGML_TYPES: dict[int, tuple[str, int, int]]` mapping a tensor type id to its name, block size and bytes per block; `tensor_bytes(dims: Sequence[int], type_id: int) -> int`.
  - `gguf/source.py`: `class ByteSource(Protocol)` with `read(self, offset: int, length: int) -> bytes` and `size(self) -> int | None`; `LocalSource(path: Path)`; `HttpRangeSource(url: str, client: httpx.Client | None = None, chunk: int = 1 << 20)` which fetches in chunks and keeps what it has read; `FakeSource(data: bytes)` recording `reads: list[tuple[int, int]]`.
  - `gguf/reader.py`: `read_header(source: ByteSource) -> GgufHeader`, raising `CatalogError` with the offending offset for a bad magic, an unsupported version, a truncated read or an unknown value type.

- [ ] **Step 1: Write the fixture builder and the failing test**

`tests/fixtures/gguf_builder.py` builds a valid header in memory, so the reader is tested against bytes rather than a downloaded file:

```python
"""Build a minimal but valid GGUF header in memory for tests."""

from __future__ import annotations

import struct

MAGIC = b"GGUF"


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value_type: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", value_type) + payload


def uint32(key: str, value: int) -> bytes:
    """A UINT32 metadata entry."""
    return _kv(key, 4, struct.pack("<I", value))


def string(key: str, value: str) -> bytes:
    """A STRING metadata entry."""
    return _kv(key, 8, _string(value))


def string_array(key: str, values: list[str]) -> bytes:
    """An ARRAY of STRING metadata entry, used for the tokenizer vocabulary."""
    body = struct.pack("<I", 8) + struct.pack("<Q", len(values))
    for value in values:
        body += _string(value)
    return _kv(key, 9, body)


def tensor(name: str, dims: list[int], type_id: int, offset: int) -> bytes:
    """One tensor info record."""
    out = _string(name) + struct.pack("<I", len(dims))
    for dim in dims:
        out += struct.pack("<Q", dim)
    return out + struct.pack("<I", type_id) + struct.pack("<Q", offset)


def build(metadata: list[bytes], tensors: list[bytes], version: int = 3) -> bytes:
    """Assemble a header from encoded metadata and tensor records."""
    head = MAGIC + struct.pack("<I", version)
    head += struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(metadata))
    return head + b"".join(metadata) + b"".join(tensors)
```

`tests/unit/test_gguf_reader.py`:

```python
import pytest

from llamafit.errors import CatalogError
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from llamafit.gguf.types import tensor_bytes
from tests.fixtures import gguf_builder as b


def sample() -> bytes:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 32),
        b.uint32("llama.embedding_length", 4096),
        b.uint32("llama.attention.head_count", 32),
        b.uint32("llama.attention.head_count_kv", 8),
        b.uint32("general.alignment", 32),
        b.string_array("tokenizer.ggml.tokens", ["a", "b", "c"]),
    ]
    tensors = [
        b.tensor("token_embd.weight", [4096, 128256], 12, 0),
        b.tensor("blk.0.attn_q.weight", [4096, 4096], 12, 1024),
        b.tensor("blk.0.ffn_down_exps.weight", [4096, 14336, 8], 12, 2048),
    ]
    return b.build(metadata, tensors)


def test_reads_counts_metadata_and_tensors() -> None:
    header = read_header(FakeSource(sample()))
    assert header.version == 3
    assert header.tensor_count == 3
    assert header.alignment == 32
    assert header.metadata["general.architecture"] == "llama"
    assert header.metadata["llama.block_count"] == 32
    assert header.metadata["tokenizer.ggml.tokens"] == ["a", "b", "c"]
    assert [t.name for t in header.tensors] == [
        "token_embd.weight", "blk.0.attn_q.weight", "blk.0.ffn_down_exps.weight",
    ]
    assert header.tensors[0].dims == [4096, 128256]


def test_computes_tensor_sizes_from_dimensions_and_type() -> None:
    header = read_header(FakeSource(sample()))
    # Q4_K is type 12: 256 elements per block, 144 bytes per block.
    assert header.tensors[1].bytes_ == (4096 * 4096 // 256) * 144
    assert tensor_bytes([32], 0) == 128          # F32, one byte-per-element block of 4
    assert tensor_bytes([256], 8) == 8 * 34      # Q8_0, 32 per block, 34 bytes


def test_reads_only_what_the_header_needs() -> None:
    source = FakeSource(sample() + b"\x00" * 5_000_000)
    header = read_header(source)
    assert header.header_bytes < 4096
    assert max(offset + length for offset, length in source.reads) < 1 << 20


def test_a_bad_magic_is_a_catalog_error() -> None:
    with pytest.raises(CatalogError, match="not a GGUF file"):
        read_header(FakeSource(b"XXXX" + sample()[4:]))


def test_an_unsupported_version_is_a_catalog_error() -> None:
    with pytest.raises(CatalogError, match="version 99"):
        read_header(FakeSource(b.build([], [], version=99)))


def test_a_truncated_header_is_a_catalog_error_not_a_crash() -> None:
    with pytest.raises(CatalogError, match="truncated"):
        read_header(FakeSource(sample()[:40]))


def test_an_unknown_value_type_is_a_catalog_error() -> None:
    bad = b.build([b._kv("weird", 99, b"\x00" * 4)], [])
    with pytest.raises(CatalogError, match="value type 99"):
        read_header(FakeSource(bad))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_gguf_reader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamafit.gguf'`.

- [ ] **Step 3: Implement `gguf/types.py`**

```python
"""GGUF metadata value types and GGML tensor types.

The tensor table gives, per type id, its name, how many elements share one
quantisation block, and how many bytes that block occupies. The values follow
ggml's own type traits table; Task 3 checks them against real files, because a
wrong row would silently mis-size every model that uses that type.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum


class ValueType(IntEnum):
    """The metadata value types a GGUF header can carry."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# type id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
}


def type_name(type_id: int) -> str:
    """The GGML name of a tensor type, or ``unknown(<id>)``."""
    entry = GGML_TYPES.get(type_id)
    return entry[0] if entry else f"unknown({type_id})"


def tensor_bytes(dims: Sequence[int], type_id: int) -> int:
    """Bytes a tensor of these dimensions occupies in this type.

    Raises:
        KeyError: If the type id is not in the table.
    """
    _, block_elements, block_bytes = GGML_TYPES[type_id]
    elements = 1
    for dim in dims:
        elements *= dim
    return elements // block_elements * block_bytes
```

- [ ] **Step 4: Implement `gguf/source.py`**

```python
"""Where header bytes come from: a local file, an HTTP range request, or a test.

A GGUF file can be a hundred gigabytes; its header is a few hundred kilobytes at
the front. Every reader here fetches only the ranges the parser asks for, so a
remote header costs one or two requests rather than a download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import httpx

from llamafit.errors import CatalogError, NetworkError


class ByteSource(Protocol):
    """A random-access source of bytes."""

    def read(self, offset: int, length: int) -> bytes:
        """Return exactly ``length`` bytes from ``offset``, or fewer at the end."""
        ...

    def size(self) -> int | None:
        """The total size when it is known."""
        ...


@dataclass
class FakeSource:
    """An in-memory source that records every range it was asked for."""

    data: bytes
    reads: list[tuple[int, int]] = field(default_factory=list)

    def read(self, offset: int, length: int) -> bytes:
        """Return the slice and record the request."""
        self.reads.append((offset, length))
        return self.data[offset : offset + length]

    def size(self) -> int | None:
        """The length of the buffer."""
        return len(self.data)
```

`LocalSource` opens the file lazily, keeps the handle, and turns `OSError` into a `CatalogError` naming the path. `HttpRangeSource` keeps a `bytearray` of what it has fetched plus the offset it starts at, issues `Range: bytes=start-end` requests of at least `chunk` bytes so a sequential parser costs one request, turns a non-206 response or an `httpx.HTTPError` into a `NetworkError` naming the URL, and reads the total size from `Content-Range`.

- [ ] **Step 5: Implement `gguf/reader.py`**

A small cursor over the source drives the parse:

```python
"""Parse a GGUF header: magic, version, counts, metadata and tensor infos."""

from __future__ import annotations

import struct

from llamafit.errors import CatalogError
from llamafit.gguf.source import ByteSource
from llamafit.gguf.types import ValueType, tensor_bytes, type_name
from llamafit.models.gguf import GgufHeader, TensorInfo

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = (2, 3)
_DEFAULT_ALIGNMENT = 32
_SCALARS: dict[int, tuple[str, int]] = {
    ValueType.UINT8: ("<B", 1), ValueType.INT8: ("<b", 1),
    ValueType.UINT16: ("<H", 2), ValueType.INT16: ("<h", 2),
    ValueType.UINT32: ("<I", 4), ValueType.INT32: ("<i", 4),
    ValueType.FLOAT32: ("<f", 4), ValueType.BOOL: ("<?", 1),
    ValueType.UINT64: ("<Q", 8), ValueType.INT64: ("<q", 8),
    ValueType.FLOAT64: ("<d", 8),
}


class _Cursor:
    """Sequential reader over a ByteSource with a buffered window."""

    def __init__(self, source: ByteSource, window: int = 1 << 20):
        self._source = source
        self._window = window
        self._buffer = b""
        self._buffer_start = 0
        self.offset = 0

    def take(self, length: int) -> bytes:
        """Return the next ``length`` bytes, refilling the window as needed."""
        end = self.offset + length
        if not (self._buffer_start <= self.offset and end <= self._buffer_start + len(self._buffer)):
            fetch = max(length, self._window)
            self._buffer = self._source.read(self.offset, fetch)
            self._buffer_start = self.offset
        start = self.offset - self._buffer_start
        chunk = self._buffer[start : start + length]
        if len(chunk) < length:
            raise CatalogError(
                f"truncated GGUF header: wanted {length} bytes at offset {self.offset}, "
                f"got {len(chunk)}",
                hint="The file may still be downloading, or the URL may not be a GGUF file.",
            )
        self.offset = end
        return chunk
```

The window default must be strictly below one megabyte, because the test asserts the reader never reaches that far into the file; use `(1 << 20) - 1`.

`_scalar`, `_string` and `_value` build on `take`; `_value` dispatches on the type id, handles `ARRAY` by reading its element type and count and recursing, and raises `CatalogError(f"unknown GGUF value type {type_id} at offset ...")` for anything else. `read_header` checks the magic and version first, reads the two counts, then the metadata map, then the tensor infos, computing `bytes_` for each with `tensor_bytes` and falling back to `0` with the tensor recorded in `header.metadata["_unknown_tensor_types"]` when the type id is missing from the table, so one unknown type does not fail the whole read. `alignment` comes from `general.alignment` when present, else `_DEFAULT_ALIGNMENT`. `header_bytes` is the cursor offset when the parse finishes.

- [ ] **Step 6: Run the tests, lint and type-check**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_gguf_reader.py -v && .venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m mypy`
Expected: PASS, clean, clean.

- [ ] **Step 7: Commit**

```bash
git add src/llamafit/gguf src/llamafit/models/gguf.py tests/unit/test_gguf_reader.py tests/fixtures/gguf_builder.py
git commit -m "feat: GGUF header reader over an injectable byte source"
```

---

### Task 3: Read a real GGUF file and prove the type table

**Files:**
- Create: `tests/unit/test_gguf_real_file.py`
- Modify: `src/llamafit/gguf/types.py` only if the check below finds a wrong row

**Interfaces:**
- Consumes: `read_header`, `LocalSource`, `tensor_bytes`.
- Produces: no new interface; this task's product is confidence that the block-size table is right.

A wrong row in `GGML_TYPES` would mis-size every model using that type, and no unit test built from synthetic bytes can catch it. A real file can: the tensor data starts at the header end rounded up to the alignment, and the sum of every tensor's size plus that offset must equal the file size. That single identity checks the whole table against the types the file actually uses.

This machine has real GGUF files under `D:\llama.cpp\models`, including a small one, `Qwen3-0.6B\Qwen3-0.6B-Q8_0.gguf` at about 805 MB. The test is marked `hardware` so CI skips it, and it skips itself when the file is absent.

- [ ] **Step 1: Write the test**

```python
"""Check the tensor-size table against real files: the arithmetic must close exactly."""

from pathlib import Path

import pytest

from llamafit.gguf.reader import read_header
from llamafit.gguf.source import LocalSource

CANDIDATES = [
    Path("D:/llama.cpp/models/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf"),
    Path("D:/llama.cpp/models/Qwen3-Coder-Next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf"),
]


@pytest.mark.hardware
@pytest.mark.parametrize("path", CANDIDATES, ids=lambda p: p.name)
def test_tensor_sizes_account_for_the_whole_file(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    header = read_header(LocalSource(path))
    data_start = (header.header_bytes + header.alignment - 1) // header.alignment * header.alignment
    total = data_start + sum(t.bytes_ for t in header.tensors)
    actual = path.stat().st_size
    assert total == actual, (
        f"tensor sizes do not account for the file: computed {total}, actual {actual}, "
        f"difference {actual - total}. A row in GGML_TYPES is wrong for one of the types "
        f"used here: {sorted({t.type for t in header.tensors})}"
    )


@pytest.mark.hardware
def test_a_split_model_reports_its_own_shard_only() -> None:
    path = CANDIDATES[1]
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    header = read_header(LocalSource(path))
    assert header.tensor_count == len(header.tensors)
    assert header.metadata.get("general.architecture")
```

- [ ] **Step 2: Run it on this machine**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_gguf_real_file.py -v -m hardware`
Expected: PASS. If the first test fails, the difference in the message tells you how far off the table is; find the type ids the file uses, check those rows against ggml's `type_traits` table in the llama.cpp source, correct them, and re-run until the arithmetic closes exactly. Record in your report which rows you corrected, if any.

- [ ] **Step 3: Confirm the whole suite still passes and commit**

Run: `.venv\Scripts\python.exe -m pytest -q -m "not hardware" && .venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m mypy`

```bash
git add tests/unit/test_gguf_real_file.py src/llamafit/gguf/types.py
git commit -m "test: check the tensor-size table against real GGUF files"
```

---

### Task 4: Derived facts

**Files:**
- Modify: `src/llamafit/models/gguf.py` (fill in `GgufFacts`)
- Create: `src/llamafit/gguf/facts.py`
- Test: `tests/unit/test_gguf_facts.py`

**Interfaces:**
- Consumes: `GgufHeader`, `TensorInfo`.
- Produces:
  - `GgufFacts` with `arch: str`, `n_layer: int | None`, `n_embd: int | None`, `n_vocab: int | None`, `n_head: int | None`, `n_head_kv: int | None`, `head_dim: int | None`, `attention_layers: int | None`, `attention_layers_source: Literal["tensors", "all-layers", "unknown"]`, `n_expert: int | None`, `n_expert_used: int | None`, `has_shared_experts: bool`, `bytes_expert_weights: int`, `bytes_attention_weights: int`, `bytes_output_head: int`, `bytes_token_embd: int`, `bytes_lazy_tables: int`, `bytes_total: int`, `kv_bytes_per_token_f16: int | None`, `recurrent_state_bytes: int | None`.
  - `derive_facts(header: GgufHeader, *, lazy_tensor_names: Sequence[str] = ()) -> GgufFacts`.
  - `kv_bytes_per_token(facts: GgufFacts, kv_type: str) -> int | None` where `kv_type` is one of `f16`, `q8_0`, `q4_0`.

**How `attention_layers` is determined, and why it improves on the specification.** Section 7.2 says the full-attention layer set comes from "the architecture's interval or layer-type array, else the family rule in the catalog". The tensors say it directly and exactly: a hybrid model's linear-attention layers carry `ssm_*` tensors and no `attn_k`, while its full-attention layers carry `attn_k` or `attn_v`. Count the distinct `blk.N` prefixes that own an `attn_k.weight` or `attn_v.weight`. That needs no family rule and cannot drift when a vendor changes the interval. When no block carries either tensor, fall back to `n_layer` and record `attention_layers_source="all-layers"`. Update section 7.2 of the specification to describe this, since the code is the better rule.

- [ ] **Step 1: Write the failing test**

```python
from llamafit.gguf.facts import derive_facts, kv_bytes_per_token
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from tests.fixtures import gguf_builder as b


def dense_header() -> bytes:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 2),
        b.uint32("llama.embedding_length", 4096),
        b.uint32("llama.attention.head_count", 32),
        b.uint32("llama.attention.head_count_kv", 8),
        b.uint32("llama.attention.key_length", 128),
        b.string_array("tokenizer.ggml.tokens", ["a"] * 32000),
    ]
    tensors = [
        b.tensor("token_embd.weight", [4096, 32000], 8, 0),
        b.tensor("output.weight", [4096, 32000], 8, 1),
        b.tensor("blk.0.attn_k.weight", [4096, 1024], 8, 2),
        b.tensor("blk.0.ffn_down.weight", [11008, 4096], 8, 3),
        b.tensor("blk.1.attn_k.weight", [4096, 1024], 8, 4),
    ]
    return b.build(metadata, tensors)


def hybrid_moe_header() -> bytes:
    metadata = [
        b.string("general.architecture", "qwen3next"),
        b.uint32("qwen3next.block_count", 4),
        b.uint32("qwen3next.embedding_length", 2560),
        b.uint32("qwen3next.attention.head_count", 24),
        b.uint32("qwen3next.attention.head_count_kv", 2),
        b.uint32("qwen3next.attention.key_length", 128),
        b.uint32("qwen3next.expert_count", 512),
        b.uint32("qwen3next.expert_used_count", 10),
        b.string_array("tokenizer.ggml.tokens", ["a"] * 1024),
    ]
    tensors = [
        b.tensor("token_embd.weight", [2560, 1024], 8, 0),
        b.tensor("per_layer_token_embd.weight", [2560, 1024, 4], 8, 1),
        # three linear-attention layers and one full-attention layer
        b.tensor("blk.0.ssm_out.weight", [2560, 2560], 8, 2),
        b.tensor("blk.0.ffn_down_exps.weight", [512, 2560, 512], 8, 3),
        b.tensor("blk.1.ssm_out.weight", [2560, 2560], 8, 4),
        b.tensor("blk.2.ssm_out.weight", [2560, 2560], 8, 5),
        b.tensor("blk.3.attn_k.weight", [2560, 256], 8, 6),
        b.tensor("blk.3.ffn_down_shexp.weight", [512, 2560], 8, 7),
    ]
    return b.build(metadata, tensors)


def test_dense_model_facts() -> None:
    facts = derive_facts(read_header(FakeSource(dense_header())))
    assert facts.arch == "llama"
    assert (facts.n_layer, facts.n_embd, facts.n_head, facts.n_head_kv) == (2, 4096, 32, 8)
    assert facts.head_dim == 128
    assert facts.n_vocab == 32000
    assert facts.attention_layers == 2
    assert facts.attention_layers_source == "tensors"
    assert facts.n_expert is None and facts.has_shared_experts is False
    assert facts.bytes_expert_weights == 0
    assert facts.bytes_token_embd > 0 and facts.bytes_output_head > 0


def test_hybrid_moe_counts_only_the_full_attention_layers() -> None:
    facts = derive_facts(read_header(FakeSource(hybrid_moe_header())),
                         lazy_tensor_names=["per_layer_token_embd"])
    assert facts.n_layer == 4
    assert facts.attention_layers == 1
    assert facts.attention_layers_source == "tensors"
    assert facts.n_expert == 512 and facts.n_expert_used == 10
    assert facts.has_shared_experts is True
    assert facts.bytes_expert_weights > 0
    assert facts.bytes_lazy_tables > 0
    # the per-layer table carries a third dimension, so it cannot equal the embedding table
    assert facts.bytes_lazy_tables != facts.bytes_token_embd


def test_head_dimension_falls_back_to_embedding_over_heads() -> None:
    metadata = [
        b.string("general.architecture", "llama"),
        b.uint32("llama.block_count", 1),
        b.uint32("llama.embedding_length", 4096),
        b.uint32("llama.attention.head_count", 32),
        b.uint32("llama.attention.head_count_kv", 32),
    ]
    facts = derive_facts(read_header(FakeSource(b.build(metadata, []))))
    assert facts.head_dim == 128


def test_kv_bytes_per_token_by_cache_type() -> None:
    facts = derive_facts(read_header(FakeSource(dense_header())))
    # 2 caches * 2 attention layers * 8 kv heads * 128 head dim = 4096 elements per token
    assert kv_bytes_per_token(facts, "f16") == 4096 * 2
    assert kv_bytes_per_token(facts, "q8_0") == 4096 * 34 // 32
    assert facts.kv_bytes_per_token_f16 == 4096 * 2


def test_a_header_without_the_architecture_key_still_returns_facts() -> None:
    facts = derive_facts(read_header(FakeSource(b.build([], []))))
    assert facts.arch == "unknown"
    assert facts.n_layer is None
    assert facts.attention_layers_source == "unknown"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_gguf_facts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'llamafit.gguf.facts'`.

- [ ] **Step 3: Implement `gguf/facts.py`**

The shape of it:

```python
"""Turn a parsed GGUF header into the facts a memory budget needs.

Every number here is read from the file, not estimated: tensor sizes come from the
dimensions and type in the tensor table, and the layer counts come from the keys the
architecture writes. The one derivation that is not a direct read is the number of
full-attention layers, which is counted from the tensors that only such a layer owns.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from llamafit.models.gguf import GgufFacts, GgufHeader

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")
_ATTENTION_MARKERS = ("attn_k.weight", "attn_v.weight")
_KV_TYPE_BYTES = {"f16": (1, 2), "q8_0": (32, 34), "q4_0": (32, 18)}


def _int(header: GgufHeader, *keys: str) -> int | None:
    """The first key present in the metadata whose value is an integer."""
    for key in keys:
        value = header.metadata.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
```

`derive_facts` then:

1. Reads `arch = str(header.metadata.get("general.architecture", "unknown"))` and builds the prefixed keys with it.
2. Reads `n_layer`, `n_embd`, `n_head`, `n_head_kv`, `n_expert`, `n_expert_used` through `_int`, accepting both `{arch}.attention.head_count_kv` and `{arch}.attention.head_count` for the second when the first is missing (a model with no grouped-query attention omits it).
3. `head_dim` from `{arch}.attention.key_length`, else `n_embd // n_head` when both are known, else `None`.
4. `n_vocab` from the length of `tokenizer.ggml.tokens` when it is a list, else `{arch}.vocab_size`.
5. Walks the tensors once, accumulating: `bytes_total`; `bytes_token_embd` and `bytes_output_head` by exact name; `bytes_lazy_tables` for any tensor whose name starts with one of `lazy_tensor_names`; `bytes_expert_weights` for names containing `_exps`; everything else that matches `_BLOCK_RE` into `bytes_attention_weights`. Sets `has_shared_experts` when any name contains `_shexp`. Collects the block indices whose tensor names end with one of `_ATTENTION_MARKERS` into a set.
6. `attention_layers` is the size of that set with source `"tensors"` when it is non-empty; else `n_layer` with source `"all-layers"` when `n_layer` is known; else `None` with source `"unknown"`.
7. `kv_bytes_per_token_f16` is `kv_bytes_per_token(facts, "f16")`, computed after the rest.
8. `recurrent_state_bytes` from `{arch}.ssm.state_size` times `{arch}.ssm.inner_size` times the number of non-attention layers when all three are known, else `None`.

`kv_bytes_per_token` is

```python
def kv_bytes_per_token(facts: GgufFacts, kv_type: str) -> int | None:
    """Bytes both KV caches grow by per token, or ``None`` when the shape is unknown."""
    if facts.attention_layers is None or facts.n_head_kv is None or facts.head_dim is None:
        return None
    block_elements, block_bytes = _KV_TYPE_BYTES[kv_type]
    elements = 2 * facts.attention_layers * facts.n_head_kv * facts.head_dim
    return elements // block_elements * block_bytes
```

- [ ] **Step 4: Run the tests, lint and type-check**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_gguf_facts.py -v && .venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m mypy`
Expected: PASS, clean, clean.

- [ ] **Step 5: Check the facts against a real model**

Add to `tests/unit/test_gguf_real_file.py`:

```python
@pytest.mark.hardware
def test_facts_from_the_reference_machines_coder_model() -> None:
    from llamafit.gguf.facts import derive_facts

    path = CANDIDATES[1]
    if not path.exists():
        pytest.skip(f"{path} is not on this machine")
    facts = derive_facts(read_header(LocalSource(path)))
    assert facts.n_expert is not None and facts.n_expert > 1
    assert facts.attention_layers is not None
    assert facts.attention_layers < (facts.n_layer or 0), "a hybrid model has fewer attention layers"
    assert facts.bytes_expert_weights > facts.bytes_attention_weights
```

Run it and put the derived facts for that model in your report: the layer count, the attention-layer count, the expert count and the byte split. Phase 1C's budget is built on exactly these numbers, so a reader of your report should be able to sanity-check them against the model card.

- [ ] **Step 6: Commit**

```bash
git add src/llamafit/gguf/facts.py src/llamafit/models/gguf.py tests/unit/test_gguf_facts.py tests/unit/test_gguf_real_file.py
git commit -m "feat: derive architecture facts from a GGUF header"
```

---

### Task 5: Header cache and the public reading interface

**Files:**
- Create: `src/llamafit/gguf/cache.py`
- Modify: `src/llamafit/gguf/__init__.py`
- Test: `tests/unit/test_gguf_cache.py`

**Interfaces:**
- Produces:
  - `cache_key_for_path(path: Path) -> str` and `cache_key_for_url(url: str, etag: str | None) -> str`, both returning a hex digest.
  - `HeaderCache(directory: Path)` with `get(key: str) -> GgufHeader | None` and `put(key: str, header: GgufHeader) -> None`, storing one JSON file per key and treating any read error or schema mismatch as a miss.
  - `read_header_cached(source: ByteSource, key: str, cache: HeaderCache | None) -> GgufHeader`.
  - `read_facts(target: Path | str, *, lazy_tensor_names: Sequence[str] = (), cache: HeaderCache | None = None, client: httpx.Client | None = None) -> GgufFacts` — the one function the rest of the codebase calls. A `Path` or a string without a scheme reads locally; a string starting with `http://` or `https://` reads over ranges.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from llamafit.gguf import read_facts
from llamafit.gguf.cache import HeaderCache, cache_key_for_path, cache_key_for_url
from llamafit.gguf.reader import read_header
from llamafit.gguf.source import FakeSource
from tests.fixtures import gguf_builder as b
from tests.unit.test_gguf_facts import dense_header


def test_a_cached_header_round_trips(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    header = read_header(FakeSource(dense_header()))
    assert cache.get("k") is None
    cache.put("k", header)
    again = cache.get("k")
    assert again is not None
    assert again.tensor_count == header.tensor_count
    assert [t.name for t in again.tensors] == [t.name for t in header.tensors]
    assert again.metadata["general.architecture"] == "llama"


def test_a_corrupt_cache_entry_is_a_miss_not_a_crash(tmp_path: Path) -> None:
    cache = HeaderCache(tmp_path)
    cache.put("k", read_header(FakeSource(dense_header())))
    next(tmp_path.glob("*.json")).write_text("{not json", encoding="utf-8")
    assert cache.get("k") is None


def test_the_path_key_changes_with_size_and_time(tmp_path: Path) -> None:
    target = tmp_path / "m.gguf"
    target.write_bytes(b"a" * 10)
    first = cache_key_for_path(target)
    target.write_bytes(b"a" * 20)
    assert cache_key_for_path(target) != first


def test_the_url_key_changes_with_the_etag() -> None:
    assert cache_key_for_url("https://x/y.gguf", "abc") != cache_key_for_url("https://x/y.gguf", "def")


def test_read_facts_from_a_local_file(tmp_path: Path) -> None:
    target = tmp_path / "m.gguf"
    target.write_bytes(dense_header())
    facts = read_facts(target)
    assert facts.arch == "llama" and facts.n_layer == 2


def test_read_facts_uses_the_cache_the_second_time(tmp_path: Path) -> None:
    target = tmp_path / "m.gguf"
    target.write_bytes(dense_header())
    cache = HeaderCache(tmp_path / "cache")
    first = read_facts(target, cache=cache)
    target.write_bytes(b"XXXX" + dense_header()[4:])  # would fail to parse if re-read
    second = read_facts(target, cache=cache)
    assert second.n_layer == first.n_layer
```

The last test needs the key to stay stable across that rewrite, so write the same number of bytes and restore the modification time with `os.utime(target, (stat.st_atime, stat.st_mtime))` captured before the rewrite; add those two lines to the test.

- [ ] **Step 2: Run it, implement, run again**

Run the test, watch it fail on the missing module, then write `cache.py` and the `__init__` wiring. `HeaderCache.put` writes `directory / f"{key}.json"` with `header.model_dump_json()`, creating the directory on first use; `get` returns `None` on `OSError`, `ValueError` or a pydantic `ValidationError`. `read_facts` resolves the target, picks the source, builds the key, consults the cache, parses on a miss, stores, and derives.

- [ ] **Step 3: Lint, type-check and commit**

```bash
git add src/llamafit/gguf tests/unit/test_gguf_cache.py
git commit -m "feat: cache parsed GGUF headers and expose read_facts"
```

---

### Task 6: Catalog loader, custom overrides and validation

**Files:**
- Create: `src/llamafit/catalog/__init__.py`, `src/llamafit/catalog/loader.py`, `src/llamafit/catalog/validate.py`
- Modify: `pyproject.toml` (add `pyyaml>=6.0` to `[project] dependencies`)
- Test: `tests/unit/test_catalog_loader.py`

**Interfaces:**
- Consumes: `CatalogModel`, `Catalog`, `CatalogError`, `get_paths`.
- Produces:
  - `Problem(file: str, model_id: str | None, location: str, message: str)`
  - `load_models_from_file(path: Path) -> tuple[list[CatalogModel], list[Problem]]` — a YAML file holds a list of entries; a syntax error or a failed validation becomes a `Problem` and never an exception.
  - `bundled_catalog_dir() -> Path` returning the packaged `data/catalog` directory through `importlib.resources`.
  - `custom_models_path(env: Mapping[str, str] | None = None) -> Path` honouring `LLAMAFIT_CUSTOM_MODELS` then the data directory.
  - `load_catalog(*, bundled_dir: Path | None = None, custom_path: Path | None = None, strict: bool = False) -> tuple[Catalog, list[Problem]]` — bundled files first in sorted order, then the custom file; a custom entry whose id already exists replaces it, a new id is appended; a duplicate id inside the bundled set is a `Problem`. With `strict=True` any problem raises a `CatalogError` listing every one of them.
  - `validate_files(paths: Sequence[Path]) -> list[Problem]`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pytest

from llamafit.catalog.loader import load_catalog, load_models_from_file
from llamafit.errors import CatalogError

ENTRY = """
- id: tiny-1b
  name: Tiny 1B
  vendor: Example
  family: tiny
  release_date: 2026-01-01
  license: {spdx: MIT, url: "https://example.invalid/l"}
  params: {total_b: 1.0, active_b: 1.0}
  architecture: {class: dense, gguf_arch: llama}
  context: {native: 8192}
  capabilities: [coding]
  use_cases: [coding]
  quality: {baseline: 60}
  sources:
    - repo: example/tiny-1b-GGUF
      trust: community
      quants:
        - {name: Q4_K_M}
"""


def write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_a_valid_file(tmp_path: Path) -> None:
    models, problems = load_models_from_file(write(tmp_path, "tiny.yaml", ENTRY))
    assert problems == []
    assert [m.id for m in models] == ["tiny-1b"]


def test_broken_yaml_becomes_a_problem_not_an_exception(tmp_path: Path) -> None:
    models, problems = load_models_from_file(write(tmp_path, "bad.yaml", "- id: [unclosed"))
    assert models == []
    assert len(problems) == 1 and "bad.yaml" in problems[0].file


def test_an_invalid_entry_names_its_field(tmp_path: Path) -> None:
    text = ENTRY.replace("baseline: 60", "baseline: 900")
    models, problems = load_models_from_file(write(tmp_path, "x.yaml", text))
    assert models == []
    assert any("baseline" in p.location for p in problems)


def test_a_custom_entry_replaces_a_bundled_one_and_new_ids_are_added(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "tiny.yaml", ENTRY)
    custom = write(tmp_path, "custom.yaml", ENTRY.replace("Tiny 1B", "Mine") + ENTRY.replace(
        "tiny-1b", "other-2b").replace("Tiny 1B", "Other"))
    catalog, problems = load_catalog(bundled_dir=bundled, custom_path=custom)
    assert problems == []
    assert catalog.by_id["tiny-1b"].name == "Mine"
    assert "other-2b" in catalog.by_id
    assert len(catalog.models) == 2


def test_a_duplicate_id_inside_the_bundled_set_is_a_problem(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "a.yaml", ENTRY)
    write(bundled, "b.yaml", ENTRY)
    _, problems = load_catalog(bundled_dir=bundled, custom_path=None)
    assert any("duplicate" in p.message for p in problems)


def test_strict_mode_raises_with_every_problem_listed(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "bad.yaml", "- id: [unclosed")
    with pytest.raises(CatalogError) as info:
        load_catalog(bundled_dir=bundled, custom_path=None, strict=True)
    assert "bad.yaml" in str(info.value)


def test_a_missing_custom_file_is_not_a_problem(tmp_path: Path) -> None:
    bundled = tmp_path / "bundled"
    write(bundled, "tiny.yaml", ENTRY)
    catalog, problems = load_catalog(bundled_dir=bundled, custom_path=tmp_path / "absent.yaml")
    assert problems == [] and len(catalog.models) == 1
```

- [ ] **Step 2: Run it, implement, run again**

`load_models_from_file` reads with `yaml.safe_load`, requires a list at the top level, and validates each entry with `CatalogModel.model_validate`, converting a `ValidationError` into one `Problem` per error with `location` built from the error's `loc` tuple joined with dots. `load_catalog` composes them. Add `pyyaml>=6.0` to the runtime dependencies and reinstall the environment.

- [ ] **Step 3: Lint, type-check and commit**

```bash
git add src/llamafit/catalog pyproject.toml tests/unit/test_catalog_loader.py
git commit -m "feat: load the catalog from YAML with custom overrides"
```

---

### Task 7: The seed catalog and its schema

**Files:**
- Create: `src/llamafit/data/catalog/qwen3.yaml`, `src/llamafit/data/catalog/qwen3-coder.yaml`, `src/llamafit/data/catalog/llama3.yaml`, `src/llamafit/data/catalog/gemma3.yaml`, `src/llamafit/data/schema/catalog.schema.json`, `scripts/gen_schema.py`
- Test: `tests/unit/test_catalog_data.py`

**Interfaces:**
- Consumes: `load_catalog`, `CatalogModel`.
- Produces: a bundled catalog that loads with zero problems, and a JSON schema generated from the pydantic tree so contributors and editors can validate a file before opening a pull request.

**Sourcing rule, from specification section 6.3.** Every fact comes from a primary source: the vendor's model card or technical report, the vendor's announcement, or the Hugging Face repository that publishes the GGUF files. No third-party catalog. Each entry's `quality.benchmarks` carries at least two published scores with their source URLs, and `license.url` points at the actual licence file. Leave `bytes`, `sha256` and `gguf_facts` empty; Task 9 fills them.

Seed these families, chosen because two of them are measured on the reference machine and the other two cover the dense and vision cases the estimator needs to exercise:

| File | Entries |
|---|---|
| `qwen3-coder.yaml` | `qwen3-coder-next` (80B total, 3B active, MoE hybrid, coding, tools) |
| `qwen3.yaml` | `qwen3.8-flash-next` (125B total, 6B active, MoE hybrid, thinking, vision, tools), `qwen3-0.6b` (dense, tiny, useful as a smoke test) |
| `llama3.yaml` | `llama-3.1-8b-instruct` (dense, general, the reference dense model everyone knows) |
| `gemma3.yaml` | `gemma-3-27b-it` (dense, vision, a second vision entry so the multimodal path is not tested by one model alone) |

For the two measured models, add the `measured` block from `docs/calibration/2026-09-09-reference-machine.md`, quoting the flags and the figures exactly and pointing `source` at that file.

- [ ] **Step 1: Write the failing test**

```python
from llamafit.catalog.loader import bundled_catalog_dir, load_catalog


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

    text = resources.files("llamafit.data.schema").joinpath("catalog.schema.json").read_text("utf-8")
    assert json.loads(text) == CatalogModel.model_json_schema(by_alias=True), (
        "regenerate with: python scripts/gen_schema.py"
    )
```

- [ ] **Step 2: Write `scripts/gen_schema.py`**

```python
"""Write the catalog JSON schema from the pydantic models.

Usage:
    python scripts/gen_schema.py
"""

from __future__ import annotations

import json
from pathlib import Path

from llamafit.models.catalog import CatalogModel

DEST = Path(__file__).resolve().parents[1] / "src/llamafit/data/schema/catalog.schema.json"


def main() -> int:
    """Regenerate the schema file and report where it went."""
    DEST.parent.mkdir(parents=True, exist_ok=True)
    schema = CatalogModel.model_json_schema(by_alias=True)
    DEST.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The test compares parsed JSON, so the indentation and key order in the file do not have to match the in-memory dict, only the content.

- [ ] **Step 3: Write the catalog files, run the tests until they pass**

Research each model before writing its entry, and put the URL you used in the file. Do not invent a benchmark score; if you cannot find two published scores with sources for a model, replace that model with one you can source and say so in your report.

- [ ] **Step 4: Add the data files to the wheel**

Confirm `[tool.hatch.build.targets.wheel] packages = ["src/llamafit"]` already carries the YAML and JSON files; if a build excludes them, add `[tool.hatch.build] include` rules and note it. Check with `python -m build --wheel` and by listing the wheel's contents, or by `pip install -e .` and reading the file through `importlib.resources` in a fresh interpreter.

- [ ] **Step 5: Commit**

```bash
git add src/llamafit/data scripts/gen_schema.py tests/unit/test_catalog_data.py
git commit -m "feat: seed catalog with five models from primary sources"
```

---

### Task 8: The Hugging Face metadata client

**Files:**
- Create: `src/llamafit/catalog/hf.py`
- Test: `tests/unit/test_catalog_hf.py`, `tests/fixtures/hf_responses.py`

**Interfaces:**
- Produces:
  - `RepoFile(path: str, size: int | None, sha256: str | None)`
  - `class HfClient(Protocol)` with `list_files(repo: str) -> list[RepoFile]` and `file_url(repo: str, path: str) -> str`
  - `HttpHfClient(client: httpx.Client | None = None, token: str | None = None)` querying `https://huggingface.co/api/models/{repo}?blobs=true`, reading `siblings[].rfilename`, `siblings[].size` and `siblings[].lfs.sha256`, turning a non-200 or an `httpx.HTTPError` into a `NetworkError` naming the repository, and building the URL as `https://huggingface.co/{repo}/resolve/main/{path}`
  - `FakeHfClient(files: Mapping[str, list[RepoFile]])` for tests
  - `match_quant_files(files: Sequence[RepoFile], quant_name: str) -> list[RepoFile]` — the files belonging to one quant, sorted by name, matching a single file `*<quant>*.gguf` or a shard set `*<quant>-00001-of-000NN.gguf`, and ignoring files in a subdirectory whose name is a different quant

- [ ] **Step 1: Write the failing test**

```python
import httpx
import pytest

from llamafit.catalog.hf import HttpHfClient, RepoFile, match_quant_files
from llamafit.errors import NetworkError

API = {
    "siblings": [
        {"rfilename": "README.md", "size": 100},
        {"rfilename": "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf", "size": 10,
         "lfs": {"sha256": "aa"}},
        {"rfilename": "UD-Q4_K_XL/M-UD-Q4_K_XL-00002-of-00002.gguf", "size": 20,
         "lfs": {"sha256": "bb"}},
        {"rfilename": "UD-Q2_K_XL/M-UD-Q2_K_XL.gguf", "size": 5, "lfs": {"sha256": "cc"}},
        {"rfilename": "mmproj-F16.gguf", "size": 7, "lfs": {"sha256": "dd"}},
    ]
}


def client_for(handler: object) -> HttpHfClient:
    return HttpHfClient(client=httpx.Client(transport=httpx.MockTransport(handler)))  # type: ignore[arg-type]


def test_lists_files_with_sizes_and_checksums() -> None:
    files = client_for(lambda request: httpx.Response(200, json=API)).list_files("org/model")
    by_name = {f.path: f for f in files}
    assert by_name["mmproj-F16.gguf"].size == 7
    assert by_name["mmproj-F16.gguf"].sha256 == "dd"
    assert by_name["README.md"].sha256 is None


def test_a_failure_is_a_network_error_naming_the_repository() -> None:
    with pytest.raises(NetworkError, match="org/model"):
        client_for(lambda request: httpx.Response(404, json={})).list_files("org/model")


def test_builds_a_resolve_url() -> None:
    url = client_for(lambda request: httpx.Response(200, json=API)).file_url("org/m", "a/b.gguf")
    assert url == "https://huggingface.co/org/m/resolve/main/a/b.gguf"


def test_matches_a_shard_set_in_order() -> None:
    files = [RepoFile(path=s["rfilename"], size=s.get("size"),
                      sha256=(s.get("lfs") or {}).get("sha256")) for s in API["siblings"]]
    matched = match_quant_files(files, "UD-Q4_K_XL")
    assert [f.path.rsplit("/", 1)[-1] for f in matched] == [
        "M-UD-Q4_K_XL-00001-of-00002.gguf", "M-UD-Q4_K_XL-00002-of-00002.gguf",
    ]


def test_matches_a_single_file_quant_and_ignores_the_other_quant() -> None:
    files = [RepoFile(path=s["rfilename"], size=s.get("size"),
                      sha256=(s.get("lfs") or {}).get("sha256")) for s in API["siblings"]]
    matched = match_quant_files(files, "UD-Q2_K_XL")
    assert [f.path for f in matched] == ["UD-Q2_K_XL/M-UD-Q2_K_XL.gguf"]


def test_an_unknown_quant_matches_nothing() -> None:
    files = [RepoFile(path="a/b-Q4_K_M.gguf", size=1, sha256=None)]
    assert match_quant_files(files, "Q8_0") == []
```

- [ ] **Step 2: Run it, implement, run again**

`match_quant_files` filters to `.gguf` files whose name contains the quant name as a whole token, then, when any matches the shard pattern, keeps only the shards of the largest shard set and sorts them by index. `HttpHfClient` sends a `User-Agent` of `llamafit/<version>` and an `Authorization` header when a token is given, and honours `HF_TOKEN` from the environment when no token is passed.

- [ ] **Step 3: Lint, type-check and commit**

```bash
git add src/llamafit/catalog/hf.py tests/unit/test_catalog_hf.py
git commit -m "feat: Hugging Face metadata client"
```

---

### Task 9: Refresh the volatile fields

**Files:**
- Create: `src/llamafit/catalog/refresh.py`
- Test: `tests/unit/test_catalog_refresh.py`

**Interfaces:**
- Consumes: `HfClient`, `assign_files_to_quants`, `read_facts`, `HeaderCache`, `load_models_from_file`. Call `assign_files_to_quants(files, [q.name for q in source.quants])` once per source rather than matching each quant on its own, so a repository publishing both `Q4_K_XL` and `UD-Q4_K_XL` gives every file to exactly one of them.
- Produces:
  - `RefreshResult(model_id: str, file: str, changed: bool, fields: list[str], error: str | None = None)`
  - `refresh_file(path: Path, *, hf: HfClient, read_facts_fn: Callable[..., GgufFacts], dry_run: bool = False, only: str | None = None) -> list[RefreshResult]`
  - `dump_models(models: Sequence[CatalogModel]) -> str` — the deterministic YAML writer: `yaml.safe_dump` with `sort_keys=False`, a two-space indent, `allow_unicode=True`, `width=100`, dumping `model_dump(by_alias=True, exclude_none=True, mode="json")` so a diff shows only what changed
  - The rule that a repository whose listing fails leaves its model untouched and yields a `RefreshResult` with `error` set, so one dead repository cannot blank a file

What refresh fills, per quant: `files` (the file names in shard order), `bytes_` (their total), `sha256` (one per file, in the same order), `bpw` (`bytes_ * 8 / (params.total_b * 1e9)`, rounded to two decimals) and `gguf_facts` (from the first file's header, read over ranges). For each `extras` entry it fills `bytes_` and `sha256` by exact file name. Nothing curated is ever touched.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import yaml

from llamafit.catalog.hf import FakeHfClient, RepoFile
from llamafit.catalog.refresh import dump_models, refresh_file
from llamafit.catalog.loader import load_models_from_file
from llamafit.models.gguf import GgufFacts
from tests.unit.test_catalog_loader import ENTRY, write

FILES = {
    "example/tiny-1b-GGUF": [
        RepoFile(path="tiny-1b-Q4_K_M.gguf", size=700_000_000, sha256="abc123"),
        RepoFile(path="README.md", size=10, sha256=None),
    ]
}


def facts_stub(*args: object, **kwargs: object) -> GgufFacts:
    return GgufFacts(arch="llama", n_layer=16, n_embd=2048, n_vocab=32000, n_head=16,
                     n_head_kv=8, head_dim=128, attention_layers=16,
                     attention_layers_source="tensors", bytes_expert_weights=0,
                     bytes_attention_weights=1, bytes_output_head=1, bytes_token_embd=1,
                     bytes_lazy_tables=0, bytes_total=3, has_shared_experts=False)


def test_refresh_fills_sizes_checksums_and_facts(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    assert [r.changed for r in results] == [True]
    models, problems = load_models_from_file(path)
    assert problems == []
    quant = models[0].sources[0].quants[0]
    assert quant.files == ["tiny-1b-Q4_K_M.gguf"]
    assert quant.bytes_ == 700_000_000
    assert quant.sha256 == ["abc123"]
    assert quant.bpw is not None and 5.5 < quant.bpw < 5.7
    assert quant.gguf_facts is not None and quant.gguf_facts.n_layer == 16


def test_a_dry_run_changes_nothing_on_disk(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    before = path.read_text(encoding="utf-8")
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub, dry_run=True)
    assert results[0].changed is True
    assert path.read_text(encoding="utf-8") == before


def test_refreshing_twice_is_a_no_op(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    first = path.read_text(encoding="utf-8")
    results = refresh_file(path, hf=FakeHfClient(FILES), read_facts_fn=facts_stub)
    assert [r.changed for r in results] == [False]
    assert path.read_text(encoding="utf-8") == first


def test_a_failing_repository_leaves_the_file_untouched(tmp_path: Path) -> None:
    path = write(tmp_path, "tiny.yaml", ENTRY)
    before = path.read_text(encoding="utf-8")
    results = refresh_file(path, hf=FakeHfClient({}), read_facts_fn=facts_stub)
    assert results[0].error is not None
    assert path.read_text(encoding="utf-8") == before


def test_the_dump_keeps_curated_fields_and_their_order(tmp_path: Path) -> None:
    models, _ = load_models_from_file(write(tmp_path, "tiny.yaml", ENTRY))
    text = dump_models(models)
    data = yaml.safe_load(text)
    assert list(data[0])[:4] == ["id", "name", "vendor", "family"]
    assert data[0]["architecture"]["class"] == "dense"
    assert "gguf_facts" not in data[0]["sources"][0]["quants"][0]
```

- [ ] **Step 2: Run it, implement, run again**

`refresh_file` loads the file, and for each model: lists the repository's files once, matches each quant, computes the fields, and compares them with what is already there to decide `changed`. It writes only when something changed and `dry_run` is false, and it writes the whole file through `dump_models` so formatting stays stable. `only` filters to one model id.

- [ ] **Step 3: Lint, type-check and commit**

```bash
git add src/llamafit/catalog/refresh.py tests/unit/test_catalog_refresh.py
git commit -m "feat: refresh catalog sizes, checksums and GGUF facts"
```

---

### Task 10: Catalog services

**Files:**
- Create: `src/llamafit/services/catalog.py`
- Test: `tests/unit/test_services_catalog.py`

**Interfaces:**
- Produces:
  - `ModelFilters(use_case: UseCase | None = None, capabilities: list[Capability] = [], licenses: list[str] = [], vendor: str | None = None, search: str | None = None)`
  - `filter_models(catalog: Catalog, filters: ModelFilters) -> list[CatalogModel]` — every filter is an and, `capabilities` requires all of them, `search` matches case-insensitively against the id, name, vendor and family, and the result is sorted by `quality.baseline` descending then by id
  - `ModelSummary(id, name, vendor, params_total_b, params_active_b, capabilities, context_native, license_spdx, quant_names: list[str], largest_quant_bytes: int | None, is_local: bool)`
  - `summarise(model: CatalogModel, local_files: Sequence[LocalModel] = ()) -> ModelSummary` — `is_local` is true when any of the model's quant file names matches a file llama.cpp already has on disk
  - `ModelDetail(model: CatalogModel, quants: list[QuantDetail], local_paths: list[str])` and `QuantDetail(name, bytes_, bpw, facts, files, downloaded: bool)`
  - `describe(model: CatalogModel, local_files: Sequence[LocalModel] = ()) -> ModelDetail`

The budget of each quant on this host belongs to phase 1C; `describe` reports facts and sizes only. Say so in the docstring so the next phase knows where to add it.

- [ ] **Step 1: Write the failing test**

Cover: a use-case filter; a capability filter requiring two capabilities; a search that matches the vendor; ordering by quality then id; `summarise` marking a model local when a matching file name is among the local files and not otherwise; `describe` listing every quant with its facts. Build the catalog in the test from two or three entries constructed with the `minimal()` helper from `tests/unit/test_models_catalog.py`, imported directly, rather than from the bundled files, so the test does not break when the catalog grows.

- [ ] **Step 2: Run it, implement, run again, lint, type-check**

- [ ] **Step 3: Commit**

```bash
git add src/llamafit/services/catalog.py tests/unit/test_services_catalog.py
git commit -m "feat: catalog services for listing, filtering and describing models"
```

---

### Task 11: The catalog commands

**Files:**
- Create: `src/llamafit/cli/catalog_cmd.py`
- Modify: `src/llamafit/cli/app.py` (import the new command module at the bottom, beside the existing two), `src/llamafit/cli/render.py` (add the catalog tables)
- Test: `tests/unit/test_cli_catalog.py`

**Interfaces:**
- Produces the commands `list`, `search <text>`, `info <model>`, and the `catalog` group with `validate [FILE]`, `refresh [--model ID] [--dry-run] [--check]` and `show <model> [--yaml]`, each honouring the global `--json`.

Behaviour that the tests must pin:

- `list` prints a table with id, vendor, parameters as `total/active B`, capabilities, native context, licence and quant count, and accepts `--use-case`, `--capability` (repeatable), `--license` (repeatable), `--vendor`, `--search`, `--limit`.
- `search <text>` is `list --search <text>`.
- `info <model>` prints the model's facts, its sources and every quant with its size, bits per weight and, when present, its architecture facts; an unknown id exits 1 with a `CatalogError` naming the closest ids by simple prefix match.
- `catalog validate` validates the bundled files and the custom file, prints one line per problem and exits 1 when there is any, 0 otherwise.
- `catalog refresh` prints one line per model with the fields it changed; `--check` makes it exit 1 when anything would change and change nothing, for the scheduled CI job; `--dry-run` prints without writing.
- `catalog show <model>` prints the raw entry, as JSON by default and as YAML with `--yaml`.
- `list --json` prints a JSON array of `ModelSummary`, `info --json` prints a `ModelDetail`, `catalog validate --json` prints the problem list.

`list` is a Python builtin name and a Typer command name; name the function `list_command` and pass `name="list"` to the decorator.

- [ ] **Step 1: Write the failing test**

Use `typer.testing.CliRunner` as `tests/unit/test_cli.py` does, monkeypatching `llamafit.cli.catalog_cmd.load_catalog` to return a small fixed catalog so the tests do not depend on the bundled data. Assert: the table contains a model's id and vendor; `--json` parses and carries the ids; a capability filter narrows the rows; `info` on an unknown id exits 1 and names a close id; `catalog validate` exits 0 on a clean catalog and 1 with the problem text on a broken one; `catalog refresh --check` exits 1 when the fake refresh reports a change.

- [ ] **Step 2: Run it, implement, run again**

Follow the existing command modules exactly: import `app` and `CliState` from `llamafit.cli.app`, take `ctx: typer.Context`, read `state = ctx.obj`, print JSON with `typer.echo(model.model_dump_json(indent=2))` and tables with `state.console.print(...)`. Build every table cell that carries catalog text with `rich.text.Text`, as `render.py` already does for probe errors, so a bracket in a model name cannot crash the command.

- [ ] **Step 3: Run the real commands and read the output**

```
.venv\Scripts\python.exe -m llamafit list
.venv\Scripts\python.exe -m llamafit list --use-case coding --capability tools
.venv\Scripts\python.exe -m llamafit info qwen3-coder-next
.venv\Scripts\python.exe -m llamafit catalog validate
.venv\Scripts\python.exe -m llamafit --json list
```

Paste all five in your report and judge them as a stranger would: is the table readable at 80 columns, does `info` show what a person needs to decide, does anything look wrong?

- [ ] **Step 4: Lint, type-check and commit**

```bash
git add src/llamafit/cli tests/unit/test_cli_catalog.py
git commit -m "feat: list, search, info and catalog commands"
```

---

### Task 12: Documentation, MODELS.md and the refresh check in CI

**Files:**
- Modify: `docs/cli.md`, `docs/catalog.md`, `docs/custom-models.md`, `docs/development.md`, `docs/specs/2026-09-09-llamafit-design.md`, `README.md`, `CHANGELOG.md`, `.github/workflows/ci.yml`
- Create: `MODELS.md`, `scripts/gen_models_md.py`
- Test: `tests/unit/test_docs.py` (extend)

**Interfaces:**
- Produces: documentation that matches the code, a generated model list, and a CI step that fails when the committed catalog is stale.

- [ ] **Step 1: Extend the documentation test**

Add to `tests/unit/test_docs.py`: every command in the Typer app appears in `docs/cli.md` (the existing test already does this and will now cover the new commands); `MODELS.md` names every id in the bundled catalog; `docs/catalog.md` documents every top-level field name of `CatalogModel`, read from `CatalogModel.model_fields` so the test cannot drift.

- [ ] **Step 2: Write `scripts/gen_models_md.py`**

It loads the bundled catalog and writes `MODELS.md`: a short header saying the file is generated and how, then one table per vendor, sorted, with columns model, parameters, architecture, context, capabilities, licence and quants. Add a test that running it leaves `MODELS.md` unchanged, the same shape as the schema test, so a stale file fails CI.

- [ ] **Step 3: Update the documentation**

- `docs/cli.md`: replace the phase-1B command entries with what was built, including the real output of `llamafit list` and `llamafit info` on this machine, and mark them as shipped rather than planned.
- `docs/catalog.md`: correct the entry schema to the fields as implemented, including `architecture.class`, the `Quant` field names, and the note that `gguf_facts` is filled by `refresh`. Document the `LLAMAFIT_CUSTOM_MODELS` path and the `catalog validate` workflow.
- `docs/custom-models.md`: check every claim against the loader, especially the override-by-id rule and the local-source form.
- `docs/development.md`: move `pyyaml` from the planned table to the runtime table.
- `docs/specs/...`: correct section 7.2 to describe how the full-attention layer count is actually determined, and note in section 5.1 that `LLAMA_CPP_PATH` is checked before `PATH`, which the code has done since phase 1A and the specification still describes the other way round.
- `README.md`: move the 1B row of the roadmap table to done and add `llamafit list` and `llamafit info` to the quick start.
- `CHANGELOG.md`: one entry under *Unreleased* per user-visible addition.

- [ ] **Step 4: Add the staleness check to CI**

Add a step to `.github/workflows/ci.yml`, after the tests and only on `ubuntu-latest` with Python 3.13, that runs `python scripts/gen_schema.py && python scripts/gen_models_md.py && git diff --exit-code`, so a pull request that changes the models without regenerating the derived files fails. Do not add a network-dependent `catalog refresh --check` to the pull-request workflow; put it in a separate scheduled workflow file, `.github/workflows/catalog.yml`, running weekly, which runs `llamafit catalog refresh --check` and opens nothing, only reports. Mention both in `docs/development.md`.

- [ ] **Step 5: Run everything and commit**

```
.venv\Scripts\python.exe -m pytest --cov -m "not hardware"
.venv\Scripts\python.exe -m pytest -m hardware
.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy
```

```bash
git add -A
git commit -m "docs: catalog reference, generated model list and staleness checks"
```

---

## Plan self-review (done while writing)

- **Specification coverage.** Section 6.1 principles map to Tasks 6, 7 and 9; 6.2 the entry schema to Task 1; 6.3 the seed to Task 7; 6.4 custom models to Task 6; 6.5 refresh to Task 9. Section 7.1 header reading maps to Tasks 2, 3 and 5; 7.2 derived facts to Task 4. Section 13.1's `list`, `search`, `info` and `catalog validate|refresh|show` map to Task 11. Section 14's cache and custom-file locations are used in Tasks 5 and 6.
- **Two places where the plan corrects the specification rather than following it**, both recorded in Task 12 step 3: the full-attention layer count is read from the tensors instead of a family rule, and `LLAMA_CPP_PATH` precedence in section 5.1 already disagrees with shipped code.
- **Deliberately out of scope**, because it belongs to phase 1C and would otherwise leak in: no memory budget, no placement, no speed estimate, no scoring. `info` shows facts and sizes only, and Task 10 says where the budget will attach.
- **Placeholders:** none; every step carries its code, its data or its exact editing instruction.
- **Type consistency:** `GgufFacts` is defined in Task 2 and filled in Task 4, and Task 1 imports it, which is why Task 1's step 3 says to write Task 2 first or use a forward reference. `RepoFile`, `HfClient` and `match_quant_files` from Task 8 are used with the same signatures in Task 9. `load_catalog` and `Problem` from Task 6 are used unchanged in Tasks 7, 10 and 11. `read_facts` from Task 5 is the only entry point Task 9 uses.
- **Risk worth naming:** the tensor-type table in Task 2 is the one piece of data that cannot be verified by reasoning alone, which is why Task 3 exists and runs against real files before anything depends on it.

