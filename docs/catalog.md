# The model catalog

The catalog is the list of models LlamaFit knows about: what they are, what they can do, where
their GGUF files live, and what llama.cpp needs to run them. It is curated by hand from primary
sources, kept in YAML in the repository, validated against a JSON schema in CI, and refreshed
from the Hugging Face API for the facts that change.

## Principles

1. **Primary sources only.** Model cards, technical reports, vendor announcements, and the
   Hugging Face repositories that publish the GGUF files. Every number has a source URL in the
   entry itself. Nothing is copied from other catalogs.
2. **Exact where the file allows it.** Sizes, checksums and architecture facts come from the
   GGUF files' headers and the repository's file metadata, filled in by `llamafit catalog
   refresh`, never typed by hand.
3. **Reviewable diffs.** One file per model family, deterministic key order, so a pull request
   shows exactly what changed.
4. **Honest quality.** The quality baseline is a curator's judgement from published benchmarks,
   with the benchmarks listed. It is a ranking aid, not a claim.

## Where the files are

```
src/llamafit/data/catalog/
  qwen3.yaml
  qwen3-coder.yaml
  glm5.yaml
  ...
src/llamafit/data/schema/catalog.schema.json
```

A user's own entries go in a separate file; see [custom-models.md](custom-models.md).

## An entry

```yaml
id: qwen3.8-flash-next                      # stable slug: lowercase, digits, dots, hyphens; unique
name: Qwen3.8-Flash-Next
vendor: Alibaba Qwen
family: qwen3
release_date: 2026-08-26
license: {spdx: Apache-2.0, url: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE}
params: {total_b: 125, active_b: 6, ngram_table_b: 51}
architecture:
  class: moe-hybrid                         # dense | moe | dense-hybrid | moe-hybrid
  gguf_arch: qwen4exp                       # general.architecture in the GGUF header
  notes: "Gated DeltaNet on 3 of every 4 layers; the KV cache must stay F16"
context: {native: 262144, extended: 1048576, extended_method: yarn}
capabilities: [coding, thinking, vision, tools, multilingual, long-context]
use_cases: [coding, reasoning, multimodal]
quality:
  baseline: 84
  benchmarks:
    - {name: SWE-bench Pro, score: 62.5, source: https://qwen.ai/blog/qwen3.8}
    - {name: LiveCodeBench v6, score: 91.9, source: https://qwen.ai/blog/qwen3.8}
sampling: {temp: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 0.0}
chat_template: {reasoning_format: deepseek, thinking_toggle: enable_thinking}
llama_cpp:
  min_build: 10800
  kv_types_allowed: [f16]
  requires: {mmproj: optional, mtp: optional, lazy_mode: recommended}
  quirks: ["--lazy-mode on streams the n-gram table from disk instead of holding it in RAM"]
sources:
  - repo: unsloth/Qwen3.8-Flash-Next-GGUF
    kind: gguf
    trust: unsloth                          # official | unsloth | bartowski | community
    quants:
      - name: UD-Q4_K_XL
        files: [Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf, ...]
        bytes: 111323630080
        bpw: 4.98
        sha256: [...]
        gguf_facts: {...}                   # written by `catalog refresh`
    extras:
      - {role: mmproj, file: mmproj-F16.gguf, bytes: 904004000}
      - {role: mtp, file: mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf, bytes: 2786568256}
measured:
  - {profile: rtx4060-8gb-ddr5-128gb, quant: UD-Q4_K_XL, gen_tps: 13.5, pp_tps: 52,
     flags: "-c 32768 -ub 1024 --no-mmproj-offload", source: docs/calibration/2026-09-09-reference-machine.md}
```

## Field reference

| Field | Required | Meaning |
|---|---|---|
| `id` | yes | stable identifier used by every command; never renamed once published |
| `name`, `vendor`, `family`, `release_date` | yes | display and grouping |
| `license.spdx`, `license.url` | yes | SPDX identifier (or `Other`) and the license text |
| `params.total_b`, `params.active_b` | yes | billions; `active_b` equals `total_b` for dense models |
| `params.ngram_table_b` and similar | no | auxiliary tables budgeted separately when they can be streamed |
| `architecture.class`, `architecture.gguf_arch` | yes | drives the KV and placement rules |
| `context.native` | yes | tokens; `extended` and `extended_method` when the vendor documents one |
| `capabilities` | yes | from: `coding`, `thinking`, `vision`, `tools`, `multilingual`, `long-context`, `embeddings`, `audio` |
| `use_cases` | yes | from: `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding`; first is primary |
| `quality.baseline`, `quality.benchmarks` | yes | see the rubric below |
| `sampling` | no | vendor-recommended sampling, passed to `llama-server` by `plan` |
| `chat_template` | no | reasoning format and the template flag that toggles thinking |
| `llama_cpp.*` | no | minimum build, allowed KV types, required extras, quirks shown by `plan` |
| `sources[]` | yes | at least one GGUF repository with at least one quant |
| `quants[].bytes`, `sha256`, `gguf_facts` | yes after refresh | never hand-typed |
| `measured[]` | no | measurements with the profile and flags they were taken with |

## The quality rubric

`quality.baseline` is 0 to 100 on the model's primary use case:

| Range | Meaning |
|---|---|
| 90 and above | frontier open weights on that task at the time of the entry |
| 80 to 89 | strong current generation |
| 70 to 79 | solid previous generation |
| 50 to 69 | small or dated; usable |
| below 50 | experimental or specialised |

Cite at least two published benchmarks. Prefer benchmarks that match the primary use case
(SWE-bench for coding, GPQA or AIME for reasoning, MMMU for vision). When a vendor's own
numbers are the only ones available, say so in the source URL's context. A pull request that
changes a baseline explains why.

## Refreshing

```
llamafit catalog refresh                 # every model
llamafit catalog refresh --model ID      # one
llamafit catalog refresh --dry-run       # show what would change
llamafit catalog refresh --check         # exit non-zero when the committed data is stale (CI)
```

`refresh` queries `https://huggingface.co/api/models/{repo}?blobs=true` for file sizes and LFS
checksums, reads each quant's GGUF header with HTTP range requests (only the header, a few
megabytes at most), writes `bytes`, `sha256` and `gguf_facts`, and rewrites the YAML with the
canonical key order. It never changes curated fields.

## Validating

```
llamafit catalog validate                # bundled catalog and the custom models file
llamafit catalog validate path/to.yaml   # one file
```

Checks: schema, unique ids, enum values, well-formed URLs, every quant with bytes after
refresh, `active_b ≤ total_b`, and that files named in `quants[].files` follow the split-file
pattern when there is more than one. CI runs it on every pull request.

## Adding a model

1. Open a *Model request* issue or go straight to a pull request with the YAML.
2. Fill the curated fields with sources. Leave `bytes`, `sha256` and `gguf_facts` empty.
3. Run `llamafit catalog refresh --model ID` and commit the result.
4. Run `llamafit catalog validate`.
5. If you measured it, add a `measured` entry with the exact flags and a link to your notes.

Entries for models whose weights are not publicly downloadable, or whose license forbids
redistribution of derivatives such as GGUF conversions, are not accepted.
