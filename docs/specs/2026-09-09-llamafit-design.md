# LlamaFit — design specification

**Status:** approved design, 2026-09-09
**Scope of this document:** the whole system (phases 1 to 3), with phase 1 specified to implementation depth.
**Companion documents:** phase plans under `docs/plans/`.

## 1. Purpose

LlamaFit answers one question for a person with a computer and llama.cpp: *which open-weight language models can I run here, at what speed, with which settings, and how do I get them running?* It scans the host, reads a curated catalog of models, computes exact memory budgets from GGUF facts, ranks candidates for the capabilities the user asked for (coding, thinking, vision, tool calling, long context), and turns the winner into a concrete llama.cpp launch configuration. In later phases it installs llama.cpp and the model, writes launch presets, measures real speed, and calibrates its own estimates from the measurements.

LlamaFit never uses a language model to do any of this. Every number is computed deterministically and carries a confidence label.

### 1.1 Positioning

Tools that recommend local models usually stop at "this one fits": a coarse memory minimum per model, a use-case label, and a speed guess. LlamaFit is built around what happens after that sentence, because that is where people lose their evenings:

| Topic | The usual approach | LlamaFit |
|---|---|---|
| Memory model | one RAM and VRAM minimum per model at one quant | per-component budget from GGUF header facts and real file sizes: weights by tensor class, KV per 1K tokens, compute buffer by micro-batch, projector |
| Result of `plan` | required VRAM, RAM and cores | a runnable `llama-server` command line and preset files |
| Runtime | many runtimes, each detected and no more | llama.cpp only, and all the way: detect, install, launch, tune, verify |
| Verification | none, or crowd-sourced numbers from other machines | local llama-bench and server runs, estimate-versus-measured, per-host calibration, detection of driver paging |
| Needs | a use-case label | use case plus capability requirements and a minimum context |
| Form | a binary | a Python package, importable as a library, with CLI, TUI and web dashboard |

### 1.2 Goals

1. Run on Windows, macOS and Linux with a single `pip install llamafit`, Python 3.10 or newer, no compiler, no Node.
2. Detect the host and llama.cpp reliably, degrading gracefully when a probe is unavailable, and say what was and was not detected.
3. Produce recommendations that are explainable: every score, every memory number and every speed estimate can be expanded to show its inputs.
4. Turn a recommendation into a working llama.cpp setup with the fewest possible manual steps.
5. Be a good open-source citizen: curated catalog maintained by pull request, schema validation in CI, documentation for every command, tests that do not require hardware.

### 1.3 Non-goals

- Managing Ollama, LM Studio, vLLM or MLX. They are detected and reported so the user knows they exist; nothing more.
- Remote hosts, multi-user web access, authentication. The web dashboard binds to localhost.
- Any use of an LLM inside the tool.
- Fine-tuning, conversion or quantisation of models.
- A GUI desktop application.

## 2. Users and scenarios

1. **First-time local LLM user.** Runs `llamafit`, sees the TUI board, picks the top coding model, presses Install, gets a running server and a URL. Never reads a flag.
2. **Practitioner with an 8 GB card and lots of RAM.** Wants the strongest thinking model that still generates above 10 tokens per second, needs at least 32K context, has 100 GB of disk. Uses the Needs panel, reads the explanations, exports the preset, tunes `-c` by hand.
3. **Scripter.** Calls `llamafit recommend --use-case coding --min-context 32768 --json` from a shell script or CI job to choose a model for a fleet of machines using hardware profiles.
4. **Catalog contributor.** Adds a new model family in a YAML file, runs `llamafit catalog validate`, opens a pull request.
5. **Benchmark contributor (phase 3).** Runs `llamafit bench`, sees estimate versus measured, and the tool becomes more accurate on that machine.

## 3. Architecture

### 3.1 Layering

```
┌──────────────────────────────────────────────────────────────┐
│ Interfaces        cli/ (Typer)    tui/ (Textual)    web/ (FastAPI + static dashboard) │
├──────────────────────────────────────────────────────────────┤
│ Services          scan · recommend · plan · install · bench · catalog · doctor        │
├──────────────────────────────────────────────────────────────┤
│ Core              hardware · hwprofile · llamacpp · catalog · gguf · budget ·        │
│                   placement · speed · quality · scoring · presets · download          │
├──────────────────────────────────────────────────────────────┤
│ Foundation        models (pydantic) · constants · paths · logging · errors · units    │
└──────────────────────────────────────────────────────────────┘
```

Rules:

- Core modules are pure functions over typed data as far as possible. Anything that touches the OS (subprocess, files, network) lives behind a small interface so tests can substitute recorded outputs.
- Services compose core modules into the operations the interfaces expose. Each service function takes plain inputs and returns a pydantic model; it never prints.
- Interfaces render. The three interfaces call the same service functions; the web API serialises the same pydantic models the TUI displays.
- Constants that the estimator uses (efficiencies, overheads, penalties, weights) live in one module, documented with their source, and can be overridden by a hardware profile or a calibration record.

### 3.2 Package layout

```
src/llamafit/
  __init__.py            version, public API re-exports
  __main__.py            `python -m llamafit`
  constants.py           estimator constants with provenance comments
  errors.py              LlamaFitError hierarchy with user-facing messages
  paths.py               platform directories (config, data, cache, downloads)
  units.py               GiB/GB parsing and formatting, token counts
  logging.py             structured logging to file, quiet console
  models/                pydantic models: Host, Gpu, LlamaCpp, CatalogModel, Quant,
                         GgufFacts, Budget, Placement, Estimate, Scored, Plan, Preset,
                         BenchResult, HardwareProfile, Needs
  hardware/              os_common.py, windows.py, macos.py, linux.py, gpu_nvidia.py,
                         gpu_amd.py, gpu_apple.py, gpu_vulkan.py, bandwidth.py, __init__.py (scan)
  hwprofile/             bundled profiles, loading, matching, validation, simulation overrides
  llamacpp/              detect.py, version.py, backends.py, server.py (running instance),
                         releases.py (phase 2), install.py (phase 2), launch.py (phase 2)
  catalog/               loader.py, schema.py, refresh.py (Hugging Face), custom.py, seed/ (YAML)
  gguf/                  header.py (local and remote range reads), facts.py (derived facts), cache.py
  budget/                weights.py, kv.py, compute.py, projector.py, budget.py
  placement/             modes.py, planner.py, context_tiers.py, flags.py
  speed/                 generation.py, prompt.py, confidence.py
  quality/               baseline.py, quant_penalty.py, alignment.py
  scoring/               fit_score.py, speed_score.py, context_score.py, weights.py, rank.py
  presets/               windows_cmd.py, posix_sh.py, models_ini.py, render.py (phase 2)
  download/              ranges.py, verify.py, hf.py (phase 2)
  bench/                 llama_bench.py, server_bench.py, store.py, calibrate.py (phase 3)
  services/              scan.py, recommend.py, plan.py, catalog.py, doctor.py, install.py, bench.py
  cli/                   app.py and one module per command group
  tui/                   app.py, screens/, widgets/, styles.tcss
  web/                   api.py, server.py, static/ (index.html, app.js, styles.css)
  data/                  catalog/*.yaml, profiles/*.json, schema/*.json, calibration/*.json
```

### 3.3 Data flow of a recommendation

1. `scan` builds a `Host` (section 4) and a `LlamaCpp` status (section 5).
2. `catalog.load` yields `CatalogModel` entries with their `Quant` list (section 6); `gguf.facts` supplies or confirms architecture facts per quant (section 7).
3. For each candidate (model × quant), `budget.compute` produces a `Budget` for a placement and a context (section 8); `placement.plan` searches placements and context tiers for the best that fits (section 9).
4. `speed.estimate` produces generation and prompt estimates with confidence (section 10).
5. `quality` and `scoring` produce the four scores and the composite (section 11).
6. `rank` orders candidates, applies the user's `Needs` filters, and attaches explanations (section 12).
7. Interfaces render the board; `plan` renders the winner as flags and, in phase 2, presets.

## 4. Host detection

### 4.1 The `Host` model

| Field | Source | Notes |
|---|---|---|
| `os`, `os_version`, `arch` | `platform` | `windows`, `macos`, `linux`; `x86_64`, `arm64` |
| `cpu.model`, `cpu.physical_cores`, `cpu.logical_cores` | `py-cpuinfo`, `psutil`, `/proc/cpuinfo`, `sysctl`, WMI | performance versus efficiency cores on hybrid Intel and Apple parts when the OS exposes them |
| `cpu.isa` | `py-cpuinfo` flags | `avx2`, `avx512`, `avx512_vnni`, `amx`, `neon`, `sve` |
| `memory.total_bytes`, `memory.available_bytes` | `psutil` | available means free right now |
| `memory.type`, `memory.speed_mts`, `memory.channels` | WMI `Win32_PhysicalMemory`, `dmidecode` when readable, `system_profiler SPMemoryDataType` | optional; used to estimate bandwidth |
| `memory.bandwidth_gbps` | measured (section 4.3) or table | with a `source` label |
| `gpus[]` | section 4.2 | may be empty |
| `unified_memory` | Apple Silicon, AMD APUs | when true the memory pool is shared |
| `disk.free_bytes` per relevant path | `shutil.disk_usage` | download directory and llama.cpp directory |
| `probes[]` | every probe records `name`, `ok`, `duration_ms`, `error` | shown by `doctor` |

### 4.2 GPU detection

Probes run in order; the first that succeeds for a vendor wins, others add information when they can.

| Vendor | Probe | Fields |
|---|---|---|
| NVIDIA | `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv,noheader,nounits` | name, VRAM total and used, driver, compute capability |
| AMD | `rocm-smi --showmeminfo vram --json`, `amd-smi metric --mem --json` | name, VRAM total and used |
| Apple | `system_profiler SPDisplaysDataType -json` | chip name, core count; memory comes from the unified pool |
| Intel Arc | `sysfs` on Linux, WMI on Windows | name, VRAM |
| Any | `vulkaninfo --summary` when present | device names and heap sizes as a fallback |
| Any | WMI `Win32_VideoController`, `lspci -nn` | name only, VRAM unknown |

Each GPU records `vendor`, `name`, `vram_total_bytes`, `vram_used_bytes`, `bandwidth_gbps` (from the profile table or unknown), `compute_tflops_fp16` (table or unknown), `backend_hint` (`cuda`, `hip`, `metal`, `vulkan`, `sycl`, `cpu`). Multi-GPU is recorded; phase 1 plans on the largest single GPU and reports the others.

### 4.3 Memory bandwidth

Bandwidth dominates generation speed, so LlamaFit measures it rather than guessing when it can:

- **RAM:** a 30 to 60 millisecond multi-threaded copy of a buffer larger than the last-level cache, using NumPy if present or a pure-Python `memoryview` fallback with a documented correction factor. The result is labelled `measured`. When the probe is unavailable or the result is implausible, the DDR type, speed and channel count give `estimated`; failing that, a per-OS default gives `assumed`.
- **VRAM:** never measured by LlamaFit; taken from the bundled GPU table (name match) or the hardware profile, else `unknown`, which forces the backend fallback constants in section 10.
- **PCIe:** taken from the GPU table (generation and lanes when known) with a conservative default of 12 GB/s effective, labelled `assumed`, until phase 3 measures it during a benchmark.

### 4.4 Hardware profiles and simulation

A hardware profile is a JSON document that describes a machine well enough to score against it without being on it. The schema is small enough to write by hand:

```json
{
  "schema_version": 1,
  "name": "rtx4060-8gb-ddr5-128gb",
  "match": {"gpu_name_contains": "RTX 4060"},
  "total_ram_gb": 128,
  "unified_memory": false,
  "gpu_memory_gb": 8,
  "gpu_memory_bandwidth_gbps": 272,
  "ddr_bandwidth_gbps": 60,
  "pcie_bandwidth_gbps": 12,
  "gpu_compute_tflops_fp16": 60,
  "cpu_cores": 24,
  "calibration": {"ram_efficiency": 0.70, "vram_efficiency": 0.55, "layer_overhead_ms": 0.2}
}
```

`--profile <name|file>` scores against a profile instead of the live scan. `--memory`, `--ram`, `--cpu-cores` override single values (simulation). The TUI exposes the same as a Simulate panel with a visible `SIM` badge.

## 5. llama.cpp integration

### 5.1 Detection

| Step | Method |
|---|---|
| Binaries | `LLAMA_CPP_PATH` first, so an explicit choice always wins; then `PATH` for `llama-server`, `llama-cli`, `llama-bench`, `llama-gguf`; then well-known directories (`~/.llamafit/llama.cpp`, `/usr/local/bin`, `/opt/homebrew/bin`, `C:\llama.cpp`, `D:\llama.cpp`, LM Studio and Ollama bundles are reported as *not usable directly*) |
| Version | `llama-server --version` parsed for build number and commit; `bin/VERSION.txt` when present |
| Backends | shipped `ggml-*` shared libraries (`ggml-cuda`, `ggml-hip`, `ggml-metal`, `ggml-vulkan`, `ggml-sycl`, `ggml-cpu-*`), and `llama-server --list-devices` when the build supports it |
| Running server | `GET http://127.0.0.1:{port}/health`, `/v1/models`, `/props` on the default port 8080 and `LLAMA_SERVER_PORT`; records the loaded model and context size |
| Model files | GGUF files under the llama.cpp models directory, `LLAMA_CACHE`, and directories the user registered; matched to catalog entries by SHA-256 when known, otherwise by file name |

The `LlamaCpp` model records `installed`, `path`, `build`, `commit`, `backends[]`, `running_servers[]`, `local_models[]`, and `problems[]` (for example a CUDA build without a usable driver).

### 5.2 Version requirements

Catalog entries may declare `llama_cpp.min_build`. The recommender marks candidates whose requirement exceeds the installed build as `needs newer llama.cpp` rather than hiding them, and `doctor` says which build would unlock them.

### 5.3 Install and launch (phase 2)

Specified in section 15. Phase 1 only needs the detection above and the `flags` renderer in section 9.5.

## 6. Model catalog

### 6.1 Principles

- Curated YAML in the repository, one file per model family, reviewed by pull request. No scraping in the shipped data.
- Every factual field has a `source` (a URL) or is derived from GGUF headers.
- A JSON schema validates every file in CI and through `llamafit catalog validate`.
- A `refresh` command updates volatile fields (file sizes, SHA-256, GGUF facts) from the Hugging Face API and rewrites the YAML deterministically so diffs stay reviewable.
- Users can override or add entries locally with the same schema (section 6.4).

### 6.2 Entry schema

```yaml
id: qwen3.8-flash-next                      # stable slug, lowercase, unique
name: Qwen3.8-Flash-Next
vendor: Alibaba Qwen
family: qwen3
release_date: 2026-08-26
license: {spdx: Apache-2.0, url: https://...}
params: {total_b: 125, active_b: 6, ngram_table_b: 51}   # active_b equals total_b for dense models
architecture:
  class: moe-hybrid                         # dense | moe | moe-hybrid | dense-hybrid
  gguf_arch: qwen4exp                       # value of general.architecture
  notes: "Gated DeltaNet on 3 of 4 layers; F16 KV only"
context: {native: 262144, extended: 1048576, extended_method: yarn}
capabilities: [coding, thinking, vision, tools, multilingual, long-context]
use_cases: [coding, reasoning, multimodal]  # general | coding | reasoning | chat | multimodal | embedding
quality:
  baseline: 84                              # 0-100, section 11.1
  benchmarks:
    - {name: SWE-bench Pro, score: 62.5, source: https://...}
    - {name: LiveCodeBench v6, score: 91.9, source: https://...}
sampling: {temp: 1.0, top_p: 0.95, top_k: 20, min_p: 0.0, presence_penalty: 0.0}
chat_template: {reasoning_format: deepseek, thinking_toggle: enable_thinking}
llama_cpp:
  min_build: 10800
  kv_types_allowed: [f16]                   # architectures that assert on quantised KV
  requires: {mmproj: optional, mtp: optional, lazy_mode: recommended}
  quirks: ["--lazy-mode on streams the n-gram table from disk"]
sources:
  - repo: unsloth/Qwen3.8-Flash-Next-GGUF
    kind: gguf
    trust: unsloth                          # official | unsloth | bartowski | community
    quants:                                 # curated: the name only; the rest is refreshed
      - {name: UD-Q4_K_XL}
      - {name: UD-Q2_K_XL}
    extras:
      - {role: mmproj, file: mmproj-F16.gguf}
      - {role: mtp, file: mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}
measured:                                   # optional, from the calibration set
  - {profile: rtx4060-8gb-ddr5-128gb, quant: UD-Q4_K_XL, gen_tps: 13.5, pp_tps: 52,
     flags: "-ub 1024 --no-mmproj-offload", source: docs/calibration/2026-09-09.md}
```

Field rules:

- `params.active_b` drives speed; `params.total_b` and the quant `bytes` drive memory. Optional `ngram_table_b` and similar auxiliary tables are budgeted separately when `llama_cpp.requires.lazy_mode` allows streaming.
- `quants[].bpw` is the quant's bits per weight, so it divides the weight bytes by `params.total_b`, not the whole download. A file may carry a large auxiliary table that `params.total_b` deliberately excludes, and counting it would report a four-bit quant as a seven-bit one. The weight bytes are the file bytes less whatever the facts report as streamable tables.
- `quality.baseline` is the model's quality on its primary use case, 0 to 100, set by the curator from published benchmarks with the sources listed. The contribution guide defines the rubric (section 11.1).
- `quants[].gguf_facts` is filled by `refresh` (section 7) and is what the budget uses; when absent, the budget falls back to family-level formulas and lowers confidence.

### 6.3 Seed

The initial catalog is written from primary sources only: model cards and technical reports, vendor announcements, the Hugging Face repositories that publish the GGUF files, and the models measured on the reference machine. No third-party catalog is imported. Every file names its sources in the entries themselves, and `refresh` reaches Hugging Face directly for the volatile fields.

### 6.4 Custom models

`{data_dir}/custom_models.yaml` (or `LLAMAFIT_CUSTOM_MODELS`) holds entries in the same schema. An entry with an existing `id` replaces the bundled one; new ids are appended. `catalog validate` checks the custom file too.

### 6.5 Refresh

`llamafit catalog refresh [--model ID] [--dry-run]` queries `https://huggingface.co/api/models/{repo}?blobs=true` for file sizes and LFS SHA-256, and fetches GGUF header facts for each quant (section 7). Network failures leave the data untouched and are reported per repository. A `--check` mode exits non-zero when the committed data is stale, for a scheduled CI job.

The curated `<family>.yaml` is never written by a machine. It is hand-written prose and data with comments and sources in it, and a program that rewrites it destroys that on the first run. The volatile fields instead live in a sibling `<family>.facts.json`, keyed by model id and then by quant or extra name, written atomically through a temporary file in the same directory, and merged back onto the models by the loader. Two consequences follow and both are required: a refresh that changes nothing rewrites nothing, and a facts file that does not say what it appears to say produces a `Problem` naming the model, the quant and the field, never a quietly half-filled model. A model whose sources reuse a quant name cannot be represented in a name-keyed document at all, so it is reported and receives no merged facts rather than having its colliding quants filled from one shared entry.

## 7. GGUF facts

### 7.1 Reading headers

`gguf.header` parses the GGUF binary header (magic, version, tensor count, key-value metadata, tensor infos) from a local file or from a URL using HTTP range requests, reading only as many bytes as the header needs (typically under 4 MB even for very large models). Results are cached under the cache directory keyed by URL and ETag, or by path, size and mtime for local files.

A quant published as several shards is read as a set, never as its first file. The shard that
declares `split.no = 0` carries the whole key-value metadata block and the others carry three keys
each, so metadata comes from that shard alone; the tensor table is the union of every shard's, and
every derived byte figure comes from that union. A first shard holding metadata and no tensors at
all is a normal and observed layout, so reading only the first file yields complete architecture
metadata beside an empty tensor table, and every byte figure silently becomes zero. Where the
metadata declares `split.count` and `split.tensors.count`, both are checked against the set that
arrived; an incomplete set is an error, because facts derived from it are quietly too small rather
than visibly absent.

### 7.2 Derived facts

From the header LlamaFit derives, per quant:

| Fact | Derivation |
|---|---|
| `n_layer`, `n_embd`, `n_vocab`, `n_head`, `n_head_kv`, `head_dim` | `{arch}.block_count`, `{arch}.embedding_length`, tokenizer vocab size, `{arch}.attention.head_count(_kv)`, key length |
| `attention_layers` | the number of blocks that own `attn_k.weight` and `attn_v.weight` tensors, which is exact for every architecture including hybrids, since a block without a KV projection cannot hold a KV cache; the architecture's interval or layer-type array, and then the family rule in the catalog, are fallbacks for a file whose tensor names do not follow the convention. Recorded alongside as `attention_layers_source` so a reader can tell a counted figure from an assumed one |
| `sliding_window` | `{arch}.attention.sliding_window` when the architecture declares one, else null. Recorded but not yet used: a model with sliding-window attention holds a full-length KV cache on only a fraction of its layers, so treating every attention layer as full-context overstates the cache several-fold. Which layers slide is not in the header and comes from a per-architecture rule or the catalog |
| `n_expert`, `n_expert_used`, `shared_experts` | `{arch}.expert_count`, `{arch}.expert_used_count`, presence of `_shexp` tensors |
| `bytes_expert_weights` | sum of tensor sizes whose names contain `_exps` |
| `bytes_attention_weights` | sum of all other block tensors |
| `bytes_output_head`, `bytes_token_embd` | `output.weight`, `token_embd.weight` |
| `bytes_lazy_tables` | tensors the catalog marks as streamable (for example `per_layer_token_embd`) |
| `kv_bytes_per_token` | `2 × attention_layers × n_head_kv × head_dim × bytes(kv_type)` |
| `recurrent_state_bytes` | from the architecture's state dimensions, else the catalog's measured value |

Tensor sizes come from the tensor info table (dimensions and type), so they are exact for the file, not estimates.

## 8. Memory budget

### 8.1 Definitions

A `Budget` is computed for a candidate under a placement (section 9) at a context `c` tokens, micro-batch `ub`, batch `b = max(2 × ub, 2048)`, KV type `kv`, with the vision projector present or not.

| Component | Pool | Formula |
|---|---|---|
| Attention and other non-expert weights | VRAM if `ngl` covers the layer, else RAM | exact bytes from GGUF facts, prorated by layers on GPU |
| Expert weights | RAM when `n_cpu_moe` covers the layer, else VRAM | exact bytes prorated |
| Lazy tables | disk (streamed) when lazy mode is used, else RAM | exact bytes |
| KV cache | pool of the attention layers | `kv_bytes_per_token(kv) × c` |
| Recurrent state | VRAM with the layers | constant |
| Compute buffer | VRAM | `base(ub) + k(ub) × ub × c / 1024` (section 8.2) |
| Output buffer | RAM | `n_vocab × 4 × b` |
| Vision projector | VRAM when offloaded, RAM otherwise | file bytes plus `projector_compute_bytes` |
| Runtime overhead | VRAM | `cuda_context_bytes` (300 MiB default) when a GPU backend is used |
| Process overhead | RAM | 1 GiB default |

Required VRAM is the sum of the VRAM rows; required RAM likewise. Available VRAM is `vram_total − vram_used` at scan time, minus a `vram_reserve` (256 MiB default) for the desktop; available RAM is `memory.available`.

### 8.2 Compute buffer model

The compute buffer is the least predictable component. The initial model is piecewise linear in `ub` with a context term, fitted to the calibration set:

| `ub` | `base` (MiB) | `k` (MiB per 1K tokens of context per 1K `ub`) |
|---|---|---|
| 512 | 1090 | 1.5 |
| 1024 | 1240 | 3.0 |
| 2048 | 2080 | 8.0, rising to 25 above 64K context |

Values are in `constants.py` with their provenance. Any measured budget from phase 3 supersedes the formula for that model on that host.

### 8.3 Fit utilisation and verdicts

`utilisation_pool = required_pool / available_pool` for VRAM and RAM. The verdict is decided on the worst pool:

| Utilisation | Verdict | Meaning |
|---|---|---|
| ≤ 0.65 | Comfortable | room for a bigger context or a second model |
| ≤ 0.85 | Fits | the intended configuration runs as planned |
| ≤ 0.95 | Tight | runs, but a browser or a second process can push it over |
| > 0.95 | Does not fit | the driver would page or the load would fail |

A MoE-offload placement is not penalised for being MoE-offload: with all attention and KV on the GPU and experts in RAM it can be Comfortable, because measured speeds there are close to the GPU-only case for the same active parameters. CPU-only placements are capped at Fits because measured speeds there are rarely comfortable.

### 8.4 Driver paging

On Windows and Linux with NVIDIA drivers, a VRAM allocation that exceeds the card does not fail; the driver pages to system RAM and speed collapses silently. LlamaFit models this explicitly: any placement whose required VRAM exceeds available VRAM is `Too Tight`, and the explanation says why the server would still start.

## 9. Placement planner

### 9.1 Run modes

| Mode | Meaning |
|---|---|
| `gpu` | all weights and KV in VRAM (`-ngl 99`) |
| `moe-offload` | attention, KV, shared experts on GPU; routed experts in RAM (`-ngl 99 --n-cpu-moe N`) |
| `hybrid` | some layers on GPU, the rest on CPU (`-ngl n`) |
| `cpu` | everything in RAM (`-ngl 0`) |
| `unsupported` | nothing fits, or the host lacks a backend the model needs |

### 9.2 Search order

For a candidate, the planner evaluates in this order and keeps the first verdict better than Too Tight, continuing to find the best one when several fit:

1. `gpu` at the requested context; then with KV `q8_0` if allowed; then at half context down to the user's minimum.
2. `moe-offload` (MoE models only) with all routed experts in RAM; then progressively fewer experts on CPU while VRAM allows.
3. `hybrid` with the largest `-ngl` that fits.
4. `cpu`.

Within a mode, the planner also chooses `ub` from `{2048, 1024, 512, 256}` largest first, dropping when the compute buffer breaks the fit, and places the vision projector on the GPU only when it fits with margin, else on the CPU (`--no-mmproj-offload`), else omits it and reports that vision is off.

### 9.3 Context tiers

The planner reports `max_context_fit` for the chosen mode and a tier table (16K, 24K, 32K, 40K, 48K, 64K, 96K, 128K, 192K, 256K) with the free VRAM each tier needs, so launch scripts can choose a tier at start time from live free VRAM (section 15.3).

### 9.4 Threads

`-t` is the number of performance cores (or physical cores when the OS does not distinguish); `-tb` equals `-t`. On hybrid Intel parts efficiency cores are excluded because measured generation is slower with them.

### 9.5 Flag rendering

`placement.flags` turns a `Placement` into an ordered argument list for `llama-server`: model and projector paths, `--alias`, host and port, `-c`, `-fa on`, `-ctk/-ctv`, `-ngl`, `--n-cpu-moe`, `-ot` overrides, `--fit off`, `--lazy-mode` when the catalog recommends it, `-t/-tb`, `-b/-ub`, sampling from the catalog, `--jinja`, reasoning options, `-np 1`, `--cache-ram`, plus catalog quirks. The same list feeds the presets (phase 2) and the `plan` output.

## 10. Speed estimation

### 10.1 Generation

Time per token is the sum of memory traffic per pool divided by that pool's effective bandwidth, plus fixed overheads:

```
bytes_vram = attention_bytes_on_gpu + shared_expert_bytes_on_gpu + active_expert_bytes_on_gpu
           + output_head_bytes + kv_bytes_per_token × working_context
bytes_ram  = active_expert_bytes_in_ram + attention_bytes_on_cpu + kv_bytes_on_cpu × working_context
t_token    = bytes_vram / (vram_bw × eff_vram) + bytes_ram / (ram_bw × eff_ram)
           + n_layer × layer_overhead + sampling_overhead
gen_tps    = 1 / t_token
```

`active_expert_bytes = bytes_expert_weights × n_expert_used / n_expert`. `working_context` defaults to 8K tokens for the board and to the user's requested context for `plan`. Initial constants (with provenance): `eff_vram 0.60` (the fraction of peak bandwidth that decode kernels reach on consumer GPUs in published llama-bench results), `eff_ram 0.70`, `layer_overhead 0.20 ms`, `sampling_overhead 1 ms` (fitted to the reference machine: 22 to 24 tokens per second for a 3B-active MoE at 4.5 bits per weight with experts in DDR5-4200, 13 to 14 for a 6B-active one). When a pool's bandwidth is unknown, a per-backend fallback applies to the whole model, in GB/s-equivalent: CUDA 250, Metal 150, HIP 200, Vulkan 120, SYCL 100, CPU arm64 80, CPU x86_64 60. These fallbacks are deliberately conservative and always labelled `estimated`.

### 10.2 Prompt processing

```
t_ubatch = 2 × active_params × ub / (tflops_fp16 × eff_pp)          # compute on the GPU
         + streamed_expert_bytes / pcie_bw                          # experts in RAM streamed per micro-batch
         + streamed_expert_bytes_cpu_path / (ram_bw × eff_ram)      # when the CPU computes them instead
pp_tps   = ub / t_ubatch
```

`streamed_expert_bytes` is the expert bytes touched per micro-batch, which for `ub ≥ 256` is nearly all of them. The reference machine gives 52 tokens per second at `ub 1024` and 25 at `ub 512` for a 78 GB expert set, which fixes the initial PCIe effective bandwidth at 4 GB/s for that host until phase 3 measures it.

### 10.3 Confidence

Every estimate carries one label, in this precedence: `measured` (a stored benchmark on this host for this model, quant and flags), `calibrated` (formula with calibration factors derived from this host's measurements), `estimated` (formula with defaults), `unsupported` (no backend or no fit). The label travels with the number into every interface, and a measured number always shows the date it was taken.

## 11. Quality, context and the composite score

### 11.1 Quality

`quality = clamp(baseline − quant_penalty + alignment_bonus, 0, 100)`

- `baseline` from the catalog. Rubric for curators: 90+ frontier open weights on their primary task; 80 to 89 strong current generation; 70 to 79 solid previous generation; 50 to 69 small or dated; below 50 experimental. Cite the benchmarks used.
- `quant_penalty`: Q8 0, Q6 1, Q5 2, Q4_K_XL and Q4_K_M 4, IQ4 6, Q3 10, IQ3 12, Q2 20, IQ2 24, IQ1 35. Dynamic (`UD-`) quants use the penalty of their base level minus 1.
- `alignment_bonus`: +5 when the model's primary use case matches the request, +3 per required capability it has beyond the request's use case, capped at +10; a missing required capability excludes the candidate entirely.

### 11.2 Context

`context_score = 100 × min(1, max_context_fit / requested_context)`, with `requested_context` defaulting to 8K for general and chat, 32K for coding and reasoning, 16K for multimodal. Candidates whose `max_context_fit` is below the user's `--min-context` are excluded.

### 11.3 Fit score

From the worst-pool utilisation `u`: 100 for `0.50 ≤ u ≤ 0.80`; linear down to 70 at `u = 0.20` (a model far smaller than the machine wastes it); linear down to 40 at `u = 0.98`; 0 above.

### 11.4 Speed score

`speed_score = 100 × min(1, gen_tps / target_tps)` with targets by use case: chat 30, general 25, coding 20, reasoning 15, multimodal 15, embedding 200 (prompt throughput instead of generation). Prompt throughput adds a modifier for coding and reasoning: −10 when `pp_tps < 100`, −20 when below 40.

### 11.5 Composite

| Use case | Quality | Speed | Fit | Context |
|---|---|---|---|---|
| general | 0.35 | 0.25 | 0.25 | 0.15 |
| coding | 0.40 | 0.20 | 0.20 | 0.20 |
| reasoning | 0.50 | 0.15 | 0.20 | 0.15 |
| chat | 0.25 | 0.40 | 0.25 | 0.10 |
| multimodal | 0.40 | 0.20 | 0.25 | 0.15 |
| embedding | 0.30 | 0.45 | 0.20 | 0.05 |

The rationale: coding needs quality and room for real repositories in the context; reasoning is dominated by quality because thinking tokens are cheap to wait for but expensive to get wrong; chat is judged mostly by how fast it feels; embeddings are throughput jobs. Weights live in `scoring/weights.py` and can be overridden in the config file.

## 12. Recommendation and explanations

### 12.1 Needs

```
Needs:
  use_case: coding
  required: [coding, tools]          # must have
  preferred: [thinking]              # bonus
  min_context: 32768
  max_download_gb: 120
  licenses: [Apache-2.0, MIT, Llama]  # empty = any
  languages: []                      # empty = any
  prefer: balanced | quality | speed # shifts weights by ±0.10 between quality and speed
```

### 12.2 Board

The board lists ranked candidates with: rank, model, quant, mode, verdict, VRAM and RAM used and utilisation, `max_context_fit`, generation and prompt tokens per second with confidence, quality, composite score, download size, installed marker. Ties keep the higher quality. Only the best quant per model appears by default; `--all-quants` shows every one.

### 12.3 Explanations

Every row can expand into an explanation composed from templates, never free text: which pool limits it, what the budget contains, why the mode was chosen, which needs it satisfies, and what would move it up a tier (for example "free 1.2 GB of VRAM to reach 64K context" or "the Q3 quant would fit entirely in VRAM at 18 tokens per second"). The web and TUI show the same text the CLI prints with `--explain`.

## 13. Interfaces

### 13.1 CLI

| Command | Purpose | Notable flags |
|---|---|---|
| `llamafit` | opens the TUI (falls back to `recommend` when not a TTY) | |
| `llamafit system` | print the host scan | `--json` |
| `llamafit doctor` | probe-by-probe report, llama.cpp status, what would unlock more | `--json` |
| `llamafit list` / `search <text>` | catalog browsing | `--use-case`, `--capability`, `--license`, `--vendor` |
| `llamafit info <model>` | one model: facts, quants, budget on this host per quant | `--quant`, `--context`, `--json` |
| `llamafit fit` | all models ranked by fit | `--perfect`, `--min-fit`, `--limit` |
| `llamafit recommend` | the board for the given needs | `--use-case`, `--require`, `--prefer`, `--min-context`, `--max-download`, `--license`, `--limit`, `--all-quants`, `--explain`, `--json` |
| `llamafit plan <model>` | placement, flags and command line for one model | `--quant`, `--context`, `--ub`, `--target-tps`, `--no-vision`, `--json` |
| `llamafit catalog validate|refresh|show` | catalog maintenance | `--check`, `--dry-run`, `--model` |
| `llamafit hardware list|show|validate|path` | hardware profiles | |
| `llamafit serve` | web dashboard and API | `--host 127.0.0.1`, `--port 8765`, `--open` |
| `llamafit install llama.cpp|model <id>` | phase 2 | `--backend`, `--dir`, `--quant`, `--yes` |
| `llamafit preset <model>` | phase 2, write launch scripts and `models.ini` | `--dir`, `--port` |
| `llamafit bench [model]` | phase 3 | `--all`, `--json` |

Global flags: `--json`, `--profile`, `--memory`, `--ram`, `--cpu-cores`, `--max-context`, `--no-color`, `--verbose`. Exit codes: 0 success, 1 user error with a message, 2 environment problem (documented in `docs/cli.md`).

### 13.2 TUI

Textual application with a header showing the host summary and a tab bar:

| Screen | Content | Keys |
|---|---|---|
| Board | the ranked table; detail pane on `Enter`; explanation toggle | `/` search, `f` fit filter, `s` sort, `u` use case, `c` capabilities, `l` license, `a` installed only |
| Needs | form: use case, required and preferred capabilities, minimum context, size and license limits, prefer slider; the board re-ranks live | `Enter` apply, `r` reset |
| Host | scan details, probes, llama.cpp status, running servers | `R` rescan |
| Plan | for the selected row: budget breakdown, context tier table, flags, command line, copy to clipboard | `p` from Board, `+/-` context, `v` toggle vision |
| Simulate | override VRAM, RAM, cores or pick a profile; `SIM` badge everywhere | `S` |
| Downloads (phase 2) | active download with progress, queue, history | `D` |
| Benchmarks (phase 3) | estimate versus measured, run a bench | `b` |

`t` cycles a small set of themes; `?` shows keys; `q` quits. All screens work at 80 columns.

### 13.3 Web

`llamafit serve` starts FastAPI on `127.0.0.1:8765` and serves a static single-page dashboard (plain HTML, CSS and JavaScript, no build step) with the same five panels. The API:

| Endpoint | Returns |
|---|---|
| `GET /health` | ok and version |
| `GET /api/v1/system` | `Host` and `LlamaCpp` |
| `GET /api/v1/models` | catalog with per-host scoring; query parameters mirror the CLI (`use_case`, `require`, `min_context`, `limit`, `sort`, `include_too_tight`, `max_context`, `profile`) |
| `GET /api/v1/models/top` | the board |
| `GET /api/v1/models/{id}` | `info` |
| `POST /api/v1/plan` | `Plan` for a model, quant and context |
| `POST /api/v1/scan` | rescan |
| `GET /api/v1/profiles` | bundled and user profiles |
| phase 2 and 3 | `/api/v1/install/*` and `/api/v1/bench/*` with server-sent events for progress |

Responses are the JSON of the same pydantic models the CLI prints, so `--json` and the API never disagree. The server refuses to bind to non-loopback addresses unless `--host` is given explicitly, and prints a warning when it is.

## 14. Data and state

| Item | Location (via `platformdirs`, app name `llamafit`) |
|---|---|
| Config | `config.toml`: default use case, download directory, llama.cpp directory, weights overrides, theme |
| Catalog cache | `cache/catalog/` (refresh results, GGUF header facts keyed by URL and ETag) |
| Hardware profiles | `data/profiles/*.json` (user), bundled in the package |
| Custom models | `data/custom_models.yaml` |
| Benchmarks (phase 3) | `data/benchmarks.sqlite` |
| Downloads (phase 2) | `~/llamafit/models` by default, configurable |
| Logs | `log/llamafit.log`, rotated |

## 15. Phase 2 — installer and presets (summary, to be planned separately)

1. **llama.cpp releases.** Query the GitHub releases API, pick the asset for OS, architecture and backend (Windows: `cuda-12.x`, `vulkan`, `cpu`; macOS: `arm64` Metal, `x64`; Linux: `cuda`, `vulkan`, `cpu`, `rocm` when published), download with resume, verify the published checksum, extract into the managed directory, write `VERSION.txt`, optionally add to `PATH` (user scope, with an undo note). Never overwrite an installation LlamaFit did not create without confirmation.
2. **Model download.** Parallel range downloader (16 to 32 workers, resumable, throttle option) from Hugging Face with LFS SHA-256 verification, split-file awareness, projector and MTP extras, disk-space check before starting, and a downloads history.
3. **Presets.** From a `Plan`, write `start-<id>.cmd` (Windows) or `start-<id>.sh` (POSIX) with a context tier chosen from live free VRAM at start, plus a `models.ini` section for router mode, plus a README snippet with the endpoints. Templates live in `presets/` and are tested by rendering golden outputs.
4. **Launch.** `llamafit launch <model>` runs the preset, waits for `/health`, prints the URLs, and can stop the server it started.

## 16. Phase 3 — verifier and calibration (summary, to be planned separately)

1. `llamafit bench <model>` runs `llama-bench` for `pp` and `tg` at the planned flags, then starts a server with the same flags and issues three fixed requests (short generation, one thousand token prompt, tool call), recording generation and prompt throughput, time to first token, peak VRAM from the vendor tool, and the budget lines from the server log.
2. Results go to SQLite with host fingerprint, llama.cpp build, model, quant, flags.
3. `calibrate` fits `eff_ram`, `eff_vram`, `pcie_bw`, `layer_overhead` and per-`ub` compute-buffer constants for the host from stored results and stores them in the host's profile; estimates then carry `calibrated`, and matching stored results carry `measured_local`.
4. The paging detector compares reported VRAM against the card's total during the run and flags the signature (VRAM within 3 percent of total and speed below 60 percent of estimate).
5. Sharing measurements to a community dataset is a later milestone with its own design.

## 17. Error handling and user experience

- Every failure raises a `LlamaFitError` subclass with `message`, `hint` and `command` (the exact external command that failed), rendered without a traceback unless `--verbose`.
- Probes never abort a scan; a failed probe becomes a `problems[]` entry with a hint, and `doctor` lists all of them.
- Network is optional: the bundled catalog works offline; `refresh` and `install` say clearly when they need the network.
- No silent defaults: when a number is assumed, the interface shows the confidence label and `--explain` shows the assumption.
- Long operations stream progress (download, bench) and can be cancelled cleanly.

## 18. Testing

- **Unit tests** for every core module with recorded probe outputs (nvidia-smi text, system_profiler JSON, `/proc/cpuinfo`, GGUF header bytes) as fixtures.
- **Golden host profiles** under `tests/golden/`: the reference machine (RTX 4060 8 GB, 128 GB DDR5, Windows 11) with the 2026-09-09 measurements, an Apple M3 Max 64 GB, a CPU-only 16 GB laptop, an RTX 4090 24 GB Linux box. Budget and speed tests assert within tolerance (memory ±5 percent, speed ±25 percent) against recorded measurements where they exist, and against pinned expected values elsewhere so regressions are caught.
- **Catalog tests**: schema validation of every bundled file, uniqueness of ids, every source URL well-formed, every quant with bytes.
- **Interface tests**: CLI via Typer's runner with snapshot comparison of `--json` output; API via `httpx` test client; TUI via Textual's `Pilot` for navigation and filtering.
- **CI**: GitHub Actions matrix on Ubuntu, macOS and Windows for Python 3.10 and 3.13, with ruff, mypy in strict mode, pytest with coverage, catalog validation, and a scheduled `catalog refresh --check`.

## 19. Repository and quality tooling

```
llamafit/
  pyproject.toml            hatchling build, entry point `llamafit`, dependency groups (tui, web, dev)
  README.md                 what it is, one-minute quick start, how it works in five lines, roadmap, screenshots
  LICENSE (MIT)  NOTICE     llama.cpp and model-weight license notes
  CHANGELOG.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md  SECURITY.md
  MODELS.md                 generated from the catalog by scripts/gen_models_md.py
  .github/workflows/ci.yml  release.yml (PyPI trusted publishing)  ISSUE_TEMPLATE/  PULL_REQUEST_TEMPLATE.md
  docs/                     how-it-works.md, cli.md, tui.md, web.md, platform-support.md, catalog.md,
                            custom-models.md, benchmarking.md, development.md, calibration/, specs/, plans/
  src/llamafit/             section 3.2
  tests/
  scripts/                  gen_models_md.py, record_fixtures.py
```

Dependencies, kept deliberately short: `typer`, `rich`, `textual`, `fastapi`, `uvicorn`, `pydantic`, `pyyaml`, `platformdirs`, `psutil`, `py-cpuinfo`, `httpx`. `numpy` is optional (bandwidth probe). Nothing else without a reason recorded in `docs/development.md`.

Style: ruff (line length 100, isort rules), mypy strict, docstrings on every public function in Google style, type hints everywhere, no bare `except`, no print outside interfaces.

## 20. Phases and milestones

| Phase | Deliverable | Definition of done |
|---|---|---|
| 1 | scan, catalog with seed, GGUF facts, budget, placement, speed, scoring, `system`, `doctor`, `list`, `search`, `info`, `fit`, `recommend`, `plan`, `catalog validate`, `hardware`, TUI Board/Needs/Host/Plan/Simulate, `serve` with the dashboard board | all tests green on three OSes; the reference machine's board puts Qwen3-Coder-Next and Qwen3.8-Flash-Next where the measurements say, with estimates within tolerance |
| 2 | `install llama.cpp`, `install model`, `preset`, `launch`, Downloads screen, install API | a fresh Windows, macOS and Linux VM goes from nothing to a running server through LlamaFit alone |
| 3 | `bench`, calibration, Benchmarks screen, paging detector | estimates on the reference machine within 15 percent after calibration |

## 21. Decisions log

| Decision | Choice | Reason |
|---|---|---|
| Name | LlamaFit, CLI `llamafit`, PyPI `llamafit` | user's choice; free on PyPI, says what it does |
| Web stack | FastAPI plus static HTML/JS | pip-only, one API shared with the TUI |
| Console | Textual TUI plus `--json` CLI | dashboard parity with the web UI |
| Positioning | advisor, installer, verifier for llama.cpp | nothing else takes a machine from scan to a verified running server |
| License | MIT | the same license as llama.cpp, simplest for contributors |
| Python floor | 3.10 | pattern matching and modern typing without excluding current distributions |
| Catalog format | YAML per family with JSON schema, primary sources only | reviewable diffs, human editing, no inherited errors |
| Default port | 8765 | unassigned locally, easy to remember |
| LLM inside the tool | none | user requirement; everything must be deterministic and explainable |
