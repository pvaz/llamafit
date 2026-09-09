# Command-line reference

LlamaFit is one executable, `llamafit`, with subcommands. Every command accepts the global
options that have shipped — the table marks the ones that arrive with a later phase — and every
command can print JSON instead of tables so scripts can consume it. The JSON is the serialised
data model of the command's result; the tables show the same data.

Every heading names the phase its command belongs to and says whether it has shipped.
Phases 1A and 1B have; the rest are documented so their shape is agreed before they are
built, and their flags may still change. The [roadmap](../ROADMAP.md) says what lands when.

## Global options

| Option | Effect |
|---|---|
| `--json` | Print machine-readable JSON instead of tables. |
| `--verbose`, `-v` | Log details to `llamafit.log` in the log directory (see [architecture.md](architecture.md)) at debug level and show tracebacks for unexpected errors. |
| `--no-color` | Disable colours; useful when piping into files. |
| `--version` | Print the version and exit. |
| `--profile NAME\|FILE` | Score against a hardware profile instead of the live scan (phase 1C). |
| `--memory SIZE`, `--ram SIZE`, `--cpu-cores N` | Override single values of the scan for a what-if (phase 1C). Sizes accept `8G`, `7.5GiB`, `512M`. |
| `--max-context N` | Cap the context used for budgets and scores (phase 1C). |

## Exit codes

| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | A user or configuration error; the message says what to change. |
| 2 | An environment problem: llama.cpp missing, a required tool missing, an installation missing the data that ships inside it, or `doctor` found an error. |

## Commands

### `llamafit system` — phase 1A, shipped

Scan the machine and show CPU, memory, GPUs, disks and the llama.cpp installation.

| Option | Effect |
|---|---|
| `--no-measure` | Skip the RAM bandwidth measurement (about 50 ms of reading); the estimate from DDR facts or the assumed default is used instead. |

```
$ llamafit system
Host
  OS         windows Windows-11-10.0.26200-SP0 (x86_64)
  CPU        Intel(R) Core(TM) i9-14900KF; 24 cores / 32 threads, 8 performance cores; avx2
  Memory     127.8 GiB total, 100.8 GiB available; DDR5 4200 MT/s, 4 modules, 2 channels; bandwidth 57.3 GB/s (measured)
  GPU 0      NVIDIA GeForce RTX 4060 (cuda); 8.0 GiB VRAM, 7.2 GiB free; 272.0 GB/s and 15.0 TFLOPS fp16 (spec), driver 610.88
  Disk       C:\Dev\Projectos Pessoais\2026\llamafit: 355.6 GiB free of 1.8 TiB
  Disk       D:\llama.cpp\bin: 417.9 GiB free of 1.8 TiB

llama.cpp
  Installed     yes, b10867 at D:\llama.cpp\bin
  Backends      cuda, rpc, cpu
  Local models  5
```

With `--json` the output is a `SystemReport`: `{"host": {...}, "llamacpp": {...}, "version": "..."}`.
Every bandwidth figure carries its `bandwidth_source`: `measured`, `estimated`, `assumed` or `unknown`.

### `llamafit doctor` — phase 1A, shipped

Show every probe that ran, whether it succeeded, and findings ordered worst first: errors
(llama.cpp missing), warnings (a GPU without a matching backend, an assumed bandwidth, a failed
probe, low disk space) and confirmations (llama.cpp installed, servers running). Each warning
carries a hint. Exit code 2 when there is an error, so `doctor` can gate scripts.

With `--json` the output is a `Diagnosis`: the report plus `findings`, each with `level`,
`title`, `detail` and `hint`.

### `llamafit list` — phase 1B, shipped

List the catalog, narrowed by any filters given.

| Option | Effect |
|---|---|
| `--use-case NAME` | Keep only this use case: `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding`. |
| `--capability NAME` | Keep only models with this capability; repeatable, and every one given must be present. |
| `--license SPDX` | Keep only these licence identifiers; repeatable. |
| `--vendor NAME` | Keep only this vendor, case-insensitive. |
| `--search TEXT` | Case-insensitive substring match on id, name, vendor or family. |
| `--limit N` | Show at most this many. |

```
$ llamafit list
                                   Models
┌───────────────────────┬──────────┬────────┬─────────┬─────────────────────┐
│ ID                    │ Quality* │ Params │ Context │ Capabilities        │
├───────────────────────┼──────────┼────────┼─────────┼─────────────────────┤
│ qwen3-coder-next      │       85 │  80/3B │    256K │ coding, tools +1    │
│ qwen3.8-flash-next    │       84 │ 125/6B │    256K │ coding, thinking +4 │
│ gemma-3-27b-it        │       74 │    27B │    128K │ vision +3           │
│ llama-3.1-8b-instruct │       62 │     8B │    128K │ tools +2            │
│ qwen3-0.6b            │       35 │   0.6B │     32K │ coding, tools +1    │
└───────────────────────┴──────────┴────────┴─────────┴─────────────────────┘
   Quality is the editorial baseline, before any quantisation penalty; run
   `llamafit info <model>` for the sourced benchmarks behind it. Params is
total/active billions for a mixture-of-experts model, or one number when they
                                 are equal.
```

Columns: id, quality, parameters (one number for a dense model, total/active for a
mixture-of-experts one), native context and capabilities (as many complete names as fit, plus
a `+N` marker for the rest) — sorted by quality, the column that says why. Vendor, licence and
quant count are left out of the table; all three are one `info` or `--json` away. At narrow
widths, context and then capabilities give way first so the id and quality never get cut or
blanked.

An unknown `--use-case` or `--capability` exits 1 and lists the values that would have worked.
Nothing matched prints one line saying so, and exits 0: an empty result is an answer.

With `--json` the output is a list of `ModelSummary` objects, each with every capability, the
quant names, and the largest quant size that is known (`null` until `catalog refresh` has run).

### `llamafit search` `<text>` — phase 1B, shipped

`list --search` with the text as the argument, and no other filters.

```
$ llamafit search coder
                               Models
┌──────────────────┬──────────┬────────┬─────────┬──────────────────┐
│ ID               │ Quality* │ Params │ Context │ Capabilities     │
├──────────────────┼──────────┼────────┼─────────┼──────────────────┤
│ qwen3-coder-next │       85 │  80/3B │    256K │ coding, tools +1 │
└──────────────────┴──────────┴────────┴─────────┴──────────────────┘
```

### `llamafit info` `<model>` — phase 1B, shipped

One model's curated facts (licence, parameters, context, architecture, capabilities, quality
with its sourced benchmarks, sources) and every quant it publishes, with its size, bits per
weight and GGUF architecture facts.

```
$ llamafit info qwen3-coder-next
                      Qwen3-Coder-Next (qwen3-coder-next)
Vendor            Alibaba Qwen
Family            qwen3
Release date      2026-02-03
Licence           Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
Parameters        80B total, 3B active
Context           262,144 tokens native
Architecture      moe-hybrid, gguf_arch=qwen3next
Notes             48 layers combining Gated DeltaNet linear-attention blocks
                  with Gated Attention (full-attention) blocks; 512 routed
                  experts, 10 used per token, plus 1 shared expert. Hidden
                  dimension 2,048.
Capabilities      coding, tools, long-context
Use cases         coding
Quality baseline  85
Benchmark         SWE-bench Verified: 70.6
                  (https://huggingface.co/Qwen/Qwen3-Coder-Next)
Benchmark         SWE-bench Pro: 44.3
                  (https://huggingface.co/Qwen/Qwen3-Coder-Next)
Benchmark         Terminal-Bench 2.0: 36.2
                  (https://huggingface.co/Qwen/Qwen3-Coder-Next)
Source 1          unsloth/Qwen3-Coder-Next-GGUF (gguf, trust=unsloth)

                     Quants
┌────────────┬─────────┬─────────┬──────────────┐
│ Name       │    Size │     BPW │ Facts        │
├────────────┼─────────┼─────────┼──────────────┤
│ UD-Q4_K_XL │ unknown │ unknown │ not read yet │
└────────────┴─────────┴─────────┴──────────────┘
```

Size, bits per weight and facts read `unknown` and `not read yet` until `catalog refresh` has
filled them in for that quant; the bundled catalog carries no refreshed facts today, which is
why the example shows none. See
[catalog.md](catalog.md#where-the-facts-live) for where those numbers live.

An unknown id exits 1, naming the catalog id that shares the longest prefix with it when one
shares enough of it to be worth naming:

```
$ llamafit info qwen3-coder
no model named 'qwen3-coder' in the catalog
Hint: Did you mean: qwen3-coder-next?
```

Per-host budgets per quant (`--quant`, `--context`) are phase 1C; nothing here estimates one.

With `--json` the output is a `ModelDetail`: the whole catalog entry, plus every quant with its
`bytes`, `bpw`, `files` and `facts`.

### `llamafit catalog` `validate|refresh|show` — phase 1B, shipped

- `validate [FILE]`: check the bundled catalog files and the custom models file (or just
  `FILE` when given); exit 1 on any problem, listing each. See
  [catalog.md](catalog.md#validating) for what is and is not checked.
- `refresh [--model ID] [--dry-run] [--check]`: fill file names, sizes, checksums, bits per
  weight and GGUF facts from Hugging Face, writing them to the generated `<family>.facts.json`
  files and never to the curated YAML. `--dry-run` computes without writing; `--check` does the
  same and exits 1 if anything would change, for a scheduled job. Either way it exits 1 when a
  repository could not be read. Needs network access.
- `show <model> [--yaml]`: print the raw catalog entry, as JSON by default or as YAML.

```
$ llamafit catalog validate
4 file(s) checked, no problems found.

$ llamafit catalog refresh --model qwen3-0.6b --dry-run
qwen3-0.6b: sources[0].quants[0].files, sources[0].quants[0].bytes,
sources[0].quants[0].sha256, sources[0].quants[0].gguf_facts,
sources[0].quants[0].bpw
```

`refresh` prints one line per model that changed, naming the fields; warnings and errors come
first, each naming the model. A repository that cannot be listed leaves its whole model exactly
as it was and exits 1, rather than blanking out fields that took a real read to fill in.

### `llamafit fit` — phase 1C

Every model ranked by fit on this machine, best quant per model. Options: `--min-fit
comfortable|fits|tight`, `--limit N`, `--all-quants`, `--include-does-not-fit`.

### `llamafit recommend` — phase 1C

The board for your needs.

| Option | Effect |
|---|---|
| `--use-case general\|coding\|reasoning\|chat\|multimodal\|embedding` | sets the score weights and defaults |
| `--require CAP` (repeatable) | capability the model must have: `coding`, `thinking`, `vision`, `tools`, `multilingual`, `long-context`, `embeddings` |
| `--prefer CAP` (repeatable) | capability that earns a bonus |
| `--min-context N` | exclude candidates that cannot fit this context |
| `--max-download SIZE` | exclude larger downloads |
| `--license SPDX` (repeatable) | allowed licenses |
| `--prefer-quality`, `--prefer-speed` | shift 0.10 of weight between quality and speed |
| `--limit N`, `--all-quants`, `--explain` | output control |

### `llamafit plan <model>` — phase 1C

Placement, budget by component, context tier table, chosen flags and the full `llama-server`
command line for one model on this host. Options: `--quant NAME`, `--context N`, `--ub N`,
`--no-vision`, `--target-tps N` (what would be needed to reach a speed).

### `llamafit hardware list|show|validate|path` — phase 1C

Manage hardware profiles; see [hardware-profiles.md](hardware-profiles.md).

### `llamafit serve` — phase 1D

Start the web dashboard and JSON API. Options: `--host 127.0.0.1`, `--port 8765`, `--open`
(open the browser). Binding to any host other than loopback prints a warning; there is no
authentication. See [web.md](web.md).

### `llamafit` (no command) — phase 1D

Today it prints the help and exits. From phase 1D it opens the terminal dashboard, and in a
non-interactive terminal behaves like `recommend`. See [tui.md](tui.md).

### `llamafit install llama.cpp|model <id>` — phase 2

- `install llama.cpp [--backend cuda|vulkan|metal|hip|cpu] [--dir PATH] [--add-to-path] [--yes]`
- `install model <id> [--quant NAME] [--dir PATH] [--workers N] [--yes]`

### `llamafit preset <model>` — phase 2

Write launch scripts and a router preset section for a plan. Options: `--dir PATH`,
`--port N`, `--quant NAME`, `--context N`.

### `llamafit launch <model>` — phase 2

Run a preset, wait for `/health`, print the endpoints; `--stop` stops a server LlamaFit started.

### `llamafit bench [model]` — phase 3

Measure generation and prompt throughput at the planned flags, store the result, and print
estimated versus measured. Options: `--all`, `--quant NAME`, `--context N`, `--json`.

## Environment variables

| Variable | Effect |
|---|---|
| `LLAMAFIT_HOME` | Put config, data, cache, logs and downloads under one directory. |
| `LLAMAFIT_CUSTOM_MODELS` | Path of the custom models file (default under the data directory). |
| `LLAMA_CPP_PATH` | Directory (or install root) that holds `llama-server`; checked before `PATH`. |
| `LLAMA_SERVER_PORT` | First port to probe for a running server (then 8080, 8081, 8098). |
| `LLAMA_CACHE` | llama.cpp's model cache directory; its GGUF files are listed as local models. |
| `NO_COLOR` | Same as `--no-color`. |

## Files

See [architecture.md](architecture.md) for where configuration, caches, profiles, downloads
and logs live on each operating system.
