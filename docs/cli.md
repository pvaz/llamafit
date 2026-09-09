# Command-line reference

LlamaFit is one executable, `llamafit`, with subcommands. Every command accepts the global
options, and every command can print JSON instead of tables so scripts can consume it. The
JSON is the serialised data model of the command's result; the tables show the same data.

The **Phase** column says when a command lands (see the [roadmap](../ROADMAP.md)). Commands
from phases that are not released yet are documented so their shape is agreed before they
are built; their flags may still change.

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
| 2 | An environment problem: llama.cpp missing, a required tool missing, or `doctor` found an error. |

## Commands

### `llamafit system` — phase 1A

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

### `llamafit doctor` — phase 1A

Show every probe that ran, whether it succeeded, and findings ordered worst first: errors
(llama.cpp missing), warnings (a GPU without a matching backend, an assumed bandwidth, a failed
probe, low disk space) and confirmations (llama.cpp installed, servers running). Each warning
carries a hint. Exit code 2 when there is an error, so `doctor` can gate scripts.

With `--json` the output is a `Diagnosis`: the report plus `findings`, each with `level`,
`title`, `detail` and `hint`.

### `llamafit list` — phase 1B

List the catalog. Filters: `--use-case`, `--capability` (repeatable), `--license` (repeatable),
`--vendor`, `--search TEXT`, `--limit N`. Table columns: id, quality, parameters (one number
for a dense model, total/active for a mixture-of-experts one), native context and capabilities
(as many complete names as fit, plus a `+N` marker for the rest) — sorted by quality, the
column that says why. Vendor, licence and quant count are left out of the table; all three are
one `info` or `--json` away. At narrow widths, context and then capabilities give way first so
the id and quality never get cut or blanked.

### `llamafit search` `<text>` — phase 1B

`list --search` with the text as the argument.

### `llamafit info` `<model>` — phase 1B

One model's curated facts (licence, parameters, context, architecture, capabilities, quality,
sources) and every quant it publishes, with its size, bits per weight and, once `catalog
refresh` has run, its GGUF architecture facts. An unknown id exits 1 naming the catalog id
that shares the longest prefix with it, when one shares enough of it to be worth naming.
Per-host budgets per quant (`--quant`, `--context`) are phase 1C.

### `llamafit catalog` `validate|refresh|show` — phase 1B

- `validate [FILE]`: check the bundled catalog and the custom models file against the schema
  (or just `FILE` when given); exit 1 on any problem, listing each.
- `refresh [--model ID] [--dry-run] [--check]`: update file names, sizes, checksums and GGUF
  facts from Hugging Face; `--dry-run` prints without writing; `--check` does the same and
  exits 1 if anything would change, for a scheduled CI job.
- `show <model> [--yaml]`: print the raw catalog entry, as JSON by default or as YAML.

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

Open the terminal dashboard. In a non-interactive terminal it behaves like `recommend`.
See [tui.md](tui.md).

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
