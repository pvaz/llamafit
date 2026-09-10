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
| `--language TAG` | Speak this language, for example `pt_PT`. It is read before anything is rendered, so `--language pt_PT --help` comes out in Portuguese too. Without it LlamaFit reads `LLAMAFIT_LANGUAGE`, then the operating system's locale, then falls back to English. A language it does not have falls back to English and names the ones it does have; a request served by another region's catalog says so. The notice goes to stderr, so `--json` stays machine-readable. See [translations.md](translations.md). |
| `--version` | Print the version and exit. |
| `--profile NAME\|FILE` | Score against a hardware profile instead of the live scan. The profiles ship now (`llamafit hardware`); the flag arrives with the commands that score, in phase 1C. |
| `--memory SIZE`, `--ram SIZE`, `--cpu-cores N` | Override single values of the scan for a what-if (phase 1C, as above). Sizes accept `8G`, `7.5GiB`, `512M`. |
| `--max-context N` | Cap the context used for budgets and scores (phase 1C). |

### What `--json` does and does not translate

A JSON document holds two kinds of string, and LlamaFit treats them differently on purpose.

**Keys and enumerated values never change with the language.** Field names, a finding's
`level` (`ok`, `warn`, `error`), `bandwidth_source` (`measured`, `estimated`, `assumed`,
`unknown`), probe names, backend names, capability and use-case ids, and model ids are
identifiers. A script matches on them, and the same script works whatever the operator's
locale says.

**Prose does change with the language.** A finding's `title`, `detail` and `hint`,
`llamacpp.problems` and a probe's `error` are sentences written for a person, and they are
the same strings the tables print. Numbers inside that prose carry the language's own
separators too, so a Portuguese `detail` reads `67,2 GB/s`. Do not parse them: read the
typed fields beside them, which is what those are for.

A pipeline that needs the prose to stay put should pass `--language en`, which pins it
whatever the machine's locale says.

**No direction marks, ever.** The tables put Unicode isolates around identifiers when the
language is written right to left, so that `--verbose` inside an Arabic sentence does not
reach a reader as `verbose--`. `--json` gets none of them: the marks are added by the
renderer at the last moment, and the services that build the JSON never see one, so
`llamafit --language ar --json doctor` differs from the English one in its words and in
nothing else. [translations.md](translations.md#right-to-left-languages) explains what the
tables do and how far it goes.

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

### `llamafit fit` — phase 1C, shipped

Every model in the catalog ranked by how well it uses this machine, and nothing else. There is
no use case, no weights and no score here: a coding model is not left out of a fit listing for
being a coding model. Models are sized for 32,768 tokens of context, and the context column is
the largest each one holds in the mode shown.

| Option | Effect |
|---|---|
| `--min-fit comfortable\|fits\|tight` | The worst verdict to list. Defaults to `tight`. |
| `--perfect` | Only configurations inside section 11.3's ideal band, between half and four fifths of the tightest pool. |
| `--limit N` | Show at most this many rows. |
| `--all-quants` | Show every quantisation instead of the best-fitting one per model. |

```
$ llamafit fit
                            Fit on this machine
 #   Model                  Quant        Fit     Runs             Ctx    Card      RAM
 1   qwen3-coder-next       UD-Q4_K_XL   fits    experts in RAM   61K    5.4 GiB   47.3 GiB
 2   qwen3-0.6b             Q8_0         fits    GPU              32K    5.5 GiB    2.2 GiB
 3   llama-3.1-8b-instruct  Q4_K_M       tight   split            35K    6.1 GiB    7.8 GiB
Sized for 32K tokens. The context column is the largest each one holds in the mode shown.
```

Models that cannot be placed at all are listed under **Not placed** with the reason, never
dropped. `--limit` cuts the ranked rows and never the reasons.

### `llamafit recommend` — phase 1C, shipped

The board for your needs: every model planned, sized, estimated and scored, in order. This is
the command the program exists for, and it scans the machine to answer.

| Option | Effect |
|---|---|
| `--use-case general\|coding\|reasoning\|chat\|multimodal\|embedding` | Sets the score weights, the speed target and the context candidates are scored against. Defaults to `general`. |
| `--require CAP` (repeatable) | Capability the model must have: `coding`, `thinking`, `vision`, `tools`, `multilingual`, `long-context`, `embeddings`, `audio`. A model without it is excluded, by name. |
| `--prefer balanced\|quality\|speed` | Moves a tenth of the weight between quality and speed. It leans the board; it does not replace the weights. Write the four numbers into the config file for that. |
| `--min-context N` | Exclude candidates that cannot hold at least this many tokens. |
| `--max-download SIZE` | Exclude larger downloads. Sizes look like `40G`, `7.5GiB`, `512M`. |
| `--license SPDX` (repeatable) | Licences the request will accept. A model with another one is excluded and still shown, with the licence it actually has. |
| `--limit N` | Show at most this many rows. Defaults to 10. |
| `--all-quants` | Show every quantisation instead of the best one per model. |
| `--no-vision` | Plan without a vision projector, freeing its memory for context. |
| `--explain` | Expand every row shown into the four scores, the weights, the quality it was built from, the memory budget line by line, the context ladder and where a token's time goes. Combine with `--limit 1` for one model. |

```
$ llamafit recommend --use-case coding
                                 Recommended
 #   Model                Quant        Score   Gen/s   Fit     Runs             Ctx
 1   qwen3-coder-next     UD-Q4_K_XL    93.9    24.1   fits    experts in RAM   61K
 2   qwen3-0.6b           Q8_0          72.1   114.6   fits    GPU              32K
 3   qwen3.8-flash-next   UD-Q4_K_XL    67.5    13.9   tight   experts in RAM   17K
Speeds are for 8K tokens of context so every row compares like with like; the context
column is the largest each one holds. Sized and scored for coding at 32K tokens.
Speeds are section 10's formula on its default constants. Nothing has been benchmarked
on this machine yet, so no figure here is a measurement.
Weights: quality 0.40, speed 0.20, fit 0.20, context 0.20.

                                  Not ranked
 Model                   Quant    Why not
 gemma-3-27b-it          Q4_K_M   not a coding model; its entry lists general, multimodal,
                                  chat, so ask for one of those
```

Which columns appear depends on the width of the terminal. The rank, model, quantisation and
score are never dropped; the rest are added in the order gen/s, fit, run mode, context,
quality, card, prompt/s, download size, RAM, each only while its whole content fits. A column
that cannot fit is dropped rather than shrunk.

**No row is ever dropped for failing.** A candidate the request excludes appears under **Not
ranked** with the reason: the job its entry does not offer, the capability it lacks, the
licence it carries, the download it exceeds, or the memory it needs. A shorter list would say
none of that.

**Nothing is labelled `measured`.** Section 10.3 reserves that word for a benchmark taken on
*this* machine, and phase 3's `bench` is what will store one. A catalog entry's own `measured`
block is a record from the curator's machine; `plan` shows it beside the estimate, and neither
is ever allowed to become the other.

### `llamafit plan` — phase 1C, shipped

`llamafit plan <model>`: place one model on this machine and print the command line that runs
it. The output is the memory budget component by component with the source of every figure,
the context ladder a launch script chooses from, where a token's time goes, any run the
catalog records for comparison, and last, on a line of its own, the `llama-server` command.

| Option | Effect |
|---|---|
| `--quant NAME` | Plan this quantisation, matched case-insensitively, instead of the one that scores best on this machine. |
| `--context N` | Size for this many tokens. When it does not fit, the plan sizes down **and** shows what the context you asked for would have cost, so the overflow is visible rather than merely retreated from. |
| `--ub N` | Set the micro-batch by hand. The whole budget is rebuilt around it, because it moves the compute buffer and with it the verdict, the largest context that fits and every rung of the ladder. |
| `--target-tps N` | A generation speed to answer against: whether this reaches it, and which context would. |
| `--no-vision` | Leave the vision projector out, freeing its memory for context. |

The context ladder has three states, not two. `fits` is a rung a launch script may take;
`pages` is a rung the driver will accept and then quietly page to system memory, which is
section 8.4's failure and the one a person cannot diagnose for themselves; `no room` is a rung
system memory could not absorb either. A table showing only yes and no would file the second
under the third and lose the only one worth warning about.

### `llamafit hardware` `list|show|validate|path` — phase 1B, shipped

Hardware profiles: a machine described in a file, so LlamaFit can answer for a machine
that is not this one. See [hardware-profiles.md](hardware-profiles.md) for the format.

- `list`: every profile LlamaFit can reach, bundled ones first, with the machine each
  describes and where its file is.
- `show <name|file> [--as-host]`: one profile, with the `provenance` line that says where
  its figures came from. `--as-host` prints the `Host` the profile substitutes — the same
  table `llamafit system` prints, opening with the line that says it is not this machine.
- `validate [FILE]`: check the bundled profiles and your own (or just `FILE`); exit 1 on
  any problem, listing each. Beyond the schema it reports a name two files claim, a GPU
  figure that contradicts the bundled specification table, and `match` rules that would
  never recognise the machine the profile itself describes.
- `path`: the directory your own profiles go in.

```
$ llamafit hardware list
                                Hardware profiles
┌─────────────────────────┬─────────┬───────────────────────┬───────────────────────┐
│ Name                    │ From    │ Machine               │ Description           │
├─────────────────────────┼─────────┼───────────────────────┼───────────────────────┤
│ reference-rtx4060-128gb │ bundled │ 128.0 GiB RAM, NVIDIA │ Reference machine:    │
│                         │         │ GeForce RTX 4060 with │ RTX 4060 8 GB,        │
│                         │         │ 8.0 GiB               │ i9-14900KF, 128 GiB   │
│                         │         │                       │ DDR5-4200, Windows 11 │
└─────────────────────────┴─────────┴───────────────────────┴───────────────────────┘

$ llamafit hardware show reference-rtx4060-128gb --as-host
Host
  SIMULATED  these figures come from the hardware profile reference-rtx4060-128gb; they
             are not this machine
  OS         windows Windows-11-10.0.26200-SP0 (x86_64)
  ...
```

`--json` gives the profile itself, the list of profiles, the `Host` (with `--as-host`) or
the problems, whichever the subcommand is about. A host built from a profile carries
`"simulated": true` and a `simulation` object naming the profile and its file, so a script
can tell a what-if from a scan without reading a heading.

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

### `llamafit preset` — phase 2, shipped

`llamafit preset <model>`: turn the plan for one model into files you can run and edit. A
plan is a command line somebody has to paste; a preset is a script they can double-click,
and it is what makes a recommendation survive contact with a real desktop.

Three files are written, into LlamaFit's own presets directory unless `--dir` says otherwise:

| File | What it is |
|---|---|
| `start-<id>.cmd` or `start-<id>.sh` | the launch script, for this operating system |
| `models-<id>.ini` | a section for a llama.cpp router's `models.ini` |
| `README-<id>.md` | the endpoints, the context ladder, and what changes when the machine does |

| Option | Effect |
|---|---|
| `--dir PATH` | Write the files here instead of in the presets directory. |
| `--port N` | Bind this port instead of llama.cpp's default 8080. |
| `--quant NAME` | Plan this quantisation instead of the one that scores best on this machine. |
| `--context N` | Size for this many tokens. It becomes the **top** of the ladder; the script never goes above it. |
| `--force` | Overwrite files you have edited, saying which ones it discarded. |

**The script chooses its context when it runs.** The plan behind it was computed while the
machine was idle; it runs when a browser has taken a gigabyte of the card. A configuration
that asks for more card memory than is free does not fail on an NVIDIA driver — it starts,
pages the overflow into system memory, and runs at a fraction of its speed while `/health`
answers and the log looks healthy. So the script carries section 9.3's ladder rather than a
number: it reads how much card memory is free, keeps 256 MiB back for the desktop, and takes
the largest rung that still fits. It never goes *above* the planned context, because the
planner weighed system memory and your request too and the script can only measure the card.

How free memory is read, and what happens when it cannot be:

| Machine | How | When it cannot |
|---|---|---|
| NVIDIA, any OS | `nvidia-smi --query-gpu=memory.free` | says so, uses the planned context, warns that it may page |
| AMD on Linux | `/sys/class/drm/card*/device/mem_info_vram_{total,used}` | same |
| Apple silicon | `vm_stat` free, inactive and speculative pages | same |
| AMD or Intel on Windows, Intel on Linux | nothing a script can ask | the script's header says so once, rather than warning on every run |

Exit codes of the generated script: **0** as llama-server exited, **3** the model file or
llama.cpp is not where the script expects it, **4** not even the smallest rung fits the free
card memory, so it stopped rather than page. `LLAMAFIT_CONTEXT` overrules the ladder for one
run.

**A file you have edited is never overwritten.** Each file carries a checksum of itself; a
file whose checksum no longer matches has been changed, and LlamaFit keeps it and names it.
`--force` replaces it and says that it did.

### `llamafit launch` — phase 2, shipped

`llamafit launch <model>`: run the model's preset **script** — not a command line rebuilt
for the occasion, so the ladder still chooses the context and your own edits still apply —
wait for `/health`, and print the endpoints. If no preset exists yet, one is written first.

| Option | Effect |
|---|---|
| `--dir PATH` | Look for the preset here. |
| `--port N` | The port to expect; otherwise the one the script itself names, which is what makes a hand-edited port work. |
| `--timeout N` | Seconds to wait for `/health`. The default of 180 is set by how long reading twenty gigabytes of weights takes. |
| `--stop` | Stop the server LlamaFit started for this model, and everything it started. |

A server already answering on that endpoint is reported rather than joined by a second one.
What was started is remembered by process id **and start time**, so `--stop` cannot aim at
whatever inherited a recycled id.

### `llamafit bench [model]` — phase 3

Measure generation and prompt throughput at the planned flags, store the result, and print
estimated versus measured. Options: `--all`, `--quant NAME`, `--context N`, `--json`.

## Environment variables

| Variable | Effect |
|---|---|
| `LLAMAFIT_HOME` | Put config, data, cache, logs and downloads under one directory. |
| `LLAMAFIT_CUSTOM_MODELS` | Path of the custom models file (default under the data directory). |
| `LLAMAFIT_PROFILES` | Directory holding your own hardware profiles (default `profiles` under the data directory). |
| `LLAMA_CPP_PATH` | Directory (or install root) that holds `llama-server`; checked before `PATH`. |
| `LLAMA_SERVER_PORT` | First port to probe for a running server (then 8080, 8081, 8098). |
| `LLAMA_CACHE` | llama.cpp's model cache directory; its GGUF files are listed as local models. |
| `NO_COLOR` | Same as `--no-color`. |

## Files

See [architecture.md](architecture.md) for where configuration, caches, profiles, downloads
and logs live on each operating system.
