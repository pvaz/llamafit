# Command-line reference

LlamaFit is one executable, `llamafit`, with subcommands. Every command accepts the global
options below, and every
command can print JSON instead of tables so scripts can consume it. The JSON is the serialised
data model of the command's result; the tables show the same data.

Every heading names the phase its command belongs to. Every command on this page has shipped.
What has not is named in the [roadmap](../ROADMAP.md): two screens of the terminal dashboard,
the install and benchmark endpoints of the web API, and the path by which a benchmark corrects
the estimator.

## Global options

| Option | Effect |
|---|---|
| `--json` | Print machine-readable JSON instead of tables. |
| `--verbose`, `-v` | Log details to `llamafit.log` in the log directory (see [architecture.md](architecture.md)) at debug level and show tracebacks for unexpected errors. |
| `--no-color` | Disable colours; useful when piping into files. |
| `--language TAG` | Speak this language, for example `pt_PT`. It is read before anything is rendered, so `--language pt_PT --help` comes out in Portuguese too. Without it LlamaFit reads `LLAMAFIT_LANGUAGE`, then the operating system's locale, then falls back to English. A language it does not have falls back to English and names the ones it does have; a request served by another region's catalog says so. The notice goes to stderr, so `--json` stays machine-readable. See [translations.md](translations.md). |
| `--version` | Print the version and exit. |
| `--profile NAME\|FILE` | Answer for the machine a hardware profile describes instead of this one. Nothing about this machine is probed. `llamafit hardware list` names the profiles you have; a path to a `.json` file works too. |
| `--memory SIZE`, `--ram SIZE`, `--cpu-cores N` | Substitute one pool of the live scan for a what-if: the card's memory, the system memory, the physical core count. Everything else stays as scanned. Sizes accept `8G`, `7.5GiB`, `512M`. A size names a machine: `--ram 16GiB` is a 16 GiB machine, not this one with the difference subtracted. |
| `--max-context N` | Plan, report and score no context longer than this. It is not a machine; see below. |

### The four flags that replace a machine

`--profile`, `--memory`, `--ram` and `--cpu-cores` change *what the answer is about*.
Whatever they produce is marked, and marked in the data rather than only in a heading:

- The `Host` carries a `simulation` object naming the profile, its file and the pools that
  were overridden, and a `simulated` boolean computed from it.
- `recommend`, `fit` and `plan` carry the same two fields on their own documents, because
  a board carries no host and a script reading `--json` would otherwise have nothing to
  read. `"simulated": false` on a scan is part of the promise: absence is not evidence.
- On the terminal the first line above the table is red, says `SIMULATED`, and says which
  profile or which pools, before the reader meets a figure. It is above the *answer*, not
  above the table, so a board that ranked nothing carries it too — that is the answer most
  worth being warned about, because there is no figure in it to be suspicious of.

**What the sizes mean.** `--ram 16GiB` describes a 16 GiB machine and `--memory 4G` a 4 GiB
card. What this machine has in use is carried across unchanged onto a pool the same size or
larger, because a bigger card does not empty itself; onto a smaller one the same *share* is
carried, because this machine's open applications could not have fitted on it. A flag naming
the size the machine already has moves nothing at all.

They apply to `system`, `fit`, `recommend`, `plan` and `preset`, and to `llamafit` with no
arguments, which opens the dashboard already simulating that machine with its badge
showing. `doctor` **refuses** them: every line it prints is a probe that ran on the machine
LlamaFit is running on, a profile carries no probes, and a report about this machine under
a heading claiming another's would be worse than an error. `bench` refuses a simulated
machine for the same reason — a benchmark measures what it runs on. The commands that read
no machine at all (`list`, `search`, `info`, `catalog`, `hardware`, `llamacpp`, `install`,
`serve`) ignore them.

`llamafit --profile NAME system` is the shortest way to see what a profile becomes; it is
`llamafit hardware show NAME --as-host` with this machine's llama.cpp printed beside it.
The llama.cpp half is never substituted, because the binary and the GGUF files on this
disk are real whichever machine the numbers describe.

### `--max-context` is not a machine

It caps the context, so it belongs with `--min-context` rather than with the four above,
and it changes the question rather than the subject. It caps four things at once, which is
the only way it can be honest: the context the planner sizes for, the ladder
`plan` prints for a launch script to choose from, the largest context reported
(`max_context_fit`), and the context the section 11.2 score is measured against. A ceiling
below `--min-context` is refused by name rather than answered with "this model is too
short", which would blame the model for the request.

The web API spells it `max_context` on `/api/v1/models` and `/api/v1/models/top`, where it
sits beside `min_context` for the same reason.

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
| 2 | An environment problem: llama.cpp missing, a required tool missing, an installation missing the data that ships inside it, or `doctor` found an error. A command line that cannot be parsed at all — an unknown option, a missing argument — also exits 2, which is the convention every tool built on this argument parser follows. |

The global options above belong to `llamafit` itself, so they come **before** the command: `llamafit --json recommend`, not `llamafit recommend --json`. The second form exits 2 and says where the option goes.

## Commands

### `llamafit system` — phase 1A, shipped

Scan the machine and show CPU, memory, GPUs, disks and the llama.cpp installation.

| Option | Effect |
|---|---|
| `--no-measure` | Skip the RAM bandwidth measurement; the estimate from DDR facts or the assumed default is used instead. |
| `--refresh-bandwidth` | Measure RAM bandwidth again rather than reading back the figure kept for this machine, and keep the new one. |

```
$ llamafit system
                                               Host
OS      windows Windows-11-10.0.26200-SP0 (x86_64)
CPU     Intel(R) Core(TM) i9-14900KF; 24 cores / 32 threads, 8 performance cores; avx2
Memory  127.8 GiB total, 95.2 GiB available; DDR5 at 4200 MT/s, 4 modules across 2 channels;
        bandwidth 38.3 GB/s (measured)
GPU 0   NVIDIA GeForce RTX 4060 (cuda); 8.0 GiB VRAM, 7.3 GiB free; 272.0 GB/s and 60.4 TFLOPS fp16
        (spec), driver 610.88
Disk    C:\projects\llamafit: 356.5 GiB free of 1.8 TiB
Disk    D:\llama.cpp\bin: 417.9 GiB free of 1.8 TiB

                  llama.cpp
Installed     yes, b10867 at D:\llama.cpp\bin
Backends      cuda, rpc, cpu
Local models  5
```

The paths on the disk lines are the ones LlamaFit cares about — where it was run, where it
downloads to, where llama.cpp is — reported once per distinct volume, so a machine with all
three on one drive shows one line.

With `--json` the output is a `SystemReport`: `{"host": {...}, "llamacpp": {...}, "version": "..."}`.
Every bandwidth figure carries its `bandwidth_source`: `measured`, `estimated`, `assumed` or `unknown`.

RAM bandwidth is measured as the best of five short windows — contention can only make the
reading low, never high, so the largest of several is the machine rather than its load — and
then kept in the platform's cache directory, filed under the processor, the pool and the
modules' type, speed and channel count. Every later scan reads it back, which is why the same
board no longer reports a different tokens-per-second from one minute to the next. A figure
read back says so (`bandwidth_cached` in the JSON, *cached* in the table); `--refresh-bandwidth`
times it again.

`recommend --json` and `fit --json` carry a `machine` object with the same figures — what was
free, and the bandwidth with its source — so two answers taken minutes apart can be told apart
by reading them. The table says the same thing under the rows.

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
$ llamafit list --limit 5
                                         Models
┌────────────────────────┬──────────┬───────────┬─────────┬─────────────────────────────┐
│ ID                     │ Quality* │    Params │ Context │ Capabilities                │
├────────────────────────┼──────────┼───────────┼─────────┼─────────────────────────────┤
│ glm-5.3                │       92 │   744/40B │      1M │ coding, thinking, tools +1  │
│ kimi-k3                │       92 │ 2800/104B │      1M │ coding, thinking, vision +2 │
│ deepseek-v4-pro-0813   │       91 │  1600/49B │      1M │ coding, thinking, tools +1  │
│ glm-5.3-flash          │       89 │   320/18B │      1M │ coding, thinking, vision +2 │
│ deepseek-v4-flash-0731 │       88 │   284/13B │      1M │ coding, thinking, tools +1  │
└────────────────────────┴──────────┴───────────┴─────────┴─────────────────────────────┘
 Quality is the editorial baseline, before any quantisation penalty; run `llamafit info
  <model>` for the sourced benchmarks behind it. Params is total/active billions for a
              mixture-of-experts model, or one number when they are equal.
```

Without `--limit` that table is sixty-two rows long, one per model in the catalog. `list` adds
no caption saying what a limit cut, where `fit` and `recommend` do: there the rows below the
cut were ranked against this machine and their absence is worth knowing about, and here the
whole list is one command away.

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
┌──────────────────┬──────────┬────────┬─────────┬─────────────────────────────┐
│ ID               │ Quality* │ Params │ Context │ Capabilities                │
├──────────────────┼──────────┼────────┼─────────┼─────────────────────────────┤
│ qwen3-coder-next │       85 │  80/3B │    256K │ coding, tools, long-context │
└──────────────────┴──────────┴────────┴─────────┴─────────────────────────────┘
    Quality is the editorial baseline, before any quantisation penalty; run
    `llamafit info <model>` for the sourced benchmarks behind it. Params is
 total/active billions for a mixture-of-experts model, or one number when they
                                   are equal.
```

### `llamafit info` `<model>` — phase 1B, shipped

One model's curated facts (licence, parameters, context, architecture, capabilities, quality
with its sourced benchmarks, sources), every quant it publishes with its size, bits per weight
and GGUF architecture facts, and what each of those quants would cost on this machine.

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
Notes             48 layers combining Gated DeltaNet linear-attention blocks with Gated Attention
                  (full-attention) blocks; 512 routed experts, 10 used per token, plus 1 shared
                  expert. Hidden dimension 2,048.
Capabilities      coding, tools, long-context
Use cases         coding
Quality baseline  85
Benchmark         SWE-bench Verified: 70.6 (https://huggingface.co/Qwen/Qwen3-Coder-Next)
Benchmark         SWE-bench Pro: 44.3 (https://huggingface.co/Qwen/Qwen3-Coder-Next)
Benchmark         Terminal-Bench 2.0: 36.2 (https://huggingface.co/Qwen/Qwen3-Coder-Next)
Source 1          unsloth/Qwen3-Coder-Next-GGUF (gguf, trust=unsloth)

                                Quants
┌────────────┬──────────┬──────┬──────────────────────────────────────┐
│ Name       │     Size │  BPW │ Facts                                │
├────────────┼──────────┼──────┼──────────────────────────────────────┤
│ UD-Q4_K_XL │ 46.2 GiB │ 4.96 │ qwen3next, 48 layers, 512/10 experts │
└────────────┴──────────┴──────┴──────────────────────────────────────┘
```

Those three columns are filled in by `catalog refresh`, and the bundled catalog ships them
already refreshed: a generated `<family>.facts.json` sits beside every `<family>.yaml` in the
wheel, so the figures above are what a fresh install prints. A quant nobody has refreshed —
one you added yourself, or one added to the catalog since the last refresh — reads `unknown`,
`unknown` and `not read yet` until it has been. See
[catalog.md](catalog.md#where-the-facts-live) for where those numbers live.

An unknown id exits 1, naming the catalog id that shares the longest prefix with it when one
shares enough of it to be worth naming:

```
$ llamafit info qwen3-coder
no model named 'qwen3-coder' in the catalog
Hint: Did you mean: qwen3-coder-next?
```

`info` reads the machine as well as the catalog, so a third table follows the two above: one
line per quantisation, sized for this host.

```
                              On this machine
┌────────────┬────────────────┬─────────┬──────────┬───────┬─────┬───────┐
│ Quant      │ Runs           │    Card │      RAM │ Tok/s │ Ctx │ Fit   │
├────────────┼────────────────┼─────────┼──────────┼───────┼─────┼───────┤
│ UD-Q4_K_XL │ experts in RAM │ 6.3 GiB │ 46.4 GiB │  17.6 │ 48K │ tight │
└────────────┴────────────────┴─────────┴──────────┴───────┴─────┴───────┘
Sized for 32,768 tokens, with every figure from a placement computed for this machine.
`llamafit plan MODEL --quant NAME` opens one of these rows into the budget line by line, the
context ladder and the command line that runs it.
```

| Option | Effect |
|---|---|
| `--quant NAME` | Show only this quantisation, matched case-insensitively, in both the `Quants` table and the one above. An unknown name exits 1 and lists the ones the model publishes — the same sentence `plan --quant` gives, from the same matcher. |
| `--context N` | Size every quantisation for this many tokens instead of the 32,768 the planner defaults to, never above what the model holds or what `--max-context` allows. The caption names the figure that was actually used, not the one you typed. |

**One line each, and no more than that — the depth belongs to `plan`.** The question `info`
answers is the one before the plan: *which of these quantisations can this machine run, and how
fast*. The memory budget component by component, the context ladder, where a token's time goes
and the `llama-server` command line are all about a quantisation already chosen, and printing
them here for three quants would be three copies of `plan` under a different heading. So this
table stops at the verdict, both pools, a speed and the largest context each one holds, and the
caption says where the rest is.

Every figure comes from a placement the planner actually searched for, not from arithmetic on
another one. A quantisation nobody can size — no GGUF header read yet — is named under the table
with the reason rather than dropped from it, and one that fits nowhere gets a row saying
`nowhere` and `no room` rather than a blank. Its speed column reads `unknown`, not `0.0`: a
configuration that does not run is not a configuration that runs slowly.

The four flags that replace a machine work here as they do everywhere else, and the answer is
marked: a red `SIMULATED` line above the whole page, and `simulation` with `simulated` on the
JSON.

With `--json` the output is a `ModelDetail`: the whole catalog entry, plus every quant with its
`bytes`, `bpw`, `files` and `facts`, and now its `placement`, `speed` and `unplaceable_because`
as well. The web API's `GET /api/v1/models/{id}` returns the same shape; it does not size the
quants today, so those three fields come back empty there.

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
25 files checked, no problems found.

$ llamafit catalog refresh --model qwen3-0.6b --dry-run
No changes.
```

`No changes.` is what a refreshed catalog says, and the bundled one ships refreshed. A model
whose repository has moved on prints the fields instead, one line per model:
`qwen3-0.6b: sources[0].quants[0].bytes, sources[0].quants[0].sha256`.

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
| `--perfect` | Only configurations that score a perfect fit under section 11.3: at least half the machine's memory held in the model's own weights, and no pool past four fifths full. |
| `--limit N` | Show at most this many rows. |
| `--all-quants` | Show every quantisation instead of the best-fitting one per model. |
| `--sort`, `--search`, `--installed`, `--runs`, `--columns`, `--wide`, `--hide-excluded` | The view flags `recommend` has, with the same meanings; `--sort` takes the keys a fit row has a figure for: `score` (the fit score, the listing's own order), `context`, `size`, `card`, `ram`, `fit`, `model`, `quant`. `--min-fit` is a request flag here, so there is no view filter of that name. |

The listing has the board's shape: its columns are chosen the same way (`Runs`, `Ctx`, `Card`,
`RAM`, `Size`, `Have`, admitted in that order after `#`, `Model`, `Quant` and `Fit`), the
models with no placement are rows of the same table, dimmed, with `no room` or `no facts`
where the verdict would be, and the reasons are under it once each.

```
$ llamafit fit --limit 5
                                     Fit on this machine
┌───┬──────────────────────────────┬────────────┬────────────────┬─────────┬──────────┬─────┐
│ # │ Model                        │ Quant      │ Runs           │    Card │ Fit      │ Ctx │
├───┼──────────────────────────────┼────────────┼────────────────┼─────────┼──────────┼─────┤
│ 1 │ gpt-oss-120b                 │ UD-Q4_K_XL │ experts in RAM │ 5.9 GiB │ fits     │ 44K │
│ 2 │ mistral-small-4-119b         │ UD-Q2_K_XL │ experts in RAM │ 5.8 GiB │ fits     │ 54K │
│ 3 │ kimi-linear-48b-a3b-instruct │ Q6_K       │ split          │ 4.7 GiB │ fits     │ 72K │
│ 4 │ nemotron-3-super-120b-a12b   │ UD-Q4_K_XL │ split          │ 6.1 GiB │ fits     │ 74K │
│ 5 │ qwen3.6-35b-a3b              │ UD-Q6_K_XL │ experts in RAM │ 6.0 GiB │ fits     │ 61K │
│   │ command-a-plus-05-2026       │ Q4_K_M     │ nowhere        │       – │ no room  │   – │
│   │ deepseek-v4-flash-0731       │ UD-Q4_K_XL │ nowhere        │       – │ no room  │   – │
└───┴──────────────────────────────┴────────────┴────────────────┴─────────┴──────────┴─────┘
Sized for 32K tokens. The context column is the largest each one holds in the mode shown.
Showing the best 5 of 54 that qualified; --limit sets how many, and the rest are neither
worse-behaved nor hidden, only further down.
Computed with 100.0 GiB of 128.0 GiB system memory free and 7.5 GiB of 8.0 GiB free on the NVIDIA
GeForce RTX 4060.

Not placed
no room (11): no placement of it fits this machine at any context.
  command-a-plus-05-2026 Q4_K_M, deepseek-v4-flash-0731 UD-Q4_K_XL, deepseek-v4-pro-0813
```

That second caption is the point of `--limit`: a cut list says so and says how much it cut,
because a reader who was not told would take five rows for the whole answer. It appears only
when something was actually cut.

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
| `--min-tps N` | Exclude candidates generating fewer tokens per second than this, in place of the speed a person reads at (about 6). `--min-tps 0` says nobody is waiting on the tokens — a batch run — and excludes nothing for being slow. Whatever the figure, the board says under itself which one it used and every excluded row names it. |
| `--max-download SIZE` | Exclude larger downloads. Sizes look like `40G`, `7.5GiB`, `512M`. |
| `--license SPDX` (repeatable) | Licences the request will accept. A model with another one is excluded and still shown, with the licence it actually has. |
| `--limit N` | Show at most this many rows. Defaults to 10. |
| `--all-quants` | Show every quantisation instead of the best one per model. |
| `--no-vision` | Plan without a vision projector, freeing its memory for context. |
| `--explain` | Expand every row shown into the four scores, the weights, the quality it was built from, the memory budget line by line, the context ladder, where a token's time goes and where a prompt token's goes — and then what would change the answer. Combine with `--limit 1` for one model. |

The flags above change what is *ranked*, and the board says so: a candidate they exclude is
listed with the reason. The flags below change only what is *drawn*. The ranking, the count
that qualified and the `--json` document are untouched by any of them but `--sort`, which
reorders the document's rows and leaves every `rank` where the scorer put it, exactly as the
web API's `sort=` does. Whenever one of them is in force the board says so in a line under
the table, in the terminal dashboard's own words: `2 of 71 shown, by speed, showing the ones
that run, matching qwen.`

| Option | Effect |
|---|---|
| `--sort KEY[:asc\|:desc]` | Order the rows drawn by `score` (the default), `speed`, `quality`, `context`, `size`, `prompt`, `card`, `ram`, `fit`, `model` or `quant`. A figure comes largest first and a verdict best first; a word comes A to Z; `:asc` or `:desc` turns that round. A row with nothing to order by goes last either way. The `#` column keeps the ranking. |
| `--min-fit comfortable\|fits\|tight` | Draw only rows at or above this verdict. The same name as `fit`'s flag, but here a view filter: the rows hidden are still counted as qualified. |
| `--search TEXT` | Draw only rows whose id, name or quantisation contains the text. |
| `--installed` | Draw only rows whose file is already on this machine — the `Have` column's `yes`. |
| `--runs gpu\|moe-offload\|hybrid\|cpu` | Draw only rows that run in this mode. |
| `--columns LIST` | Draw exactly these columns, comma-separated, in this order, named as `--json` names them: `rank, model, quant, size, have, score, quality, gen, prompt, confidence, mode, vram, ram, verdict, context`. |
| `--wide` | Draw every column whatever the width of the terminal; a cell that does not fit folds. `COLUMNS=200 llamafit recommend` is the other way to get a wide board. |
| `--hide-excluded` | Leave the candidates that were not ranked off the table. The reasons under it stay. |

There are no `--min-speed`, `--max-size` or `--max-card` flags, though the web page has a box
for each under its headings. `--min-tps` excludes and says so; a `--min-speed` that merely hid
would sit one line under it and differ in a word, and a reader who picked the wrong one would
get a board whose caption disagreed with its rows for a reason nothing named. The page needed
boxes because a pointer cannot type a flag; a terminal has `--sort`, `--limit` and `--json`,
and the terminal dashboard's `/` box takes the same terms the page's boxes do.

```
$ llamafit recommend --use-case coding --limit 3
                                           Recommended
┌───┬──────────────────────────────┬────────────┬─────────────┬───────┬────────────────┬─────────┐
│ # │ Model                        │ Quant      │       Score │ Tok/s │ Runs           │ Fit     │
├───┼──────────────────────────────┼────────────┼─────────────┼───────┼────────────────┼─────────┤
│ 1 │ qwen3-coder-next             │ UD-Q4_K_XL │        89.6 │  23.9 │ experts in RAM │ tight   │
│ 2 │ qwen3.6-35b-a3b              │ UD-Q6_K_XL │        88.3 │  24.4 │ experts in RAM │ fits    │
│ 3 │ north-mini-code-1.0          │ UD-Q4_K_XL │        83.7 │  22.7 │ experts in RAM │ tight   │
│   │ command-a-plus-05-2026       │ Q4_K_M     │ unsupported │     – │ nowhere        │ no room │
│   │ deepseek-v4-flash-0731       │ UD-IQ2_XXS │    too slow │   4.6 │ split          │ tight   │
│   │ deepseek-v4-pro-0813         │ UD-Q4_K_XL │ unsupported │     – │ nowhere        │ no room │
│   │ deepseek-r1-0528-qwen3-8b    │ Q8_0       │    too slow │   6.0 │ split          │ tight   │
└───┴──────────────────────────────┴────────────┴─────────────┴───────┴────────────────┴─────────┘
Speeds are for 8K tokens of context so every row compares like with like; the context column is the
largest each one holds. Sized and scored for coding at 32K tokens.
Computed with 100.0 GiB of 128.0 GiB system memory free and 7.5 GiB of 8.0 GiB free on the NVIDIA
GeForce RTX 4060.
Every speed above is derived from 57.0 GB/s (measured) of memory bandwidth.
Showing the best 3 of 27 that qualified; --limit sets how many, and the rest are neither
worse-behaved nor hidden, only further down.
Speeds are section 10's formula on its default constants. Nothing has been benchmarked on this
machine yet, so no figure here is a measurement.
Weights: quality 0.40, speed 0.20, fit 0.20, context 0.20.

Not ranked
too slow (19): generates fewer than the 6 tokens per second a person reads at: a batch tool on this
machine and not one to sit in front of; a smaller model or quantisation would keep up.
  deepseek-v4-flash-0731 UD-IQ2_XXS (4.6 tok/s), deepseek-r1-0528-qwen3-8b Q8_0 (6.0 tok/s),
  devstral-small-2-24b-instruct Q4_K_M (2.9 tok/s), gemma-3-12b-it Q4_K_M (4.6 tok/s),
  gemma-4-31b-it UD-Q3_K_XL (1.3 tok/s), gemma-4-26b-a4b-it UD-Q3_K_XL (3.8 tok/s), gemma-4-12b-it
```

The list is cut off above; the real one runs to a row for every candidate in the catalog,
and `--limit` cuts only the ranked rows at the top of it.

Which columns appear depends on the width of the terminal, and on nothing else: not on
`--limit`, not on which models happen to rank. The rank, model, quantisation and score are
never dropped; the rest are admitted in the order `Tok/s`, `Fit`, `Runs`, `Ctx`, `Qual`,
`Card`, `Size`, `Have`, `How`, `Prompt tok/s`, `RAM`, each only while its whole content fits
(`How` is admitted together with `Tok/s`, or not at all, when the rows disagree about how
their speeds were arrived at). A column that cannot fit is dropped rather than shrunk, and a
column that is admitted is drawn in the place the web page gives it — `Size` between `Quant`
and `Score`, `Fit` and `Ctx` last — so a figure lives in the same place at every width. The
model column is measured from the longest id in the catalog and capped at 28 cells, where a
longer name folds onto a second line; below about 68 columns it folds harder rather than
anything being cut. The terminal dashboard chooses with the same function, so it and the
command line draw the same columns at the same width. On this catalog, in English:

| Width | Columns |
|---|---|
| 80 | `# Model Quant Score Tok/s Fit` |
| 100 | + `Runs` |
| 120 | + `Qual Ctx` |
| 140 | + `Size Card` |
| 160 | + `Have How` |
| 182 and up | + `Prompt tok/s RAM`, all fifteen |

A translated label is wider than its English — `experts in RAM` is fourteen cells and its
Portuguese is longer — and the budget is measured from the labels in force, so a Portuguese
board admits the next column a little later rather than overflowing.

**No row is ever dropped for failing.** A candidate the request excludes is a row of the same
list, dimmed, below the ranked ones, with every column filled with whatever was computed for
it — a model excluded for running at four tokens a second was placed, sized and estimated
first, and those figures are what tell a reader whether a smaller quantisation would rescue it
— and the word for the reason where its score would be: `too slow`, `no room`, `unsupported`,
`no coding`, `too big`, `short context`, `unknown quant`, `no estimate`. The sentence goes
under the table, once per reason, with the ids and their failing figures after it, rather
than once per row: on the reference machine the old **Not ranked** table said the same forty
words twenty-six times. `--explain` still prints the whole sentence under an expanded row.

**A model is filtered on what it can do, never on what it is offered for.** `use_cases` is the
curator's emphasis and sets the primary-use-case bonus; `capabilities` is what the weights can
actually do, and only that carries a filter. Asking for coding asks for the coding capability,
reasoning for thinking, multimodal for vision, embedding for embeddings; `general` and `chat`
ask for nothing, so a model built for one job is ranked on its merits for either of those. The
`--use-case` filter on `llamafit list` is a different thing: there you are browsing the catalog
and asking to see what an entry offers itself for.

**`--explain` ends with what would change the answer.** Everything above it explains a decision
already taken; the last block is the only part a reader can act on, and section 12.3 of the
design asks for it by name. There are exactly two things that can change without changing the
machine or the model — how much of the card is free, and which quantisation you fetch — so there
are at most two lines:

```
What would change it
To reach 64K tokens, free 61.6 MiB more on the graphics card: that rung needs 6.9 GiB, which
is 96% of what is free and over the line where the driver starts paging.
UD-Q4_K_XL, the next quantisation down, was planned here too: experts in RAM, fits, up to
62,464 tokens at 25.1 tokens per second.
It generates 25.1 tokens per second rather than 18.9.
```

**Neither line is arithmetic on the row above it.** The context line quotes a rung of the ladder
whose budget was computed for that rung; the quantisation line quotes a placement the planner
searched for that quantisation, at the same 8K working context every speed on the board uses. A
file four fifths the size does not make a budget four fifths the size — the cache, the compute
buffer and the overheads do not shrink with the weights — so nothing is scaled and nothing is
guessed. That is also why each line can be absent: a model publishing one quantisation is offered
no other, and a placement already at the top of its ladder is offered no rung. **A sentence
saying a smaller quantisation would fit, when it would not, is worse than no sentence.**

The figure to free is what the rung needs to come back inside the band this program will
recommend, which is a *tight* fit and not a comfortable one. It is only offered where freeing the
card is the answer at all: a rung system memory could not absorb either is named as out of reach,
with no figure attached, because closing a browser would not get there. And the second line says
one thing, not four — the whole model reaching the card beats a better verdict, a better verdict
beats more room, and more room beats a speed the estimator only claims to within five per cent.
When nothing changes it says so, which saves a reader a download.

This costs one extra placement search per row explained, which is why it is printed under
`--explain` and nowhere else. It is a terminal expansion: `recommend --json` carries the board,
and the counterfactuals are not on it.

**Nothing on this board is labelled `measured`.** Section 10.3 reserves that word for a
benchmark taken on *this* machine. `llamafit bench` takes and stores one, but nothing reads
the benchmark database back into the board: `recommend`, `fit` and `plan` run the formula on
the constants that ship with LlamaFit, so every speed here is `estimated`. A catalog entry's
own `measured` block is a record from the curator's machine; `plan` shows it beside the
estimate, and neither is ever allowed to become the other.

### `llamafit plan` — phase 1C, shipped

`llamafit plan <model>`: place one model on this machine and print the command line that runs
it. The output is the memory budget component by component with the source of every figure,
the context ladder a launch script chooses from, where a token's time goes and where a
prompt token's goes, any run the catalog records for comparison, and last, on a line of its
own, the `llama-server` command.

Both speed tables are read the same way: each is the terms of the formula behind one figure,
and each trio adds up to one over that figure, so a reader can see which term dominates
rather than being asked to believe a total. The prompt table's link row is the term this
project knows least about — its rate was fitted on the single model whose expert set does
not fit in system memory, and a model whose set stays in the page cache streams about three
times faster — so it is printed as its own row, with a note saying so, instead of being
folded into a number nobody can question.

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

### `llamafit hardware` `list|show|validate|path` — phase 1C, shipped

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
┌─────────────────────────┬─────────┬──────────────────────────────┬──────────────────────────────┐
│ Name                    │ From    │ Machine                      │ Description                  │
├─────────────────────────┼─────────┼──────────────────────────────┼──────────────────────────────┤
│ reference-rtx4060-128gb │ bundled │ 128.0 GiB RAM, NVIDIA        │ Reference machine: RTX 4060  │
│                         │         │ GeForce RTX 4060 with 8.0    │ 8 GB, i9-14900KF, 128 GiB    │
│                         │         │ GiB                          │ DDR5-4200, Windows 11        │
└─────────────────────────┴─────────┴──────────────────────────────┴──────────────────────────────┘

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

### `llamafit serve` — phase 1D, shipped

Start the web dashboard and JSON API on this machine.

```
llamafit serve                     # http://127.0.0.1:8765
llamafit serve --port 9000 --open  # another port, and open the browser
```

| Option | What it does |
|---|---|
| `--host ADDRESS` | Bind somewhere other than `127.0.0.1`. Anything but loopback puts the dashboard on your network, where there is no password and no login; the consequence is printed before the server starts. Leaving the flag alone is what keeps the server on this machine — the address is not merely defaulted to loopback, it is refused elsewhere unless the flag was actually typed. |
| `--port N` | Listen on this port. Default 8765. |
| `--open` | Open the page in your browser once the server is up. |

It needs the `web` extra: `pip install "llamafit[web]"`. Without it the command says so and
gives you that line.

The dashboard speaks the language `--language` chose, like everything else, and its numbers
carry the same labels the tables above use. See [web.md](web.md) for the API, the safety
rules and what the page does about languages.

### `llamafit` (no command) — phase 1D, shipped

Opens the terminal dashboard: the board, the request as a form, the machine, the plan for a
chosen row and the simulation controls, over one scan and one catalog. It is what somebody
who has just installed LlamaFit meets, so nothing on it has to be found out from a flag.

Where there is no terminal to draw one in — a pipe, a redirect, a CI job — it says so on
standard error and prints exactly what `llamafit recommend` would print, with the same
defaults, read off that command rather than repeated here. See [tui.md](tui.md).

### `llamafit install` — phase 2, shipped

The `install` group puts onto the machine what it takes to run a model: llama.cpp itself,
and the weights. Both commands have shipped, and both keep the same rule — the whole plan
is printed, and only then is the question asked.

#### `llamafit install llama.cpp` — phase 2, shipped

Downloads a published llama.cpp build from the project's GitHub releases and installs it into
`~/.llamafit/llama.cpp`, which is the first directory the detector looks in, so nothing has to
be configured afterwards.

| Option | Effect |
|---|---|
| `--backend cuda\|vulkan\|hip\|sycl\|metal\|cpu` | Install this backend instead of the one chosen for your GPU. It is honoured exactly: if the release publishes no such build the command says so, rather than installing something else. |
| `--dir PATH` | Install here instead of `~/.llamafit/llama.cpp`. |
| `--tag TAG` | Install this release, for example `b6100`, instead of the newest. |
| `--add-to-path` | Offer to add the `bin` directory to your `PATH`, in user scope, printing the exact way to undo it before asking. |
| `--force` | Install over a directory LlamaFit did not create. |
| `--allow-unverified` | Install an archive that publishes no checksum. |
| `--dry-run` | Print the plan and write nothing. |
| `--yes`, `-y` | Do not ask; assume yes. |

**What it prints before writing anything.** The release and its build number, the backend, the
archive and any companion archive, how much has to be downloaded and how much is already on
disk, whether a checksum is published, the directory it will install into and what is already
there, and the free space on both the cache and the install volume. Only then does it ask.
`--dry-run` stops there, and so does `--json` without `--yes`, because a machine-readable run
should not be the one that opens a prompt nobody is watching.

**Which archive.** Chosen from the asset names, for the operating system, the architecture and
the backend: on Windows with an NVIDIA card, `llama-<tag>-bin-win-cuda-<ver>-x64.zip` plus the
matching `cudart-...` archive, which holds the CUDA runtime libraries the build cannot start
without; on Apple silicon, `llama-<tag>-bin-macos-arm64.tar.gz`, which *is* the Metal build,
since Metal is compiled in and has no archive of its own; on Linux with an AMD card,
`llama-<tag>-bin-ubuntu-rocm-<ver>-x64.tar.gz`, falling through to the Vulkan archive on a
release that publishes no ROCm build. A machine with no GPU gets the CPU build, and so does a
Linux machine with an NVIDIA card, because llama.cpp publishes no Linux CUDA archive — there it
falls to Vulkan.

**Which CUDA.** A release publishes CUDA archives for two or three CUDA majors at once, and the
newest is not automatically right: a CUDA 13 build does not start at all on a driver older than
580. LlamaFit reads the driver version from the card and takes the newest archive that driver
can load; where the driver is unknown it takes the *oldest* published CUDA, which is the one
with the widest support, on the same principle that prefers an AVX2 build to an AVX512 one.

**Which release.** The newest `bNNNN` tag, found by listing the releases. Not GitHub's
`/releases/latest`, which skips prereleases and so names llama.cpp's semantic-version tag,
whose only attachment is a text file.

**Rules it keeps.** A directory that does not carry LlamaFit's own `.llamafit-install.json`
marker is never overwritten without `--force`: people build llama.cpp themselves, with their own
flags, and that is not something a download can restore. The published checksum is verified on
the partial file *before* anything is unpacked, and a file that does not match is deleted rather
than kept. An interrupted download continues from where it stopped the next time the command is
run. Your `PATH` is never changed unless you ask with `--add-to-path` and then answer yes.

**Exit codes.** `0` on success or after a dry run; `1` when you decline, when the directory is
not LlamaFit's, when a volume has no room, or when an archive fails its checksum.

#### `llamafit install model <id>` — phase 2, shipped

Download a model's weights from the repository the catalog names, resuming anything an
earlier run left behind and checking every file against the SHA-256 the catalog holds.

```
llamafit install model <id> [--quant NAME] [--dir PATH] [--workers N] [--limit-rate RATE]
                            [--yes] [--dry-run] [--no-extras] [--recheck]
                            [--allow-unverified]
llamafit install history [--limit N]
```

Before a socket is opened it prints the model, the quantisation, the repository, every
file it will fetch, what the set weighs, where it is going, how much is already there and
what would be left on the volume afterwards. In a terminal it then asks; outside one it
does not, because there is nobody there to answer.

| Option | Effect |
|---|---|
| `--quant NAME` | Fetch this quantisation instead of the first the catalog publishes. |
| `--dir PATH` | Put the files here instead of under the downloads directory. |
| `--workers N` | Requests in flight, 1 to 32. The default is 8: the specification's 16–32 will take every bit of a domestic connection, and nobody asked for their video call to stop working. |
| `--limit-rate RATE` | A ceiling in bytes per second — `5M`, `500K` — shared across every worker, so the figure is the figure whatever `--workers` says. |
| `--yes`, `-y` | Do not ask before starting. |
| `--dry-run` | Print the plan and the disk arithmetic, fetch nothing. Needs no network. |
| `--no-extras` | Weights only: no vision projector, no draft model. |
| `--recheck` | Re-read files that are already here and hash them against the catalog. |
| `--allow-unverified` | Fetch files the catalog holds no checksum for. Refused by default, because a model that downloaded wrong does not fail loudly — it answers nonsense. |

What is on disk while a download is running: `<name>.gguf.part`, created at the file's
final size, and `<name>.gguf.part.state`, a small JSON record of which chunks have
arrived. A chunk is recorded only once its last byte is written, so `Ctrl+C` or a dropped
connection costs at most one chunk and the next run continues rather than restarts. When
every file of a model has arrived and been checked, `llamafit-install.json` is written
into the directory; its presence is what "this model is installed" means, and a directory
holding three shards of four does not have one.

A file whose checksum does not match is deleted along with its part file and its record,
and the command exits 1. Nothing that could be resumed into the wrong bytes is kept.

`install history` lists what has been downloaded, newest first, finished or not.
`--json` gives the plan (with `--dry-run`), the outcome, or the history records.

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

### `llamafit bench` — phase 3, shipped

`llamafit bench <model>`: measure the model on this machine at the flags `plan` would launch
it with, and print the measurement beside the estimate. It runs `llama-bench` for a prompt
row and a generation row, then starts a real `llama-server` and asks it three fixed
questions — a short prompt on a cold server, a thousand-token prompt on a warm one, and a
tool call — recording the throughput each reports about itself, the time to first token, the
peak VRAM during the run and the buffer sizes from the server's log.

| Option | Effect |
|---|---|
| `--quant NAME` | Benchmark this quantisation instead of the planned one. |
| `--context N` | Size for this many tokens. |
| `--ub N` | Set the micro-batch by hand; the whole budget is rebuilt around it. |
| `--sweep` | Measure prompt processing at every micro-batch of the ladder. Section 10.2 has two free parameters and one micro-batch is one equation, so without this the prompt half of a calibration cannot be fitted at all. |
| `--no-server` | Run `llama-bench` only. Gives up the paging check, the buffer sizes and the tool-call test. |
| `--no-store` | Print the result without writing it to the database, so no estimate is relabelled. |
| `--dry-run` | Print the two command lines and run nothing. |
| `--calibrate` | Fit the estimator's constants to this machine from every stored result, and name the ones the measurements do not determine. Works on its own, with no model. |
| `--show` | List what this machine has already measured, and stop. |

The last column of the table is the point of the command. A tool that measured a model and
then showed only the measurement would leave you better informed about that model and no
better informed about the next one, so the estimate stays on the page — and the estimate
shown is the one made *before* the run, never one recomputed afterwards from a database that
by then contains the answer. The first two columns are what make the last one an error rather
than an artefact: each row's estimate is the formula run at the context that row filled and
the micro-batch it used, and the plan's own figure — for a context no row reaches — is quoted
under the table and never given a ratio. [benchmarking.md](benchmarking.md) has the
arithmetic and the comparison that was rejected.

**The paging check** has three states and not two: it fires when peak VRAM was within three
percent of the card's total **and** generation came in below three fifths of the estimate
(section 16.4), it clears a card that stayed well short of full, and it says *unchecked* when
there was no vendor tool, no card total or no estimate. A configuration nobody could check
has not been cleared.

**A result is refused rather than stored** when the run did not do what it was told — a
context llama.cpp clamped, a micro-batch it did not use — and when the host is a simulated
one. The flags a stored result hands out are built from what the tool reported about itself
rather than from the command line it was given, so a micro-batch sweep's rows are each filed
under the size that row actually ran at.

**`--calibrate` fits only what the data determines.** Four measurements pinned four constants
on the reference machine and two would not have pinned three, so a constant whose column is
empty, whose configurations are too few, or whose fit comes out impossible is refused by name
with the reason. `layer_overhead_ms` is never fitted at all: section 10.1 removed the
per-layer term, so a number fitted for it would be read by nothing.

**A fit is printed and stored, and nothing reads it back yet.** `recommend`, `fit` and `plan`
run the formula on the constants that ship in the package, so a calibrated machine's board
still says `estimated`. [benchmarking.md](benchmarking.md) says what is missing and why it was
left rather than bodged.

## Environment variables

| Variable | Effect |
|---|---|
| `LLAMAFIT_HOME` | Put config, data, cache, logs and downloads under one directory. |
| `LLAMAFIT_CUSTOM_MODELS` | Path of the custom models file (default under the data directory). |
| `LLAMAFIT_PROFILES` | Directory holding your own hardware profiles (default `profiles` under the data directory). |
| `LLAMA_CPP_PATH` | Directory (or install root) that holds `llama-server`; checked before `PATH`. |
| `LLAMA_SERVER_PORT` | First port to probe for a running server (then 8080, 8081, 8098). |
| `LLAMA_CACHE` | llama.cpp's model cache directory; its GGUF files are listed as local models. |
| `GITHUB_TOKEN` | Sent when reading the llama.cpp releases API, so a shared address does not run into the anonymous hourly limit. Not required, and no permission beyond reading a public repository is used. |
| `NO_COLOR` | Same as `--no-color`. |

## Files

See [architecture.md](architecture.md) for where configuration, caches, profiles, downloads
and logs live on each operating system.
