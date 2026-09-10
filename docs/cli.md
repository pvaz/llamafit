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
| `--perfect` | Only configurations that score a perfect fit under section 11.3: at least half the machine's memory held in the model's own weights, and no pool past four fifths full. |
| `--limit N` | Show at most this many rows. |
| `--all-quants` | Show every quantisation instead of the best-fitting one per model. |

```
$ llamafit fit
                            Fit on this machine
 #   Model                  Quant        Fit     Runs             Ctx    Card      RAM
 1   gemma-3-27b-it         Q4_K_M       tight   split            33K    6.2 GiB   32.1 GiB
 2   qwen3.8-flash-next     UD-Q4_K_XL   tight   experts in RAM   17K    6.3 GiB   75.5 GiB
 3   qwen3-coder-next       UD-Q4_K_XL   tight   experts in RAM   32K    6.3 GiB   46.4 GiB
 4   llama-3.1-8b-instruct  Q4_K_M       tight   split            35K    6.1 GiB    7.8 GiB
 5   qwen3-0.6b             Q8_0         fits    GPU              32K    5.5 GiB    2.2 GiB
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
| `--min-tps N` | Exclude candidates generating fewer tokens per second than this, in place of the speed a person reads at (about 6). `--min-tps 0` says nobody is waiting on the tokens — a batch run — and excludes nothing for being slow. Whatever the figure, the board says under itself which one it used and every excluded row names it. |
| `--max-download SIZE` | Exclude larger downloads. Sizes look like `40G`, `7.5GiB`, `512M`. |
| `--license SPDX` (repeatable) | Licences the request will accept. A model with another one is excluded and still shown, with the licence it actually has. |
| `--limit N` | Show at most this many rows. Defaults to 10. |
| `--all-quants` | Show every quantisation instead of the best one per model. |
| `--no-vision` | Plan without a vision projector, freeing its memory for context. |
| `--explain` | Expand every row shown into the four scores, the weights, the quality it was built from, the memory budget line by line, the context ladder, where a token's time goes and where a prompt token's goes. Combine with `--limit 1` for one model. |

```
$ llamafit recommend --use-case coding
                                 Recommended
 #   Model                Quant        Score   Tok/s   Fit     Runs             Ctx
 1   qwen3-coder-next     UD-Q4_K_XL    84.8    23.7   tight   experts in RAM   32K
 2   qwen3.8-flash-next   UD-Q4_K_XL    67.2    13.5   tight   experts in RAM   17K
 3   qwen3-0.6b           Q8_0          54.0   114.6   fits    GPU              32K
Speeds are for 8K tokens of context so every row compares like with like; the context
column is the largest each one holds. Sized and scored for coding at 32K tokens.
Speeds are section 10's formula on its default constants. Nothing has been benchmarked
on this machine yet, so no figure here is a measurement.
Weights: quality 0.40, speed 0.20, fit 0.20, context 0.20.

                                  Not ranked
 Model                   Quant    Why not
 gemma-3-27b-it          Q4_K_M   not a coding model; its entry lists general, multimodal,
                                  chat, so ask for one of those
 llama-3.1-8b-instruct   Q4_K_M   not a coding model; its entry lists general, chat,
                                  reasoning, so ask for one of those
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

### `llamafit serve`

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

### `llamafit` (no command) — shipped

Opens the terminal dashboard: the board, the request as a form, the machine, the plan for a
chosen row and the simulation controls, over one scan and one catalog. It is what somebody
who has just installed LlamaFit meets, so nothing on it has to be found out from a flag.

Where there is no terminal to draw one in — a pipe, a redirect, a CI job — it says so on
standard error and prints exactly what `llamafit recommend` would print, with the same
defaults, read off that command rather than repeated here. See [tui.md](tui.md).

### `llamafit install` — phase 2

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
by then contains the answer.

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
per-layer term, so a number fitted for it would be read by nothing. See
[benchmarking.md](benchmarking.md).

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
