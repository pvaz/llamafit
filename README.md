# LlamaFit

[![CI](https://github.com/pvaz/llamafit/actions/workflows/ci.yml/badge.svg)](https://github.com/pvaz/llamafit/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](docs/platform-support.md)
[![Languages](https://img.shields.io/badge/interface-37%20languages-brightgreen.svg)](docs/translations.md)

**Find, size, install and verify open-weight LLMs for llama.cpp on your own machine.**

LlamaFit scans your computer, keeps a curated catalog of GGUF models, computes exact memory
budgets from the model files themselves, ranks what fits for what you need (coding, thinking,
vision, tool calling, long context), and turns the winner into a working `llama-server`
configuration. Then it installs llama.cpp and the model, writes tuned launch scripts, and
measures the real speed so you can see what the estimate got wrong.

It works on Windows, macOS and Linux, installs with `pip`, speaks English and 37 other
languages, and never uses a language model to do any of this: every number is computed,
labelled with how it was obtained, and explainable.

```console
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

Sixty-two models, strongest first; `--limit` is what keeps five of them on this page.

> **Status.** 0.1.0. Every command in the table below ships and is tested on Windows, macOS
> and Linux: the host scan and the diagnostics, the catalog, the memory budget, the placement
> planner, the speed estimator and the ranking, the terminal and web dashboards, the installer
> and the benchmark verifier. Three pieces named in the design are not built — the terminal
> dashboard's Downloads and Benchmarks screens, the install and benchmark endpoints of the web
> API, and the path by which a benchmark corrects the estimator, which is why every speed is
> still labelled `estimated`. The [roadmap](ROADMAP.md) says so line by line; the design is
> in [`docs/specs/`](docs/specs/2026-09-09-llamafit-design.md).

## How it differs

| | The usual approach | LlamaFit |
|---|---|---|
| **Memory** | one RAM and VRAM minimum per model | per-component budget from the file's own tensor table: weights by class, cache per 1K tokens, compute buffer by micro-batch, projector |
| **Result** | "you need 16 GB" | a runnable `llama-server` command line and launch scripts |
| **Speed** | a guess, or numbers from someone else's machine | computed from your measured bandwidth, then checked against real benchmarks on your machine |
| **Honesty** | one confident number | every figure labelled measured, calibrated or estimated, and expandable into its inputs |
| **Failure** | silence | names the case where the driver pages to system memory and the server starts anyway at half speed |

## Why

Running a model locally with llama.cpp involves a chain of decisions that are easy to get
wrong and hard to see when you do: which model family has the capability you need, which
quantisation of it fits, how much of it goes on the GPU, how big a context the remaining
memory allows, which micro-batch size keeps prompt processing fast, where the vision
projector should live, and whether the result is actually running at the speed the hardware
can deliver. A wrong answer rarely fails loudly. On NVIDIA drivers a configuration that asks
for more VRAM than the card has simply starts, pages silently into system RAM, and runs at
half speed for weeks before anyone notices.

LlamaFit exists to make that chain explicit, correct, and reproducible.

## What it does

1. **Scans the host.** CPU model, cores and instruction sets; RAM totals, type, speed and
   measured bandwidth; every GPU with its free VRAM, bandwidth and compute; free disk space.
   Each probe reports whether it worked so you know what the numbers rest on.
2. **Detects llama.cpp.** Binaries on `PATH` or in the usual places, build number, compiled
   backends (CUDA, Metal, HIP, Vulkan, SYCL, CPU), local GGUF files, and any `llama-server`
   already running.
3. **Reads the catalog.** Curated YAML, one file per model family, with parameters, architecture
   class, capabilities, context, license, published benchmark scores and the Hugging Face
   repositories that carry the GGUF files. What changes — file names, sizes, checksums, bits per
   weight and GGUF header facts — is refreshed from the Hugging Face API into a generated file
   beside it, without downloading a single weight. Every model it ships is listed in
   [MODELS.md](MODELS.md).
4. **Computes the budget.** From the GGUF header of each file: weight bytes by tensor class,
   KV cache per thousand tokens, compute buffer as a function of micro-batch and context,
   projector cost. It then searches placements (all on GPU; attention on GPU with experts in
   RAM; hybrid; CPU) and context sizes for the best one that fits, and estimates generation
   and prompt speed from memory bandwidth and PCIe throughput.
5. **Ranks and explains.** Four scores from 0 to 100 (quality, speed, fit, context) weighted by
   your use case, a verdict (Comfortable, Fits, Tight, Does not fit), and a plain-language
   explanation of why each candidate is where it is and what would move it up.
6. **Makes it run.** `plan` prints the exact `llama-server` command line. `install` fetches
   llama.cpp release builds and model files with verification. `preset` writes launch scripts
   that pick the context size from the VRAM that is free at start. `bench` measures the result,
   stores it beside the estimate it is compared with, and fits the estimator's constants to
   your machine — refusing by name the ones your measurements do not determine.

## The command line

Every command has a `--json` form for scripts. The Textual terminal dashboard opens with plain
`llamafit`; the web dashboard with `llamafit serve`.

| Command | What it does | Phase |
|---|---|---|
| `llamafit system` | show the host scan and the llama.cpp installation | 1A (done) |
| `llamafit doctor` | every probe, what failed, what would unlock more | 1A (done) |
| `llamafit list`, `search`, `info` | browse the catalog; `info` shows one model in full, with every quant it publishes | 1B (done) |
| `llamafit catalog validate`, `refresh`, `show` | maintain the catalog | 1B (done) |
| `llamafit fit` | every model ranked by fit on this machine | 1C (done) |
| `llamafit recommend` | the board for your needs: use case, required capabilities, minimum context, size and license limits; `--explain` shows the working | 1C (done) |
| `llamafit plan <model>` | placement, memory budget, context tiers, flags and the command line | 1C (done) |
| `llamafit hardware` | hardware profiles, to score against a machine you are not on | 1C (done) |
| `llamafit` | the terminal dashboard | 1D (done) |
| `llamafit serve` | the web dashboard and JSON API on localhost | 1D (done) |
| `llamafit install llama.cpp`, `install model` | install the runtime and download models | 2 (done) |
| `llamafit preset`, `launch` | write launch scripts and start a server | 2 (done) |
| `llamafit bench` | measure, compare with the estimate, calibrate | 3 (done) |

## An example

The machine LlamaFit was designed on: an RTX 4060 with 8 GB of VRAM, an i9-14900KF and 128 GB
of DDR5. The interesting models for it are mixture-of-experts models whose experts live in RAM
while attention and the KV cache live on the GPU. LlamaFit's budget for a 125B MoE with 6B
active parameters at 4-bit on that card says: 5.8 GB of VRAM before any context, 36 MiB per
thousand tokens of context, so 40K tokens fit with 7.6 GB free and 128K would need 13 GB.
The original hand-written launcher asked for 128K, started without complaint, and ran at
6 tokens per second. With the budget respected it runs at 14, and prompt processing goes from
6 to 50 tokens per second. Those measurements are the first entries in the
[calibration set](docs/calibration/2026-09-09-reference-machine.md).

## Install

```
pip install llamafit            # add [fast] for the NumPy bandwidth measurement
pip install "llamafit[web]"     # and the browser dashboard, if you want one
```

Requirements: Python 3.10 or newer. No compiler, no Node, no account. The optional
`llamafit[fast]` extra adds NumPy for a more accurate memory-bandwidth measurement, and
`llamafit[web]` adds FastAPI and uvicorn for `llamafit serve` — the dashboard itself is
plain HTML, CSS and JavaScript that ships in the package and fetches nothing from the
internet, so there is still no build step anywhere.

### Or without a Python at all

Every release also carries a single self-contained file — interpreter, dependencies,
catalog and all 37 message catalogs inside it — for Windows, macOS and Linux on both
x86-64 and ARM. Take the one for your machine from the
[releases page](https://github.com/pvaz/llamafit/releases), and run it:

```
tar -xzf llamafit-0.1.0-linux-x86_64.tar.gz && ./llamafit recommend
```

It is about 25 MB, it is not code-signed, and your operating system will say something
about that the first time. [docs/standalone.md](docs/standalone.md) covers the checksums,
what Windows and macOS will ask you, and the one thing the binary leaves out.

## Use

```
llamafit recommend --use-case coding   # the board for what you want to do
llamafit recommend --explain --limit 1 # and every number that produced the first row
llamafit plan qwen3-coder-next         # budget, context tiers and the command line to paste
llamafit fit                           # every model ranked by how well it uses this machine
llamafit list                          # the catalog, strongest first
llamafit info qwen3-coder-next         # one model: facts, quants, what each costs here
llamafit system                        # CPU, memory, GPUs, disks, llama.cpp installation
llamafit doctor                        # what was detected, what failed, what would help
llamafit --json recommend              # the same as JSON for scripts
```

Speeds on the board are estimates from the formulas in
[how-it-works.md](docs/how-it-works.md), never benchmarks of your machine: `plan` prints the
runs the catalog records beside its own estimate rather than in place of them, and
`llamafit bench` measures yours and prints the measurement beside the estimate. What a
benchmark records does not yet reach the board, which is why every speed there stays labelled
`estimated`; [benchmarking.md](docs/benchmarking.md) says what is missing.

Sizes and GGUF facts ship filled in: a generated `<family>.facts.json` beside each catalog
file carries them, so `llamafit info` prints real figures on a fresh install.
`llamafit catalog refresh` brings them up to date from Hugging Face; nothing is downloaded
but file metadata and headers.

## Your language

The interface ships in English and 37 other languages, chosen with `--language`, the
`LLAMAFIT_LANGUAGE` variable, or your operating system's own setting, in that order. Ask
for one that is not there and LlamaFit falls back to English **and says so**, because
quietly ignoring what you asked for is its own kind of bug.

Six of the 37 are close to complete: Portuguese, Brazilian Portuguese, Spanish, French,
German and Italian. The rest carry about a quarter of the messages — the column headings
and the short phrases, not the sentences that explain anything — **and LlamaFit tells you
so the first time you use one**, with the figure and where to help. A tool that lists a
language it barely speaks is making a promise; saying how far it got is keeping one.

If the console you are writing to cannot represent the script — a Windows code page and
Japanese, say — LlamaFit speaks English instead and tells you which encoding refused
which language, rather than printing a screen of question marks. `PYTHONUTF8=1` fixes it,
except in the standalone binary, which reads no `PYTHON*` variable at all:
[docs/standalone.md](docs/standalone.md) says what does.

No catalog has been read by a native speaker, and every one of them says so at the top and
asks for that review. Correcting a line of your own language is the easiest useful
contribution this project has: see [docs/translations.md](docs/translations.md).

## Documentation

- [How it works](docs/how-it-works.md): the memory budget, placement search, speed model and scoring, with the formulas.
- [Command-line reference](docs/cli.md)
- [Terminal dashboard](docs/tui.md) and [web dashboard and API](docs/web.md)
- [Platform support and probes](docs/platform-support.md) and the [standalone binary](docs/standalone.md)
- [The catalog](docs/catalog.md) and the models it ships ([MODELS.md](MODELS.md)), [custom models](docs/custom-models.md) and [hardware profiles](docs/hardware-profiles.md)
- [Benchmarking and calibration](docs/benchmarking.md)
- [Architecture](docs/architecture.md) and the [design specification](docs/specs/2026-09-09-llamafit-design.md)
- [Development](docs/development.md) and [contributing](CONTRIBUTING.md)

## Principles

- **No number without a label.** Measured, calibrated, estimated or assumed; the label travels
  with the number into every table, screen and JSON document.
- **Exact where it can be.** Memory budgets come from the GGUF headers and real file sizes,
  not from a size class.
- **Nothing hidden.** Every recommendation expands into the inputs that produced it.
- **The host is the truth.** Estimates are starting points; measurements on your machine
  replace them.
- **Deterministic.** No model in the loop, no randomness, the same inputs give the same board.
- **Small dependency surface.** A handful of well-known Python libraries, listed in
  [`docs/development.md`](docs/development.md) with a reason for each.

## Contributing

Catalog entries, hardware fixtures, GPU table rows, translations, bug reports and
documentation are all welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md); catalog rules are
in [docs/catalog.md](docs/catalog.md) and translation rules in
[docs/translations.md](docs/translations.md). Please read the
[code of conduct](CODE_OF_CONDUCT.md) and the [contributor licence agreement](CLA.md), which a
pull request confirms in one line.

## License

LlamaFit is free software under the **GNU Affero General Public License, version 3 or later**
(`AGPL-3.0-or-later`). The full text is in [LICENSE](LICENSE).

### What that means if you just want to use it

Nothing. Using LlamaFit costs nothing and requires nothing: run it on any machine, on any
number of machines, at home or at work, for a hobby or for money, script it, wrap it, put it
in your pipeline. You owe no notice, no payment and no publication. The AGPL places
obligations on *giving the software to other people*, not on using it, and the reports, launch
scripts and JSON it produces are yours.

| What you do | What you owe |
|---|---|
| Run it, anywhere, for anything, including commercially | nothing |
| Modify it and keep the modification to yourself | nothing |
| Give it to someone else, modified or not | the source, under this same licence |
| Run a **modified** version as a network service others use | that modified source, offered to those users |

Model weights are not part of this project. They are downloaded from the repositories the
catalog names and keep their own licences, which the catalog records; the AGPL says nothing
about them.

### Why AGPL and not GPL or MIT

LlamaFit ships a web dashboard, and a hosted dashboard is exactly what a plain GPL does not
reach: someone could take LlamaFit, sharpen the estimator, run it as a service, and never
publish a line, because they never *distribute* anything. Section 13 of the AGPL — the network
clause — closes that gap: if people interact with a modified LlamaFit over a network, they are
entitled to its source. LlamaFit's whole value is numbers that can be checked and corrected,
so corrections should come back to the people relying on them. MIT asked for nothing; this
project asks for that one thing and nothing more.

### Commercial licensing

The AGPL is not workable for everybody. The copyright holder can also license LlamaFit on
other terms, which is possible because contributors grant the rights described in
[CLA.md](CLA.md). Open an issue to ask.

Third-party notes are in [NOTICE](NOTICE); every runtime dependency is under a permissive
licence that combines with the AGPL.
