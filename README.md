# LlamaFit

**Find, size, install and verify open-weight LLMs for llama.cpp on your own machine.**

LlamaFit scans your computer, keeps a curated catalog of GGUF models, computes exact memory
budgets from the model files themselves, ranks what fits for what you need (coding, thinking,
vision, tool calling, long context), and turns the winner into a working `llama-server`
configuration. Then it installs llama.cpp and the model, writes tuned launch scripts, measures
the real speed, and uses those measurements to sharpen its own estimates.

It works on Windows, macOS and Linux, installs with `pip`, and never uses a language model to
do any of this: every number is computed, labelled with how it was obtained, and explainable.

> **Status: phase 1A (host scan and diagnostics) and phase 1B (the model catalog) are done.**
> `llamafit system`, `llamafit doctor`, `llamafit list`, `search`, `info` and the `catalog`
> commands work on Windows, macOS and Linux. Memory budgets, scoring, the terminal dashboard
> and the web dashboard follow. The design is in
> [`docs/specs/`](docs/specs/2026-09-09-llamafit-design.md) and the implementation plans are in
> [`docs/plans/`](docs/plans/).
> Watch the repository or read the [roadmap](ROADMAP.md) to see what lands when.

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
   stores it, and calibrates the estimator for your machine.

## The command line

Every command has a `--json` form for scripts. The Textual terminal dashboard opens with plain
`llamafit`; the web dashboard with `llamafit serve`.

| Command | What it does | Phase |
|---|---|---|
| `llamafit system` | show the host scan and the llama.cpp installation | 1A (done) |
| `llamafit doctor` | every probe, what failed, what would unlock more | 1A (done) |
| `llamafit list`, `search`, `info` | browse the catalog; `info` shows one model in full, with every quant it publishes | 1B (done) |
| `llamafit catalog validate`, `refresh`, `show` | maintain the catalog | 1B (done) |
| `llamafit fit` | every model ranked by fit on this machine | 1C |
| `llamafit recommend` | the board for your needs: use case, required capabilities, minimum context, size and license limits | 1C |
| `llamafit plan <model>` | placement, memory budget, context tiers, flags and the command line | 1C |
| `llamafit hardware` | hardware profiles, to score against a machine you are not on | 1C |
| `llamafit` | the terminal dashboard | 1D |
| `llamafit serve` | the web dashboard and JSON API on localhost | 1D |
| `llamafit install llama.cpp`, `install model` | install the runtime and download models | 2 |
| `llamafit preset`, `launch` | write launch scripts and start a server | 2 |
| `llamafit bench` | measure, compare with the estimate, calibrate | 3 |

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
```

Requirements: Python 3.10 or newer. No compiler, no Node, no account. The optional
`llamafit[fast]` extra adds NumPy for a more accurate memory-bandwidth measurement.

## Use

```
llamafit list                   # the catalog, strongest first
llamafit info qwen3-coder-next  # one model in full: facts, sources, every quant
llamafit system                 # CPU, memory, GPUs, disks, llama.cpp installation
llamafit doctor                 # what was detected, what failed, what would help
llamafit --json system          # the same as JSON for scripts
```

Sizes and GGUF facts read `unknown` until `llamafit catalog refresh` fills them in from
Hugging Face; nothing is downloaded but file metadata and headers.

## Documentation

- [How it works](docs/how-it-works.md): the memory budget, placement search, speed model and scoring, with the formulas.
- [Command-line reference](docs/cli.md)
- [Terminal dashboard](docs/tui.md) and [web dashboard and API](docs/web.md)
- [Platform support and probes](docs/platform-support.md)
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
[code of conduct](CODE_OF_CONDUCT.md).

## License

MIT, see [LICENSE](LICENSE). Third-party notes are in [NOTICE](NOTICE). Model weights are not
part of this project and keep their own licenses, which the catalog records.
