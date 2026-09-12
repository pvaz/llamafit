# Roadmap

LlamaFit is built in phases. Each phase ships working software with tests on Windows, macOS
and Linux, and each has a written plan under `docs/plans/` that lists its tasks.
The design for all phases is in `docs/specs/2026-09-09-llamafit-design.md`.

The order is promised; the dates are not.

## Phase 1: advisor

The tool answers "what can I run here, at what speed, with which settings" without touching
the machine beyond reading it.

| Part | Delivers | Done when |
|---|---|---|
| 1A foundation and host scan (done) | package, CI, error handling, host detection (CPU, memory, bandwidth, GPUs, disks), llama.cpp detection (binaries, build, backends, local models, running servers), `llamafit system`, `llamafit doctor` | tests green on three operating systems; the reference machine's scan is reproduced from recorded fixtures |
| 1B catalog and GGUF facts (done) | YAML catalog with schema validation, GGUF header reader for local files and remote URLs, `refresh` from the Hugging Face API, custom model overrides, `list`, `search`, `info`, `catalog validate`, `catalog refresh` | the first catalog covers the current open-weight families with sources for every number; `info` shows exact per-quant facts |
| 1C budget, placement, speed and scoring (done) | memory budget by component, placement search with context tiers, speed estimator, quality and context scores, composite ranking, explanations, hardware profiles and simulation, `fit`, `recommend`, `plan`, `hardware` | the reference machine's board places the two measured models where the measurements say, within tolerance |
| 1D interfaces (done) | Textual terminal dashboard (Board, Needs, Host, Plan, Simulate), FastAPI JSON API and static web dashboard on localhost, `serve` | both interfaces show the same data as the CLI's `--json` |

## Phase 2: installer (shipped)

The tool makes the recommendation run.

- `install llama.cpp` (done): official release builds matched to OS, architecture and backend,
  checksum-verified, into a managed directory, optionally on `PATH`.
- `install model` (done): parallel resumable downloads from Hugging Face with SHA-256
  verification, split files, projector and MTP extras, disk-space check.
- `preset` (done): launch scripts per OS that choose the context size from the VRAM free at
  start, plus router `models.ini` sections.
- `launch` (done): start a preset, wait for health, print the endpoints, stop it again.
- The Downloads screen in the TUI and the install endpoints in the API are **not built**.
  Those four commands are the command line only; `llamafit` with no arguments and `llamafit serve` still read
  the machine and never change it.

The commands ship. What the phase still owes is its own proof: this phase is done when a fresh
Windows, macOS and Linux virtual machine goes from nothing to a running server through LlamaFit
alone, and nobody has yet run that.

## Phase 3: verifier (shipped)

The tool checks its own work and learns from the machine.

- `bench` (done): llama-bench and real server requests at the planned flags; generation and
  prompt throughput, time to first token, peak VRAM, budget lines from the server log.
- A local results database (done). The Benchmarks screen showing estimated versus measured is
  **not built**.
- Per-host calibration of the estimator constants from stored results. `bench --calibrate`
  fits what the measurements determine, refuses by name what they do not, and stores the fit
  beside the runs it came from. **Nothing reads it back yet**: `fit`, `recommend` and `plan`
  run the formula on the constants that ship in `llamafit/constants/`, so every speed on the
  board is still `estimated`. [benchmarking.md](docs/benchmarking.md) says what is missing.
- A paging detector for the silent NVIDIA driver fallback, VRAM pinned at the ceiling plus a
  speed drop (done). It reports three states: paging, not paging, and unchecked.

Here too the commands ship without the proof. Phase 3 is done when estimates on the
reference machine are within 15 percent after calibration, and they are not, because the
calibration a run produces is not yet applied to anything.

## Later

Ideas with a place in the design but no plan yet, in rough order of value:

- Opt-in sharing of measurements to a community dataset, with a hardware picker to browse
  results from machines like yours.
- Multi-GPU placement.
- Ollama and LM Studio as read-only sources of already-downloaded models.
- A `--target-tps` planner that answers "what would I need to run this at 30 tokens per second".
- Packaged binaries for people who do not have Python.

## Not planned

- Running any model inside LlamaFit to produce its answers.
- Fine-tuning, conversion or quantisation of models.
- A hosted service, accounts, or telemetry.
