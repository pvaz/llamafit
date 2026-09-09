# Roadmap

LlamaFit is built in phases. Each phase ships working software with tests on Windows, macOS
and Linux, and each has a written plan under `docs/plans/` that lists its tasks.
The design for all phases is in `docs/specs/2026-09-09-llamafit-design.md`.

Dates are not promised. The order is.

## Phase 1 — advisor

The tool answers "what can I run here, at what speed, with which settings" without touching
the machine beyond reading it.

| Part | Delivers | Done when |
|---|---|---|
| **1A foundation and host scan** | package, CI, error handling, host detection (CPU, memory, bandwidth, GPUs, disks), llama.cpp detection (binaries, build, backends, local models, running servers), `llamafit system`, `llamafit doctor` | tests green on three operating systems; the reference machine's scan is reproduced from recorded fixtures |
| **1B catalog and GGUF facts** | YAML catalog with schema validation, GGUF header reader for local files and remote URLs, `refresh` from the Hugging Face API, custom model overrides, `list`, `search`, `info`, `catalog validate`, `catalog refresh` | the first catalog covers the current open-weight families with sources for every number; `info` shows exact per-quant facts |
| **1C budget, placement, speed and scoring** | memory budget by component, placement search with context tiers, speed estimator, quality and context scores, composite ranking, explanations, hardware profiles and simulation, `fit`, `recommend`, `plan`, `hardware` | the reference machine's board places the two measured models where the measurements say, within tolerance |
| **1D interfaces** | Textual terminal dashboard (Board, Needs, Host, Plan, Simulate), FastAPI JSON API and static web dashboard on localhost, `serve` | both interfaces show the same data as the CLI's `--json` |

## Phase 2 — installer

The tool makes the recommendation run.

- `install llama.cpp`: official release builds matched to OS, architecture and backend,
  checksum-verified, into a managed directory, optionally on `PATH`.
- `install model`: parallel resumable downloads from Hugging Face with SHA-256 verification,
  split files, projector and MTP extras, disk-space check.
- `preset`: launch scripts per OS that choose the context size from the VRAM free at start,
  plus router `models.ini` sections.
- `launch`: start a preset, wait for health, print the endpoints, stop it again.
- The Downloads screen in the TUI and the install endpoints in the API.

Done when a fresh Windows, macOS and Linux virtual machine goes from nothing to a running
server through LlamaFit alone.

## Phase 3 — verifier

The tool checks its own work and learns from the machine.

- `bench`: llama-bench and real server requests at the planned flags; generation and prompt
  throughput, time to first token, peak VRAM, budget lines from the server log.
- A local results database and a Benchmarks screen showing estimated versus measured.
- Per-host calibration of the estimator constants from stored results.
- A paging detector for the silent NVIDIA driver fallback (VRAM pinned at the ceiling plus a
  speed drop).

Done when estimates on the reference machine are within 15 percent after calibration.

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
