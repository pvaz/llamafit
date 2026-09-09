# Architecture

This page is the map for someone about to change the code. The full design, including the
formulas and the parts not yet built, is the
[design specification](specs/2026-09-09-llamafit-design.md); this page does not
repeat it.

## Layers

```
Interfaces    cli/ (Typer)      tui/ (Textual)      web/ (FastAPI + static dashboard)
              render only; call services; never compute

Services      scan · recommend · plan · install · bench · catalog · doctor
              compose core modules; take plain inputs; return pydantic models; never print

Core          hardware · hwprofile · llamacpp · catalog · gguf · budget · placement
              speed · quality · scoring · presets · download
              pure functions over typed data; OS access behind small protocols

Foundation    models (pydantic) · constants · paths · logging · errors · units
```

Rules that keep it that way:

- **Interfaces do not compute.** If a screen needs a number, a service returns it. The three
  interfaces render the same pydantic models, so `--json`, the TUI and the API cannot disagree.
- **Services do not print.** They return models and raise `LlamaFitError` subclasses.
- **Core is testable without a machine.** Every subprocess goes through `Runner`, every HTTP
  call through `HttpClient`; tests inject recorded output. Anything else that touches the OS
  (file listing, disk usage) takes paths as arguments.
- **Constants have provenance.** Every estimator constant lives in `constants.py` with a
  comment saying where it came from, and can be overridden by a hardware profile.

## Packages

| Package | Responsibility | Depends on |
|---|---|---|
| `models` | every data structure, as pydantic models | nothing internal |
| `errors`, `units`, `paths`, `logging`, `constants` | foundation | `models` |
| `i18n` | choose a language and translate a message; see [translations.md](translations.md) | nothing internal |
| `hardware` | host detection: CPU, memory, bandwidth, GPUs, disks; `scan()` | foundation |
| `hwprofile` | hardware profiles: load, match, validate, simulate | foundation |
| `llamacpp` | find the installation, read version and backends, discover servers; later install and launch | `hardware.runner` |
| `catalog` | load, validate, refresh and merge the YAML catalog | `gguf` |
| `gguf` | read GGUF headers locally or over HTTP range requests; derive architecture facts | foundation |
| `budget` | memory need by component for a candidate under a placement | `gguf`, `catalog` |
| `placement` | search placements and context tiers; render `llama-server` flags | `budget` |
| `speed` | generation and prompt estimates with confidence labels | `budget`, `hwprofile` |
| `quality`, `scoring` | the four scores and the composite | `catalog`, `speed` |
| `presets` | launch scripts and router sections (phase 2) | `placement` |
| `download` | resumable parallel downloads with verification (phase 2) | foundation |
| `bench` | measurements, storage, calibration (phase 3) | `llamacpp`, `hwprofile` |
| `services` | the operations the interfaces expose | everything above |
| `cli`, `tui`, `web` | rendering | `services` |

## Data flow of a recommendation

1. `services.scan.scan_system()` builds a `Host` and a `LlamaCpp`.
2. `catalog.load()` yields `CatalogModel` entries with `Quant` lists; `gguf.facts()` supplies
   or confirms architecture facts per quant.
3. For each candidate (model × quant) `placement.plan()` searches placements and contexts,
   calling `budget.compute()` for each, and keeps the best that fits.
4. `speed.estimate()` produces generation and prompt figures with a confidence label.
5. `quality` and `scoring` produce the four scores and the composite.
6. `scoring.rank()` orders candidates, applies the user's `Needs`, and attaches explanations.
7. An interface renders the board; `services.plan` renders the winner's flags and, in
   phase 2, presets.

## State on disk

Directories come from `platformdirs` with the application name `llamafit`; `LLAMAFIT_HOME`
puts everything under one directory.

| Item | Windows | macOS | Linux |
|---|---|---|---|
| Config (`config.toml`) | `%LOCALAPPDATA%\llamafit` | `~/Library/Application Support/llamafit` | `~/.config/llamafit` |
| Data (profiles, custom models, benchmarks database) | `%LOCALAPPDATA%\llamafit` | `~/Library/Application Support/llamafit` | `~/.local/share/llamafit` |
| Cache (catalog refresh results, GGUF headers) | `%LOCALAPPDATA%\llamafit\Cache` | `~/Library/Caches/llamafit` | `~/.cache/llamafit` |
| Logs (`llamafit.log`, rotated) | `%LOCALAPPDATA%\llamafit\Logs` | `~/Library/Logs/llamafit` | `~/.local/state/llamafit/log` |
| Downloads (models) | `~\llamafit\models` | `~/llamafit/models` | `~/llamafit/models` |

The downloads directory and the llama.cpp directory are configurable in `config.toml` and by
command options.

## Error handling

`LlamaFitError(message, hint, command)` and its subclasses (`ProbeError`, `ConfigError`,
`CatalogError`, `NetworkError`, `NotInstalledError`, `PackagedDataError`) are the only
exceptions that reach the user, rendered as message, command and hint without a traceback
unless `--verbose`. Probes never raise; they record a `Probe` with `ok=False` and an error
string that `doctor` shows with a hint.

`PackagedDataError` is the one raised for a broken installation rather than for anything the
user did. The catalog, the GPU table and the translations are read from inside the package
through `importlib.resources`, so a wheel built without one of them installs cleanly and fails
on the first command that needs it. Every reader goes through `llamafit/data/__init__.py`,
which turns `importlib`'s `ModuleNotFoundError` — and a directory that shipped empty, which the
catalog loader would otherwise read as a catalog with no models in it and no problem to report
— into a sentence naming what is missing, with a reinstall as the hint. It is the one error
where telling someone to reinstall is honest advice rather than a shrug, and it exits 2 rather
than 1: an installation missing its own data is an environment problem, not something the user
wrote wrong.

## Testing strategy

Unit tests per module with recorded fixtures; golden host profiles for the estimator with
tolerances (memory 5 percent, speed 25 percent) against recorded measurements; catalog schema
tests; CLI snapshot tests through Typer's runner; API tests through the httpx test client; TUI
tests through Textual's `Pilot`. The CI matrix covers three operating systems and two Python
versions. Details in [development.md](development.md).
