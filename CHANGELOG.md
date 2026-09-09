# Changelog

All notable changes to LlamaFit are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may change interfaces;
the changelog says so when they do.

## [Unreleased]

### Added
- Project scaffold: packaging, CI matrix, lint, type-check and test configuration.
- Design specification covering host detection, llama.cpp integration, the model catalog,
  GGUF facts, memory budgets, placement planning, speed estimation, scoring, the CLI, TUI
  and web dashboard, the installer, and the verifier (`docs/specs/`).
- Implementation plan for phase 1A, foundation and host scan (`docs/plans/`).
- Repository documentation: README, roadmap, contributing guide, code of conduct, security
  policy, support guide, issue and pull request templates, and the `docs/` set.
- Reference-machine calibration measurements from 2026-09-09
  (`docs/calibration/2026-09-09-reference-machine.md`).
- Host scan: OS, CPU (cores, performance cores, instruction sets), memory (totals, DDR facts,
  measured or estimated bandwidth), GPUs (NVIDIA, AMD, Apple, generic), disks.
- llama.cpp detection: binaries, build, backends, local GGUF files, running servers.
- `llamafit system` and `llamafit doctor` with Rich tables, `--json`, hints and exit codes.
- Documentation: CLI reference, platform support, development guide, contributing guide.

### Fixed
- RAM bandwidth now measures sequential read throughput, not a copy: a copy moves each
  byte twice (read and write), which is not what llama.cpp's read-dominated weight
  streaming does, and a single thread also cannot saturate a multi-channel memory
  controller. With NumPy, a memory-bound reduction runs concurrently across a thread
  pool (physical core count, capped at 8); the pure-Python fallback, which cannot
  parallelise because it holds the GIL and still measures a copy, keeps its own
  (re-derived) correction factor and is labelled `estimated` rather than `measured`.
- Memory channel count is parsed from Windows (`BankLabel`/`DeviceLocator`) and Linux
  (`dmidecode`'s `Bank Locator`/`Locator`) slot labels when they encode a channel
  identifier, so the theoretical bandwidth estimate runs again on boards that report one;
  `llamafit system` shows the channel count next to the module count when it is known.

[Unreleased]: https://github.com/pvaz/llamafit/commits/main
