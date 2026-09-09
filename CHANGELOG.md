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
- Model catalog: curated YAML per model family, with parameters, architecture, context,
  capabilities, licence, published benchmark scores and the Hugging Face repositories that
  publish the GGUF files. The volatile fields live in a generated `<family>.facts.json` beside
  each file, so nothing ever rewrites hand-written YAML.
- GGUF header reader for local files and for remote URLs over HTTP range requests, and the
  architecture facts derived from a header: layer, head and vocabulary counts, expert counts,
  the byte breakdown by tensor class, KV bytes per token, the declared context length and
  sliding window. A quant published as several shards is read as a set, not from its first file.
- `llamafit list` and `llamafit search`: browse the catalog, filtered by use case, capability,
  licence, vendor or text.
- `llamafit info <model>`: one model in full, with every quant it publishes and the GGUF facts
  read from it.
- `llamafit catalog validate|refresh|show`: check the catalog files, fill their volatile fields
  from Hugging Face without downloading any weights (`--dry-run`, `--check` and `--model`), and
  print one entry as JSON or YAML.
- Custom models: entries in `custom_models.yaml` are merged with the bundled catalog, an entry
  whose id matches a bundled one replacing it.
- Translated interface: every message a person reads goes through the GNU gettext layer, and
  a global `--language TAG` option chooses the language. The order is the option, then
  `LLAMAFIT_LANGUAGE`, then the operating system's locale, then English; a language LlamaFit
  does not have falls back to English and names the ones it does have, and a request served by
  another region's catalog says which variety the reader is getting. The option is read before
  anything is rendered, so `--language pt_PT --help` is translated too, and the notice goes to
  stderr so `--json` stays machine-readable. `docs/translations.md` says how to add a language.
- European Portuguese catalog (`pt_PT.po`) covering the wrapped interface. It has had no
  second reader, and both the file and `docs/translations.md` say so.
- Numbers are written the way the reader's language writes them. The group separator, the
  decimal separator and the abbreviation for a thousand million are catalog entries, so a
  language sets its own without a locale database or a dependency, and one that has not
  filled them in falls back to English rather than to nothing. English writes a model's
  context as `32,768`, which a reader who groups with a point would otherwise read as a
  fraction. A language that groups with a space says so by writing that space: those three
  entries are read with `pgettext_literal`, which takes a translation of nothing but
  whitespace at its word, while every other message still falls back to English rather than
  reach a screen as a blank line.
- `scripts/gen_messages.py` copies a `# Translators:` comment block above a call into the
  template as `#.` lines, for what a message cannot say about itself.

### Fixed
- RAM bandwidth now measures sequential read throughput, not a copy: a copy moves each
  byte twice (read and write), which is not what llama.cpp's read-dominated weight
  streaming does, and a single thread also cannot saturate a multi-channel memory
  controller. With NumPy, a memory-bound reduction runs concurrently across a thread
  pool (physical core count, capped at 8); the pure-Python fallback, which cannot
  parallelise because it holds the GIL and still measures a copy, keeps its own
  (re-derived) correction factor and is labelled `estimated` rather than `measured`.
- Memory channel count is parsed from Windows (`BankLabel`/`DeviceLocator`) and Linux
  (`dmidecode`'s `Bank Locator`/`Locator`) slot labels when they encode a channel letter
  or a `ControllerN-DIMMx` per-controller locator, so the theoretical bandwidth estimate
  runs again on boards that report one; `llamafit system` shows the channel count next to
  the module count when it is known.
- Three hints named what they were about instead of pointing at it with a pronoun whose
  antecedent was in a neighbouring message, which a translator, and a reader taking in one
  line, never sees.
- The `list` table's column budget measures terminal cells rather than characters. A
  Japanese heading is two characters and four columns wide, and a Devanagari one counts
  combining marks that occupy no column at all, so a budget in characters admitted columns
  it had no room for.

[Unreleased]: https://github.com/pvaz/llamafit/commits/main
