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
- Memory budget: for a model, a quant, a placement and a context, what every component needs
  and which pool it lands in — weights prorated by the layers on the card, the key-value cache
  and the recurrent state, the compute and output buffers, the vision projector and the two
  fixed overheads. Every line says whether its bytes came from the file's own tensor table or
  from a fitted formula, because a budget that presented both in one voice would lend the
  measured figures' authority to the modelled ones. A configuration that needs more than the
  card has free is reported as paging rather than as failing: an NVIDIA driver does not refuse
  the allocation, it pages to system memory and the speed collapses while the server starts and
  looks healthy, and that is the failure a person cannot see for themselves. Attention and the
  cache on the card with the experts in system memory is judged like any other placement.
  Where an architecture allocates memory its GGUF header does not describe — Qwen3.8-Flash-Next
  allocates about 9 MiB of cache per 1,024 tokens beyond the shapes it declares, 27 percent of
  its cache at 32K and 1.1 GB at 128K — the budget carries it on a line naming the architecture
  and the component rather than leaving it out. Reporting less memory than a configuration will
  use is the one way of being wrong that turns into "this fits" when it does not.
- `llamafit list` and `llamafit search`: browse the catalog, filtered by use case, capability,
  licence, vendor or text.
- `llamafit info <model>`: one model in full, with every quant it publishes and the GGUF facts
  read from it.
- `llamafit catalog validate|refresh|show`: check the catalog files, fill their volatile fields
  from Hugging Face without downloading any weights (`--dry-run`, `--check` and `--model`), and
  print one entry as JSON or YAML.
- Custom models: entries in `custom_models.yaml` are merged with the bundled catalog, an entry
  whose id matches a bundled one replacing it.
- Placement planner: for one model and quantisation on one machine it searches the run modes
  in order -- everything on the card, routed experts in system memory, some layers on the card,
  the processor alone -- and keeps the best configuration that fits rather than the first,
  choosing the context, the micro-batch, the KV cache type, the number of layers on the card
  and where the vision projector lives. A faster mode wins over a longer context, and a floor
  under the context ladder is what keeps that from recommending a working configuration nobody
  could work in. It takes the memory budget as an injected function, so it can be read, tested
  and changed apart from the thing that sizes a configuration.
- Context tier table: what each of ten context lengths costs on the card for the chosen
  placement, so a launch script can pick the largest one that fits the free VRAM it actually
  sees at the moment it starts, and a recommendation survives a browser being opened.
- `llama-server` flag rendering: a placement becomes the ordered argument list a person can
  paste, with the model and projector paths, the endpoint, the context, the cache type, the
  layer placement, the thread count, the batch sizes, the vendor's sampling and the catalog's
  own quirks. Numbers on a command line are never localised, and a curated quirk that is prose
  rather than arguments reaches the reader as a note instead of being pasted onto the line.
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
  reach a screen as a blank line. The `B` of `27B` is the one that does not vary: it is how
  these models are named, so localising it would make one table row disagree with the model
  identifier beside it.
- `scripts/gen_messages.py` copies a `# Translators:` comment block above a call into the
  template as `#.` lines, for what a message cannot say about itself.
- `CLA.md`, the contributor licence agreement: a contributor keeps their copyright and grants
  the owner a licence broad enough to keep offering LlamaFit commercially alongside the AGPL.
  `CONTRIBUTING.md` and the pull request template say how to agree to it, in one line, once.

### Changed
- **Licence: MIT is now the GNU Affero General Public License, version 3 or later**
  (`AGPL-3.0-or-later`). `LICENSE` holds the Free Software Foundation's text verbatim, with
  the notice the licence asks a program to carry appended and filled in. Using LlamaFit still
  costs nothing and requires nothing; the obligations begin only on distributing a modified
  version or running one as a network service. Section 13, the network clause, is why AGPL
  rather than GPL: the phase 1D web dashboard is exactly the case a plain GPL does not reach.
  The `README` says what this means for a user, and `NOTICE` records that every runtime
  dependency is permissively licensed and combines with the AGPL.
- Every module under `src/llamafit/` carries a three-line copyright and
  `SPDX-License-Identifier` header, so a file copied out of the project on its own still says
  what it is under.
- Packaging metadata declares `license = "AGPL-3.0-or-later"` as a PEP 639 expression with a
  matching trove classifier, and ships `LICENSE` and `NOTICE` in the wheel. The build backend
  floor moves to `hatchling>=1.27`, the first release that writes `License-Expression`.
- The capability names a reader sees — `coding`, `thinking`, `vision`, `tools`,
  `multilingual`, `long-context`, `embeddings`, `audio` — are translatable, each with a
  context of its own. They reached every language in English because they are catalog enum
  values rather than messages, and they are ordinary words. Thirty of the thirty-seven
  catalogs are filled; `ar`, `bn`, `he`, `hi`, `ta`, `th` and `ur` fall back to English
  until somebody who reads them fills them in. `--capability` still takes the English id.

### Fixed
- Messages six translators independently reported as impossible to translate well are
  fixed in the English rather than worked around in thirty-six catalogs. Lines built from
  optional pieces — the memory row's details, the GPU row's specifications and driver, a
  model's context, a quant's facts summary — are one whole message per shape instead of
  fragments joined behind a separator no catalog could reach; one of them appended a
  message that opened with a comma. Counts each have an entry of their own, so cores,
  threads, bytes, files, shards and tensors agree with the number beside them, and the
  experts ratio selects its form on the number the noun actually stands next to. The
  timeout message spells its unit out instead of welding a bare `s` to the placeholder.
  The list table's column headings and the row labels that shared their messages each
  carry a context, and the two captions are handed their heading rather than spelling its
  English word out a second time. `isa unknown`, `type unknown`, `VRAM unknown`,
  `unknown build`, `unknown model`, `unknown error`, `no specs`, `no server` and
  `not read yet` carry a context naming their row, like the entries beside them. The
  parameters row welded its `B` to a placeholder while the list table asked
  `billions_suffix()` for the same mark. Translations whose English is unchanged were
  carried across; a message that became a different sentence is untranslated everywhere
  and falls back to English rather than keeping a translation of something else.
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
- Arabic, Hebrew and Urdu no longer show a command flag backwards. A leading hyphen is
  direction-neutral, so `--verbose` inside a right-to-left sentence was drawn as
  `verbose--`, and a reader who retyped what they saw got a command that does not run.
  Every value the code treats as an identifier — a flag, a command name, a path, a model or
  repository id, a URL, a size with its unit — is now wrapped in Unicode isolates as it is
  interpolated; an identifier a translator kept verbatim inside prose is isolated by the
  renderer; each line holding a right-to-left letter is given an explicit base direction,
  which keeps a sentence's final full stop with the sentence; and a table's columns and
  alignments are mirrored so the first column is the first one read. Nothing changes for a
  left-to-right language, nothing was added to any catalog, and `--json` receives no
  direction mark in any language. `docs/translations.md` records what this cannot fix and
  which terminals act on it.

[Unreleased]: https://github.com/pvaz/llamafit/commits/main
