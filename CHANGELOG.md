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
  use is the one way of being wrong that turns into "this fits" when it does not. The always-on
  shared experts are a line of their own, so a tensor override that sends just those to system
  memory can be costed: on the reference machine it frees 239 MiB and turns Qwen3.8-Flash-Next
  at 32,768 tokens from 29 MiB over an 8 GB card into 211 MiB inside it, which is the best
  configuration anyone has measured on that machine. What a GPU backend costs before it
  allocates anything is a lookup keyed by the card, with the conservative 300 MiB default
  underneath it: a card somebody has measured gets the measured figure and every other card
  keeps the safe one, and the budget line says which of the two a reader is looking at,
  because 180 MiB on an 8 GB card is two rungs of the context ladder.
- `llamafit list` and `llamafit search`: browse the catalog, filtered by use case, capability,
  licence, vendor or text.
- `llamafit info <model>`: one model in full, with every quant it publishes and the GGUF facts
  read from it.
- `llamafit catalog validate|refresh|show`: check the catalog files, fill their volatile fields
  from Hugging Face without downloading any weights (`--dry-run`, `--check` and `--model`), and
  print one entry as JSON or YAML.
- Custom models: entries in `custom_models.yaml` are merged with the bundled catalog, an entry
  whose id matches a bundled one replacing it.
- `llamafit install llama.cpp`: downloads a published llama.cpp build and installs it into
  `~/.llamafit/llama.cpp`, which is the first directory the detector looks in, so nothing has to
  be configured afterwards. The archive is chosen from the asset names for the operating system,
  the architecture and the backend, with the CUDA runtime archive a Windows CUDA build cannot
  start without brought along beside it, and with the CUDA version chosen against the card's
  actual driver rather than by taking the newest — a CUDA 13 build does not run slowly on a
  driver below 580, it does not run at all. The whole plan is printed before anything is
  written: which release, which archive, how large, how much is already on disk, whether a
  checksum is published, where it will go, what is already there and how much room the volume
  has. A directory that does not carry LlamaFit's own marker file is never overwritten without
  `--force`, because people build llama.cpp themselves with their own flags and no download can
  put that back. The published checksum is verified on the partial file before anything is
  unpacked, and a file that fails it is deleted rather than left to be resumed onto; an
  interrupted download continues where it stopped; and `PATH` is changed only on an explicit
  `--add-to-path` and a yes, with the undo printed in the same breath as the offer.
- Placement planner: for one model and quantisation on one machine it searches the run modes
  in order -- everything on the card, routed experts in system memory, some layers on the card,
  the processor alone -- and keeps the best configuration that fits rather than the first,
  choosing the context, the micro-batch, the KV cache type, the number of layers on the card
  and where the vision projector lives. A faster mode wins over a longer context, and a floor
  under the context ladder is what keeps that from recommending a working configuration nobody
  could work in. It takes the memory budget as an injected function, so it can be read, tested
  and changed apart from the thing that sizes a configuration. For a mixture-of-experts model
  holding its routed experts in system memory it also tries sending the always-on shared
  experts after them, which is what `-ot ffn_.*_shexp=CPU` does: they run for every token, so
  it is tried second and never preferred, but on the reference machine those 239 MiB are what
  turns the best configuration anybody has measured there from withheld into recommended. A
  placement that moved them says so on its own command line.
- `llamafit.services.plan`: the join between the memory budget and the placement search, which
  is what lets the planner run against real models on a real machine rather than only against
  an injected fake.
- Hardware profiles: a machine described in a JSON file instead of probed, so LlamaFit can
  answer for a machine that is not this one -- a card somebody is thinking of buying, or the
  twenty identical machines a model is being chosen for from a build server that is none of
  them. A profile becomes the same `Host` a scan produces, so every service downstream takes
  it without knowing which it got, and single-value overrides substitute one pool of the live
  scan for a what-if. A simulated host cannot be mistaken for a measured one: it carries a
  `simulation` object naming the profile, its file and the pools that were overridden, and a
  `simulated` flag computed from it, so the fact reaches a `--json` consumer as a field rather
  than only a reader as a heading. On the terminal it is the first row of the host table, in
  red, before any figure.
- `llamafit hardware list|show|validate|path`: the profiles LlamaFit can reach, one profile
  with the `provenance` line that says where its figures came from, the host a profile stands
  in for (`show --as-host`), a check that follows the catalog's habit of returning problems
  rather than raising, and the directory your own profiles go in (`LLAMAFIT_PROFILES`
  overrides it). Beyond the schema, `validate` reports a name two files claim, a GPU figure
  that contradicts the bundled specification table, and match rules that would never
  recognise the machine the profile itself describes. The published JSON schema is generated
  from the model beside the catalog's, at `data/schema/hwprofile.schema.json`.
- One bundled profile, `reference-rtx4060-128gb`: the machine every estimator constant was
  fitted to and every `measured` block in the catalog was taken on, so a stranger can
  reproduce the documentation's numbers without owning it. A test holds it to that -- the host
  it produces is compared field for field against the reference machine the speed tests use,
  and the estimator reads the same effective bandwidths off both. Nothing aspirational is
  bundled: a profile with an invented memory bandwidth reads exactly like a measured one to
  everything downstream, so a profile must state where each figure came from, a memory
  bandwidth is refused without the label saying whether it was measured, estimated or assumed,
  and a card's bandwidth and compute figures are left out so they come from the bundled
  specification table rather than from a second copy that can drift.
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

- `llamafit recommend`: the board. Every model and quantisation in the catalog is planned
  (section 9), sized (section 8), estimated (section 10) and scored (section 11), and the
  result is one ordered list with `--use-case`, `--require`, `--prefer`, `--min-context`,
  `--max-download`, `--license`, `--limit`, `--all-quants`, `--no-vision`, `--explain` and
  `--json`. A candidate the request excludes is shown with the reason and what to change
  about the request, never dropped: a shorter list tells a reader nothing. `--explain`
  expands a row into the four scores, the weights that combined them, the quality it was
  built from, the memory budget line by line, the context ladder and where a token's time
  goes, which is the project's standing promise that no number has to be taken on faith.
- `llamafit fit`: every model ranked by how well it uses this machine and nothing else, with
  `--perfect`, `--min-fit`, `--limit` and `--all-quants`. It deliberately does not apply
  section 11's exclusions: a coding model is not left out of a fit listing for being one.
- `llamafit plan <model>`: one model placed on this machine, with `--quant`, `--context`,
  `--ub`, `--target-tps`, `--no-vision` and `--json`. It prints the memory budget component
  by component with the source of every figure, the context ladder with what each rung costs
  the card, where a token's time goes, the runs the catalog records for comparison, and last,
  on a line of its own, the `llama-server` command line. `--ub` rebuilds the whole budget
  rather than relabelling it, because a micro-batch moves the compute buffer and with it the
  verdict; `--context` beyond what fits sizes down **and** shows what the context asked for
  would have cost, so the overflow is met rather than merely retreated from.
- The context ladder has three states rather than two. A rung that overflows the card is
  marked `pages`, not `no room`: section 8.4's failure is that the driver accepts the
  allocation, moves the overflow to system memory, and lets the server start and look healthy
  while generation collapses. `ContextTier` now carries the verdict its own budget produced,
  which was being computed and thrown away, so the ladder no longer has to guess.
- `--prefer balanced|quality|speed` moves a tenth of the weight between quality and speed
  (section 12.1), capped by what the other part has to give so no weight goes below nothing.
### Changed
- **The placement planner steps down section 9.3's context ladder instead of halving.**
  Section 9.2 says so in as many words, and the difference is not cosmetic: from 40,960
  tokens, halving reaches 20,480 and then the floor and never 32,768 or 24,576, so on the
  reference machine with a desktop open the search stepped past every context that fits in
  the right mode and settled on a hybrid placement with two layers on the card and seventy
  gigabytes of experts on the memory bus. It now finds the mixture-of-experts placement the
  machine was measured running.
- **`-t` is the threads the performance cores provide, not the number of those cores.**
  Section 9.4: on the reference machine's i9-14900KF the eight performance cores carry
  simultaneous multithreading and provide sixteen of its thirty-two threads, sixteen is the
  measured optimum, and it is about four percent faster than eight. Efficiency cores are
  still excluded — generation measures slower with them — but excluding them is not the same
  as counting cores. The rendered command line now matches the flags the winning
  configuration was actually measured with.
- A quantisation's files are matched against the disk by bare file name on both sides. A
  catalog `files` entry is a path inside the publishing repository and often carries a
  directory, so a sharded model that was on the machine read as missing from it.
- `llamafit list` and `llamafit recommend` share one catalog loader, one model lookup and one
  pair of value checks (`llamafit/cli/common.py`), so a mistyped id, use case or capability
  fails the same way whichever command met it. The capability message names the flag that was
  actually typed, `--capability` or `--require`, which cost the thirty-seven catalogs their
  translation of the old wording.
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
