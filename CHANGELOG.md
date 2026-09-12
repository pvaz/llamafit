# Changelog

All notable changes to LlamaFit are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Until 1.0, minor versions may change interfaces;
the changelog says so when they do.

## [Unreleased]

## [0.1.1] - 2026-09-12

### Added
- Standalone executables for Windows, macOS and Linux on x86-64 and ARM: one
  self-contained file per platform, with the interpreter, the dependencies, the catalog,
  the generated facts, the hardware profiles, the schemas and all 37 message catalogs
  inside it. Nothing to install and no Python needed. Built by
  `.github/workflows/binaries.yml` from `packaging/llamafit.spec` on five machines, because
  PyInstaller is not a cross-compiler, and attached to the GitHub release with SHA-256
  checksums. The Linux builds are made on Ubuntu 22.04 so that they need glibc 2.35 and
  not 2.39, which is the difference between running on Debian 12 and RHEL 9 and refusing
  to start there. `docs/standalone.md` covers which file to take, how to verify it, what
  Windows and macOS will say about a binary nobody has paid to sign, and the one thing
  the binary leaves out.
- `.github/workflows/binaries-attach.yml` puts binaries on a release that already exists,
  by hand, calling the same build. It touches nothing else: a version number on PyPI can
  be used once and never again, while a release asset can be replaced any number of
  times, so a platform can be rebuilt after a fix without going near an upload that
  cannot be repeated.

### Changed
- **All 37 translation catalogs are complete.** Thirty-one of them carried about a
  quarter of the messages, the column headings and the short phrases without the sentences
  that explain anything, and now carry all 1,153. Every one of them still says at the
  top that it has not been read by a speaker of its language, because it has not, and
  that is still the most useful contribution anyone can make here.
- Every number a reader sees uses their own decimal separator. Eleven messages formatted
  floats with a numeric conversion, which writes a point whatever the language, so a
  sentence saying `0.92 GB` sat directly underneath a table that had just written
  `0,0165` for the same reader. No wording changed in any language.
- Eleven catalogs no longer guess the curator's gender in the one message whose English
  says "their machine". Each uses its own way of saying "one's own" instead.

### Fixed
- `multiprocessing.freeze_support()` is called before anything else in `main()`. Nothing
  in LlamaFit starts a process, since the bandwidth measurement uses threads, but
  py-cpuinfo does, and in a frozen build a child process is started by re-running the
  executable with `--multiprocessing-fork`, which Click meets as an option it has never
  heard of and answers with a usage error printed across whatever the real run was
  drawing. It affects the standalone binary only, on the first command most people try.
- `llamafit install llama.cpp` no longer fails against a release whose archives have not
  finished uploading. llama.cpp tags several builds a day and the files arrive after the
  tag, so for a few minutes the newest release carries nothing; the installer took it and
  said "release b10931 publishes nothing for windows x86_64", which is true of that tag
  and reads as a statement about the platform. A tag with no files is not a build anybody
  can install, so the newest one that publishes files is chosen, and the plan names it.
- The hint that says where a global option goes no longer blames the wrong option.
  `llamafit --gpu-vram 0 --ram 8GiB plan …` is a usage error about `--gpu-vram`, and the
  hint declared `--ram` misplaced and offered a command line that was not one: it took
  the value of the first option for the command name, and moved an option without its
  value.

## [0.1.0] - 2026-09-11

### Added
- `recommend` and `fit` take view flags that change what is drawn and never what is
  ranked: `--sort KEY[:asc|:desc]` over eleven keys (`score`, `speed`, `quality`,
  `context`, `size`, `prompt`, `card`, `ram`, `fit`, `model`, `quant`), `--search`,
  `--installed`, `--runs`, `--min-fit` on `recommend`, `--columns LIST`, `--wide` and
  `--hide-excluded`; each says so in a line under the table. The same eleven words are
  the dashboard's `s` cycle and the API's `sort=`, and a test asserts the three agree.
- Dashboard keys: `S` the previous sort key, `o` reverse the order, `e` hide or show the
  unranked rows, `c` every column with the table scrolling sideways; `n` now moves the
  cursor to the first unranked row, whose reason the explanation pane shows. The Needs
  form gained the slowest generation worth having, the command line's `--min-tps`.
- `llamafit --json` with no command prints `recommend --json` rather than opening a
  full-screen application over a document, and `llamafit --help` says what the bare command
  does: it opens the dashboard, and in a pipe prints what `recommend` would.
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
- `llamafit info <model>`: one model in full, with every quant it publishes, the GGUF facts
  read from it, and what each of those quants would cost on this machine — the run mode, both
  pools, an estimated speed, the largest context it holds and the verdict, one line each, with
  `--quant` to narrow it to one and `--context` to size it for a length you name. One line and
  no more: the budget component by component, the ladder and the command line are `plan`'s, and
  two commands printing the same screen is a defect of its own.
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
  `--min-tps`, `--max-download`, `--license`, `--limit`, `--all-quants`, `--no-vision`,
  `--explain` and `--json`. A candidate the request excludes is shown with the reason and
  what to change about the request, never dropped: a shorter list tells a reader nothing.
  `--explain` expands a row into the four scores, the weights that combined them, the
  quality it was built from, the memory budget line by line, the context ladder and where a
  token's time goes, which is the project's standing promise that no number has to be taken
  on faith. It then says what would change the answer: the next rung of the context ladder
  this machine cannot take and how much of the card would have to be free to reach it, and
  the next quantisation down with what it would run as. Both come from a placement actually
  computed rather than from arithmetic on the row above, and a row with no such answer — a
  model publishing one quantisation, a placement already at the top of its ladder — is given
  none, because a sentence promising a fit that is not there is worse than no sentence.
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
- **`--min-tps` lets a person with nobody waiting say so.** Section 11.4 excludes a candidate
  that generates more slowly than a person reads, which is right for somebody sitting in
  front of it and wrong for somebody running a batch — and until now there was no way to
  disagree with the figure, in either direction. `Needs.min_tps` carries the request's own
  speed and `--min-tps` sets it: `0` says nobody is waiting on these tokens and excludes
  nothing for being slow, and a larger figure raises the bar for a reader who skims. Unset
  means the reading floor, so a request that says nothing is judged exactly as it was
  before. The figure goes into the floor's place for both the exclusion and the ramp, from
  one call, so a board cannot exclude on one number while scoring against another. It
  applies to a throughput job too: the default answers "who is waiting", which is a fact
  about the job, and a typed figure answers "what will you accept", which is not the
  program's to overrule. Because a board that quietly stopped excluding would be the same
  failure the exclusion exists to prevent, a board whose floor was moved says so under
  itself, and every row excluded by a figure somebody typed names that figure and the flag
  that set it rather than the reading rate.
- Prompt processing shows its working, as generation already did. Section 10.2's three
  terms — the arithmetic, the expert set crossing the link to the card, and the same set
  read out of system memory when there is no card — arrive on `SpeedEstimate` per prompt
  token and are drawn as their own table wherever the token-time table is drawn, so each
  trio adds up to one over the figure above it. The link term is deliberately a row of its
  own rather than a share of a total: section 10.2's rate was fitted on the one model whose
  expert set does not fit in system memory, so part of its read comes off the disk, while a
  set that stays in the page cache streams about three times faster. Two physical paths,
  one constant, and on the reference machine's own best configuration that term is 88
  percent of a prompt token — which is the point of printing it. The gap is recorded, not
  fixed: the reader looking at the terms is the person most likely to catch it first.
- `llamafit serve`: a JSON API and a browser dashboard, on this machine only. The API calls the
  same service functions the commands call and hands back the identical
  `model_dump_json(by_alias=True)` that `--json` prints, so the two cannot disagree — a test
  runs both and compares the documents. `/health`, `/api/v1/system`, `/scan`, `/doctor`,
  `/models`, `/models/top`, `/models/{id}`, `/plan`, `/profiles`, `/catalog/schema`,
  `/catalog/problems` and `/ui`, with query parameters that mirror `recommend`'s options and
  are refused, by name, when they are not ones LlamaFit knows.
- The dashboard itself: the board with every row expanding into the score, the speed, the
  budget and the context ladder behind it, a Needs form that re-ranks it, the machine with
  every finding `doctor` reports, a plan with a copyable command line, and simulation against
  a hardware profile or a substituted pool. Plain HTML, CSS and JavaScript served from the
  package — no build step, no package manager, no framework, and nothing fetched from the
  internet when the page opens, so it works offline and opening it tells nobody. It says once,
  under the header, that nothing has been benchmarked on this machine and every speed on it is
  therefore a computed estimate.
- The dashboard is translated. It holds no words of its own: `GET /api/v1/ui` serves every
  label, built by the same gettext calls the rest of LlamaFit uses and extracted by
  `scripts/gen_messages.py` like any other message, so filling in a `.po` file translates the
  page too. The words for verdicts, run modes, confidence labels, memory pools and budget
  components are imported from the terminal's own renderer rather than written again.
- The server binds to loopback by refusal rather than by default: `serve()` will not bind
  elsewhere unless the caller says separately that it means to, which the command line says
  only when `--host` was actually typed, and it prints what that exposes first. The `Host`
  header is checked, so a website cannot point a name of its own at `127.0.0.1` and read the
  API through a visitor's browser; there are no CORS headers; and the `profile` parameter takes
  a profile's name rather than a path, so a request cannot ask the server to open a file.
- `fastapi` and `uvicorn` as the optional `web` extra, with their rows in
  `docs/development.md`: somebody who only wants the command line should not have to install a
  web server, and `llamafit serve` without them says so and gives the line that fixes it.
- `llamafit install model <id>`: the parallel, resumable model downloader (section 15.2).
  The files are tens or hundreds of gigabytes, so everything about it is about doing it once.
  Before a socket is opened it prints the model, the quantisation, the repository, every file
  it will fetch, what the set weighs, where it is going, how much is already there and what
  would be left on the volume afterwards; in a terminal it then asks, and outside one it does
  not, because there is nobody there to answer. The disk is checked against the catalog's own
  byte count first and the run refused with the shortfall named, since a hundred-gigabyte
  download that fills a volume at ninety percent has cost hours and left nothing behind. Each
  file is cut into 32 MiB chunks fetched by a pool of workers into a part file created at its
  final size, with a sidecar recording each chunk only once its last byte is written, so an
  interruption at eighty gigabytes continues rather than restarts — a test proves it by
  interrupting a transfer and then refusing, from the server side, to serve a byte that was
  already on disk. Every file is checked against the SHA-256 the catalog holds before it is
  moved into place, and one that does not match is deleted along with its part file and its
  record, because a model that downloaded wrong does not fail loudly: it loads and answers
  nonsense. A quant with no checksums in the catalog is refused outright unless
  `--allow-unverified` says so in as many words. A split model is one thing: every shard and
  every extra the entry declares are planned together under one total, and the manifest that
  means "this model is installed" is written only once all of them have arrived and been
  checked, so nobody ends up with three files of four and no warning. A server that answers a
  range request with the whole body is caught before a byte is written and that file restarts
  as one sequential stream; a server that rate-limits is waited for and given one fewer
  connection, permanently, rather than retried harder. `--limit-rate` is one token bucket
  shared by every worker, and `--workers` defaults to eight rather than the specification's
  sixteen, because the line is somebody's home connection and they did not ask for their
  video call to stop working. `Ctrl+C` sets a flag the workers read between pieces instead of
  raising into whichever one was running, and the last line says the run can be carried on.
- `llamafit install history`: what has been downloaded, when, how big it was and whether it
  finished, carrying the sentence a failed run gave at the time.
- **Launch presets: `llamafit preset <model>` writes a script that chooses its context when
  it runs.** A plan is a command line somebody has to paste; a preset is a file they can
  double-click, and it is what makes a recommendation survive contact with a real desktop. The
  plan behind it was computed while the machine was idle and the script runs when a browser
  has taken a gigabyte of the card, so the script does not carry a context: it carries section
  9.3's ladder — every rung with what it costs the card — reads how much is free at start
  time, keeps 256 MiB back for the desktop, and takes the largest rung that still fits. This
  is the failure the project was born from: a configuration over the card does not fail on an
  NVIDIA driver, it pages into system memory and runs at a fraction of its speed for weeks
  while `/health` answers and the log looks healthy. The ladder never climbs above the planned
  context, because the planner weighed system memory and the user's request too and the script
  can measure only the card, and when not even the smallest rung fits the script stops with
  exit code 4 rather than starting something that would page.
- Free card memory is read with the tool that ships with the driver: `nvidia-smi` on every
  platform, amdgpu's sysfs files on Linux, `vm_stat` on Apple silicon, where the graphics
  memory is the system memory. A card with no such tool — AMD or Intel on Windows, Intel on
  Linux — makes the script say so once in its header rather than apologise on every run, and a
  probe that answers with something that is not a number falls back to the planned context and
  says out loud that it did.
- A preset is three files: `start-<id>.cmd` or `start-<id>.sh` for the machine that asked,
  `models-<id>.ini` for a llama.cpp router, and `README-<id>.md` with the endpoints and the
  ladder. The router section says in its own comments that it is the one artifact that cannot
  run the ladder, since a router starts a model with no shell in between.
- **A file you have edited is never overwritten.** Each generated file carries a checksum of
  itself, and one whose checksum no longer matches is kept and named, with `--force` the only
  way to replace it — and `--force` says which files it discarded. A file that has lost its
  stamp altogether counts as edited, because both a hand-written and a hand-cut file are
  somebody's work.
- `llamafit launch <model>` runs the preset **script** rather than a command line rebuilt for
  the occasion, so the ladder still chooses and hand edits still apply; it waits for
  `/health`, prints the endpoints, and `--stop` stops the server and everything it started. A
  server already answering is reported rather than joined by a second one, and what was
  started is remembered by process id *and* start time so `--stop` cannot aim at whatever
  inherited a recycled id.
- `llamafit` with no arguments opens the terminal dashboard (section 13.2), and prints the
  board `llamafit recommend` would print when there is no terminal to draw one in. Five
  screens over one scan and one catalog: **Board**, the ranked table with the explanation
  under it; **Needs**, section 12.1's request as a form that offers the six use cases, the
  eight capabilities and the licences the catalog actually carries, so nobody has to know a
  value before they can type it; **Host**, `system` and `doctor` on one page, probe by probe;
  **Plan**, the budget, the context ladder, the flags and the command line, with `+` and `-`
  stepping along the ladder the planner produced rather than a step the screen invented; and
  **Simulate**, section 4.4's overrides and profiles, with `SIMULATED` in the header for as
  long as the figures are about another machine.
- The dashboard shows the command line's own renderables rather than a second drawing of the
  same facts, so section 12.3's promise that the two interfaces show the same explanation is
  kept by construction. The explanation is open by default: on a terminal it costs a page per
  row and sits behind `--explain`, and on a screen it costs nothing.
- No speed on the dashboard appears without saying what kind of number it is. Nothing has been
  benchmarked on any machine yet, so every figure is a formula on default constants, and the
  sentence saying so is a band above the table that does not scroll away rather than a caption
  under it. When rows disagree about how their speeds were arrived at, the label is a column
  beside each figure and a terminal too narrow for the pair shows neither.
- **Section 13.1's five global flags now exist**: `--profile`, `--memory`, `--ram`,
  `--cpu-cores` and `--max-context`. The first four ask what a machine that is not this one
  would run -- a card somebody is thinking of buying, or the twenty identical machines a
  model is being chosen for from a build server that is none of them -- and `--profile`
  answers without probing this machine at all, which is what makes it work on a machine
  whose graphics driver hangs a probe. They apply to `system`, `fit`, `recommend`, `plan`
  and `preset`, and to `llamafit` with no arguments, which opens the dashboard already
  simulating that machine with its badge showing. `doctor` refuses them rather than printing
  this machine's probes under a heading claiming another's, and `bench` refuses a simulated
  machine because a benchmark measures what it runs on.
- **A board, a fit listing and a plan now say in their own JSON that they are not about this
  machine.** A `Host` has carried `simulation` and `simulated` since profiles landed, but a
  board carries no host, so a program reading `--json` had nothing to read: the red
  `SIMULATED` line above the table is no use to a script. `Board`, `FitBoard` and
  `PlanReport` carry the same two fields, set from the host they were computed against, with
  `simulated` computed from `simulation` so neither can drift or be left out. `"simulated":
  false` on a scan is part of the promise, because absence is not evidence. On the terminal
  the banner is the first line above each of the three, and `fit`'s own heading stops reading
  "Fit on this machine" one line under "they are not this machine".
- `--max-context` caps every context the run will size for, offer or score against: the
  planner's search, the context ladder a launch script chooses from, the largest context
  reported, and the denominator section 11.2's score is measured against. It is not a host
  property and does not sit with the four above -- it changes the question, not the subject --
  so it lives on `Needs` beside `--min-context`, and a ceiling below that floor is refused by
  name rather than answered with "this model is too short", which would blame the model for
  the request. The web API gains the `max_context` query parameter section 13.3 always
  promised, beside `min_context`.

### Changed
- **The terminal's board is the web page's board.** One list: the ranked rows, then the
  candidates that were not ranked, dimmed, every column filled with whatever was computed
  for them and the word for the reason -- `too slow`, `no room`, `unsupported` -- where the
  score would be, in the command line and in the terminal dashboard alike. The separate
  `Not ranked` and `Not placed` tables are gone; on the reference machine the first said the
  same forty words twenty-six times under a twenty-line board. The reasons are under the
  table, once each, with the ids and the figure that failed after them. `--explain` still
  prints the whole sentence under an expanded row, and `--json` is unchanged.
- **One column chooser for three interfaces.** The command line and the dashboard each had
  one, with different orders, a model column that folded at 28 in one and was cut at 24 in
  the other, and different padding; the command line's also budgeted the model column from
  the rows a limit left, so `--limit 2` drew nine columns where the default ten drew seven.
  `board_columns` in `llamafit/cli/render_board.py` is now the only one: a fixed priority
  (`Tok/s`, `Fit`, `Runs`, `Ctx`, `Qual`, `Card`, `Size`, `Have`, `How`, `Prompt tok/s`,
  `RAM`), a column admitted only whole, the model column measured from the catalog, the
  labels measured in the language in force so a Portuguese board admits a column later
  rather than overflowing, and the admitted columns drawn in the web page's order so each
  keeps its place at every width. A test draws the command line's board, the dashboard's
  table and the page's heading list at 80, 100, 120, 140, 160 and 200 columns and asserts
  the same headings in the same order. The dashboard gained the `Have` column the command
  line lacked.
- The dashboard's `/` box is a filter box: it takes the page's terms (`fit>=fits`,
  `speed>=20`, `size<=30G`, `card<=6G`, `ram<=32G`, `ctx>=32K`, `quality>=70`, `runs=gpu`,
  `have`) beside plain text, `f` and `a` set terms in the same model, and the state line
  names every term in force. The API's `sort=` accepts the same eleven keys `--sort` does,
  with `:asc` and `:desc`, from the same function; `sort=size` now puts the largest download
  first, as the page's heading does, and `sort=size:asc` the smallest.
- **`Gen/s` is now `Tok/s`, in the command line, the terminal dashboard and the browser.**
  `Gen/s` was an abbreviation of something the reader had to already know, and the figure it
  named is the one the whole tool exists to produce; a person looking for tokens per second
  did not find the column. It is one catalog entry read by all three interfaces, so it was
  renamed in all three at once. The prompt-throughput column, `PP/s` on the board and
  `Prompt/s` in the measurements table, is now `Prompt tok/s` in both: two figures that are
  both tokens per second, sitting side by side, cannot have only one of them say so.
- **The browser dashboard is one page.** Board, Needs, Host, Plan and Simulate were five tabs
  and are now a machine on a line, a row of controls that is always visible, and a list. Every
  control re-asks the server as it changes rather than waiting for an Apply button. A plan
  belongs to a model, so it is built inside the row that names one; the machine and the
  machine to simulate are disclosures behind the line, not screens.
- **The list is seven columns and carries the candidates that were not ranked.** It kept the
  first seven of the priority the terminal's board had already settled twice, and moved
  `Runs`, `Qual`, `Card`, `Prompt tok/s`, `Size` and `RAM` into the row, which opens onto them.
  A candidate that did not qualify is a row in the same list, quieter, keeping the columns
  that identify it and spending the rest of its width on the reason it was refused; the whole
  reason is in the row when it is opened. Fourteen columns of one weight was a table that hid
  its own answer.
- **The dashboard is dark, and does not follow the viewer's theme.** One palette, whose
  neutrals carry a constant blue bias rather than being flat grey, and colour on two things
  only: how fast a configuration runs, and whether it fits. Borders, fills, radii and shadows
  are spent on the three things that are separate objects — an expanded row, a block of text
  to copy, and a control a pointer has to hit — and everything else is separated by space.
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
- **Section 10.2 of the specification described a prompt-processing formula that has not
  been the one running since the tensor-throughput correction.** It showed one compute term
  charged wholly to the graphics card; the code splits it by the share of the layers each
  device holds and applies the card's efficiency only to the card's half. The section now
  says what runs, including where the CPU's rate came from -- 0.44 TFLOP/s per performance
  core, from Qwen3-0.6B Q8_0 at `-ngl 0 -t 8` reaching 2,924 prompt tokens per second, which
  is 3.51 TFLOP/s across eight of them -- and that it is effective rather than a peak and is
  not scaled by `eff_pp` a second time. `docs/calibration/2026-09-09-reference-machine.md`
  records the two prompt figures the constants are fitted to and retires the superseded
  pair, `eff_pp` 0.30 and the invented `CPU_FP16_TFLOPS_PER_CORE` 0.05, the way it already
  retires the per-layer overhead and the 0.36 scattered efficiency: named, explained, with
  the run they came from kept. `docs/how-it-works.md` carried the superseded generation
  formula too -- one system-memory efficiency and a 0.2 ms per-layer overhead -- and now
  carries the one in force.
- One verdict was called `roomy` in the board's column and `Comfortable:` in the sentence
  explaining it, which asked a reader to work out that two words were one verdict and asked
  thirty-seven translators to keep two unrelated words in step. Every other verdict opens its
  sentence with the word in the column; this one now does too, and its six translations
  already did.
- The web dashboard's string table carried `any`, which nothing asked for and whose source
  gave a translator no way to know what it qualified, so six languages left it in English.
  It is gone, and with it the entry in each of those six catalogs.
- **The fit score's left-hand slope reads the model, not the pool.** Section 11.3 has a
  slope that penalises a model far smaller than the machine, precisely so the tool does not
  always recommend the smallest safe option — and it did not work, because it was measured
  on the worst pool's utilisation. On a machine with a small card the worst pool is the
  card, and most of what sits on it is the compute buffer, the output buffer, the key-value
  cache and the backend's own overhead: a 0.6B at Q8 puts 0.6 GB of weights on an
  eight-gigabyte card and 6 GB of cache and buffers on top, so the card reads ninety percent
  full and a 27B model reads the same. Worse, the cache is elastic — the planner grows the
  context until the card is full — so the score was reading the planner's appetite rather
  than the model, and every candidate came out equally well fitted. On the reference
  machine's 128 GB, `llamafit recommend` put a 0.6B first for a general request and second
  for coding, and `llamafit fit` called it the second best-fitting model on the machine.
  The waste question is now asked of the model's own resident weight tensors against every
  byte of memory the machine has to hold them in, both pools added, on a scale of halvings:
  full marks at half the machine, and the price the specification already put on waste —
  thirty points for a shortfall of two and a half times, about 23 points per halving —
  applied per halving all the way down instead of flattening at a fifth. The floor goes with
  the quantity that needed it: this measure is relative to the machine, so a 0.6B is half a
  small laptop and scores 100 there while scoring nothing on a workstation. The crowding
  question is untouched — same shape, same constants, same worst-pool quantity — and the two
  are combined by taking the worse, since neither complaint excuses the other. The 0.6B now
  comes last for general, last for coding and last on `fit`, and `--explain` names both
  quantities rather than the one.
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
- **A benchmark is no longer called `measured` for a run it does not describe.** The
  estimator treated a flag a recorded run does not mention as agreement, so a run that
  never says where its layers went matched every placement in sight. The catalog's own
  winning row for Qwen3.8-Flash-Next is such a row — it recorded `-ub 1024`, the projector
  off the card and the shared experts in system memory, and nothing about `-ngl` or
  `--n-cpu-moe` — and it matched a hybrid split holding two layers on the card as readily
  as the full offload it was actually taken on. Six flags are now load-bearing, and they
  are exactly the ones that decide which bytes are read out of which pool: `-ngl`,
  `--n-cpu-moe`, `-ot ffn_.*_shexp=CPU`, `--no-mmproj-offload`, `-ctk`/`-ctv`, and `-ub`
  for prompt processing. An unrecorded switch is read as the default it stands for — no
  override, `f16`, `-ub 512` — but an unrecorded `-ngl` is read as nothing at all, because
  that one flag separates 78 tokens per second from 279.5 for the same file on the same
  machine and this project's own records keep it in a base line the catalog rows quoted
  only the delta of. A run that does not describe the placement still calibrates the
  formula and says which run it came from; it cannot be labelled `measured`, and when its
  layer split is unknown the estimate says that the correction assumed this placement.
  Nothing printed differently today — no command feeds catalog measurements to the
  estimator, and section 10.3 reserves `measured` for a benchmark taken on *this* machine —
  but phase 3 stores exactly such benchmarks and would have walked straight into it. The
  Flash-Next row now records the whole command line rather than the delta.
- **A model nobody can wait for scores zero rather than a fraction.** Section 11.4's straight
  line to the origin gave Gemma 3 27B at 2.1 tokens per second 8.4 points out of 100 for a
  general request, which put it above Llama 3.1 8B at 9.7 on the same machine. A person
  waiting three and a half minutes for a five-hundred-token answer is not getting eight
  percent of a good experience. The score now runs between two speeds instead of one: the
  target, which is a property of the job and where the score stops, and a floor of six
  tokens per second, which is a property of the *reader* — silent reading at about 240 words
  a minute, three quarters of a word to a token — and where it starts. Between them the axis
  is doublings and not tokens per second, because six to twelve is the difference between
  waiting and not waiting while twenty-one to twenty-five is a difference nobody can feel,
  and a straight line prices them the same; it is the axis section 11.3's capacity arm
  already chose, for the same reason. The floor does not move with the use case, because a
  person reading a reasoning trace reads no faster than one reading a chat reply, and
  embedding has no floor at all: nobody reads an embedding, so it keeps the straight line
  and is scored on prompt throughput. What it costs is named rather than hidden — where
  nothing reaches six tokens per second the speed term stops separating candidates at all.
- **A model slower than its reader is excluded from an interactive board, not ranked on
  it.** The speed score already knew that six tokens per second is the rate a person reads
  at and already scored everything below it zero, and zero was not enough: on the reference
  machine Gemma 3 27B at 2.2 tokens per second led a general board at 54.92 over Llama 3.1
  8B at 51.34 with its speed column reading nothing at all. That is not the speed term being
  outvoted — it is the speed term at its rail, unable to say anything worse about a model
  that cannot be used, while quality and fit go on being right about a model nobody can wait
  for. Reweighting only demotes: on the bundled reference profile, where the ordering was
  never inverted, the same model still sat at number two of three for general, chat and
  multimodal. So a candidate below the floor is now excluded with its speed and the floor in
  the reason, keeping its placement and its estimate the way every other exclusion does, and
  a throughput job with no reader can never be excluded this way. The rule adds no constant:
  it reuses the reading floor, which was derived from reading rates rather than fitted to a
  board, and it fires on the machine rather than on the model — the same entry at the same
  quantisation leads a general board on a machine with a card that holds it.
- **A request for a job asks for the ability the job needs, not for a curator's wording.**
  Section 11.1 briefly excluded a model whose `use_cases` did not contain the requested one.
  It fixed a real defect — Llama 3.1 8B ranking second for a coding request — and broke a
  commoner case: ask for a general model on the reference machine and Qwen3-Coder-Next, the
  fastest thing that fits it, was not ranked at all, on the grounds that "its entry lists
  coding". A coding model is a perfectly reasonable thing to hold a general conversation
  with, and the rule was hiding the best answer. The mistake was the field. `use_cases` says
  what a model is *for*, which is a curator's emphasis; `capabilities` says what it can *do*,
  which is a fact about the weights, and only a fact can carry a gate — excluding on emphasis
  throws away good answers and turns every judgement call into a hard filter nobody knew they
  were setting. So a requested use case now implies a capability where one exists —
  `CAPABILITY_FOR_USE_CASE` beside the two vocabularies in `models/catalog.py`, exhaustive and
  refusing to load until a new use case has been decided about — and the capability gate does
  the work: coding implies coding, reasoning implies thinking, multimodal implies vision,
  embedding implies embeddings, while general and chat imply nothing because they are the
  absence of a specialisation rather than one. The case that started it survives unharmed:
  Llama 3.1 8B lacks the coding capability as well as the coding use case, so a coding request
  still excludes it, now naming the ability rather than a list of jobs — "no coding capability,
  which coding needs; ask for a different use case, or add it to the entry when the model
  really has it". `use_cases` keeps its job: the first entry is still worth +5 when the request
  names it, so a model built for the task still outranks one that merely can do it, and
  `llamafit list --use-case` still browses on the whole list.

- **`--ram` and `--memory` below what this machine is using produced a machine with nothing
  free on it.** The overrides kept the absolute bytes this machine had in use and subtracted
  them from the pool the flag named, so `--ram 16GiB` on a 128 GiB machine with 28 GiB in use
  computed 16 minus 28 and clamped to zero: `16.0 GiB total, 0 B available`, every board
  empty, including for a four-billion-parameter model that would have sat entirely on the
  card. It survived because the obvious thing to test is a bigger machine, where the old rule
  was right. A size names a machine now — `--ram 16GiB` is a 16 GiB machine, not this one with
  the difference taken off the top — and one rule, `_carried_load`, serves both pools: the
  same load on a pool the same size or larger, because a bigger card does not empty itself,
  and the same *share* of a smaller one, because this machine's open applications could never
  have fitted on it. A flag naming the size the machine already has moves nothing, byte for
  byte, in either direction.
- **The red `SIMULATED` line was drawn inside the board's table, so a board that ranked
  nothing carried no warning at all** — an empty answer about somebody else's machine, which
  is the answer most in need of one, because there is no figure in it to be suspicious of.
  The line is a property of the answer now rather than of one renderer: `simulated_answer` in
  `cli/render.py` is the single place it is attached, `recommend` and `fit` render their own
  "nothing was ranked" sentence through it instead of the command choosing between a renderer
  and a bare print, and `preset` and `launch` — which write launch scripts to disk sized for
  the substituted machine — print it too, where they printed nothing before.
- **The board did not say which machine produced it, and two of its inputs move on their
  own.** Seven runs in forty minutes moved the top row between two quantisations, the speed
  between 21 and 29 tokens per second and the ranked count between 30 and 38, with nothing in
  the output admitting anything had changed. `recommend` and `fit` now carry `MachineFacts` —
  what was free in both pools, and the RAM bandwidth with its source — on the board and in
  `--json`, so two answers can be told apart by reading them. And the bandwidth itself has
  stopped wandering: it is measured as the best of five short windows rather than one
  (contention can only make a reading low, never high, so the maximum of several walks
  towards the machine while the mean walks towards its load), and then kept in the platform's
  cache directory under a key naming the processor, the pool and the modules' type, speed and
  channel count. A figure read back says so — `bandwidth_cached`, *cached* in the table — and
  `llamafit system --refresh-bandwidth` times it again. Free memory is left to move, because
  reading it is the point of reading it; what that needed was the board saying what it used.

[Unreleased]: https://github.com/pvaz/llamafit/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/pvaz/llamafit/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/pvaz/llamafit/releases/tag/v0.1.0
