# The terminal dashboard

Branch `feat/tui`, worktree `llamafit-wt-tui`. Two commits:

| SHA | Subject |
|---|---|
| `d951ca8` | `build: declare textual, the terminal dashboard's toolkit` |
| `6937095` | `feat(tui): open a dashboard when llamafit is given nothing to do` |

`pytest --cov -m "not hardware and not network"`: 2098 passed, 1 skipped, 11 deselected,
97.12 % total coverage. `src/llamafit/tui/` is at 100 % of statements with one partial
branch. `ruff check`, `ruff format --check`, `mypy --strict` clean; the three generators
produce no diff.

## What was built

`src/llamafit/tui/`, 21 modules, about 1,000 statements.

```
tui/
  state.py        Dashboard: the only caller of a service in the package
  board_view.py   which columns fit, which rows show, in what order (no Textual)
  summary.py      the machine line and the band that says what a speed is
  explain.py      the command line's --explain, plus the runs that check it
  format.py       four display helpers
  labels.py       the words on the Needs form
  keys.py         every key, declared once
  entry.py        `llamafit` with no arguments; the non-terminal fallback
  app.py          header, tab bar, five screens, one key bar, the scan worker
  screens/        board, needs, host, plan, simulate, help
  widgets/        RichPane, MachineBar
  styles.tcss
```

`cli/app.py` gains one import line and one call: `typer.echo(ctx.get_help())` in the root
callback became `open_dashboard(ctx)`. Nothing else outside `tui/` changed except
`pyproject.toml`, `docs/development.md`, `docs/cli.md`, `docs/tui.md`, `CHANGELOG.md` and
the regenerated `messages.pot`.

### The shape

`tui/state.py` holds a `Dashboard`: the scan, the catalog, the request, the substitution
and the board that came back. It is the only file in the package that calls
`services/`. Every screen reads fields off it; no screen has a catalog or a host, so no
screen can compute anything even by accident. The scan and the catalog loader are injected
callables, which is what makes the whole interface testable against a recorded machine
without a probe running, and what makes the empty catalog and the failed scan arguments
rather than accidents.

Almost everything drawn is a Rich renderable the command line already builds:
`render_explanation`, `render_plan`, `render_host`, `render_llamacpp`, `render_probes`,
`render_findings`, `render_measurements`, `render_excluded`. Section 12.3 says the TUI shows
the same text the CLI prints with `--explain`; it is the same object, so the two cannot
drift. `RichPane` keeps the renderable on a public attribute so a test can draw it through a
Rich console at a fixed width and assert on text.

## 1. What does a newcomer see in the first five seconds, and what can they do next?

At 80 × 24, on the reference machine, with no arguments typed:

```
 ⭘                                 LlamaFit
 NVIDIA GeForce RTX 4060, 7.5 GiB free of 8.0 GiB · 100.0 GiB free of 128.0 GiB
 Board  Needs  Host  Plan  Simulate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Move with the up and down arrows. Enter says why a row is where it is; p gives
 the command line that runs it.
 Speeds are section 10's formula on its default constants. Nothing has been
 benchmarked on this machine yet, so no figure here is a measurement.
 3 of 3 shown, by score, showing every candidate.
 #    Model                     Quant        Score  Gen/s   Fit
 1    llama-3.1-8b-instruct     Q4_K_M       75.8   33.9    tight
 2    qwen3-0.6b                Q8_0         70.7   114.6   tight
 3    gemma-3-27b-it            Q4_K_M       56.8   1.6     tight
────────────────────────────────────────────────────────────────────────────────
Llama 3.1 8B Instruct Q4_K_M, Meta, licensed llama3.1; it can: tools,
multilingual, long-context

1. Llama 3.1 8B Instruct Q4_K_M
            Score 75.8
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Part    ┃ Score ┃ Weight ┃ Adds ┃
 ↑↓ choose   Enter why   p command line   / search   f fit   s sort   a on disk
 A all quants   x explanation   n not ranked   ? keys   t theme   q quit
```

Three lines come before the table and none of them is a footnote. The first says what to
do. The second says what a speed is. The third says what is on the screen and why —
four of this screen's keys can hide a row, and somebody who cannot find the model they came
for is owed the reason. Below the table the explanation is already open, scrolled to the
score breakdown; the budget, the ladder and where a token's time goes follow.

**What they can do next**, without reading a flag: move the cursor and watch the
explanation follow it; press `p` and land on the Plan screen with the memory budget
component by component, the context ladder with what each rung costs, the runs the catalog
records for comparison, whether the file is on this machine, and last, on a line of its
own, the `llama-server` command line — `y` copies it. `Tab` reaches the Needs form, where
the six use cases, the eight capabilities and the licences the catalog actually carries are
offered as choices rather than as values to be typed. Everything else is one key on the
screen it belongs to.

Two things do **not** happen in the first five seconds and I think that is right. The
dashboard does not scan before it draws: the first paint uses whatever it has, the scan runs
on a worker thread, and the header says `scanning this machine…` until it lands. And nothing
on any of the five screens runs, downloads or changes anything.

## 2. Which numbers does the screen show without a way to check them?

Every speed. `Gen/s`, `PP/s`, and the three shares of a token's time in the explanation are
section 10's formula on its default constants; nothing has been benchmarked on any machine,
`measured` is reserved for a run taken *here*, and phase 3 is what will take one. The screen
says so in a band above the table that does not scroll away, in the command line's own
sentence, and the `How` column carries the label per row whenever the rows disagree.

The one check that exists today is the catalog's own recorded runs, taken on the curator's
machine. `recorded_measurements` hands them over and the explanation shows them beside the
estimate with the flags, the context and the date — but **only for the entries that carry a
`measured` block**. For a model with none, the estimate stands alone with nothing to compare
it against.

Beyond speed, and in decreasing order of how much it bothers me:

- **The modelled budget lines.** The compute buffer, the backend overhead and the process
  overhead are formulas fitted to ten measurements on one graphics card. The `From` column
  says `formula` rather than `file` and the notes under the table say what was modelled, so
  they are *labelled* — but the constants themselves are not on screen, and a reader cannot
  tell a well-fitted formula from a badly fitted one from here.
- **`max_context_fit` and every rung of the context ladder.** Both come out of the same
  budget, so they inherit whatever the modelled lines get wrong. The ladder at least states
  what each rung costs the card, which is the figure a launch script will compare against.
- **The quality baseline.** It is the curator's editorial score. The catalog entry cites the
  benchmarks behind it and `llamafit info` prints them; the dashboard shows the number and
  the arithmetic done to it (`62 from the curator, less 4.0 for this quantisation, plus 5
  because general is the job it was built for`) but not the citation. A reader who wants it
  has to leave the dashboard.
- **Memory bandwidth when it was not measured.** It reaches the screen as a speed note —
  "System memory bandwidth was not measured; 40 GB/s assumed for x86_64" — which is honest,
  and unfalsifiable from here.
- **Free VRAM.** One reading, from one moment, from the vendor tool. The ladder exists
  precisely because that number stops being true when a browser opens, but the board's own
  verdicts are computed against the one reading.

Checkable from the screen: the download size, whether a file is on this machine (the Plan
screen prints the path), the weights, the licence, the capabilities, every arithmetic step
of the composite, and every budget line marked `file`.

## 3. What did you need from the services that they do not return?

Six things. None of them was computed in the interface.

1. **The "what would move it up a tier" sentence.** Section 12.3's own example is "free
   1.2 GB of VRAM to reach 64K context". No service produces it. The ingredients are on the
   screen — `ContextTier.vram_required` for the next rung and `Budget.vram_available` — and
   subtracting them is one line, which is exactly why I did not write it: it would be a
   number the interface invented. The dashboard shows the ladder and stops. This is the
   largest gap between section 12.3 and what shipped, and it belongs in `services/plan.py`
   beside `TargetCheck`, which already answers the same shape of question for speed.

2. **`Needs` is not section 12.1's `Needs`.** The spec lists `preferred`, `licenses`,
   `languages` and `prefer`. The model has none of them: `licenses` and `prefer` are
   arguments to `build_board`, and `preferred` and `languages` do not exist anywhere. So a
   request is split across two objects, `tui.state.Request` exists only to carry both, and
   the Needs form cannot offer preferred capabilities or a language filter at all. A single
   `Needs` carrying all six would remove a class here and let the form grow two fields.

3. **No cached scan.** `docs/tui.md` says the dashboard should show the board within a
   second from the last scan and rescan in the background. Nothing in `services/` stores or
   reads one. The dashboard does the next best thing — draws immediately, scans on a thread,
   redraws when it lands — but the first paint on a cold start has no board on it. A
   `scan_cache` service returning `(Host, taken_at)` would close this.

4. **No one-line host summary.** `render_host` produces a table. The header band needs a
   sentence, so `tui/summary.py` assembles one from `Host` fields, which added six
   translated messages that another interface will want too. This is display, not
   arithmetic, but it is display the web dashboard will duplicate.

5. **`BoardRow` does not carry the vendor, the licence or the capabilities.** The board's
   detail line and `recorded_measurements` both need the `CatalogModel`, so the dashboard
   holds the catalog and looks the model up by id. That works, and it means a board
   serialised to JSON is not enough to draw this screen from — which matters for the web
   dashboard, whose whole premise is that the API returns what the interfaces render.

6. **No way to ask whether a substitution can apply.** `resolve_host` raises `ConfigError`
   for, say, a card resize on a machine with no card. There is no query, so
   `Dashboard.substitute` applies, catches, and puts the previous machine back before
   returning the error to the form. It is correct and it reads like an apology.

Two smaller ones: nothing returns the set of licences present in the catalog (the form
derives it with a set comprehension), and `Board` carries no count of what the request
excluded, so the "not ranked" table is the only way to see it.

## Deviations from section 13.2, deliberately

- **Downloads and Benchmarks are not built.** Phase 2 and phase 3; there is nothing to
  download with and nothing to benchmark with.
- **The Needs and Simulate forms apply with `Ctrl+S` and reset with `Ctrl+R`,** not `Enter`
  and `r`. Both keys sit among text inputs that consume them, and a form where `r` cleared
  the licence field the moment you typed "r" would be worse than an unfamiliar chord. Both
  screens also have buttons.
- **The Board has no `u`, `c` or `l` keys.** In the spec they open the use case, capability
  and licence choosers; all three live on the Needs form, one `Tab` away, and three keys that
  all do "open the form" would have pushed the key bar to a third line at 80 columns.
- **`A` shows all quantisations, not `q`.** `q` quits, in the same table.
- **The `How` column is dropped before the `Fit` column when every row carries the same
  confidence label**, which is the only case where the band above the table is true of every
  figure. When the labels differ, `Gen/s` and `How` are admitted together or not at all: a
  speed never reaches a narrow screen without the word that says what kind of number it is.

## Concerns

- **The board is only as good as the catalog.** On the reference machine a general request
  ranks `llama-3.1-8b-instruct` first at 75.8 and every row comes back `tight`. The
  dashboard is showing that honestly, and it is not a flattering first screen. Not this
  layer's problem to fix, but it is what a newcomer meets.
- **Textual is now a hard runtime dependency**, and it is not a small one: it pulls
  `linkify-it-py` and `mdit-py-plugins`, and it requires `rich >= 14.2` where LlamaFit's own
  floor is `rich >= 13.7`. The resolver installs the intersection, so nothing breaks and both
  declarations stay true of the code that made them — but in practice no LlamaFit install can
  now use a Rich older than 14.2, and the CLI's tables have never been run against one.
- **`tui/format.py` repeats four helpers from `cli/render_board.py`** because the originals
  are that module's private names. Twelve lines of duplication; the right fix is to move them
  somewhere both interfaces can import, which is a change to `cli/` I was asked not to make.
- **The scan worker has no cancel.** `R` starts an exclusive worker, so a second rescan
  replaces the first, but a scan that hangs on a broken driver hangs the worker for as long
  as the probe takes. The screen stays usable; the machine line just stays stale.
- **Nothing tests the dashboard against a right-to-left language.** Every string goes through
  `isolate`/`for_display` the way `cli/render.py` does, and the Rich renderables are the
  command line's own, which `test_cli_rtl.py` already covers — but Textual's own chrome (the
  tab bar, the key bar, the header) is not mirrored and nobody has looked at it in Arabic.
- **`App.copy_to_clipboard` does not work in macOS Terminal.** `y` will silently do nothing
  there, and the notification will still say the command line is on the clipboard. The
  command line is also on the screen, unwrapped, so nothing is lost — but the notification
  is a small lie on one platform.
