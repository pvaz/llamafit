# The flaky dashboard test, and the thousand-line install module

Branch `fix/tuirace`, worktree `llamafit-wt-tuirace`, two commits.

| | |
|---|---|
| `01d5ad9` | `fix: lay the board's table out for the terminal, not for a guessed eighty` |
| `245a84e` | `refactor: split the install group along what it says, to whom, and about what` |

Checks at `245a84e`: 2886 passed, 2 skipped, 14 deselected; coverage 96.62% (floor 85%).
`ruff check`, `ruff format --check`, `mypy` clean. `gen_messages.py`, `gen_schema.py`,
`gen_models_md.py` followed by `git diff --exit-code` clean. All 37 `.po` catalogs report
zero problems. `test_simulating_a_bigger_card_puts_the_badge_up_and_re_ranks` run twenty
times in a row: 20 passed, 0 failed. Before the fix, the same twenty runs gave 18 passed,
2 failed.

---

## 1. Was the race in the test or in the widget, and what should a screen draw before it knows its size?

**In the widget.** The test was reading the screen at a moment the widget was entitled to
be asked about, and it got a wrong answer, not a half-drawn one.

### What was actually happening

The trace I was handed said `BoardScreen._width()` falls back to 80 before the first
resize arrives. That is true, but it is not where the test was failing. Running it to a
failure shows the assertion that breaks is the *second* `column_of(app, "Ctx")`, at line
401 — after the Simulate screen has applied a bigger card and the app has gone back to the
Board. The first `column_of` at line 393 had already succeeded, so the pane had been laid
out and did know its width.

The real mechanism is the tab, not the first paint:

```
app.query_one("#tabs").active = "tab-simulate"   # Board is now hidden; its size is (0, 0)
simulate.action_apply()                          # posts Applied
  -> LlamaFitApp.on_simulate_pane_applied
       self.repaint()      # BoardPane._draw_table() runs while the pane is HIDDEN
       self.go_to("tab-board")
```

A Textual widget's `size` is a report about the last layout pass. A pane sitting behind
another tab has not been laid out, so `self.size.width` is `0` — I confirmed this directly
(`board Size(width=0, height=0)` while `app Size(width=120, height=44)`). So `repaint()`
asked `_width()` for a width, got the constant 80, and laid the table out for 80 columns.
At 80 the column budget admits `rank, model, quant, score, gen, verdict` and stops; `Ctx`
costs nine cells it does not have. Then the tab came forward, Textual laid the pane out,
`on_resize` fired, and the table was rebuilt at 120 with `Ctx` back.

Both tables are real, and which one the test reads depends on whether the resize was
processed inside the `await pilot.pause()` that follows. About one time in ten it was not.

This is not only a test problem. A reader who applies a simulated machine sees the table
lose four columns and get them back, having done nothing to it. The dashboard's own
`board_view` docstring says a narrow terminal loses a column *honestly* rather than
truncating a figure — a column that vanishes and returns is neither.

### What a screen should draw before it knows its size

The question is a good one and the answer turned out to be that the screen was never in
that state. Two things were being conflated:

- **the pane's laid-out width** — genuinely unknown before the first layout pass, and
  genuinely zero for as long as the pane is behind another tab; and
- **the terminal's width** — known before any widget is mounted, updated by the driver
  before any widget hears it has been resized, and never zero in a running application.

`_width()` was asking the first and, on not getting an answer, inventing one. The fix is
to ask the second when the first has nothing to say:

```python
return self.size.width or self.app.size.width
```

`styles.tcss` sets `TabPane { padding: 0 }`, so the Board pane is the full width of the
screen; I verified the two numbers are equal at runtime (both 120 in the harness). The
measured width answers when there is one and the terminal's answers when there is not, and
they agree — which is what makes a redraw from behind another tab produce the same table as
a redraw in front of it. No fallback constant, no deferred draw, no window in which the
table says something different.

I considered the other honest answer — draw nothing until a width arrives, and let
`on_resize` do the first draw — and rejected it. It removes the *wrong* table but replaces
it with an *empty* one for exactly the same window, so the test would still be racing, just
against a different wrong answer. Worse, it would make the pane's contents depend on
Textual's frame timing, which is the thing that was flaky in the first place.

If both widths really are zero — an application that has not started — `columns_for_width(0)`
already has a defined answer: the four columns in `REQUIRED`, which a row cannot be
identified or acted on without, in a table that scrolls. That is the documented behaviour
for a terminal too narrow for anything else, and it is the honest thing to draw when
nothing is known. No code path in the suite reaches it.

The coupling to `TabPane { padding: 0 }` is now written down at both ends: in the
`_width()` docstring and as a comment on the CSS rule itself, so somebody adding padding to
a tab pane is told what else reads it.

### The regression test

`tests/unit/test_tui_degrading.py::test_a_board_behind_another_tab_is_laid_out_for_the_terminal_it_will_come_back_to`
pins the mechanism rather than the symptom: it asserts the pane really has no width of its
own while it is hidden (`board.size.width == 0`), then redraws it from there and asserts the
column set is unchanged. It fails deterministically against the old code and needs no
timing to do it.

---

## 2. Where else did you find a width assumed before it was known?

Three more, all in the command line's renderers, and all latent rather than live:

- `src/llamafit/cli/render_board.py:784` — `render_board(board, *, console_width: int = 80)`
- `src/llamafit/cli/render_board.py:992` — `render_fit(board, *, console_width: int = 80)`
- `src/llamafit/cli/render.py:843` — `render_catalog_list(summaries, *, console_width: int = 80)`

Same shape as the one I fixed: a constant standing in for a width the caller was supposed
to supply. The difference is that here it is a *default parameter*, so it only fires when a
caller forgets. Every production caller passes a real one —
`board_cmd.py:204`, `board_cmd.py:278`, `catalog_cmd.py:107`, all `console_width=console.width`
— so today the constant is exercised only by tests. It is still the shape that bites: a
fourth caller added tomorrow gets an 80-column table on a 200-column terminal, silently and
with no failing test, because 80 is a width that *works*.

I did not change them. They are in `cli/render.py` and `cli/render_board.py`, outside the
files this task was scoped to, and both are shared with the terminal dashboard and with the
scoring agent's work. My recommendation is to make `console_width` a required keyword
argument in all three: there is no caller that legitimately does not know it, and a
`TypeError` at the call site is the failure a silent 80 should have been all along.

One related, non-identical finding, also left alone:
`src/llamafit/download/progress.py:153` (`RichProgress._columns`) chooses its progress-bar
columns from `console.width` **once, in `__init__`**, and never looks again. That is not a
width assumed before it was known — it is a width read correctly and then frozen — but the
consequence is the same family of bug: resize the terminal during a two-hour model download
and the bar keeps the column set it picked at the start. It is in `download/`, outside this
task's scope, and it is cosmetic.

Nothing else in `tui/` reads a size to make a layout decision. Every other pane holds a Rich
renderable inside a `RichPane`, and Textual re-renders those against the widget's real width
on every resize, so there is no width for them to assume.

---

## 3. What seams did you split the install module along, and why those?

`cli/install_cmd.py` was 1022 lines. It is now four modules:

| Module | Lines | Holds |
|---|---:|---|
| `cli/install_cmd.py` | 341 | the `install` group, `install model`, `install history` |
| `cli/llamacpp_cmd.py` | 361 | `install llama.cpp` |
| `cli/render_install.py` | 325 | every table and sentence a person reads |
| `cli/install_json.py` | 149 | the four `--json` documents |

### The seam the project already uses: what a command decides vs. what a reader sees

I read `cli/catalog_cmd.py` and `cli/render.py` first, as instructed, and the seam is
plain: a `*_cmd.py` parses flags, calls services and decides; a `render*.py` turns what came
back into Rich renderables and returns them without printing. `catalog_cmd.py` imports
`render_catalog_list`, `render_model_facts` and `render_quants` and prints them itself.

`install_cmd.py` had that seam and had not used it. Its two plan tables, its ownership
sentence, its install report, its download-plan block, its stopped and finished lines and
its history table were all built inline, and about half of them printed directly to a
console they had been handed. `cli/render_install.py` is those, in the project's existing
shape: every function returns a renderable and prints nothing, so the command keeps the
decision about whether there is a console at all — which is what stops a `--json` run from
drawing a table nobody is watching.

Two of them changed shape slightly in the move. `_report` and `_print_plan` each made four
to six `console.print` calls; they are now `render_release_result` and `render_download_plan`
returning a `rich.console.Group`, which the command prints in one call. Rich renders a Group
child by child exactly as sequential prints do, and the install tests assert on the output
substrings that would have caught a difference.

### The second seam: a person and a script are different readers

`cli/render.py`'s own docstring makes a point of it — "none of it is anywhere near `--json`:
the marks are added here, at the last moment before Rich draws, and the services that build
the machine-readable output never see one." The install group has four hand-built `--json`
documents, and they want the opposite of everything the renderers do: byte counts instead of
`4.1GiB`, bare identifiers instead of isolated ones, no sentences at all. `cli/install_json.py`
is those four. Keeping them beside the tables that describe the same facts would be one edit
away from a localised figure reaching a parser, and the two shapes need to be able to move
independently — a column can be added to a table without breaking a script.

### The third seam: a program is not data

That leaves the commands themselves, and this is where I could have split by command count
and deliberately did not. `install history` is not its own module: it is twenty lines, it
reads the record that `install model` writes, and separating them would put a reader and a
writer of the same file in different places.

`install llama.cpp` is its own module because it is a different job, not because it is a
different command. It fetches a **program**: it probes the machine's card to pick a backend,
resolves which of a release's published archives suits this OS and driver, verifies a
checksum, unpacks over a directory that may belong to somebody else, and offers to change a
PATH that outlives the process. `install model` fetches **data**: a set of files named by the
catalog, resumable, verified against checksums the catalog already holds, over two hours,
interruptible. They share the `install` group and the rule that nothing large starts before
its cost is printed. They share no code at all — no helper, no option, no import that both
need. Two responsibilities that touch in one Typer group and nowhere else.

`install_cmd.py` owns the group and `llamacpp_cmd.py` imports it, so the dependency runs one
way and `cli/app.py` imports both, the way it imports every other command module.

### What it cost the tests

One line. `tests/unit/test_cli_install.py` did `from llamafit.cli import install_cmd` and
then `monkeypatch.setattr(install_cmd, "HttpReleaseClient", ...)` and similar for
`HttpFetcher`, `current_os`, `current_arch`, `detect_gpus` and `managed_dir` — all six of
which are names the llama.cpp command looks up in its own module's namespace. The module
moved, so those references moved with it: the import line changed and the eleven
`install_cmd` mentions became `llamacpp_cmd`. Not one assertion, argument, expectation or
test name changed; `git diff` on that file is eleven identical substitutions of a module
name.

`tests/unit/test_cli_install_model.py` is untouched. That was a constraint on the split, and
a useful one: it patches `"llamafit.cli.install_cmd.HttpRangeReader"`,
`"...DownloadOptions"` and `"...install_model"` by string, and imports `_StoppedByInterrupt`
from `install_cmd` by name. Anything that moved `_run` or the interrupt guard out of
`install_cmd.py` would have broken all four, which is why the download machinery stayed with
the command that uses it rather than becoming a fifth module. That constraint and the
responsibility argument pointed the same way.

Coverage of the four modules after the split: `install_cmd.py` 94%, `llamacpp_cmd.py` 99%,
`render_install.py` 96%, `install_json.py` 100%.

---

## Concerns

- **`messages.pot` is 260 lines changed and none of them is a message.** Moving the install
  group's strings between files rewrote every `#:` reference and reordered the entries.
  I verified the message set is identical — 1020 before, 1020 after, nothing added, nothing
  removed — by parsing both with the project's own `llamafit.i18n.po.read_po` and diffing
  the key sets. The 37 `.po` catalogs carry no reference lines, so none of them needed
  touching, and all 37 report zero problems. **This file will conflict with the translations
  agent's branch**, and the conflict will be pure noise: whoever merges second should
  re-run `scripts/gen_messages.py` and take its output rather than resolving hunks by hand.

- **I did not touch `CHANGELOG.md`.** The dashboard fix is user-visible — a table that
  stopped flickering its columns — and probably earns a line under *Unreleased*, but the
  changelog is the likeliest conflict with two other agents in flight and it was outside the
  scope I was given.

- **The three `console_width: int = 80` defaults are still there** (finding 2 above). They
  are inert today. If nobody picks them up, the next person to add a caller to
  `render_board`, `render_fit` or `render_catalog_list` will get an 80-column table on a
  wide terminal and no test will notice.

- **`cli/render.py` (1304 lines) and `cli/render_board.py` (1196 lines)** are both well past
  the rule that sent me to `install_cmd.py`, and `cli/bench_cmd.py` (568) and
  `cli/preset_cmd.py` (546) are over it too. Out of scope here, but the rule is not being
  applied evenly.

- **`_width()` now leans on `TabPane { padding: 0 }`.** The Board pane gets the whole width
  of the screen, and the column budget is arithmetic on that number, so a pane that stopped
  being full-width would lay its table out a few cells too wide and clip the last column.
  I wrote the dependency down at both ends — in the `_width()` docstring and as a comment on
  the CSS rule — but it is a coupling that did not exist before and no test would catch its
  violation. A stricter alternative would be to compare `self.size.width` against
  `self.app.size.width` in a test whenever both are non-zero; I judged the comment sufficient
  for a rule that is one line of CSS in a file with four rules in it.
