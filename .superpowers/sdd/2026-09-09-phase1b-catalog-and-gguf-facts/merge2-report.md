# Merging five branches into one tree

Worktree `llamafit-wt-merge2`, branch `chore/merge-phase23`, cut from `main` at `405c1f8`.
Five merges, in the order asked for, each committed only once the whole gate was green:

| SHA | Branch merged | Subject |
|---|---|---|
| `4cd9f553ceb511955d7457042e9bf365bfe68ce2` | `feat/downloads` | `merge: bring the model downloader into the install group` |
| `54dc8f232fbb598b98d66ccda955c6279c88ac27` | `feat/presets` | `merge: add the launch presets beside the install group` |
| `633b74ec652014501404aa8860e22131a8e25476` | `feat/tui` | `merge: open the terminal dashboard when there is nothing else to do` |
| `316b641a48940e6295fc955d22639fd7fd237844` | `feat/bench` | `merge: add \`llamafit bench\` beside the rest of the commands` |
| `411d536d9f78f56f60852e8ef1d47df0e28a9f7b` | `fix/speedscore` | `merge: take the speed score's new shape and the refitted prompt compute` |

Nothing was pushed. No other branch or worktree was touched.

## Verification

Baseline on `main` before starting: 2072 passed, 2 skipped, 12 deselected.

Final, on `411d536`:

| Command | Result |
|---|---|
| `pytest --cov -m "not hardware and not network"` | 2885 passed, 2 skipped, 14 deselected; total coverage 96.61 % (floor 85 %) |
| `ruff check .` | All checks passed |
| `ruff format src tests scripts` | 301 files left unchanged |
| `ruff format --check .` | 328 files already formatted |
| `mypy` | no issues in 161 source files |
| `gen_messages.py && gen_schema.py && gen_models_md.py && git diff --exit-code` | clean |
| `check_po.py` over all 37 catalogs | 0 problems each |

`llamafit --help` lists, in the order it prints them: `plan`, `bench`, `recommend`, `fit`,
`list`, `search`, `info`, `doctor`, `preset`, `launch`, `serve`, `system`, `catalog`,
`hardware`, `install`. `llamafit install --help` lists `llama.cpp`, `model` and `history`.
`llamafit` with no arguments takes the dashboard path and, into a pipe, says so and prints
what `llamafit recommend` would have printed.

The suite was run after every merge, not only at the end, so a failure could be attributed
to the merge that caused it.

## 1. How the install module was composed, and what each side contributed

`src/llamafit/cli/install_cmd.py` existed on both sides and neither knew about the other.
It is not two implementations of one command; it is two commands that belong to one group,
so the file is composed rather than chosen. I read both in full first and never touched the
conflicted file itself — the merged module was assembled from the two clean sides extracted
with `git show`, so no marker or half-line could survive into it.

**One docstring.** Both sides had written the same governing idea in their own words. The
merged docstring says it once, at the top, as the rule that holds the group together —
nobody should start something large and irreversible without being told what it costs
first, so each command prints its whole plan before a byte is written and only then asks —
and then gives each command a paragraph saying what *its* plan contains. The three
paragraphs that were peculiar to one side are kept where they belong: `--json` without
`--yes` being a dry run belongs to `llama.cpp`; `Ctrl+C` keeping the part files and their
records belongs to `model`. The deferred-`help=` note, which both sides had, is said once
at the end.

**What `main` contributed.** `VALID_BACKENDS`; the `--backend`, `--dir` and `--tag` option
singletons; `_check_backend`, `_primary_gpu`, `_release_for`, `_build_plan`,
`_result_payload`, `_ownership_line`, `_ownership_style`, `_progress_bar`, `_report`,
`_offer_path`; and `llamacpp_command` with its five inline options. Also the B008 rationale
comment, which both sides had written in slightly different words and which now appears
once.

**What `feat/downloads` contributed.** The `MODEL` argument and the `--quant`, `--dir`,
`--limit-rate`, `--workers`, `--yes`, `--dry-run`, `--no-extras`, `--recheck`,
`--allow-unverified` and `--limit` singletons; `_status_text`, `_print_plan`, `_reporter`,
`_outcome_payload`, `_run`, `_say_stopped`, `_say_done`, the `_StoppedByInterrupt` context
manager; and `install_model_command` and `install_history_command`.

**Shared.** One `install_app` and one `app.add_typer` call. Its `help=` was written twice,
each side describing only the half it had built — `main` said "llama.cpp, and later
models", `feat/downloads` said "the weights, for now". Both are now false, so the group's
help is new text covering both: *Install what it takes to run a model: llama.cpp itself,
and the weights.* Neither of the old strings appears in any `.po` catalog, so no
translation was orphaned by replacing them.

**Three names collided** — the same identifier meaning different things on the two sides.
Each is now named after what it describes rather than after which file it came from:

| Both sides called it | `llama.cpp` half | `model` half |
|---|---|---|
| `_DIR_OPTION` | `_LLAMACPP_DIR_OPTION` | `_MODEL_DIR_OPTION` |
| `_plan_table` | `_release_plan_table` | `_download_plan_table` |
| `_plan_payload` | `_release_plan_payload` | `_download_plan_payload` |

No test names any of these, so the renames are internal. Everything the tests *do* patch by
name — `HttpReleaseClient`, `HttpFetcher`, `current_os`, `current_arch`, `detect_gpus`,
`managed_dir` on one side, `HttpRangeReader`, `DownloadOptions`, `install_model` and
`_StoppedByInterrupt` on the other — is still a module-level name in the merged file.
`__all__` now lists all three command functions and `install_app`.

The two halves are separated by a section banner each, and the file is 1,023 lines. The
project's own rule says to split a file that grows past a few hundred lines; this one is
over that, and the natural split is `install_cmd.py` keeping the group and two modules
beside it. I did not do it in a merge commit, because a merge that also moves code is a
merge nobody can review. It is listed under concerns.

**The test file was written twice as well** and this was not mentioned in the brief.
`tests/unit/test_cli_install.py` is an add/add conflict: `main`'s covers
`install llama.cpp`, `feat/downloads`'s covers `install model` and `install history`. They
cannot share a module. Each installs an `autouse` fixture that points `LLAMAFIT_HOME` at a
different directory — `tmp_path/"llamafit-home"` and `tmp_path/"home"` — and in one module
the later-defined fixture would win for every test, so the llama.cpp suite would look for a
cache under a directory the model suite had moved. Merging them would have produced a file
that passes only by accident of definition order.

So they stay two modules. The llama.cpp suite keeps `tests/unit/test_cli_install.py`; the
model and history suite becomes `tests/unit/test_cli_install_model.py`, which is also what
its own docstring already says it is. **Not one line of either suite's content changed** —
both files are byte-identical to their branch versions. All 26 llama.cpp tests and all 19
model and history tests pass together.

## 2. Which conflicts were genuinely both sides, and which were one side stale

**Genuinely both sides — 8 conflicts.**

- `src/llamafit/cli/install_cmd.py` (add/add) and `tests/unit/test_cli_install.py`
  (add/add), covered above.
- `CHANGELOG.md`, three times: `feat/downloads`, `feat/presets` and `feat/tui` each added
  entries to the end of the same *Unreleased → Added* list. No entry contradicts another;
  all are kept, in the order the branches were merged, which is also the order the work
  landed. `feat/bench` and `fix/speedscore` added no CHANGELOG entries at all, so there was
  nothing to merge and nothing was invented for them.
- `src/llamafit/cli/app.py`, twice, both from `feat/presets`: one line in the import block
  that registers a command group, one in `__all__`. `preset_cmd` goes beside `serve_cmd` in
  the alphabetical order those two lists already keep. `feat/downloads`, `feat/tui` and
  `feat/bench` touched the same two lists without conflicting, and git got those right.
- `docs/cli.md`, once, from `feat/downloads` — but only half of it was both-sided; see
  below.

**One side stale — 3 places, all of them a branch describing work another branch had since
finished.** These are the ones where keeping both sides would have shipped a lie.

- `docs/cli.md`, the `llamafit install` section. Both sides wrote a real section and both
  ended it with a sentence about the *other* command: `main` said `install model` "is still
  to come", `feat/downloads` said `install llama.cpp` "has not shipped". Both sentences are
  now false. The result is one `### llamafit install — phase 2` heading, an intro saying
  both have shipped and both keep the same rule, and a `####` subsection per command
  carrying that side's full text. The two stale sentences are gone.
- `docs/development.md`, the runtime-dependency table. `feat/tui` added the `textual` row
  that `main` had already written a different version of, so both `fastapi`/`uvicorn` and
  `textual` are now rows there, each with the reason its own branch gave.
- `docs/development.md`, the *planned, not installed* table below it. That table loses a
  row as a package ships. `main` had removed `fastapi`/`uvicorn` and left `textual`;
  `feat/tui` had removed `textual` and left `fastapi`/`uvicorn`. Correctly merged it is
  **empty**, which is a header with no rows. It is now a sentence saying there is nothing
  outstanding and that `fastapi`, `uvicorn` and `textual` were the last three to move up.
  This is the one place where I wrote prose neither branch wrote; the alternative was an
  empty table.

**`src/llamafit/data/locale/messages.pot` conflicted on four of the five merges and was
never resolved by choosing lines.** Each time the sources were settled first and then
`python scripts/gen_messages.py` rewrote it: 792 messages after `feat/downloads`, 833 after
`feat/presets`, 932 after `feat/tui`, 1,019 after `feat/bench`, 1,020 after
`fix/speedscore`. All 37 `.po` catalogs report zero problems at the end — in particular no
catalog holds a message the template lost, which is what `check_po.py` fails on.

**`fix/speedscore` merged with no conflict at all**, which is worth saying rather than
passing over. It rewrites constants, the speed estimate and the score that reads it; the
four branches before it added commands and screens *beside* that code rather than inside
it. The one place they could have collided is `llamafit bench`, which fits `eff_pp` and
`pcie_effective_gbps` — two of the constants `fix/speedscore` refits. It fits them by name,
not by value, so the refit lands underneath it and all 8 bench test modules still pass.

## 3. Did any test fail, and what was it telling me

**One did, once, and it was not telling me a resolution was wrong.**
`tests/unit/test_tui_app.py::test_simulating_a_bigger_card_puts_the_badge_up_and_re_ranks`
failed with a `ValueError` on the first full-suite run after the `feat/bench` merge. It has
not failed since: it passed on its own, passed with the bench modules loaded before it,
passed eight consecutive whole-file runs, and passed both full `--cov` runs afterwards
(2,885 tests each).

It is a race in `feat/tui`'s own code, and I found the mechanism rather than waiting for it
to go away. `BoardScreen._width()` returns `self.size.width or _DEFAULT_WIDTH`, and
`_DEFAULT_WIDTH` is 80 — the width used before the first resize event arrives. The test
mounts at 120 columns and then reads a column by name:

```
80  -> ('rank', 'model', 'quant', 'score', 'gen', 'confidence')
120 -> ('rank', 'model', 'quant', 'score', 'gen', 'confidence', 'verdict', 'mode',
        'context', 'have', 'quality')
```

`Ctx` is the `context` column, and at 80 columns it is not drawn. The helper does
`headings(app).index("Ctx")`, which is a `ValueError: 'Ctx' is not in list` on any run where
the board is read between its first draw and its first post-resize redraw. Under the load of
a 2,800-test run that window is occasionally open.

I did not touch it, and I want to be explicit about why. `git diff feat/tui -- src/llamafit/tui
tests/unit/test_tui_app.py tests/fixtures/dashboard.py` is empty: every file involved is
byte-identical to the branch that wrote it, so no resolution of mine can be implicated. The
fix belongs in `feat/tui`'s own code — either the screen redrawing on resize, or the test
awaiting a `pause()` before it reads the table — and doing it inside a merge commit would
hide a real bug in a diff nobody expects to contain one. It is a concern, not a resolution.

**No test was edited to make a merge pass.** The only change to any test file in this whole
merge is that one of the two `test_cli_install.py` files was written to a different path;
neither file's contents were altered.

## Concerns

1. **The TUI board race above.** Intermittent, pre-existing on `feat/tui`, and it will
   surface again in CI. `BoardScreen` draws at a default width of 80 before its first
   resize, and any test that reads a column by name in that window fails. Worth a fix on
   the screen rather than the test, since a real terminal opens the same way.
2. **`src/llamafit/cli/install_cmd.py` is 1,023 lines**, over the project's own "split it
   past a few hundred" rule. It holds one command group and reads as one; the split I would
   make is `install_cmd.py` keeping the group and its two halves moving out beside it. Left
   for a commit of its own, because code motion inside a merge commit is unreviewable.
3. **`feat/bench` shipped no CHANGELOG entry.** `llamafit bench` is a new top-level command
   and there is nothing under *Unreleased* about it. I did not write one, because inventing
   a changelog entry for somebody else's feature is guessing at what they thought was worth
   saying. It needs one before release.
4. **`docs/development.md`'s planned-dependency table is now a sentence.** If the next
   dependency arrives before anyone notices, the table it was supposed to be listed in no
   longer exists. The sentence says what to do, but a reviewer who liked the table should
   know it is gone.
5. **Translations fell further behind, as expected.** The template went from 466 untranslated
   messages per catalog to 775. Every catalog is still structurally clean — no orphaned
   message, no invented placeholder — and the suite treats a missing translation as a
   warning, so nothing fails. But five branches of new user-facing text landed in English.
6. **`docs/benchmarking.md` says "the Benchmarks screen is phase 1D".** The dashboard from
   `feat/tui` has five screens and none of them is Benchmarks, so the sentence is still true
   — but the two branches now sit in one tree and somebody should decide whether the
   dashboard is supposed to grow one.
