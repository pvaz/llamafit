# Merging `feat/rtl` into `main`: what each conflict was, and what was kept

Worktree `llamafit-wt-merge`, branch `chore/merge-rtl`, cut from `main` at `870e1d9`.
Nothing was pushed and no other branch or worktree was touched.

| | |
|---|---|
| Merge commit | `cc8cda77d86077982c88136ce33d0001e48c8d6e` |
| First parent | `870e1d9` fix: rebuild the constants file as the union of three branches |
| Second parent | `07f444f` docs: what right-to-left rendering fixes, and what it cannot |
| Merge base | `8fdac06` feat: the types phase 1C's modules hand to one another |

Conflicts: fifteen in `src/llamafit/cli/render.py`, one in `docs/translations.md`, one in
the generated `src/llamafit/data/locale/messages.pot`.

## Verification

| Check | Result |
|---|---|
| `pytest --cov -m "not hardware and not network"` | 1489 passed, 5 failed, 1 skipped, 11 deselected; coverage **95.87%** (floor 85%) |
| `ruff check .` | All checks passed |
| `ruff format src tests scripts` / `--check .` | 177 files already formatted, nothing rewritten |
| `mypy` | Success: no issues found in 85 source files |
| `gen_messages.py`, `gen_schema.py`, `gen_models_md.py` then `git diff --exit-code` | clean |
| `check_po.py` over all 37 catalogs | **0 problems** in every one |
| English output vs `870e1d9` | byte for byte identical |
| Portuguese output vs `870e1d9` | byte for byte identical |

**The five failures are pre-existing and are not this merge's.** They are
`test_budget.py::test_a_real_models_weight_lines_add_up_to_its_file_in_every_mode`
(`qwen3.8-flash-next`, `qwen3-coder-next`), `::test_the_reference_machine_sizes_flash_next_at_32k`,
`::test_the_shared_expert_override_moves_a_bucket_between_pools`, and
`test_budget_parts.py::test_shared_experts_stay_in_the_dense_bucket_until_the_facts_split_them_out`.
All five are about the shared-expert bucket in `llamafit.budget`. This merge touches no
file under `src/llamafit/budget/` and no budget test — `git diff HEAD --stat` against the
pre-merge tree lists only the nine files below. To be certain rather than merely confident,
`git archive HEAD` was extracted to a scratch directory and the same two test files run
against it with the same interpreter: **the identical five failures, same assertions, same
numbers.** They fail on `870e1d9` with or without this merge.

Files changed by the merge:

```
 CHANGELOG.md                          |  12 +
 docs/cli.md                           |   8 +
 docs/translations.md                  | 102 +++++++
 src/llamafit/cli/render.py            | 482 +++++++++++++++++++++++++---------
 src/llamafit/data/locale/messages.pot | 242 ++++++++---------
 src/llamafit/i18n/__init__.py         |  34 +++
 src/llamafit/i18n/bidi.py             | 293 +++++++++++++++++++++
 tests/unit/test_cli_rtl.py            | 303 +++++++++++++++++++++
 tests/unit/test_i18n_bidi.py          | 255 ++++++++++++++++++
```

### The template's message ids did not move

The template regenerated from the merged source has **exactly the same set of
`(msgctxt, msgid)` pairs as `870e1d9`** — nothing added, nothing removed. Only the
`#:` source-line references moved, because `render.py` grew. That is the check that
matters: it says the merge kept `main`'s message shapes and did not let any of
`feat/rtl`'s older wordings back in. Against `feat/rtl` the template differs by 116 added
and 25 removed ids, which is the English round landing, and is the intended direction.

All 37 catalogs report 0 problems and their translated counts are unchanged from `main`
(247 for thirty of them, 239 for the seven that predate the round, 246 for `pt_PT` which
deliberately leaves one entry empty to prove the fallback).

---

## 1. Every conflict, and which side was kept

`H` is `main` (HEAD), `R` is `feat/rtl`.

### `src/llamafit/cli/render.py`

| # | Where | Kept | Why, in one line |
|---|---|---|---|
| 1 | module docstring | **both** | `H`'s three habits verbatim, then `R`'s isolate paragraph renumbered to a fourth, plus a new paragraph saying how the two interact. |
| 2 | `from typing import Any` | `R` | `_add_columns` needs it for its `Mapping[str, Any]` column specs; `H` had no reason to import it. |
| 3 | `render_host`, CPU row opener | **both** | `H`'s `_cores(host.cpu)` helper (two counted entries, one per number) called from inside `R`'s `_add_row`/`_cell`; `R`'s inline `%(physical)d cores / %(logical)d threads` is the exact string `H` deleted for being one entry with two counts. |
| 4 | `render_host`, CPU values | **both** | `R`'s `isolate(host.cpu.model)` and `isolate(" ".join(host.cpu.isa))` kept, with `H`'s `pgettext("CPU instruction sets", "isa unknown")` replacing `R`'s bare `_("isa unknown")`. |
| 5 | `render_host`, memory details | `H` | `R`'s nested `", ".join` generator is precisely the joined-in-Python shape `H` replaced with `_memory_kind`/`_memory_details`; the isolate `R` had on `mem.type` was moved into `_memory_kind`, which is where the type is now interpolated. |
| 6 | `render_host`, memory row values | **both** | `R`'s `_size()` for the two byte figures, `H`'s `_memory_details(mem)` (which carries its own *type unknown* with a context) for the middle. |
| 7 | `render_host`, VRAM and GPU specs | `H` | same story: `H`'s `_gpu_specs` and `pgettext("GPU VRAM", "VRAM unknown")` over `R`'s joined list and bare `_("VRAM unknown")`; `R`'s `isolate(gpu.driver)` moved into `_gpu_specs`, in both shapes that name a driver. |
| 8 | `render_llamacpp`, unknown build | **both** | `H`'s `pgettext("llama.cpp build", "unknown build")` inside `R`'s `_add_row`. |
| 9 | `render_llamacpp`, running server | **both** | `R`'s `isolate(server.url)` and `isolate(server.model)` with `H`'s `pgettext("model name", "unknown model")` as the fallback. |
| 10 | `render_probes`, no-server status | **both** | `H`'s `pgettext("probe result", "no server")` wrapped in `R`'s `for_display`, matching the two neighbouring statuses that already had both. |
| 11 | `render_catalog_list`, columns | **both** | `R`'s list-of-specs handed to `_add_columns` (so the order and alignment can be mirrored), with `H`'s `_quality_heading(headings)` supplying the quality header instead of `headings["quality"]`. |
| 12 | `render_model_facts`, Parameters | **both, reshaped** | `H`'s `%(total)s total, %(active)s active` with the suffix inside the value, and `R`'s isolate now around the *whole* `27B` — see §2. |
| 13 | `render_model_facts`, Context | **both, reshaped** | `H`'s single whole sentence `%(tokens)s tokens native, %(extended)s extended via %(method)s` with `R`'s isolates on all three values — see §2. |
| 14 | `render_model_facts`, Notes / Capabilities / Use cases | **both** | `H`'s `pgettext("table row label", ...)` labels and `_capability_label` translation, inside `R`'s `_add_row`/`_cell`, with the isolate kept on the joined list. |
| 15 | `_facts_summary` | `H` | `R`'s `", ".join(parts)` is the joined shape `H` replaced with four written-out messages plus `_experts`; `R`'s `isolate(facts.arch)` moved into all four of them. |

### `docs/translations.md`

| # | Where | Kept | Why |
|---|---|---|---|
| 16 | the opening blockquote | **both** | `H` rewrote the identifier list because *capability* stopped being one; `R` appended a pointer to its new right-to-left section. They are consecutive sentences, not alternatives, so both are in, in that order. |

### `src/llamafit/data/locale/messages.pot`

| # | Where | Kept | Why |
|---|---|---|---|
| 17 | 21 conflicting hunks | **neither** | It is generated. Resolved by running `python scripts/gen_messages.py` once `render.py` was settled, never by choosing lines. |

---

## 2. Which conflicts were genuinely both — a resolution neither side wrote

Four, and they are the point of the exercise.

**The parameter count (conflict 12).** `feat/rtl` wrote, in code and in
`docs/translations.md`, that this one could not be fixed: the message was
`%(total)sB total, %(active)sB active`, the `B` was welded on outside the placeholder, and
an island round the digits alone would have parted `27` from its `B` and let the two swap
places in an Arabic line — worse than doing nothing. The English round removed exactly
that weld: the suffix became `billions_suffix()`, a catalog entry of its own, and what
reaches `%(total)s` is now the finished `27B`. So the merged line is
`isolate(f"{_fmt_billions(...)}{billions_suffix()}")` — an isolate `feat/rtl` could not
write, made possible by a change `main` made for an unrelated reason. Arabic now renders
`⟨27B⟩ total, ⟨3B⟩ active` with the island round both characters. The residue bullet in
`docs/translations.md` that named `%(total)sB total` has been rewritten to say it is done
and to keep the three figures that genuinely remain (`%(gbps)s GB/s`,
`%(compute)s TFLOPS fp16`, `%(speed)d MT/s`, and `b%(build)d` in front).

**The extended context (conflict 13).** `feat/rtl` isolated `format_grouped(...)` inside
two messages, the second of which it flagged in a comment as a fragment opening with a
comma that a translator cannot work with, and deliberately left alone so as not to move a
message id. `main` fixed precisely that, into one whole sentence with three placeholders.
The merged line puts the isolates on all three values of the *new* sentence — including
`method`, which keeps `pgettext("context extension method", "unspecified method")`
un-isolated because that one is prose in the reader's own language.

**Three helpers that did not exist on `feat/rtl` (conflicts 5, 7, 15).** `main` extracted
`_memory_kind`, `_gpu_specs` and `_facts_summary`/`_experts` out of joined lists. The
values `feat/rtl` isolated — `mem.type`, `gpu.driver`, `facts.arch` — moved with them, so
the isolate had to move too, and into *every* branch of each helper rather than the one
place the old code interpolated it: `_gpu_specs` names the driver in two shapes and
`_facts_summary` names the architecture in four. Taking either side wholesale would have
lost one thing or the other silently.

**The capabilities cell (conflict 14, and the list table's).** `feat/rtl` isolated
`", ".join(model.capabilities)` because a capability was an enum value — an identifier.
`main` made them ordinary words with a `pgettext("model capability", ...)` each. The
merged code translates them *and* keeps the isolate, and the reason is the reversal, not
an oversight: a cell that is Latin until a catalog answers those eight entries and the
reader's own script once it has is a run whose direction is not known in advance, which is
exactly what U+2068 FIRST STRONG ISOLATE is defined for, and the mark keeps the commas
between the names out of the surrounding Arabic either way. That reasoning is written into
`_capability_label`'s docstring, because the old comment ("never words") is now false.

Two smaller things were also neither side's:

- `src/llamafit/i18n/bidi.py` is new, and `feat/rtl` was cut before the AGPL relicence, so
  it arrived with no notice. It gained the three-line header every module carries —
  `test_licensing.py::test_every_source_module_carries_the_notice` catches this, and did.
- `_from_table` and `_experts` gained a sentence each saying *why* their figures stay bare,
  so the next reader does not "fix" them into the failure mode `feat/rtl` warned about.

---

## 3. Did a test fail, and what was it telling me?

Yes — one, and it was telling me something real about a message, not about the merge
machinery.

`tests/unit/test_cli_rtl.py::test_the_first_column_is_added_last_so_it_lands_against_the_right_edge`
failed at index 3 of five: drawn `الجودة*` against expected `الجودة`. Everything else —
the reversal itself, which is what the test's name and body are about — was exactly right.

The test built its expectation as `[headings[name] for name in (...)]` from
`_list_headings()`. On `feat/rtl` that was sound, because the quality heading's message was
literally `_("Quality*")` with the footnote marker inside it. `main` pulled the marker out
into `_FOOTNOTE_MARKER` and `_quality_heading()`, on the stated grounds that an asterisk is
punctuation and not a word, and a translator who met it inside a message would have to
decide whether it is theirs to keep. So `_list_headings()["quality"]` no longer is what the
table is given.

There were two ways to make it green. Putting the marker back inside the message would have
undone a deliberate `main` change and moved a message id, which is the one thing every
comment in this file says not to do. The other is to ask the renderer for what it actually
prints. **I edited the test**, replacing the list comprehension with an explicit list whose
quality entry is `_quality_heading(headings)`, and left a comment saying why. The assertion
the test exists to make — that the five columns are drawn in reverse order — is untouched
and still the thing being asserted; what changed is only where the expected quality header
is read from. This is the one test edit in the merge and it is called out here because it
is one.

Nothing else failed. The other 17 tests in `test_cli_rtl.py` and all 33 in
`test_i18n_bidi.py` passed against the merged renderer without modification, including the
two that matter most for the reshaped messages:
`test_english_output_carries_no_direction_marks_at_all`,
`test_portuguese_output_carries_no_direction_marks_either`, the four parametrised
`test_json_output_never_holds_a_direction_mark` cases, and
`test_a_model_id_and_its_capabilities_stay_whole_in_the_list`, which asserts
`⟨coding, tools⟩` and is what pinned down the capability decision in §2.

## What a reader should still know

- `main`'s output in English and Portuguese is byte for byte what it was, checked by
  rendering all seven tables at width 200 against a pristine `870e1d9` checkout and
  diffing, not only by the no-marks assertions.
- The five budget failures were on `870e1d9` before this merge and are unrelated to it.
  They are somebody's to fix, and this merge neither caused nor hid them.
- What is still unmarked in a right-to-left language, and why, is listed under
  *What is still wrong, and why it was left* in `docs/translations.md`. That list is one
  bullet shorter than `feat/rtl` left it.
