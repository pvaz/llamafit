# The web dashboard, cleaned

Branch `feat/clean-web`, off `main`. Nothing merged, rebased or pushed; no other branch or
worktree touched.

The brief arrived in three parts: four pieces of design feedback, then a decision on the
name of the speed column, then a change of direction that replaced the five-tab layout
with a single page built for a catalog about to grow from five models to fifty. This
report answers the three questions asked, and then says what was actually built.

---

## 1. Which columns did I drop, and where did each one go?

The board had fourteen columns of one weight. It now has seven, and an eighth that appears
only when it has something to say.

| Kept | Why it earns a column across fifty rows |
|---|---|
| `#` | The rank the composite score gave it. Two characters. |
| `Model` | Which model. The only thing that cannot be recovered from anywhere else. |
| `Quant` | Which file. A model without its quantisation is not a thing you can run. |
| `Score` | Why the rows are in that order, and what a sort by speed is departing from. |
| `Tok/s` | The number the tool exists to produce. |
| `Fit` | Whether it runs at all. |
| `Ctx` | How much context it holds — a constraint people actually decide on. |
| `How` | **Conditional.** Drawn only when the rows disagree about how their speeds were arrived at. |

Seven is not a number I picked. It is the first seven of the priority the terminal's own
board had already settled twice — `REQUIRED` plus the head of `_board_columns`'s optional
list in `src/llamafit/cli/render_board.py`, and of `MIXED_ORDER` in
`src/llamafit/tui/board_view.py`. The terminal had done this exercise under real pressure
(eighty columns) and I had no better information than its conclusion.

`How` behaves the way the TUI's `UNIFORM_ORDER` explains: when every row carries the same
confidence label, the sentence under the header says it once, in full, and a column would
be that one word repeated fifty times. When the labels differ, no sentence above the table
is true of all of them, so the word has to sit beside the figure. Today every row is
`estimated`, so the column is absent and the sentence carries it.

**Where the six that left went — all of them into the row, none of them deleted:**

| Dropped | Where it is now |
|---|---|
| `Runs` (mode) | New facts strip at the top of the expanded row. |
| `Size` (download) | Same facts strip, beside it. |
| `Qual` | Already in the expanded row's score table, as the `quality` part with its weight and what it added. |
| `Card` (VRAM) | Already in the expanded row's budget: *"Card: 5.5 GiB of 6.4 GiB free, 86% used."* |
| `RAM` | Same sentence's twin, for system memory. |
| `Prompt tok/s` | Already in the expanded row's speed headline: *"114.6 tokens per second generated and 21,643 read, at 8K tokens of context (estimated)."* |

Four of the six were already in the detail view and were being printed twice. Only `Runs`
and `Size` needed somewhere to go, and they got a two-line `dl` at the top of the row along
with whether the file is already on this machine and where — which the board never showed
at all and the TUI has as a `Have` column.

**Sorting.** The page still sorts on the headings that remain: `Score`, `Tok/s`, `Ctx`. The
API's `sort` parameter is untouched and still accepts `quality` and `size`; the page simply
no longer offers a heading to click for them. The machine-readable interface did not change
in any respect — no endpoint, no field, no parameter, no serialisation.

---

## 2. The name of the speed column, and what the other one is called

**`Tok/s`, and prompt throughput is called `Prompt tok/s`.**

I took the second of the two branches offered: prompt throughput leaves the grid entirely.
On the web list it is not a column at all — it is a clause in the sentence the expanded row
prints — so `Tok/s` stands alone and is unambiguous.

**I renamed it in the terminal too, and that was the larger decision.** `Gen/s` was one
catalog entry read by `cli/render_board.py`, `tui/board_view.py` and `web/strings.py`, and
`src/llamafit/web/strings.py` documents the invariant that both interfaces read one entry
per heading so that "a column meaning the same thing is spelled the same way". Renaming
only the browser would have broken a stated project rule in order to fix a wording problem
that `Gen/s` has just as badly in the terminal.

That created a second problem I had to solve rather than leave: on the terminal board, and
in the measurements table, generation and prompt throughput sit side by side, and `Tok/s`
beside `PP/s` invites the reader to think only one of them is tokens. So:

- `Gen/s` → `Tok/s` — the unmarked, headline figure.
- `PP/s` (board) → `Prompt tok/s`.
- `Prompt/s` (measurements table) → `Prompt tok/s` as well, collapsing two spellings of one
  quantity onto one message. `run.pp_tps` and `speed.pp_tps` are the same number and now
  have one name across the whole tool.

The asymmetry is deliberate: the headline figure is unqualified, the other is qualified by
direction, and the qualifier does the disambiguating work. The terminal board at 150
columns now reads `… | Score | Tok/s | Fit | Runs | Ctx | Qual | Card | Prompt tok/s | Size`.

**Priority.** I was asked to check that the speed column is among the last things a narrow
terminal drops. It already was and still is: `gen` is the *first* entry of the optional
list in `_board_columns` and the first group of both `MIXED_ORDER` and `UNIFORM_ORDER`,
which makes it the highest-priority droppable column in both terminal boards. Only the four
identity columns (`#`, `Model`, `Quant`, `Score`) outrank it, and those are never dropped.
I did not change the priority; I verified it.

**Catalogs.** Three messages were orphaned — `Gen/s`, `PP/s`, `Prompt/s` — and two are new
— `Tok/s`, `Prompt tok/s`. Only six of the thirty-seven catalogs carried the old entries at
all (the other thirty-one have never reached the column headings and correctly fall back to
English). The stale entries were removed from all six and the replacements translated:

| | `Tok/s` | `Prompt tok/s` |
|---|---|---|
| `pt_PT`, `pt_BR` | `Tok/s` | `Tok/s de prompt` |
| `es` | `Tok/s` | `Tok/s de prompt` |
| `fr` | `Tok/s` | `Tok/s de prompt` |
| `de` | `Tok/s` | `Prompt-Tok/s` |
| `it` | `Tok/s` | `Tok/s di prompt` |

`Tok/s` is left as it stands in all six, which is what those catalogs already did with
`Gen/s`, `PP/s` and `Prompt/s`: it is a unit abbreviation, like `GB/s`. The translator
comment above the entry was rewritten to say so. `prompt` is a loanword in all six —
`O tempo de um token de prompt`, `Die Zeit eines Prompt-Tokens` — so the replacements match
the register those catalogs already use.

**All thirty-seven catalogs report zero problems** from the checker, after `gen_messages.py`
regenerated the template.

---

## 3. What I looked at before and after, and what changed my mind

I ran `llamafit serve` and read the page in a browser at 1440 wide before touching
anything, in the system language (Portuguese) and then in English, collapsed and with a row
expanded. Three things I would not have got from the source:

**The detail view was broken, and had been.** `table { min-width: 34rem }` applied to every
table on the page, including the four small ones inside an expanded row, each of which sits
in a `19rem` grid track wrapped in `overflow-x: auto`. So they were silently scrolled: the
score table showed `Part` and `Score` and hid `Weight` and `Adds`; the context ladder hid
`On this machine`, which is the column that says what happens at each rung. Every figure was
present in the DOM and none of it was on the screen. Scoping the floor to the two full-width
tables fixed it. I would have read past that rule a dozen times.

**Seven columns in a `width: 100%` table look worse than fourteen.** My first cut left a
hundred empty pixels between a model's name and its speed, because the browser distributes
slack evenly and there was suddenly a lot of slack. Fewer columns did not clean the table up
on its own; it needed the width bounded and the columns pinned. That is what pushed me to
`table-layout: fixed` with a width per column, which is also what made fifty rows work.

**`table-layout: fixed` does nothing if the table's own width is `auto`.** I set
`width: auto` to make the list hug its content, measured it, and found the table 1362px wide
with a 153px name column — my declared widths ignored. CSS resolves the table box with the
*automatic* algorithm when `width` is `auto`, whatever `table-layout` says, so one long
excluded-row sentence was setting the width for every row. Capping the container and letting
the table fill it fixed it. I had this wrong until I read the computed style.

**And a real bug I only saw because I was reloading.** Chrome restores form values across a
reload. The controls came back showing `coding`/`quality`/`+coding capability` while `asked`
still held the defaults, so the controls and the request disagreed until the next
interaction. The page now reads the request off its controls at boot instead of trusting its
own defaults, and the form carries `autocomplete="off"`.

**What the 50-row list changed.** Two things, and I only saw them with the fixture on
screen. The captions that say what the figures are — *"Speeds are for 8K tokens of context
so every row compares like with like"* — were under the table, which is fine for three rows
and is never read at fifty; they moved above it. And the column headings scrolled away, so
they are now sticky. Sticky inside an `overflow-x: auto` container does not work, because
`overflow-x: auto` forces `overflow-y` to `auto` and the box never scrolls vertically; the
horizontal scroll is now applied only below `56rem`, where the headings cannot stick but the
table has to scroll.

---

## The page at fifty rows

I did not reason about this. The catalog on this machine produces two ranked rows and three
excluded, so I built a fixture in the page — `scratchpad/fifty_rows.js`, not shipped — that
clones a real row into 38 ranked and 12 unranked across twenty model families, ten sizes,
six quantisations and all four verdicts, then calls `renderBoard()`. What it looks like:

- **960 pixels wide, left-aligned, fifty rows of identical height.** Every column has a
  declared width, so nothing shifts between rows.
- **`Tok/s` reads as a single column of cyan running down the page**, descending, and it is
  the only cyan anywhere. This is the answer to the original complaint: at fifty rows the
  column is not merely findable, it is the thing your eye lands on first.
- **`Fit` bands.** Because the list is score-ordered and fit correlates with score, the
  verdict column blocks into runs of green, then amber, then red. That is legible as shape
  before it is read as words.
- **Everything else is a shade of the ground** — rank, model, quant, score, context — so the
  two coloured columns carry all the signal.
- **The twelve unranked rows sit at the bottom of the same list**, dimmer, keeping `#`
  (blank), `Model` and `Quant` aligned with every other row, and giving the remaining width
  to the reason. Ranked first and unranked after, because a candidate with no score has no
  place in an order made of scores.
- **A long model id ellipsises** rather than running under the next column. I tested with a
  52-character community-fine-tune name.

The reason cell gets about 400 pixels — roughly the first clause, which is the informative
one (*"generates 2.1 tokens per second, below the 6 a person rea…"*). The whole sentence is
in the `title` attribute and in the row when it is opened.

---

## What was built

**One page.** The five tabs are gone. The machine is a line under the title. Behind it are
two disclosures, folded by default: **Host** (the machine's facts, the llama.cpp
installation, every finding `doctor` reports, and Scan again) and **Simulate** (profile, or
an override of VRAM, RAM or cores). The request is a row of controls that is always visible.
The list is everything below it.

**The controls are live.** There is no Apply button. Every control re-asks the server on
`change`, so a person changes what they are asking for and watches the list answer. Reset is
wired to re-ask too. The default row count went from 10 to 50.

**The plan moved into the row it belongs to.** Pressing *Plan this* inside an expanded row
builds the plan for that model and quant and renders the file line and the command line with
its Copy button underneath. Two fields — Context and Micro-batch — sit beside the button so
the Plan panel's ability to ask for a different context is not lost. When the context asked
for is the one the row was already planned at, the plan does **not** redraw the speed, the
budget, the ladder or the notes the row is already showing; when it differs, it draws all of
them, because then they are different numbers.

**Dark, and committed to it.** `color-scheme: dark`, no `prefers-color-scheme` block, no
light values anywhere. The neutrals run `#101317` → `#171b21` → `#232830` → `#2f3641` →
`#7d8794` → `#98a1ad` → `#d3d9e0` → `#f2f5f8`, and every step carries about eight more parts
of blue than of red — a consistent cool bias rather than a flat grey. `--ink-faint` on the
ground is 5.1:1, so the quiet text still passes AA.

**Colour on two things.** `--speed` (`#79c6e3`) is used on exactly one column and nowhere
else on the page. The verdict colours are the terminal's own three, unchanged in meaning.
The only other uses of colour are the same three severities on the doctor's findings, and
the `SIMULATED` badge, which is red because it means every figure on the screen is invented.

I deliberately did not give the interactive accent a colour of its own. Links and the
current-state markers use ink and an underline, so that the two coloured columns are not
competing with a third hue.

**Fewer cards, fewer borders.** The tab bar, the form cards, the fieldset borders and the
notice boxes are all gone; the per-row table borders are gone, leaving one rule under the
header. Border, fill, radius and shadow are now spent on exactly three things, each of which
genuinely is a separate object: the expanded row under a board row (a fill), the block of
text meant to be copied (a fill), and the controls a pointer has to hit (a 1px border and a
2px radius). Everything else is separated by space and alignment.

**Nothing was lost from the three things that had to survive.**

- *Every number says how it was obtained.* The estimate sentence — the terminal's own
  `confidence_sentence("estimated")` — is still directly under the header, above everything
  else on the page. The `How` column still appears the moment the rows stop agreeing. The
  budget still names every line as `file` or `formula`. The speed headline still ends
  `(estimated)`.
- *An excluded candidate keeps its reason.* It is now more prominent than before, not less:
  it used to be a second table below the fold and is now a row in the one list, in place,
  with the reason on the same line and the whole of it one click away.
- *The machine-readable interface did not change.* No endpoint, parameter, field or
  serialisation was touched. The only Python changed outside the headings is a docstring.

**Headers.** The three static files each already carried the three-line copyright and licence
notice in their own comment syntax — an HTML comment, a `/* */` block, `//` lines — because
`tests/unit/test_web_static.py` enforces exactly that, `tests/unit/test_licensing.py` walking
only `*.py` and never seeing them. I kept that call: all three files were rewritten in place
with their notices intact, and the same test still checks them. The same file also caps the
three at 1500 lines total; they are at 1456, down from 1567 at my first pass, which is what
pushed the stylesheet into writing one-declaration rules on one line.

---

## Verification

| Check | Result |
|---|---|
| `pytest --cov -m "not hardware and not network"` | 2925 passed, 2 skipped, 14 deselected |
| Coverage | **96.63%** (floor 85%) |
| `ruff check .` | All checks passed |
| `ruff format --check .` | 336 files already formatted |
| `mypy` | no issues in 164 source files |
| `gen_messages.py && gen_schema.py && gen_models_md.py && git diff --exit-code` | clean |
| `check_po.py` on all 37 catalogs | zero problems each |

One test changed. `test_the_board_names_the_confidence_per_row_only_when_the_rows_disagree`
laid a board out for 200 columns and then drew it into a 100-column console, so Rich squeezed
every column and the wider `Prompt tok/s` heading pushed `measured` into `measur…`. The test
above it in the same file already passes `width=200` for exactly this reason; this one did
not. It now does. Production never creates that mismatch — `render_board` is called with the
console's real width — so the test now checks what it says it checks.

---

## Concerns

1. **I renamed two headings in the terminal, which was not literally in the brief.** The
   brief was about the browser. I judged that a web-only rename would break the project's
   own stated invariant, and the second instruction confirmed it ("Change it in the terminal
   too"). But `PP/s` → `Prompt tok/s` was mine, forced by the first instruction's rule that
   `Tok/s` beside `PP/s` is not defensible. It changes the terminal's board and its
   measurements table for anyone who has learned them. Worth the owner's eye.

2. **The terminal's column-width budget is optimistic, and I only nudged it.** While fixing
   the test I found that `_board_columns` admits columns whose combined rendered width can
   exceed the console it was given — at 200 columns it admits all ten and Rich then truncates
   `Fit` to `tig…`. That predates this work and I left it alone; widening `prompt` from 6 to
   13 cells makes it slightly easier to hit. It deserves its own fix.

3. **Fifty rows is the size I tested; two hundred is not.** `Rows to show` accepts up to 200
   and the excluded list is never capped by the API. The list renders every row eagerly with
   no virtualisation. I did not measure it at 200+.

4. **The 1500-line cap is now the binding constraint on this page.** At 1456 there are 44
   lines of headroom for three files. The next feature — Downloads in phase 2 — will not fit
   without splitting them, which the test's own message anticipates ("split it or cut it").

5. **The sticky header does not stick below 56rem**, where the container takes the horizontal
   scroll. That is a real trade and phones get the worse half of it.

6. **The unranked reason is cut to about 400 pixels.** Tooltip and expansion recover it, but
   a person scanning quickly sees a clause, not a sentence. If the reasons were written
   shortest-clause-first this would matter less; some already are.

7. **Five served strings are now asked for by nobody** — `panel.board` is used, but
   `board.title`, `board.excluded`, `board.explain`, `board.score_parts`, `needs.apply` and
   `plan.pick_first` are not, since the second table, the Apply button and the model picker
   are gone. They are still extracted from `strings.py`, so no catalog entry is orphaned and
   no translator loses work; they are simply unused keys. Removing them is a separate,
   catalog-touching decision I did not want to fold into this one.
