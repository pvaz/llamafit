# `fit`, `recommend` and `plan` — implementation report

Branch `feat/commands`, worktree `llamafit-wt-commands`. Five commits:

| SHA | Subject |
|---|---|
| `6ea2482` | fix(placement): walk section 9.3's ladder, and count the threads the cores give |
| `794e448` | feat(plan): give each context tier the verdict its own budget produced |
| `ed9d6db` | fix(services): match a quant against the disk by bare file name |
| `f75cb80` | feat(cli): add fit, recommend and plan |
| `c023748` | feat(cli): say what each part of a score was measured against |

Verification: 1,621 passed, 1 skipped, 11 deselected; coverage 95.69 percent (floor 85).
`ruff check`, `ruff format --check`, `mypy --strict` and the three generators followed by
`git diff --exit-code` are all clean.

## What was built

| Piece | Where |
|---|---|
| The board: plan, size, estimate, score, order, and keep the exclusions | `src/llamafit/services/recommend.py` |
| The plan report: placement, budget, ladder, speed, flags, command line | `src/llamafit/services/plan.py` (added to the existing budget/planner join) |
| Tables for all three, plus the budget, ladder, speed and score expansions | `src/llamafit/cli/render_board.py` |
| `llamafit fit`, `llamafit recommend` | `src/llamafit/cli/board_cmd.py` |
| `llamafit plan` | `src/llamafit/cli/plan_cmd.py` |
| The catalog loader, model lookup and value checks four commands now share | `src/llamafit/cli/common.py` |
| `--prefer balanced\|quality\|speed` | `src/llamafit/scoring/weights.py` |

Flags follow section 13.1. One departure, noted because it is a deliberate omission:
section 12.1 lists both a `preferred: [thinking]` capability list and a
`prefer: balanced|quality|speed` weight shift, and `docs/cli.md` had `--prefer CAP
(repeatable)` for the first. Only the second is implemented. Section 11.1 removed the
per-capability bonus outright — "it looked like a judgement and was arithmetic on the
length of the request" — so `--prefer thinking` would change nothing at all, and a flag
that changes nothing is worse than a flag that is missing. `docs/cli.md` now says so.

---

## 1. The two commands as they render on this machine

Both are pasted verbatim from a piped run, so Rich uses an 80-column width and the column
budget drops the run mode, context, quality and memory columns; a 100-column terminal shows
them. The machine's locale is Portuguese, so numbers carry Portuguese punctuation
(`93,9`, `6,3 GiB`) while the new headings and sentences are still English: they are
untranslated in all 37 catalogs, which is what a new message should look like. The one
Portuguese word visible, `Contexto`, is a heading whose message already existed for the
catalog listing.

### `llamafit recommend --use-case coding`

```
                          Recommended
┌───┬────────────────────┬────────────┬───────┬───────┬───────┐
│ # │ Model              │ Quant      │ Score │ Gen/s │ Fit   │
├───┼────────────────────┼────────────┼───────┼───────┼───────┤
│ 1 │ qwen3-coder-next   │ UD-Q4_K_XL │  93,9 │  23,0 │ fits  │
│ 2 │ qwen3-0.6b         │ Q8_0       │  72,2 │ 114,6 │ fits  │
│ 3 │ qwen3.8-flash-next │ UD-Q4_K_XL │  66,9 │  13,3 │ tight │
└───┴────────────────────┴────────────┴───────┴───────┴───────┘
Speeds are for 8K tokens of context so every row compares like with like; the
context column is the largest each one holds. Sized and scored for coding at
32K tokens.
Speeds are section 10's formula on its default constants. Nothing has been
benchmarked on this machine yet, so no figure here is a measurement.
Weights: quality 0,40, speed 0,20, fit 0,20, context 0,20.

                                  Not ranked
┌───────────────────────┬────────┬────────────────────────────────────────────┐
│ Model                 │ Quant  │ Why not                                    │
├───────────────────────┼────────┼────────────────────────────────────────────┤
│ gemma-3-27b-it        │ Q4_K_M │ not a coding model; its entry lists        │
│                       │        │ general, multimodal, chat, so ask for one  │
│                       │        │ of those                                   │
│ llama-3.1-8b-instruct │ Q4_K_M │ not a coding model; its entry lists        │
│                       │        │ general, chat, reasoning, so ask for one   │
│                       │        │ of those                                   │
└───────────────────────┴────────┴────────────────────────────────────────────┘
```

Section 11.1's own worked example holds: Llama 3.1 8B, which used to rank second for a
coding request "carried there by fit and a capped speed score", is now off the board with
the reason and the list of jobs its entry does claim.

The estimate for `qwen3-coder-next` moves between 22.6 and 24.1 tokens per second across
runs minutes apart, and `qwen3.8-flash-next` between 13.3 and 13.9. That is not noise in
the estimator: the memory-bandwidth probe re-measures on every scan and returned anything
from 41.7 to 60 GB/s on an otherwise idle machine, and the scattered-expert term divides by
it. Worth a note in §J below.

### `llamafit plan qwen3.8-flash-next`

```
Qwen3.8-Flash-Next UD-Q4_K_XL
experts in RAM at 16 384 tokens, micro-batch 1 024, f16 cache, 16 threads.
Routed experts are held in system memory, and so are the always-on shared
experts, which is what -ot ffn_.*_shexp=CPU does; attention and the KV cache
stay on the card.
Sized for 16 384 tokens rather than the 32 768 asked for.
16 threads, not the 32 this processor has: the efficiency cores are left out
because generation measures slower with them, and what is left is the threads
the performance cores provide.
Tight: 6,7 GiB of the graphics card is free and this needs 6,3 GiB, so another
program can push it over. The context tier table is how a launch script
recovers.

                     Memory budget
┌──────────────────────┬────────┬───────────┬─────────┐
│ Component            │ Where  │      Size │ From    │
├──────────────────────┼────────┼───────────┼─────────┤
│ dense weights        │ card   │   3,6 GiB │ file    │
│ shared experts       │ system │ 239,5 MiB │ file    │
│ routed experts       │ system │  71,7 GiB │ file    │
│ token embedding      │ system │ 644,1 MiB │ file    │
│ output head          │ card   │ 644,1 MiB │ file    │
│ other weights        │ card   │   6,7 MiB │ file    │
│ streamed tables      │ disk   │  26,8 GiB │ file    │
│ KV cache             │ card   │ 384,0 MiB │ formula │
│ KV cache, undeclared │ card   │ 144,0 MiB │ formula │
│ recurrent state      │ card   │ 112,2 MiB │ formula │
│ compute buffer       │ card   │   1,3 GiB │ formula │
│ output buffer        │ system │   1,9 GiB │ formula │
│ backend overhead     │ card   │ 120,0 MiB │ formula │
│ process overhead     │ system │   1,0 GiB │ formula │
└──────────────────────┴────────┴───────────┴─────────┘
Card: 6,3 GiB of 6,7 GiB free, 94% used.
System memory: 75,5 GiB of 99,1 GiB free, 76% used.
Tight: it runs, but a browser or a second process can push it over.

dense weights: attention projections, feed-forward weights and norms
shared experts: always-on experts, which run for every token
routed experts: routed experts, read across the bus once per micro-batch
token embedding: llama.cpp keeps the embedding table in memory even at full
offload
streamed tables: streamed from disk as it is needed, not held in memory
KV cache: qwen4exp allocates more cache than its header describes; see the line
below
KV cache, undeclared: what qwen4exp allocates beyond its header, measured
rather than derived
recurrent state: from the architecture's state dimensions; it does not grow
with context
compute buffer: fitted to one machine's measurements; the least certain line
here
output buffer: room for one logit per vocabulary entry per token of the batch
backend overhead: measured on this card with driver 610.88 and llama.cpp
b10867; one machine
process overhead: the server process before any model buffer

               Context tiers
┌──────────┬────────────┬─────────────────┐
│ Contexto │ Card needs │ On this machine │
├──────────┼────────────┼─────────────────┤
│      16K │    6,3 GiB │ tight           │
│      24K │    6,5 GiB │ pages           │
│      32K │    6,8 GiB │ pages           │
│      40K │    7,1 GiB │ pages           │
│      48K │    7,4 GiB │ pages           │
│      64K │    8,0 GiB │ pages           │
│      96K │    9,1 GiB │ pages           │
│     128K │   10,2 GiB │ pages           │
│     192K │   12,5 GiB │ pages           │
│     256K │   14,7 GiB │ pages           │
└──────────┴────────────┴─────────────────┘
A launch script compares the free memory it sees against the card column and
takes the largest rung that fits (see docs/cli.md).
Pages: this needs more of the graphics card than is free. The driver will not
refuse it — it moves the overflow to system memory, the server starts, the log
looks healthy, and generation runs at a fraction of its speed with nothing
saying why.

13,3 tokens per second generated and 46 read, at 16K tokens of context
(estimated).
┌───────────────────────┬─────────┬───────┐
│ A token's time        │ Seconds │ Share │
├───────────────────────┼─────────┼───────┤
│ reading the card      │  0,0273 │   36% │
│ reading system memory │  0,0466 │   62% │
│ everything else       │  0,0010 │    1% │
└───────────────────────┴─────────┴───────┘
1.50 GB of routed experts is read from system memory per token at 0.57 of its
57 GB/s, because each token selects a different handful of experts and the read
is a scatter of small blocks rather than a stream.

                              Recorded elsewhere
┌──────────────────────────────────┬───────┬──────────┬──────────┬────────────┐
│ Run                              │ Gen/s │ Prompt/s │ Contexto │ Date       │
├──────────────────────────────────┼───────┼──────────┼──────────┼────────────┤
│ Gen 1K / Prompt 1K, -ub 2048,    │  14,5 │       50 │      16K │ 2026-09-09 │
│ vision and shared experts on GPU │       │          │          │            │
│ winning configuration: Gen 1K /  │  13,9 │       49 │      40K │ 2026-09-09 │
│ Prompt 1K, shared experts and    │       │          │          │            │
│ vision on CPU                    │       │          │          │            │
└──────────────────────────────────┴───────┴──────────┴──────────┴────────────┘
These are the curator's own runs on their machine, not benchmarks of this one,
so nothing above is labelled measured. Compare them with the estimate;
`llamafit bench` will measure this machine in phase 3.
Gen 1K / Prompt 1K, -ub 2048, vision and shared experts on GPU: -ngl 99
--n-cpu-moe 48 --fit off -fa on -t 16 -tb 16 -b 4096 -ub 2048
winning configuration: Gen 1K / Prompt 1K, shared experts and vision on CPU:
-ub 1024 --no-mmproj-offload -ot ffn_.*_shexp=CPU

Model file:
D:\llama.cpp\models\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-0
0004.gguf.
Command line
llama-server -m
D:\llama.cpp\models\Qwen3.8-Flash-Next\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-0
0004.gguf --alias qwen3.8-flash-next --host 127.0.0.1 --port 8080 -c 16384 -fa
on -ngl 99 --n-cpu-moe 48 -ot ffn_.*_shexp=CPU --fit off -t 16 -tb 16 -b 2048
-ub 1024 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0
--repeat-penalty 1.0 --jinja --reasoning-format deepseek -np 1
```

### The comparison against the recorded winner

`src/llamafit/data/catalog/qwen3.yaml` records the reference machine's own best
configuration as `-ub 1024 --no-mmproj-offload -ot ffn_.*_shexp=CPU` at 40,960 tokens,
with the row above it supplying the flags it inherits: `-ngl 99 --n-cpu-moe 48 --fit off
-fa on -t 16 -tb 16`, peak VRAM 7.3 GB.

| | Recorded | Planned |
|---|---|---|
| mode | `-ngl 99 --n-cpu-moe 48` | same |
| shared experts | `-ot ffn_.*_shexp=CPU` | same |
| micro-batch | `-ub 1024` | same |
| threads | `-t 16 -tb 16` | same |
| cache | f16 | same |
| context | 40,960 | **16,384** |
| vision projector | `--no-mmproj-offload` | **not rendered** |

Two differences, and they are different kinds of thing.

**The context is not a defect in the plan.** The recorded run peaked at 7.3 GB of card;
this scan found 6.7 GiB free after the 256 MiB desktop reserve, because the desktop was
holding about 1.3 GB of the 8 GB card at the time. The budget's own figure for the recorded
configuration is 7,279 MiB against a measured peak of 7.3 GB — 4.5 percent high, inside
section 18's ±5 percent memory tolerance — and the tier table says exactly this: 40K needs
7.1 GiB and would page. Run the same command with a clean desktop and the planner returns
40,960. This is the case section 9.3's ladder exists for, and it now behaves.

**The projector is a catalog gap, not a planner one.** `qwen3.8-flash-next` declares the
`vision` capability and the `multimodal` use case, but its entry publishes no `mmproj`
extra, so `projector_of` finds nothing, the ladder has one rung (`None`), no projector
memory is charged and `--no-mmproj-offload` is never rendered. The recorded run used a
projector and paid for it. **The plan is therefore optimistic about card memory for anybody
who actually wants vision from this model**, by roughly the 1.9 GB the calibration record
attributes to it. The fix is a `catalog refresh` that fills in the extras, or one hand-added
`extras: [{role: mmproj, file: ...}]`; it is not a code change and I did not make it,
because inventing a file name for a repository I have not listed would be worse than the
gap.

---

## 2. Disagreements the modules had not noticed

Six, of which four are fixed and two are reported.

### A. The planner halved the context; section 9.2 says it must not (fixed, `6ea2482`)

`placement/modes.context_ladder` produced "the request, then halves". Section 9.2 spells out
the opposite in as many words — "Candidate contexts come from that ladder, not from halving.
Halving cannot reach 40960" — and gives the reason. The module's own docstring cited 9.2
while implementing its contrary.

It bit on this machine, hard. With the desktop holding 1.3 GB of the card, no
mixture-of-experts placement of `qwen3.8-flash-next` fits at 40,960 or at 20,480, and the
halving ladder offers nothing between 20,480 and the 16,384 floor it stops above. The
planner therefore found *nothing* in `moe-offload`, fell through to `hybrid`, and returned
**two layers of forty-eight on the card with seventy gigabytes of expert weights on the
memory bus**, at 40,960 tokens, verdict "tight". It is a placement that runs, so nothing
would ever have reported it. With the tier ladder the same request returns the
mixture-of-experts placement at 16,384, which is what the machine was measured running.

No test caught it because every planner test drives a scripted budget, where the search
order is verified against an exhaustive search over *the same ladder function*; the
exhaustive reference imports `context_ladder` too, so the two agreed on a shared mistake.
The one test that uses the real budget and the real catalog
(`test_the_planner_reaches_the_reference_machines_own_winning_configuration`) passes on the
*fixture* host, which records only 550 MiB of desktop and so reaches 32,768 by halving from
the default 32,768 — the one context where halving and the ladder happen to agree.

### B. `-t` counted cores, not threads; section 9.4 says threads (fixed, `6ea2482`)

`thread_count` returned `performance_cores`, with a docstring defending 8 on the reference
machine. Section 9.4 says the opposite, and says it was changed on purpose: "excluding them
is not the same as counting cores rather than threads: on the reference machine the measured
optimum is 16, which is the eight performance cores' thread count, and it is about four
percent faster than 8. An earlier revision of this section conflated the two and gave up
that four percent for no reason." The recorded winning configuration passes `-t 16 -tb 16`,
so the rendered command line disagreed with the measurement it was meant to reproduce.

The figure is now reached by subtracting the efficiency cores' threads from the total rather
than by doubling anything, so a part with no efficiency cores gets all its threads and a part
with no multithreading gets its cores. Three tests moved from 8 to 16.

### C. A context tier could not tell paging from not fitting (fixed, `794e448`)

`ContextTier` carried `tokens`, `vram_required` and one boolean. Section 8.4 requires that a
configuration over the card be named as paging rather than as failing, and a boolean cannot
carry that. My first attempt derived it in the renderer, from `vram_required >
vram_available`, and produced a ladder that read `16K fits / 24K no room / 32K pages` —
non-monotonic, because a rung can be over the 95 percent threshold without being over the
card. The verdict was already being computed for every rung (costing a rung *is* computing a
budget) and thrown away; it is now carried, and the ladder shows the paging band directly.

### D. Two different defaults for "what context to size for" (reported, not fixed)

`constants.DEFAULT_REQUESTED_CONTEXT` is 32,768 and is what `plan_placement` sizes for when
the request says nothing. `scoring/context_score.DEFAULT_CONTEXT` is 8,192 for general and
chat, 32,768 for coding and reasoning, 16,384 for multimodal, and is what the context score
is measured against. `models/plan.Needs` documents its own field as "What to size for,
defaulting by use case", which describes the second and is consumed by the first.

For a coding request the two agree and nothing is visible. For `--use-case general` the
planner sizes every candidate for 32,768 tokens and the score then measures the result
against 8,192. Neither is wrong about its own question — section 11.2 and section 9.2 are
answering different ones — but nothing joined them, and the first caption I wrote for
`llamafit fit` printed the score's number under the word "sized", which was simply false.

Rather than change either default, the board now carries both (`requested_context` and
`planned_context`) and says both when they differ: *"Sized for 32K tokens and scored for
general against 8K."* If one of them should win, it is a decision for whoever owns sections
9.2 and 11.2, not for the rendering layer.

### E. A benchmark can be labelled `measured` for a configuration it never described (reported, not fixed)

`speed/estimate._flags_agree` documents its rule as "a flag the measurement does not record
cannot disagree". The catalog's winning row for `qwen3.8-flash-next` records
`-ub 1024 --no-mmproj-offload -ot ffn_.*_shexp=CPU` — no `-ngl`, no `--n-cpu-moe`, because
it was written as a delta from the row above it. Under that rule the row agrees with *any*
offload, and generation deliberately does not compare the micro-batch either.

During development, before fix A, I watched the estimator return **13.9 tokens per second
labelled `measured`** for the hybrid `-ngl 2` placement — a configuration with two layers on
the card, which the benchmark had nothing to say about. A CPU-only placement of the same
model would have been labelled the same way.

There is a second half to this. Section 10.3 defines `measured` as "a stored benchmark **on
this host**", and a catalog `measured` block is a record from whichever machine its curator
ran it on. Phase 3's `bench` is what will store this host's own.

So the commands never pass catalog measurements to the estimator. Every figure they print is
`estimated`, the board says so once under the table rather than in a column, and `plan`
prints the catalog's records in a separate *Recorded elsewhere* table with the profile, the
flags, the context and the date, saying plainly that they are somebody else's machine.
Section 20's definition of done for phase 1 asks for estimates *within tolerance of* the
measurements, which is a comparison that only exists while the two are different numbers.

The estimator itself is untouched: `_flags_agree` is still able to mislabel, and it will
matter the moment phase 3 stores a local benchmark whose command line is written as a delta.
The narrow fix is to require that a measurement record `-ngl` and `--n-cpu-moe` before it can
match a placement that sets them; the broader one is a host fingerprint on `Measured`.

### F. Sharded quants read as missing from a machine that has them (fixed, `ed9d6db`)

`services/catalog.describe` matched `quant.files` entries against local files keyed by bare
name. A catalog `files` entry is a path *inside the publishing repository* and often carries
a directory —
`UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` is one of the seeded ones —
so the comparison was between unlike things and every sharded model reported as absent. My
own code inherited the bug by copying the pattern; the test that found it was in the new
service. Both sides are now reduced to a bare name. It is why the plan above names a real
`D:\llama.cpp\models\...` path.

### G. An observation that is not a bug: the fit score does not do what 11.3 claims (reported)

`llamafit recommend` with no use case puts `qwen3-0.6b` first, ahead of `gemma-3-27b-it`.
Section 11.3's left-hand slope exists precisely to stop this: "a tool that scored purely on
safety ... would therefore tell a person with 128 GB of memory and an eight-gigabyte card to
run a 0.6B model".

It happens anyway, and the reason is not the slope. At `-ub 2048` and 32,768 tokens the
compute buffer alone books 2.3 GiB of the card and the output buffer 2.3 GiB of system
memory, so the 0.6B model's worst-pool utilisation lands near 0.82 and its fit score near
100. On a small card the *buffers*, not the model, set utilisation, so the "far smaller than
the machine" band is nearly unreachable and the slope never engages. Section 11.3's curve is
sound; the quantity it is applied to does not discriminate at this end. Worth a look before
phase 1D, since the TUI will show the same ordering.

---

## 3. Where the output still asks a reader to take a number on faith

Ordered by how much it matters.

1. **Prompt processing has no working shown at all.** `plan` prints "46 read" and the board
   has a `PP/s` column, and nothing anywhere says how either was reached. Generation gets
   three shares that add up to it and a sentence naming the effective bandwidth and the
   traffic; prompt processing gets a bare number. Section 10.2's formula has two terms
   (compute on the card, expert bytes streamed over the link) and two constants (`EFF_PP`
   0.30, `EFF_PCIE` 0.33) whose provenance comment in `constants.py` openly says the second
   is under-predicting Qwen3-Coder-Next about twofold because residency, not the link, is the
   real variable. None of that reaches the reader. This is the largest remaining gap and the
   easiest to close: a second small table beside the token-time one.

2. **The card term of the token breakdown is unexplained.** "reading the card 0,0273 s, 36%"
   says nothing about the 4.83 GB it read or the 0.67 of 272 GB/s it read them at, while the
   system-memory line below it says both. The asymmetry is an accident of which notes
   `formula_estimate` happens to emit.

3. **The quality baseline is a person's opinion and the expansion does not link to its
   evidence.** `--explain` says "85 from the curator, less 3.0 for this quantisation, plus 5
   because coding is the job it was built for", which is the arithmetic but not the argument.
   The published benchmarks that justify the 85 are in the catalog and are shown by
   `llamafit info`; nothing in `recommend --explain` points there. One sentence would fix it.

4. **The weights are printed and never justified.** "Weights: quality 0,40, speed 0,20, fit
   0,20, context 0,20" is the most arguable set of numbers in the program — the module's own
   docstring says so — and the reasoning behind each lives only in that docstring and the
   specification.

5. **`max_context_fit` is a bisection with no stated stopping rule.** The `Ctx` column and
   "Holds 62,464 tokens" are the result of a binary search over budgets on a 1,024-token
   grid that stops at the last *acceptable* verdict; a reader is not told that "holds" means
   "holds without paging", nor that the figure is rounded to the nearest thousand tokens on
   purpose.

6. **`--target-tps` names a figure without naming its configuration.** "the fastest
   configuration tried is 13,5" does not say which rung of the ladder that was.

7. **Two numbers in one sentence disagree about punctuation.** `speed/estimate.py` formats
   its note with `%(gb).2f` and `%(raw).0f` directly instead of through
   `units.localise_number`, so a Portuguese reader is told "1.50 GB ... 57 GB/s" on the line
   under "13,3 tokens per second". Cosmetic, but it is the one place the project's own rule
   about number punctuation is broken, and it is visible in the paste above.

8. **The speed estimate is not reproducible run to run.** The bandwidth probe re-measures on
   every scan and returned 41.7 to 60 GB/s on an idle machine, moving `qwen3-coder-next`
   between 22.6 and 24.1 tokens per second. The figure is honestly labelled `estimated` and
   the note names the bandwidth it used, so nothing is hidden — but two runs a minute apart
   give two answers, and nothing tells the reader that the *input* moved rather than the
   model. A scan cache, or reporting the probe's own variance, would settle it.

---

## Concerns for whoever picks this up

- **The projector gap (§1).** Until the catalog carries an `mmproj` extra for
  `qwen3.8-flash-next`, every plan for it is optimistic about card memory for a user who
  wants vision, and `--no-mmproj-offload` can never be rendered. Same question applies to
  `gemma-3-27b-it`, which also declares `multimodal`.
- **`_flags_agree` (§2E)** is a live mislabelling risk that phase 3 will walk straight into.
- **The fit score at the small end (§2G)** will reorder the board for `general` and `chat`
  in a way section 11.3 explicitly did not intend.
- **The two context defaults (§2D)** should be reconciled by whoever owns sections 9.2 and
  11.2; the board currently reports both rather than choosing.
- **`--prefer` is a weight shift, not a capability list.** If the capability form is wanted,
  section 11.1 has to grow a bonus that can discriminate first.
- The two message ids `invalid --capability value(s): %(values)s` were removed from all 37
  catalogs, losing their translations, because the message now names the flag the reader
  actually typed (`--capability` for `list`, `--require` for `recommend`). Everything else
  added here is untranslated everywhere, which is correct.
