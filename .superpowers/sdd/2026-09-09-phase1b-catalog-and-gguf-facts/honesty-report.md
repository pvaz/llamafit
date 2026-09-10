# Two honesty gaps in the speed estimator

Branch `fix/honesty`, worktree `llamafit-wt-honesty`, off `main` at `c0fafa5`.

| Commit | Subject |
|---|---|
| `be44c9d` | `fix(speed): a benchmark is measured only for the run it describes` |
| `4c7f7ae` | `feat(speed): show prompt processing's working, term by term` |

1,765 passed, 1 skipped, 11 deselected. Coverage 96.41 percent against a floor of 85.
`ruff check`, `ruff format --check` and `mypy` clean; the four generators leave the tree
unchanged. Seven new messages reach `messages.pot` -- one in the first commit, six in the second --
and are untranslated in all 37 catalogs, which is what they should be.

Both commits verify on their own: the first was built as its own tree, checked, and
committed before the second was restored on top of it, so neither is a half state.

---

## 1. `measured` now means the run described this placement

### What was wrong

`_flags_agree` read a flag a recorded run does not mention as agreement. Its own docstring
said so out loud — "a flag the measurement does not record cannot disagree" — which turns
a benchmark with a short command line into a benchmark of everything.

The catalog carries exactly such a row. Qwen3.8-Flash-Next's winning configuration was
recorded as

```
flags: "-ub 1024 --no-mmproj-offload -ot ffn_.*_shexp=CPU"
```

because the calibration record states the base flags once under "Models" and then names
only what the winning run adds. Nothing in that string says where the layers went, so it
agreed with every placement whose micro-batch was 1024, and with every placement at all
for generation, where `-ub` is not compared. A hybrid split holding two layers on the card
would have been labelled with a benchmark taken with all forty-eight on it, dated, and
called `measured`.

Latent, because no command feeds catalog measurements to `estimate_speed` today and
section 10.3 reserves `measured` for a benchmark taken on *this* machine. Phase 3 stores
benchmarks taken on this machine.

### Which flags are load-bearing, and why those

The rule I used: **a flag is load-bearing when its value changes which bytes are read out
of which pool, or how many tokens they are read for.** That is the same set of inputs
`per_token_traffic` and section 10.2's formula actually consume, so the test is a property
of the arithmetic rather than a taste in flags.

| Flag | What it moves | Absent means |
|---|---|---|
| `-ngl` | how many layers, and with them their experts and their cache, sit on the card | **nothing** — see below |
| `--n-cpu-moe` | routed experts pinned to system memory even on offloaded layers | 0, none pinned |
| `-ot ffn_.*_shexp=CPU` | the always-on shared experts, 239 MB on the reference card | not overridden |
| `--no-mmproj-offload` / `--no-mmproj` | the vision projector, 1.9 GB of an 8 GB card | see the caveat below |
| `-ctk` / `-ctv` | bits per cached element, and the cache is read once per token | `f16` |
| `-ub` (prompt only) | the divisor of the whole prompt formula | `512` |

`-ngl` is compared after clamping to the layer count: 99 and 48 are the same instruction to
a 48-layer model. `-ub` stays generation-blind, as before — the reference machine's own
runs move four percent across micro-batch sizes and a factor of two across offloads.

Everything else a command line carries is not compared: `--temp`, `--top-p`, `-t`/`-tb`,
`-b`, `--jinja`, `--fit off`, `--lazy-mode`, `--alias`. None of them moves a byte between
pools. There is a test that says so by name.

**Why `-ngl`'s silence is different from the others'.** The rest of that table are opt-in
switches or values with a default llama.cpp has always carried, so their absence is a
statement a reader can check: no override, `f16`, `-ub 512`. `llamafit.placement.flags`
already relies on that for `-ctk`, which it deliberately does not render for `f16` so that
a printed line and a recorded line can be compared. `-ngl` has no such reading *inside this
project*: our own calibration record keeps the layer split in a base line and the catalog
rows quoted only the delta of it, so silence there demonstrably means "recorded elsewhere",
not "none". And it is the one flag that separates 78.0 tokens per second from 279.5 for the
same file on the same machine. A run that does not name it describes no placement.

The projector is the one place the record cannot be read in both directions. A run that
passed `--no-mmproj-offload` put the projector in system memory and a run that passed
`--no-mmproj` had none loaded; both say so. A run that passed neither either had no
projector or left it on the card, and the `--mmproj` path that would distinguish them is
not part of what a catalog entry quotes. So silence there settles only the case that
matters: the run was not one that moved a projector into system memory. Written down in
`_projector_agrees`.

### What a mismatched run produces instead

It drops out of the exact list and stays in the ranked list, so it becomes the calibration
anchor: the estimate is `calibrated`, the formula corrected by the ratio the run showed **at
the configuration its own flags describe** (`_as_placement` now reads every load-bearing
flag back, not just three), with a note naming the run and its date. If there is no usable
run at all the estimate is `estimated`. It is never `measured`, and `measured_on` stays
`None`.

One residue, and it gets a sentence of its own on screen. When the anchor run does not
record `-ngl`, `_as_placement` cannot build a different placement to correct against, so the
factor comes out at exactly the measured figure and the "correction" is a substitution
wearing a weaker label. The estimate now says so:

> That benchmark's command line does not record how many layers went to the card, so the
> correction assumes it ran at this placement.

### The catalog row

Fixed at the source as well as in the code. The Flash-Next winning row now records the whole
command line the calibration document describes —
`-ngl 99 --n-cpu-moe 48 --fit off -fa on -t 16 -tb 16 -b 4096 -ub 1024 --no-mmproj-offload
-ot ffn_.*_shexp=CPU` — the calibration document says in item 5 that those two lines are the
same run, and `Measured.flags` now says the field is the whole line and not a delta.

That is what keeps `measured` reachable rather than merely safe. With the row complete, the
reference tests still get `measured` for all four runs, and the two Flash-Next rows are now
told apart from each other: one left the shared experts and the vision projector on the
card, the other did not, they measured 14.5 and 13.9 tokens per second, and neither is a
benchmark of the other's placement.

### Tests

`test_a_benchmark_naming_no_flags_at_all_matches_any_configuration` asserted the bug. It is
now `..._describes_no_configuration` and asserts `calibrated`. Added: the bug in the shape it
was found in (the winning row against a two-layer hybrid), one test per load-bearing flag in
both directions, one that a flag which moves no byte is not compared, one that the two
Flash-Next runs are not benchmarks of each other, and one that `-ub` still does not stop a
generation benchmark from counting.

---

## 2. Prompt processing shows its parts

`SpeedEstimate` grew three fields in the shape the generation trio already had —
`prompt_compute_seconds_per_token`, `prompt_link_seconds_per_token`,
`prompt_ram_seconds_per_token` — one per term of section 10.2. Per prompt token rather than
per micro-batch, so each trio adds up to one over the figure above it and a reader who has
learnt to check one table has learnt to check the other. `render_speed` draws them wherever
it draws the generation table, through a helper both tables share.

### The reference machine's own best configuration

Qwen3.8-Flash-Next UD-Q4_K_XL on the bundled `reference-rtx4060-128gb` profile: mode
`moe-offload`, `-ngl 99`, `--n-cpu-moe 48`, shared experts in system memory, `-ub 1024`,
32,768 tokens of context, `f16` cache. That is the calibration record's winning
configuration, with one honest difference: the planner leaves vision out entirely at this
context, where the recorded run loaded the projector into system memory.

```
13.0 tokens per second generated and 46 read, at 32K tokens of context (estimated).
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ A token's time        ┃ Seconds ┃ Share ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ reading the card      │  0.0295 │   38% │
│ reading system memory │  0.0463 │   60% │
│ everything else       │  0.0010 │    1% │
└───────────────────────┴─────────┴───────┘
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ A prompt token's time                  ┃ Seconds ┃ Share ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ doing the arithmetic                   │  0.0027 │   12% │
│ streaming the experts across the link  │  0.0190 │   88% │
│ reading the experts from system memory │  0.0000 │    0% │
└────────────────────────────────────────┴─────────┴───────┘
1.50 GB of routed experts is read from system memory per token at 0.57 of its 57 GB/s,
because each token selects a different handful of experts and the read is a scatter of
small blocks rather than a stream.
77.0 GB of the expert set is streamed across the link once per micro-batch at 4.0 GB/s.
That rate was fitted on the one model whose expert set does not fit in system memory, so
part of its read comes off the disk; an expert set that stays in the page cache streams
about three times faster, and for one of those this term is that much too slow.
```

0.0027 + 0.0190 + 0.0000 = 0.0217, and 1 / 0.0217 is 46 prompt tokens per second, which is
the figure in the first line.

### Making the known gap visible rather than hiding it

Section 10.2's link constant covers two physical paths with one number. It was fitted on
Flash-Next, whose 111 GB of weights plus a 29 GB streamed table do not fit in 128 GB of RAM,
so part of every micro-batch's expert read comes off the disk at an effective 4 GB/s.
Coder-Next's 50 GB expert set stays in the page cache and streams at about 12.8 GB/s on the
same machine. `test_prompt_processing_is_half_the_truth_for_a_model_that_fits_in_memory`
already asserted the resulting twofold error. I did not try to fix it.

Three things make it visible instead of folding it into a total:

1. The link term is a row of its own, never summed with the arithmetic. On this
   configuration it is **88 percent of the prompt token**, so a reader can see that almost
   the whole answer rests on the constant this project knows least about. On Coder-Next it
   is 81 percent — the model that constant is wrong for.
2. The note under the table names the rate (4.0 GB/s), names why it is that rate, and says
   which direction the error runs and roughly how far: "an expert set that stays in the page
   cache streams about three times faster, and for one of those this term is that much too
   slow."
3. Section 10.2's third term, the same expert set read out of system memory when there is no
   card to stream to, is a separate row and a separate note rather than being silently
   substituted into the second. Before, one variable held either quantity and the estimate
   never said which path it had priced.

A smaller thing found on the way: at four decimals a small dense model's prompt trio prints
`0.0000` three times, and a breakdown that adds up to zero is worse than no breakdown
because it looks like one. The seconds column now picks its decimals from the total, so
Qwen3-0.6B's forty-six microseconds print as `0.000046`. The generation table is unchanged
by this, which a test pins.

---

## 3. Where a number is still printed without a way to check it

Yes — three places, none of them introduced here, all left alone deliberately.

**The two card-side terms carry no note.** "reading the card" and the new "doing the
arithmetic" are numbers with no sentence under them saying what bandwidth, what fp16
figure, and what efficiency produced them, while both system-memory terms and now the link
term do. The asymmetry is not principled; it is where the notes happened to be written. The
fix is two more notes naming `EFF_VRAM` against the card's GB/s and `EFF_PP` against its
TFLOP/s, and it would be worth doing next.

**And the compute constant is known to be wrong, on screen nowhere.**
`test_prompt_processing_is_an_order_out_on_a_small_dense_model` records that Qwen3-0.6B
managed 21,734 prompt tokens per second, which needs 26 TFLOP/s from a card the bundled
table rates at 15, because that 15 is the RTX 4060's fp32 shader throughput and not what its
tensor cores do with fp16 — so `EFF_PP` has been absorbing the difference wherever the
streaming term hid it. A dense model's prompt figure is roughly fourfold low and nothing a
reader sees says so. This is the same class of defect as the link constant and, unlike it, it
has no visible term to hang a note on until the note above exists. Recorded in a test since
the constants work; still not printed.

**The context ladder shows one total per rung with no source marker.** `render_budget` marks
every line exact or modelled, which is the project's own rule; `render_tiers` shows "Card
needs 7.2 GB" per rung with neither a breakdown nor that marker, and those totals contain
the compute buffer, the least certain line in the whole budget. The chosen context's
breakdown is printed beside it, so a reader can reconstruct one rung out of eight or ten.

Two things I checked and found sound: the board's speed column carries a `How` column with
the confidence label, and the quality figure expands into curator baseline, quantisation
penalty and alignment bonus with a pointer to `llamafit info` for the sourced benchmarks
behind the baseline.

---

## Notes for whoever picks this up

- `DEFAULT_MICRO_BATCH` (512) is new in `llamafit.constants.placement`. It exists so that an
  absent `-ub` can be read as a statement rather than a blank, and its docstring says why.
- `_as_placement` now always rebuilds the anchor's micro-batch, batch, cache type, CPU-MoE
  layers and shared-expert pool from the flags. The projector is left alone on purpose,
  because silence about it is genuinely ambiguous; there is a comment saying so.
- Running `ruff format` after `scripts/gen_messages.py` invalidates `messages.pot`, because
  the template records `file:line` for every call site and formatting moves lines. Format
  first, extract second. It cost one confusing test failure.
