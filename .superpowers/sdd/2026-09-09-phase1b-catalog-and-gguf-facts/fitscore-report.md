# The fit score measured the wrong thing at one end

Branch `fix/fitscore`, worktree `llamafit-wt-fitscore`, off `main` at `c0fafa5`.

## The defect, restated precisely

Section 11.3 gives the fit score a left-hand slope whose stated purpose is that "a model far
smaller than the machine wastes it". It did not work. On the reference machine — RTX 4060 with
8 GB, 128 GB of DDR5 — `llamafit recommend --use-case general` put Qwen3-0.6B first, and
`llamafit fit`, which is nothing *but* the machine question, called it the second best-fitting
model on a machine with 115 GB of memory.

The reason is that the slope was measured on the worst pool's utilisation, and on a host with a
small card the worst pool is the card, whose contents are mostly not the model:

| Qwen3-0.6B Q8_0, reference profile | bytes |
|---|---|
| dense weights + token embedding, on the card | 0.63 GB |
| KV cache at 40K tokens, on the card | 3.76 GB |
| compute buffer, on the card | 2.45 GB |
| backend context, on the card | 0.13 GB |
| **card total** | **6.97 GB, 90 % of what is free** |

Gemma 3 27B on the same machine also reads 93 %. Ninety percent of a card says nothing about
whether the model is 0.6 GB or 16.5 GB.

There is a second, worse property. The key-value cache is *elastic*: the planner grows the
context until the card is full. So a score read off the total budget is reading the planner's
appetite, and every candidate converges on the same utilisation by construction. Scoring waste
that way could not have worked on any machine; the small card is what made it obvious.

## What the score measures now

Two questions, answered with the worse of them.

**Crowding — untouched.** The whole budget against whichever pool is tightest. 100 up to 0.80,
linear to 40 at 0.98, 0 above. Same shape, same constants, same quantity. Paging remains named
separately by the budget's own `too-tight` verdict.

**Waste — the change.** The model's own *resident weight tensors* against *every byte of memory
the machine has to hold them in*:

```
share = resident weight bytes / (vram_available + ram_available)
```

- Weight lines only: `dense-weights`, `shared-expert-weights`, `expert-weights`,
  `token-embedding`, `output-head`, `global-weights`, `lazy-tables`, `vision-projector`.
- Not the cache, the recurrent state, the compute buffer, the output buffer, the projector's
  compute buffer, the backend context or the process overhead. Each of those is either elastic
  (the planner chose it) or fixed (it costs the same whatever is loaded). None is evidence
  about the size of the model.
- A tensor the planner streams from disk is in the `disk` pool, is not resident, and does not
  count. Qwen3.8-Flash-Next's 28.8 GB n-gram table is correctly excluded: 82.5 GB of its
  111.3 GB is what actually occupies the machine.
- Both pools added. On a machine with a small card most of a large model's weights live in
  system memory, and a denominator of VRAM alone would say a 50 GB model and a 0.6 GB model are
  equally sized. A machine with unified memory charges every line to system memory and reports
  no card, so nothing is counted twice.

The curve over `share`: 100 at 0.50 or above, then **22.7 points off for every halving**,
reaching zero at about 1/42 of the machine.

### Why that rate, and why nothing was tuned

22.7 is not a number I chose. Section 11.3 already priced waste — 100 at 0.50, 70 at 0.20 — and
that is a rate: thirty points for a shortfall of a factor of 2.5, which is 30/log₂(2.5) = 22.69
points per halving. **The new curve passes through both of the specification's own points
exactly.** Two things changed and neither is a constant: the quantity it is a curve of, and the
axis it is linear in.

The axis matters as much as the quantity. A fiftieth of a machine and a hundredth of a machine
are a ratio apart, not a difference apart; a linear-in-share curve compresses every small model
into one indistinguishable heap near the origin and spends all its resolution on the difference
between 40 % and 80 %, where there is nothing to decide. The task's own framing is
multiplicative — "differing by fifty times in what they ask of the machine" — and so is the
score now.

I checked the rate against an independent anchor before adopting it. Reasoning from the shape
of the model landscape rather than from the specification — full marks at half the machine,
nothing at a thirty-second of it, because a factor of 32 spans the entire ladder from a 1 GB
tiny model to a 100 GB frontier one — gives 25 points per halving. The two derivations land 10 %
apart. I took the specification's, because it is already written down and because a number I
can point at in a document nobody fitted to this catalog is worth more than one I derived this
afternoon.

### Why the floor is gone

The old module held the score at 70 below 0.20, arguing that a tiny model is still the right
answer for a smoke test or on a machine where nothing else fits. That argument was compensating
for a quantity that did not know how large the machine was. `share` does. The same 0.639 GB of
Q8 weights scores:

| machine | capacity | share | capacity score |
|---|---|---|---|
| laptop, no card, 1.2 GB free | 1.2 GB | 0.53 | **100** |
| laptop, no card, 12 GB free | 12.5 GB | 0.051 | 24.8 |
| reference, 8 GB card + 128 GB | 115 GB | 0.0055 | **0** |
| workstation, 8 GB card + 512 GB | 516 GB | 0.0012 | 0 |

The "machine where nothing else fits" case takes care of itself, and it does so without a
constant. A zero costs a candidate a fifth to a quarter of its composite and never its place on
the board — nothing is dropped for fitting badly.

### Why `min` and not a product or an average

A configuration that both wastes the machine and crowds the card should not have either
complaint softened by the other, and a reader shown one number deserves the complaint that is
actually biting. The `--explain` sentence now names both quantities:

> Its weights hold 15.4 GiB of this machine's 104.3 GiB and the tightest pool is 94 % full. The
> score wants at least 50 % of the memory taken and no more than 80 % of any one pool.

## Rankings, before and after

Two boards. The **profile board** is deterministic — the bundled `reference-rtx4060-128gb`
profile, 8.0 GB card free and 107 GB of system memory free — and is the one to compare against
later. The **live board** is this machine as it actually is right now, with a browser and an
editor holding memory; its free VRAM moves between runs, so before and after were computed in
one process against one scan to make the difference attributable to the score alone.

### Profile board (deterministic)

| Use case | Before | After |
|---|---|---|
| general | llama-3.1-8b **75.8**, **qwen3-0.6b 70.7**, gemma-3-27b 57.4 | llama-3.1-8b **66.9**, gemma-3-27b 57.4, **qwen3-0.6b 54.0** |
| coding | coder-next 89.6, flash-next 76.2, qwen3-0.6b 67.3 | coder-next 89.6, flash-next 76.2, qwen3-0.6b **54.0** |
| reasoning | flash-next 77.8, llama-3.1-8b 63.0 | flash-next 77.8, llama-3.1-8b 55.9 |
| chat | llama-3.1-8b 78.3, gemma-3-27b 44.5 | llama-3.1-8b 69.4, gemma-3-27b 44.5 |
| multimodal | flash-next 78.3, gemma-3-27b 60.0 | flash-next 78.3, gemma-3-27b 60.0 |
| embedding | nothing ranked (no seeded model declares it) | unchanged |

`llamafit fit`, the machine question alone:

| | Before | After |
|---|---|---|
| 1 | qwen3-coder-next 74.0 | qwen3-coder-next 74.0 |
| 2 | **qwen3-0.6b 66.7** | gemma-3-27b 55.6 |
| 3 | gemma-3-27b 55.6 | qwen3.8-flash-next 51.0 |
| 4 | llama-3.1-8b 55.1 | llama-3.1-8b 19.4 |
| 5 | qwen3.8-flash-next 51.0 | **qwen3-0.6b 0.0** |

The 0.6B goes from second to last on the pure machine question, and Qwen3.8-Flash-Next — which
holds 82.5 GB of the machine's 115 and is held back only by a card that is 95 % full — moves
from last to third. Nothing else on that board moves for a bad reason.

### Live board (one scan, both curves)

| Use case | Before | After |
|---|---|---|
| general | **qwen3-0.6b 76.9**, llama-3.1-8b 62.2, gemma-3-27b 57.0 | gemma-3-27b 57.0, **qwen3-0.6b 54.0**, llama-3.1-8b 51.8 |
| coding | coder-next 84.9, **qwen3-0.6b 72.3**, flash-next 67.2 | coder-next 84.9, flash-next 67.2, **qwen3-0.6b 54.0** |
| reasoning | flash-next 71.0, llama-3.1-8b 66.1 | flash-next 71.0, llama-3.1-8b 57.7 |
| chat | llama-3.1-8b 52.9, gemma-3-27b 43.9 | gemma-3-27b 43.9, llama-3.1-8b 42.5 |
| multimodal | flash-next 77.9, gemma-3-27b 59.4 | unchanged |

### Would I defend the new order?

**Yes for the thing this task was about.** Qwen3-0.6B leaving first place for general, and
second place for coding, is right and I would argue it with anybody. Its fit score is now 0 on
a machine that could hold a hundred and eighty times more model, and 0 is the honest number:
this configuration is not using this machine at all. It still scores 54 and still appears —
which is also right, because it is the correct answer for a draft model, a smoke test, or an
autocomplete sidecar, and the board says so by ranking it rather than hiding it.

**A small model still wins where it should.** On a laptop with no card and 1.2 GB of memory
free, Qwen3-0.6B is the only thing that runs, scores **100** on fit, and takes the general board
outright at 66.4. On a 16 GB laptop it scores 24.8 while Llama 3.1 8B scores 89.3 and wins. The
same model, the same weights, three different verdicts, decided by the machine. That is the
property I most wanted and the one the old floor was faking.

**One inversion I am less comfortable with, and it is not hidden.** On the *live* scan — where
the card is already partly occupied and Llama 3.1 8B is forced into a split placement at
9.7 tok/s — the change moves Gemma 3 27B, at **2.1 tok/s**, above it for general and for chat.
Fit is doing exactly what it should (15.4 GiB of the machine against 4.6 GiB) and the composite
is still wrong for a person, because 2.1 tokens per second is slower than reading. I do not
think that is a fault in section 11.3, and I did not fix it in section 11.3, because fixing it
there would mean smuggling speed into the fit curve and making two rows of the breakdown table
move for one reason. It is a fault in section 11.4: `speed_score` is linear to target, so
2.1 tok/s against a target of 25 scores 8.4 out of 100 rather than something near zero, and at
a weight of 0.25 that leaves eight points on the table for a model nobody can use. A knee or a
floor in the speed score is the honest place to look next. On the deterministic profile, where
Llama runs at 33.9 tok/s, the inversion does not occur and Llama wins general and chat
comfortably.

## Answers to the three questions

### 1. What does the fit score now measure, and why is that the right quantity?

The worse of two ratios:

- **crowding** = total budget / tightest pool available — how close this configuration is to
  the point where the driver starts paging and the speed collapses silently. Unchanged.
- **waste** = resident weight bytes / (VRAM available + RAM available) — how much of the machine
  the *choice of model* commits, expressed in halvings.

The waste ratio is the right quantity for three reasons.

It is the only part of the budget that is a property of the model rather than of the planner or
of the runtime. The cache is whatever context the planner asked for; the compute and output
buffers are functions of the micro-batch and the vocabulary; the backend and process overheads
are constants. A measure built on any of those measures LlamaFit's own settings.

Its denominator is the actual constraint. "This model wastes the machine" means "this machine
could have held much more model", and what a machine can hold is its memory — all of it, in
both pools, because weights genuinely live in both. It deliberately does *not* discount system
memory for being slow: how fast a placement runs is the speed score's question, it has a weight
of 0.15 to 0.45 in every use case, and encoding it twice would make the breakdown table lie
about why a candidate lost.

It is relative to the machine, which makes it portable. The same 0.6 GB model is 100 on a
1.2 GB laptop and 0 on a 115 GB desktop without any per-machine constant, and that is what lets
the score keep meaning something on hardware nobody has run it on.

### 2. Which existing tests changed, and was each one asserting the defect or something real?

**`tests/unit/test_scoring_parts.py`**

| Test | What happened | Was it real? |
|---|---|---|
| `test_the_fit_curve_at_every_threshold_and_either_side` | Split. The rows for 0.80 → 0.98 → above became `test_the_crowding_arm_at_every_threshold_and_either_side` **with identical expected values**. The rows below 0.50 — `(0.0, 70)`, `(0.199, 70)`, `(0.201, 70.1)`, `(0.35, 85)`, `(0.499, 99.9)` — were the defect written down as a table: they assert that a pool one-fifth full scores 70 whatever is in it. Replaced by `test_every_halving_of_the_model_costs_the_same` over `share`. | Right arm real and preserved; left arm was the defect |
| `test_the_curve_punishes_both_ends_and_not_the_middle` | Removed. Its payload was `fit_score(0.10) == SPARSE_SCORE` — the floor. | The defect |
| `test_the_cliff_above_the_crowded_threshold_is_a_cliff` | Kept, retargeted at `crowding_score`. Same assertions. | Real |
| `test_fit_is_decided_on_whichever_pool_is_worst` | Kept verbatim. A budget at 0.99 of system memory still scores 0. | Real |
| `test_a_machine_with_no_card_is_scored_on_system_memory_alone` | Kept; expected value still 100, now for two reasons rather than one (60 GiB of weights is 60 % of the machine *and* the pool is not crowded). A comment says so. | Real |

Eleven tests were added: the two named specification points, the per-halving rate, ratio-not-
difference behaviour, the absence of a floor, `min` of the two arms, weight lines versus buffer
lines, streamed tensors, both pools added, a placement with no weights, a machine reporting no
free memory, the 0.6B-versus-large regression at equal pool fullness, and machine-relativity.
`fit_score.py` is at 100 % line and branch coverage.

**`tests/unit/test_rank.py`**

`test_overriding_the_weights_changes_the_order` asserted `ids(machine_first)[0] == "qwen3-0.6b"`
under weights of speed 0.475 / fit 0.475. **That assertion was the defect, written into a test
as an expectation.** Its *intent* — that a weight override actually changes the order rather
than being quietly ignored — is real and is kept: the override now weights fit alone, and the
board it produces is led by Qwen3.8-Flash-Next, the model that holds 82 GB of the machine's 115.
A new test, `test_the_machine_question_alone_puts_the_smallest_model_last`, guards the defect
directly at the level where it appeared.

**`tests/fixtures/scoring.py`**

Not a test, but worth naming: every fixture budget charged its whole pool requirement to one
component called `weights`, so the fixtures modelled every placement as though all of it were
the model. That is not asserting the defect, but it is why the defect was invisible from the
fixtures. `budget()` and `placement()` now take `weight_vram` and `weight_ram`, defaulting to
the whole pool, and the five seeded placements carry the splits the real budget produces against
the reference profile. Three of them also gained more accurate system-memory totals as a
consequence (Qwen3-Coder-Next 44 → 49.8 GB, Qwen3.8-Flash-Next 75 → 81 GB, Gemma 3 27B
11 → 17 GB), which brings them closer to what `llamafit plan` actually reports.

**Nothing else changed.** 1,765 tests pass, 96.4 % coverage.

### 3. On what kind of machine would this give a worse answer than the old one?

**A machine so large that the whole catalog is beneath its notice.** On 512 GB of system memory
the capacity arm saturates at 0 for everything under about 12 GB of weights, so Llama 3.1 8B
and Qwen3-0.6B both score 0 on fit and the score stops separating them. The old curve gave
everything below a fifth of a pool a flat 70, which was equally uninformative, but it was
uninformative in a way that left the *other* three scores to decide. The new one is the same
in effect — a constant contributes nothing to an ordering — but it is a harsher-looking
constant, and on a very large machine `llamafit fit` will show a run of zeros that reads as a
verdict when it is really "none of these is in this machine's class". A person with a 512 GB
server does want to be told that, but they may want it told once rather than five times.

**A machine where every candidate is slow and the only difference is size.** This is the live
inversion above, generalised: an 8 GB card with a great deal of system memory, where dense
models above about 10 GB run from the memory bus at single-digit tokens per second. The waste
arm now actively prefers the largest of them, and the only counterweight is a speed score that
is too generous at the bottom of its range. Under the old curve, worst-pool utilisation
happened to be roughly flat across those candidates, so quality and speed decided and speed's
forgiveness did less damage. The new arm is measuring the right thing and is making a real
weakness in section 11.4 visible; on this machine, today, it costs one place on the general and
chat boards.

**A machine with a large card and little system memory.** A 48 GB card with 16 GB of RAM has a
capacity of about 62 GB, so a 24 GB model is 0.39 and scores 92. That is correct, but note the
denominator counts 16 GB of system memory the placement may not be able to use for weights at
all if the mode is GPU-only. The crowding arm catches the consequence — the card fills and the
score falls — so the composite does not go wrong, but the *reported* share is optimistic on such
a host. It is the one place where "every byte of memory the machine has to hold them in" is a
slight overstatement.

**Nowhere else that I found.** The change is monotone in the model's size and monotone in the
machine's size, which is more than the quantity it replaced could say.

## The wording section 11.3 should carry

> ### 11.3 Fit score
>
> The fit score asks two questions and answers with the worse of them, because neither complaint
> excuses the other.
>
> **Is it crowded?** From the whole budget on whichever pool is worst, `u`: 100 for `u ≤ 0.80`;
> linear down to 40 at `u = 0.98`; 0 above. A configuration filling 95 percent of a card is one
> browser window away from paging into system memory, where the speed collapses with nothing
> raising an error. That a configuration *will* page is not merely scored here, it is named: the
> budget's own verdict says `too-tight`.
>
> **Does it waste the machine?** From the model's own resident weight tensors over every byte of
> memory the machine has to hold them in,
> `w = resident weight bytes / (vram_available + ram_available)`: 100 for `w ≥ 0.50`, falling by
> `30 / log₂(2.5) ≈ 22.7` points for every halving below it — so 70 at `w = 0.20`, and 0 at
> about `w = 0.024`. A tool that scored only on safety would recommend the smallest model that
> runs, every time, and tell a person with 128 GB of memory to run a 0.6B. The user did not buy
> the machine to leave it idle.
>
> Weights only, and only the resident ones. The key-value cache and the recurrent state grow
> with whatever context the planner chose; the compute and output buffers follow the micro-batch
> and the vocabulary; the backend context and the process overhead are the same bytes whatever
> is loaded. None of them says anything about how large the model is, and the cache is worse
> than uninformative because the planner grows it until the pool is full, so a score read off
> the total budget scores the planner's appetite. A tensor streamed from disk is not resident
> and does not count. Both memory pools are added, because on a machine with a small card most
> of a large model's weights live in the second; a machine with unified memory charges every
> line to system memory and reports no card, so nothing is counted twice.
>
> **This is not the quantity the first revision of this section named, and the difference is the
> whole point.** Read on worst-pool utilisation, waste cannot be seen at all on a host with a
> small card: a 0.6B at Q8 puts 0.6 GB of weights on an eight-gigabyte card and 6 GB of cache
> and buffers on top, so the card reads 90 percent full — and so does a 27B. The price of waste
> is unchanged from that revision: thirty points for a shortfall of two and a half times, and
> the curve still passes through 100 at 0.50 and 70 at 0.20 exactly. What changed is the
> quantity, and that the interpolation is in halvings, because a fiftieth of a machine and a
> hundredth of one are a ratio apart rather than a difference apart.
>
> **There is no floor.** The first revision held the score at 70 below 0.20 on the grounds that
> a small model is still the right answer where nothing else fits. `w` is relative to the machine
> by construction, so that case needs no special rule: 0.6 GB of weights is more than half of a
> laptop with 1.2 GB free and scores 100 there, and nothing at all on a workstation that could
> hold forty times more. A zero costs a candidate a fifth to a quarter of its composite and never
> its place on the board — nothing is dropped for fitting badly, and a small model can still win
> on the other three parts, which is how a draft model or a smoke test gets recommended.

## What changed in the tree

| File | Change |
|---|---|
| `src/llamafit/scoring/fit_score.py` | `crowding_score`, `capacity_score`, `model_share`, `resident_model_bytes`, `MODEL_COMPONENTS`, `WASTE_PER_HALVING`; `fit_score` takes both quantities |
| `src/llamafit/scoring/__init__.py` | exports |
| `src/llamafit/cli/render_board.py` | `--explain` names both quantities and the bytes behind them |
| `src/llamafit/cli/board_cmd.py` | `--perfect` help, which described the old single band |
| `src/llamafit/data/locale/messages.pot` | regenerated, two messages changed |
| `tests/fixtures/scoring.py` | weight splits per placement |
| `tests/unit/test_scoring_parts.py` | the fit section rewritten, 11 tests added |
| `tests/unit/test_rank.py` | the override test's expectation, plus a direct regression guard |
| `CHANGELOG.md`, `docs/cli.md`, `docs/how-it-works.md` | the change, the `--perfect` description, and the two example boards re-run |

Both changed messages are new English, so the thirty-seven catalogs fall back to English for
them until somebody translates them; no existing translation was silently reused for a different
sentence.

## Verification

```
pytest --cov -m "not hardware and not network"   1765 passed, 1 skipped, 96.39 % (floor 85 %)
ruff check .                                     All checks passed
ruff format --check .                            214 files already formatted
mypy                                             Success: no issues found in 102 source files
```
