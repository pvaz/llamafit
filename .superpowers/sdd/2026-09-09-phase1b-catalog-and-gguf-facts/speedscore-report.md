# Speed score and prompt estimate

Branch `fix/speedscore`, two commits.

| SHA | Subject |
|---|---|
| `e6bd889` | `fix(scoring): score a model nobody can wait for at zero, not a fraction` |
| `f965604` | `fix(speed): correct the GPU table's fp16 column and refit prompt compute` |

Tests: 1,801 passed, 1 skipped, 11 deselected; coverage 96.46 percent. `ruff check`,
`ruff format --check`, `mypy` and all three generators clean, `git diff --exit-code` empty.

---

## 1. What shape the speed score has, and what it says about a person

### The shape

```
speed_score = 100 x log2(tps / floor) / log2(target / floor)      clamped to [0, 100]
```

with the target unchanged per use case and a **fixed floor of six tokens per second** for
every use case whose output a person reads. Embedding keeps the old straight line to zero.

Three claims, and they are separable, which is why I state them separately.

**The floor exists.** Silent reading of prose runs at roughly 240 words a minute — four
words a second, and a token is about three quarters of a word, so five to seven tokens per
second. A model slower than that cannot keep up with somebody reading its own output. It
is not a slow model; it is a model you wait for, one sentence at a time. Six is the middle
of a band, not a measurement of anybody in particular, and the docstring says so.

**The floor does not move with the use case.** It is the only number in the module that
describes the reader rather than the machine. A person reading a reasoning trace reads at
the same rate as a person reading a chat reply. What changes between the two is how much
faster than that they want it, and that is what the target already says.

**Between floor and target the axis is doublings, not tokens per second.** A person does
not experience tokens per second, they experience the multiple of their own reading rate,
and the steps in that multiple are not equal in tokens per second. Six to twelve is the
difference between waiting and not waiting; twenty-one to twenty-five is a difference
nobody can feel. A straight line prices them the same. Section 11.3's capacity arm already
makes exactly this argument about the model against the machine — "a fiftieth of a machine
and a hundredth of a machine are a ratio apart and not a difference apart" — and the same
reasoning applies here for the same reason, so the composite now has one idea in it rather
than two.

I checked the alternative honestly. A straight line from the floor to the target also
sends 2.1 tokens per second to zero, and it scores Llama 3.1 8B's 9.7 at 19.5 instead of
33.7. The log axis is not chosen because it flatters this machine's board — on the
inversion that prompted the work it separates the two candidates by 1.5 composite points
more than the straight line does, which changes nothing. It is chosen because 19.5 is the
wrong thing to say about a model running at one and a half times reading speed, and
because the perceptual claim behind it is the one the project already accepted next door.

**Embedding gets no floor at all**, and that is not an exemption, it is the same rule read
correctly. The floor is the speed at which the person waiting stops getting an experience.
Nobody reads an embedding; there is no person waiting on those tokens, only more work done
or less. So its score keeps the straight line to the origin, and `floor_tps("embedding")`
returns zero.

### What it says about a person

That below reading speed there is nothing to have a fraction of. Eight out of a hundred
implied a person was getting eight percent of a good experience from Gemma 3 27B at 2.1
tokens per second — three and a half minutes for a five-hundred-token answer. They are
getting none of it. Above the floor the score says how far along the road from unusable to
sufficient the model has got, measured in doublings of the reader's own rate, and past the
target it stops, because nobody experiences the difference.

### What it costs, named rather than hidden

Every candidate below six tokens per second now scores zero, so on a machine where nothing
reaches six the speed column stops separating candidates at all and the board is ordered
by quality, fit and context. I think that is the right answer to the wrong question —
when nothing is interactive, "which is fastest" is not what needs settling, "which is
worth the wait" is — but it is a real behaviour change on a slow machine and not free.
It is in the module docstring, not only here.

---

## 2. What the prompt estimate now predicts for this machine's four measured runs

| Run | Measured | Before | After |
|---|---|---|---|
| Qwen3-0.6B Q8_0, `-ngl 0` | 2,924 | 3,750 (1.28x) | **2,933 (1.00x)** |
| Qwen3-0.6B Q8_0, `-ngl 99` | 21,734 | 3,750 (0.17x) | **21,643 (1.00x)** |
| Qwen3.8-Flash-Next UD-Q4_K_XL, `-ub 1024` | 49.4 | 46.2 (0.93x) | **51.4 (1.04x)** |
| Qwen3-Coder-Next UD-Q4_K_XL, `-ub 2048` | 323 | 140 (0.43x) | **166 (0.51x)** |

Three of four inside four percent. The fourth is the link constant, which is documented as
known-wrong for a reason this change does not touch: `EFF_PCIE` was fitted on the one
model whose expert set does not fit in system memory, and Coder-Next's stays in the page
cache. It moved from 0.43 to 0.51 of the truth only because the compute term it sits
beside got smaller.

The two dense runs are exact by construction — each constant is fitted to one of them —
so their agreement is not evidence that the constants generalise. It is evidence only that
they are no longer derived one way and spent another. Flash-Next at 1.04 is the one number
here that was not fitted and did improve.

### What the new constants rest on

**`EFF_PP` = 0.43** rests on **one run**: Qwen3-0.6B Q8_0, every layer on the card, 21,734
prompt tokens per second. It is the only measurement this project has in which the compute
term is the whole of the prompt formula — nothing streams, nothing comes out of system
memory — so it reads the product `tflops_fp16 x eff_pp` off directly. At the
`2 x active_params x ub` the formula charges, that is 26.1 TFLOP/s achieved against a
table figure of 60.4, so 0.43.

It is fitted the way the formula spends it, against the catalog's `active_b`, which
matters: Qwen3-0.6B's 0.6 billion includes a tied embedding table prompt processing barely
multiplies by, so the real arithmetic is below 26.1 TFLOP/s and the real silicon
efficiency below 0.43. That is the same distinction that produced the 0.36 error in
section 10.1 — an apparent end-to-end rate is not a coefficient inside a sum of terms —
and this is the coefficient.

**I did not use the 121 TFLOPS figure in the brief, and I think it would have repeated the
defect.** An RTX 4060 has 3,072 shaders at 2.46 GHz, so 15.11 TFLOPS fp32; Ada's tensor
cores do fp16 at four times the shader rate with fp32 accumulate, which is **60.4 TFLOPS
dense**. 121 is that figure doubled by structured sparsity, which llama.cpp cannot use —
it is also, by coincidence, the card's dense INT8 rate and NVIDIA's "242 AI TOPS" halved.
Taking 121 would have put a sparsity-inflated number in a table a dense kernel divides by:
a constant derived one way and used another, which is the exact class of error this branch
exists to remove. Against 60.4 the implied efficiency is 0.43 rather than the 0.22 the
brief expected. I would rather be wrong out loud here than quietly right.

**One thing 0.43 quietly absorbs and should not carry for ever**: llama.cpp runs a
*quantised* matrix multiply, which on this card goes to the integer tensor cores at twice
the fp16 rate, and the formula counts no attention arithmetic at all. Both are in the
constant's docstring.

**The disagreement I could not resolve.** The old 0.30 came from the slope of a
`llama-bench` micro-batch sweep of Qwen3-Coder-Next (118 / 194 / 323 tokens per second at
`-ub` 512 / 1024 / 2048). Re-read against the corrected table, that slope implies about
**5 TFLOP/s** achieved where the dense run says **26** — a factor of five. The slope is
not a clean read of this constant: that model's experts stream across the link, and its
three points do not lie on a line to better than 13 percent (fitting the 512/1024 pair
gives 1.84 ms per prompt token, the 1024/2048 pair 1.03, the 512/2048 pair the 1.30 the
old docstring quoted). The dense run is the one that isolates the term, so it is the one
the constant is fitted to. I have not explained the factor of five, I have only said which
of the two measurements is entitled to set the constant. It is in the docstring as an open
disagreement, and it is the strongest argument for a second machine.

Both new constants are marked the way this project marks a figure that wants one: in the
docstring, in the prose the estimator prints in its notes, and in a test whose name says
what it does and does not prove.

### The third defect, which the refit exposed and could not ship without

Fixing the table and the constant alone would have made the estimate **worse** for most
real board rows, and provably so. The compute term was charged to the card whatever the
placement said. With an honest tensor figure, Gemma 3 27B — of which this card holds nine
layers out of sixty-two — came out at about **483 prompt tokens per second**, which needs
roughly 22 TFLOP/s out of a CPU the same machine has been measured at 3.5. That is
impossible in exactly the way 26-from-15 was impossible, and the machine's own second
measurement proves it.

So the term is now split by the share of the layers each device holds, because prompt
arithmetic runs where the weights are. `--n-cpu-moe` is deliberately not counted as a CPU
layer: prompt processing streams those experts back across the link and multiplies them on
the card, which is what section 10.2's second term already charges for.

The CPU rate that split needs was already in the codebase as `CPU_FP16_TFLOPS_PER_CORE =
0.05`, "a fifth of an AVX2 core's fp32 peak", arrived at by argument — and then multiplied
by `eff_pp` on the way out, so the rate the formula actually spent was a fiftieth of that
core's peak and about twentyfold below what this machine has been measured doing. Nothing
caught it because the term only ever fired on a host with no card at all. It is now
`CPU_PP_TFLOPS_PER_CORE = 0.44`, effective rather than peak, not multiplied by `EFF_PP`,
and fitted to the second dense run: 2,924 prompt tokens per second at `-ngl 0 -t 8` is
3.51 TFLOP/s across eight performance cores. A figure above that core's fp32 peak is not
the contradiction it looks like — a Q8_0 matrix multiply on a Raptor Lake core runs on
integer dot-product instructions, not fp32 fused multiply-adds — but it is one machine,
one thread count and one quantisation, and every other architecture inherits it by
division.

Gemma's prompt figure ends at 74.5 tokens per second, near the 83 it had before the
change, for a defensible reason instead of by two errors cancelling.

I would rather have shipped two fixes than three. This one is not separable: the second
fix creates the impossible number that the third removes, and leaving it for a later
branch would have meant shipping a figure I could prove wrong from data already in the
repository.

---

## 3. Which cards changed, and how much I trust each figure

The column mixed **three** conventions, which is the deeper defect: it had no single
meaning, so no efficiency fitted against one row transferred to another.

| Rows | Held | Now | Rule | Confidence |
|---|---|---|---|---|
| RTX 50 series (6) | fp32 shader | dense fp16 tensor, fp32 accumulate | x4 | **High** |
| RTX 40 series (9) | fp32 shader | dense fp16 tensor, fp32 accumulate | x4 | **High** |
| RTX 30 series + A6000 (6) | fp32 shader | dense fp16 tensor, fp32 accumulate | x2 | **Medium-high** |
| Apple M1–M4 (15) | fp32 shader | packed fp16 ALU rate | x2 | **Medium** |
| Apple M5 (3) | fp32 shader | packed fp16 ALU rate | x2 | **Low** |
| A100, H100 (2) | dense fp16 tensor | unchanged | — | **High** |
| RX 7000/9000 (5) | fp16 | unchanged | — | **Medium** |
| Arc A770, B580 (2) | fp16 | unchanged | — | **Low, and see below** |

Thirty-nine of forty-eight rows changed. The full before/after list is in the commit.

**Ada and Blackwell, high.** Verified against NVIDIA's own whitepaper convention on two
independent anchors. The Ada whitepaper gives the RTX 4090 330.3 TFLOPS of fp16 tensor
with fp32 accumulate against 82.58 TFLOPS fp32 — exactly four times. For Blackwell, the
RTX 5090's advertised 3,352 AI TOPS is fp4 with sparsity, which halves to fp4 dense,
halves to fp8, halves to fp16 dense = 419 TFLOPS, and 419 is four times its 104.8 fp32.
Two families, same ratio, checked from opposite ends.

**Consumer Ampere and the A6000, medium-high.** GA10x halves fp16 tensor throughput when
the accumulate is fp32, which Ada removed — so these are x2 where the newer parts are x4.
That is a real hardware difference and not an inconsistency, but it is also the choice
most exposed to which accumulate mode llama.cpp's kernels actually request: if they take
fp16 accumulate on these parts, the true figure is twice what I have written. I chose the
conservative one. The A6000 is GA102 and gets the same treatment; NVIDIA's marketing
"309.7 TFLOPS" for that card is fp16-accumulate *and* sparse, which is four times what I
put in the table.

**Apple, medium.** Apple GPUs have no separate matrix unit before M5 and issue fp16 at
twice the fp32 rate — M1 Max at 10.4 fp32 and 20.8 fp16 is the commonly published pair,
and the table's existing values were exactly the fp32 figures, which is what made the
convention identifiable. llama.cpp's Metal backend uses fp16 `simdgroup_matrix`
operations, so x2 is the right ceiling for it. What I am less sure of is whether a single
`EFF_PP` transfers: 0.43 was fitted against a *tensor-core* peak four times the shader
rate, and Metal reaches a much larger fraction of a peak only twice the shader rate, so
Apple prompt figures will read low. That is a limitation of one efficiency, not of these
rows.

**Apple M5, low, and it was low before I touched it.** M5 puts Neural Accelerators in each
GPU core and Apple claims up to four times M4's peak GPU compute for AI. I have no primary
figure for M5, let alone for an M5 Pro or Max, and the values in the table were already
estimates. I applied the same x2 as the rest for consistency; if the neural accelerators
behave as advertised these three rows are low by roughly another factor of two. They
should be treated as placeholders and are the first rows to replace when a figure exists.

**A100 and H100, unchanged and already right.** 312 is the A100's dense fp16 tensor rate
(624 with sparsity) and 989 the H100 SXM's (1,979 with sparsity). The only two rows in the
table that were already the right kind of number are the two nobody in this project's
audience owns, which is a good illustration of how the error survived.

**AMD, unchanged, medium.** All five rows are already twice their fp32 figure, which for
RDNA3 is both the packed-fp16 rate and the WMMA matrix rate — 7900 XTX at 61.4 fp32 and
122.8 fp16, table says 123. I did not touch them. The RX 9070 XT is RDNA4, which doubled
matrix throughput relative to RDNA3, and its 97 may therefore be low by about a factor of
two; AMD's published figure for that part is an fp8-with-sparsity number I could not
divide down to a dense fp16 rate with confidence, so I left it. **Said rather than
fitted.**

**Intel, unchanged, low, and I believe the B580 is wrong.** The A770's 39 matches Intel's
published 39.3 TFLOPS fp16 for its XMX engines. The B580's 29 is twice its fp32 shader
rate, which is the *shader* fp16 rate and not the XMX one; Intel publishes 233 INT8 TOPS
for that part, which implies an XMX fp16 rate of either 58 or 117 depending on the INT8:FP16
ratio, and the A770's own published pair implies 4:1, so 58. I could not confirm the ratio
from a primary source for Battlemage and did not want to put a guess in a table whose whole
problem was numbers that meant something other than what they said. **Left alone,
recorded here.**

---

## The section 11.4 wording I think the specification should carry

> ### 11.4 Speed score
>
> ```
> speed_score = 100 × log2(gen_tps / floor_tps) / log2(target_tps / floor_tps)
> ```
>
> clamped to 0 and 100, with targets by use case: chat 30, general 25, coding 20,
> reasoning 15, multimodal 15, embedding 200 (prompt throughput instead of generation).
>
> **The score runs between two speeds, and they are different kinds of number.** The
> target is a property of the job and is where the score stops: past the point where a
> person stops waiting, more speed buys nothing, so forty tokens per second in a chat
> scores what thirty-two does. The floor is a property of the *person* and is where the
> score starts: silent reading runs at roughly 240 words a minute, and a token is about
> three quarters of a word, so **six tokens per second**, the middle of a five-to-seven
> band. A model slower than that cannot keep up with somebody reading its own output.
>
> **The floor does not vary by use case.** A person reading a reasoning trace reads at the
> same rate as a person reading a chat reply. What varies between them is how much faster
> than reading they need it to be, which is what the target already says. Embedding is the
> single exception and has no floor: it is scored on prompt throughput because an
> embedding run generates nothing, and nobody reads an embedding, so there is no speed at
> which the experience stops existing — only more work done or less. Its curve is the
> straight line `100 × min(1, pp_tps / 200)`.
>
> **Between the floor and the target the score is linear in doublings, not in tokens per
> second.** A person experiences the multiple of their own reading rate, and the steps in
> that multiple are not equal in tokens per second: six to twelve is the difference
> between waiting and not waiting, twenty-one to twenty-five is a difference nobody can
> feel, and a straight line prices them the same. This is the axis section 11.3 already
> chose for the capacity arm, for the same reason.
>
> **An earlier revision ran the straight line all the way to zero tokens per second, and
> it could not be defended.** Gemma 3 27B generating 2.1 tokens per second scored 8.4 out
> of 100 for a general request and ranked above Llama 3.1 8B at 9.7 on the same machine. A
> person waiting three and a half minutes for a five-hundred-token answer is not getting
> eight percent of a good experience; they are getting none of it.
>
> **What the floor costs.** Every candidate below it scores zero, so on a machine where
> nothing reaches six tokens per second the speed term stops separating candidates and the
> board is ordered by quality, fit and context. When nothing is interactive, "which is
> fastest" is not the question a person needs settled; "which is worth the wait" is.
>
> Prompt throughput adds a modifier for coding and reasoning, unchanged: −10 when
> `pp_tps < 100`, −20 when below 40, replacing rather than stacking.

---

## The general-request ranking, before and after

**Live scan, this machine, `llamafit recommend --use-case general`.** Both boards computed
from the same estimates so the comparison is not live-scan jitter; free VRAM moves a little
between runs, which is why the generation figures differ slightly from the brief's.

| | Before | After |
|---|---|---|
| 1 | gemma-3-27b-it **56.94** (2.18 tok/s, speed 8.74) | gemma-3-27b-it **54.75** (2.18 tok/s, **speed 0.00**) |
| 2 | qwen3-0.6b 54.00 (114.6 tok/s, speed 100) | qwen3-0.6b 54.00 (114.6 tok/s, speed 100) |
| 3 | llama-3.1-8b-instruct 52.31 (10.16 tok/s, speed 40.63) | llama-3.1-8b-instruct 51.37 (10.16 tok/s, speed 36.88) |

**The speed score is fixed and the ranking is not.** This is the first concern below and I
want it stated plainly rather than buried: Gemma at two tokens a second still leads a
general board.

For `chat`, where speed carries 0.40 of the weight instead of 0.25, the order **does**
flip: gemma 43.91 / llama 43.14 before, llama 42.68 / gemma 41.00 after. `coding`,
`reasoning` and `multimodal` keep their order; nothing was reordered wrongly anywhere.

On the bundled reference profile (reproducible, no live scan) the general board is
llama 66.91 / gemma 55.14 / qwen 54.00 both before and after — that profile has 7.6 GB of
free VRAM, so Llama 3.1 8B fits entirely on the card and runs at 34 tokens per second.
The inversion is a property of the machine as it is right now, with a browser open.

---

## Concerns

**1. The composite still ranks a two-token-per-second model first for general use, and
that is now demonstrably not the speed score's fault.** Speed scores it zero. Gemma's lead
comes from quality (75 against 63, worth 4.2 composite points at weight 0.35) and fit
(55.6 against 21.7, worth 8.5 points at weight 0.25) — 12.7 points against the 9.2 that a
*perfect* speed score can contribute at weight 0.25. No shape on `[floor, target]` can
close that: I checked the straight-line-above-floor alternative and it closes 1.5 points
less than the one I chose. The fix belongs in one of three places, none of them section
11.4, and I did not go there because doing so would have been tuning a term until this
machine's board came out the way I wanted:
  - **Section 11.5's weights** for general (0.35 / 0.25 / 0.25 / 0.15), which let quality
    and fit outvote speed two to one.
  - **Section 11.3's capacity arm**, which rewards a 27B model for filling a machine that
    cannot run it at reading speed. Filling the machine and using it are not the same
    thing, and only the first is currently measured.
  - **Section 11's exclusions**, which already drop a model whose entry does not claim the
    use case. "Slower than its reader on this machine" is arguably the same kind of fact:
    not a worse answer, not an answer. That is my preference, and it is a spec change.

**2. `EFF_PP` and `CPU_PP_TFLOPS_PER_CORE` are each fitted to exactly one run on exactly
one machine**, and between them they now set every prompt figure the tool prints. The two
runs are the best kind available — the same small dense model with all of its traffic in
one place — but two points do not make a model. The `llama-bench` slope on Coder-Next
implies a card rate five times lower than the dense run does, and I have not explained
that. A second machine, ideally a different vendor, is the single highest-value
measurement this project could take next.

**3. The layer split assumes `-ngl` is the whole story.** It divides the prompt arithmetic
by the share of transformer blocks on the card. It does not know that the output head
stays in RAM until every layer is offloaded, that llama.cpp may schedule some operations
differently, or that a partially offloaded model pays a transfer cost per micro-batch that
the formula charges nowhere. It is much better than charging everything to the card; it is
not a model of what llama.cpp does.

**4. `EFF_PP` is a tensor-core efficiency applied to non-tensor-core hardware.** On Apple,
where the fp16 peak is only twice the shader rate and there are no tensor cores before M5,
0.43 will under-predict; the same is true for RDNA. One constant is doing the work of a
per-backend family of them, and the first Apple or AMD prompt measurement will show it.

**5. `EFF_PCIE` is untouched and still known wrong**, for the reason the constant's own
docstring gives. Coder-Next's prompt prediction moved from 0.43 to 0.51 of the measured
figure only because the compute term beside it shrank; the link term is still fitted on
the one model whose expert set will not fit in memory. Residency, not the link, is the
real variable, and modelling it needs a disk-bandwidth probe.

**6. Three prompt-facing card terms still print without a note**, which the honesty report
flagged before this branch and which this branch has now made more urgent: the compute
term reports one number that is the sum of two devices' work. There is a note saying what
share went to the CPU and at what rate, but `SpeedEstimate` still carries a single
`prompt_compute_seconds_per_token` field. Splitting it into two would touch `models/` and
`cli/`, which were not mine this session.

**7. Documents outside my lane now name superseded figures**, and I left them alone
deliberately:
  - `docs/specs/2026-09-09-llamafit-design.md` §10.2 — the prompt formula, which no longer
    has one compute term; and §11.4, whose replacement wording is above for you to place.
  - `docs/calibration/2026-09-09-reference-machine.md` — the constants table, which should
    gain the two prompt runs and the two rates they identify, and the note that the
    `llama-bench` slope disagrees with the dense run by a factor of five.
  - `CHANGELOG.md` — nothing added.

**8. The B580 and the RX 9070 XT are probably still wrong**, in the direction of being too
low, for reasons in section 3 above. I could not confirm either from a primary source and
chose to leave a stale number rather than add a guessed one to a table whose entire
problem was numbers meaning something other than what they said.
