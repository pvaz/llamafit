# The ranking defect: where it belongs, and the argument for putting it there

Branch `fix/ranking`, one commit.

| SHA | Subject |
|---|---|
| `9c0f819` | `fix(scoring): exclude a model slower than its reader rather than rank it` |

Tests: 2,907 passed, 2 skipped, 14 deselected; coverage 96.61 percent. `ruff check`,
`ruff format --check`, `mypy` and all three generators clean, `git diff --exit-code` empty
after the commit. One new message in `messages.pot`, untranslated in all 37 catalogs.

Every figure below is a live scan of this machine with bandwidth measured, which is what
`llamafit recommend` itself does. Free VRAM on this card moves by a few hundred megabytes
between runs, so Llama 3.1 8B reads between 9.7 and 10.5 tokens per second and its total
between 50.5 and 51.9 depending on when you look. The before and after boards below are
consecutive runs, not the same run, and none of the conclusions turns on the difference.

---

## 1. Where the fix went

**In the exclusions, in `scoring/rank.py`, as the last check `_exclusion` makes.** A
candidate whose speed falls below the reading floor for a use case somebody reads is
excluded with its number in the reason, keeping its placement and its speed the way every
other exclusion in this program does.

The test lives beside the floor it uses, in `scoring/speed_score.py`:

```python
def keeps_up_with_reader(speed: SpeedEstimate, use_case: str) -> bool:
    floor = floor_tps(use_case)
    return floor <= 0.0 or observed_tps(speed, use_case) >= floor
```

Three properties of that, each deliberate.

**It introduces no new constant.** `READING_TPS = 6.0` was already in the module, already
derived from reading rates — 240 words a minute, three quarters of a word to a token —
and already documented as "the one number in this module that describes the reader and not
the machine". The previous agent put it there and made the ramp start from it. This change
does not add a threshold; it makes an existing one load-bearing. That matters, because the
one thing everybody involved has agreed on is that no constant should be fitted until this
board comes out right, and nothing here was fitted to anything.

**It applies exactly where the floor applies.** `THROUGHPUT_ONLY` already names the use
cases with no reader — embedding — and `floor_tps` already returns zero for them. So the
exclusion can never fire on an embedding request, by construction rather than by a second
list somebody has to keep in step.

**The boundary falls on the side that keeps a candidate.** The comparison is `>=`, not
`>`: a model generating at exactly reading speed is keeping up, by the plain meaning of
the words, and stays on the board with a speed score of zero. Exclusion is the stronger
of the two claims, so it takes the stricter test.

`observed_tps` was extracted while I was there, so the exclusion and the score cannot read
different figures off one estimate — a board that excluded an embedding model on its
generation rate and then scored it on its prompt rate would be disagreeing with itself
with only one of the two numbers on screen.

### The argument against putting it there

**A hard threshold is a cliff, and I made one.** On the slow laptop I constructed
(section 4, machine 5), Qwen3-0.6B generates 5.93 tokens per second and is excluded; at
6.01 it would be ranked with a speed score of about 0.04. Those two machines are
indistinguishable to their owner and the boards they produce are not. I can soften this in
presentation — the excluded row still carries its speed, so nothing is hidden on either
side — but I cannot make it continuous, and I am not going to pretend the discontinuity is
not there. The honest defence is that it is a *stated* cliff: the reason names the boundary
and the reader's distance from it, which is more than the old behaviour offered, where 5.9
and 6.1 both scored roughly zero and the board's order was settled elsewhere without
saying so.

**The floor is a fact about people, and somebody will want to argue with it.** Six tokens
per second is the middle of a five-to-seven band and describes a reader of English prose
at leisure. It is wrong for somebody skimming, wrong for a language whose tokenizer packs
differently, wrong for a person who starts a generation and goes to make tea. There is
currently no way to say so: no `--min-tps`, no config key. I did not build one, for the
reason in concern 2, and I think that is the strongest specific objection to what I have
shipped.

**Some jobs genuinely do not have a reader, and the vocabulary cannot express them.** A
batch summarisation run is a `general` request in this program's language, and a `general`
request now assumes somebody is watching the tokens arrive. `embedding` is the only use
case that escapes, and it escapes because it generates nothing at all rather than because
nobody is reading. The right answer is a use case, or a flag, that says "nobody is
waiting"; until there is one, this rule takes options away from a batch user. It shows
them the options with their numbers, which is not nothing, but it is not the same as
ranking them.

---

## 2. Why not the other two candidates

The previous agent named three places and did not choose. I want to be specific about why
the other two are not it, because "I preferred the third" is not an argument.

The board, before the fix, on this machine:

| | quality (0.35) | speed (0.25) | fit (0.25) | context (0.15) | total | gen |
|---|---|---|---|---|---|---|
| Gemma 3 27B | 75.00 | **0.00** | 54.69 | 100.00 | **54.92** | 2.18 |
| Qwen3-0.6B | 40.00 | 100.00 | 0.00 | 100.00 | 54.00 | 114.62 |
| Llama 3.1 8B | 63.00 | 36.88 | 20.28 | 100.00 | 51.34 | 10.16 |

Gemma leads Llama by **3.58** points: quality is worth 4.20 of that and fit 8.60, against
the 9.22 Llama's speed score earns it.

**A correction to the received framing first.** The previous agent wrote that quality and
fit "outweigh what a perfect speed score can contribute at its weight", comparing 12.7
against 9.2. Those two numbers are not comparable: 9.2 was Llama's *actual* speed
contribution, not a perfect one. A perfect speed score at weight 0.25 is worth 25.00, and
Llama with one would finish at 67.12 and win by twelve points. The speed term is not
structurally outvoted, and I would not have wanted the argument for this change to rest on
a claim that it is.

**What is actually true, and it is the whole argument.** The speed score is *already at
its rail for the model that is the problem*. Gemma scores zero. There is no shape on
`[floor, target]`, no curve and no exponent, that can take it below zero, so section 11.4
has said everything it is able to say and the model still leads. And Llama cannot make up
the difference from the other side, because Llama genuinely is only 1.7 times reading
speed — 10.2 tokens per second against a general target of 25 — and 36.88 is a fair score
for that. Both ends of the range are working correctly and the answer is still wrong. That
is what places the defect outside the score.

### The composite weights

Moving weight from fit to speed does flip this board. It needs **0.051** — general's
weights going from (0.35, 0.25, 0.25, 0.15) to (0.35, 0.301, 0.199, 0.15), a fifth more
weight on speed. The result:

| | before | after the reweighting |
|---|---|---|
| Qwen3-0.6B | 54.00 | **59.10** |
| Llama 3.1 8B | 51.34 | 52.19 |
| Gemma 3 27B | 54.92 | 52.13 |

Three things wrong with it.

**It buys six hundredths of a point.** Llama takes second place from Gemma by 0.06, and
this machine's own scan-to-scan drift is twenty times that. A lever that has to be tuned
to three decimal places to settle a board is not settling anything; it is landing on the
right side today.

**It makes the other defect worse.** Qwen3-0.6B — a 0.6B model with a curator's baseline
of 35 — gains five points and extends its lead to nearly seven, because raising the speed
weight rewards exactly the candidate that is nineteen times reading speed and has nothing
else to offer. See section 5.

**It demotes rather than reclassifies.** Gemma at 2.2 tokens per second finishes third of
three and is still on the ranked list, still presented as a recommendation. On the bundled
reference profile, where Llama already led before any of this work, Gemma at 2.30 tokens
per second sat at number two on a board of three — and a reweighting fitted to the live
board would have left it exactly there. The complaint is not that the candidate was ranked
too high.

### The fit score's capacity arm

I checked this one properly and it does not say what the received framing says. Both arms,
per model (a separate scan, so Llama's fit reads 20.25 here and 20.28 in the table above;
the difference is free VRAM moving):

| | share | capacity | utilisation | crowding | fit (the worse) |
|---|---|---|---|---|---|
| Gemma 3 27B | 0.147 | 59.99 | 0.936 | **54.69** | 54.69 |
| Llama 3.1 8B | 0.044 | **20.25** | 0.914 | 61.99 | 20.25 |
| Qwen3-0.6B | 0.006 | **0.00** | 0.826 | 91.41 | 0.00 |

**Gemma's fit of 54.69 is the crowding arm, not the capacity arm.** The capacity arm gave
it 59.99 and was overridden. What put Gemma above Llama on fit is the observation that its
card is 93.6 percent full against Llama's 91.4 percent — two placements near the paging
cliff, scored about the same for being near it, and neither reading is wrong. So "the
capacity arm rewards a 27B for filling a machine that cannot run it at reading speed" is
not what happened here. **Changing that arm cannot lower Gemma's fit at all.**

It can only raise Llama's, and the required change is large and self-contradicting. Llama
needs its fit to reach **34.6**, up from 20.25, merely to *tie* Gemma at 54.92. Fourteen
points is 0.63 of a halving, which means `WASTE_PER_HALVING` falling from 22.69 to
**18.61** — and at that rate a model occupying a fifth of the machine scores **75.4**,
where section 11.3's own published anchor says 70. The constant is not a free parameter:
it was derived from the specification's two stated points, and re-fitting it to reorder one
board would put the code in contradiction with the paragraph it implements.

And it demotes rather than reclassifies, the same way. Even at that rate the result is a
tie at 54.92 with **Gemma still on the ranked list** and Qwen3-0.6B at 54.00 behind both.

### What actually separates the third candidate from the other two

The first two reorder. The third re-classifies. The complaint is a classification
complaint — "a person waiting on two tokens a second is waiting slower than they can
read" — and reordering does not answer it at any weight, because the model is still on the
ranked list either way, and on two of the five machines I tried it was already second
before anybody touched anything.

This is also the argument section 11.1 already makes, in the specification, in these
words: *"No weighting fixes that, because the defect is that an unsuitable candidate was
allowed to compete at all."* It was written about a model whose entry does not claim the
use case. It is the same shape of fact and it gets the same answer.

---

## 3. What a reader now sees for the model that used to lead

`llamafit recommend --use-case general`, this machine, verbatim:

```
                                         Recommended
┌───┬───────────────────────┬────────┬───────┬───────┬───────┬───────┬─────┬──────┬─────────┐
│ # │ Model                 │ Quant  │ Score │ Gen/s │ Fit   │ Runs  │ Ctx │ Qual │    Card │
├───┼───────────────────────┼────────┼───────┼───────┼───────┼───────┼─────┼──────┼─────────┤
│ 1 │ qwen3-0.6b            │ Q8_0   │  54,0 │ 114,6 │ fits  │ GPU   │ 32K │   40 │ 5,5 GiB │
│ 2 │ llama-3.1-8b-instruct │ Q4_K_M │  51,6 │  10,3 │ tight │ split │ 35K │   63 │ 6,1 GiB │
└───┴───────────────────────┴────────┴───────┴───────┴───────┴───────┴─────┴──────┴─────────┘

                                          Not ranked
┌────────────────────┬────────────┬────────────────────────────────────────────────────────┐
│ Model              │ Quant      │ Why not                                                │
├────────────────────┼────────────┼────────────────────────────────────────────────────────┤
│ gemma-3-27b-it     │ Q4_K_M     │ generates 2,2 tokens per second, below the 6 a person   │
│                    │            │ reads at: a batch tool on this machine and not one to   │
│                    │            │ sit in front of; a smaller model or quantisation would  │
│                    │            │ keep up                                                 │
│ qwen3-coder-next   │ UD-Q4_K_XL │ not a general model; its entry lists coding, so ask for │
│                    │            │ one of those                                            │
│ qwen3.8-flash-next │ UD-Q4_K_XL │ not a general model; its entry lists coding, reasoning, │
│                    │            │ multimodal, so ask for one of those                     │
└────────────────────┴────────────┴────────────────────────────────────────────────────────┘
```

The sentence does four things and I chose each of them: it gives the number (2.2), gives
the boundary (6), names what the model *is* rather than only what it fails ("a batch tool
on this machine"), and says what to change. "On this machine" is load-bearing — the
sentence is a claim about a pairing, not a verdict on Gemma, and machine 4 in section 4
shows the same entry at the same quantisation ranked first.

The row keeps its placement and its speed estimate, so `--explain` still expands it and
the JSON still carries everything that was learned before it was set aside. Nothing
disappeared; it moved from the half of the page that ranks to the half that explains.

---

## 4. Before and after, every use case, five machines

`total (gen tok/s)`. Bold marks a change. `—` marks a use case with no ranked rows.
Machines 1 and 2 are real: a live scan of this machine, and the bundled profile of it.
Machines 3 to 5 are constructed.

### 1. This machine, live scan

| | before | after |
|---|---|---|
| **general** | **gemma 54.9** (2.2) / qwen0.6 54.0 / llama 51.3 (10.2) | qwen0.6 54.0 / llama 50.5 (9.7) / **gemma excluded** |
| coding | coder 84.9 / flash 67.6 / qwen0.6 54.0 | 84.9 / 67.0 / 54.0, unchanged in order |
| reasoning | flash 71.2 / llama 56.7 | 70.6 / 55.9, unchanged in order |
| **chat** | llama 42.7 (10.2) / gemma 41.2 (2.2) | llama 41.5 / **gemma excluded** |
| **multimodal** | flash 78.2 / gemma 56.7 (2.2) | flash 77.4 / **gemma excluded** |
| embedding | (none ranked) | unchanged |

The small movements in the unchanged rows are the scan drift, not the fix: nothing in
`coding` or `reasoning` crosses the floor on this machine.

### 2. Bundled profile `reference-rtx4060-128gb`

Reproducible, and a materially different machine state: 7.6 GB of the card free rather
than about 6, so Llama 3.1 8B fits entirely on it and runs at 34 tokens per second.

| | before | after |
|---|---|---|
| **general** | llama 66.9 (33.9) / gemma 55.1 (2.3) / qwen0.6 54.0 | llama 66.9 / qwen0.6 54.0 / **gemma excluded** |
| coding | coder 89.6 / flash 76.2 / qwen0.6 54.0 | unchanged |
| reasoning | flash 77.6 / llama 55.9 | unchanged |
| **chat** | llama 69.4 / gemma 41.4 (2.3) | llama 69.4 / **gemma excluded** |
| **multimodal** | flash 78.0 / gemma 56.9 (2.3) | flash 78.0 / **gemma excluded** |
| embedding | (none ranked) | unchanged |

Note what this machine shows that the live one does not: **the ordering was already right
here and the defect was still present.** Gemma at 2.3 tokens per second sat at number two
on a board of three for a general request, and at number two for chat and multimodal. A
reweighting that fixed the live board would have left every one of those exactly where it
was, because there was no inversion here to fix.

### 3. A 16 GB laptop, no card, 10 GB free

**Every board unchanged.** Llama 3.1 8B runs at 8.03 tokens per second here — slower than
anywhere else it appears, and above the floor, so it stays. This is the case that would
have caught a rule written as "exclude the big model": nothing about size enters the test.

### 4. A workstation: 24 GB card, 64 GB of memory

**Every board unchanged, and Gemma 3 27B leads general at 80.7 and chat at 82.0**, at 37
tokens per second. The same entry, the same quantisation, the same catalog, first instead
of excluded. This is the evidence that the rule is a fact about a pairing and not a rule
fitted to one board: it fires on the machine, never on the model.

### 5. An old laptop: no card, two cores, single-channel DDR4 at 12 GB/s

This is the uncomfortable one and the reason I built it.

| | before | after |
|---|---|---|
| general | **llama 59.4 at 1.62 tok/s** / qwen0.6 35.0 at 5.93 | **— (both excluded)** |
| coding | qwen0.6 38.8 at 5.93 | **—** |
| reasoning | llama 61.9 at 1.62 | **—** |
| chat | llama 46.8 at 1.62 | **—** |
| multimodal | (none ranked) | unchanged |
| embedding | (none ranked) | unchanged |

Before the change, this machine's general board *recommended a model that answers at one
and a half tokens per second*, and ranked it first — because both candidates scored zero
on speed, so the board was decided by quality and fit alone. That is the same defect as
the reference machine's, worse, and on the machine where it does the most harm: somebody
on an old laptop is the least able to absorb a five-gigabyte download that turns out to be
unusable.

After the change every interactive board on this machine is empty above the line and full
below it, each row carrying its speed and the reason. I am satisfied that is the true
answer and not a regression. Three things support it:

- **An empty ranked board is not a new state.** It already happens on machine 3 for
  multimodal and embedding, and on machine 5 for multimodal, and the renderer handles it.
- **Nothing is hidden.** The excluded table is never cut by `--limit` — that is a
  deliberate rule in `build_board` — so all five models appear with their numbers.
- **The alternative I considered and rejected was a *relative* rule**: exclude a slow model
  only when a fast one exists on the same board. It never empties a board. It also
  restores the defect exactly where it is worst — on machine 5 nothing is above the floor,
  so nothing would be excluded, and the reader is back to being told that 1.62 tokens per
  second is their best option. It would additionally make exclusion depend on what else is
  in the catalog, so adding an entry would silently remove a different one from somebody's
  board, and the reason would become "something else was faster", which is a ranking
  statement dressed as an exclusion. Rejected on all three counts.

---

## 5. Which orderings I would defend, and which I would not

### I would defend

- **Gemma excluded from general, chat and multimodal on machines 1 and 2.** At 2.2 to 2.3
  tokens per second it is a batch tool, the sentence says so with the number, and the row
  is still on the page.
- **Gemma first for general and chat on machine 4.** Same entry, same quant, a card that
  holds it. If the rule had fired here it would have been fitted to the wrong thing.
- **Every board on machine 3 unchanged.** Llama at 8.03 tokens per second is slow and
  usable, and the rule correctly does not care that it is slow.
- **Every coding and reasoning board, everywhere.** Untouched, and they were already the
  boards this project gets right: Qwen3-Coder-Next leads coding on every machine that can
  run it, Qwen3.8-Flash-Next leads reasoning and multimodal.
- **Machine 5's empty boards**, for the reasons above. This is the one I expect to be
  argued with, and I would argue back.

### I would not defend

**Qwen3-0.6B first for general on machine 1, and first for general and coding on machine
3.** A 0.6B model with a curator's baseline of 35 to 40 leading a general request over
Llama 3.1 8B at 63 is not a recommendation I would give a person.

| | quality (0.35) | speed (0.25) | fit (0.25) | context (0.15) | total |
|---|---|---|---|---|---|
| Qwen3-0.6B | 40 → 14.00 | 100 → 25.00 | 0 → 0.00 | 100 → 15.00 | **54.00** |
| Llama 3.1 8B | 63 → 22.05 | 33.6 → 8.41 | 20.3 → 5.08 | 100 → 15.00 | 50.53 |

Llama wins quality by 8.05 and loses speed by 16.59. The 0.6B is nineteen times reading
speed and the 8B is 1.7 times, and only one of those differences is something a person can
experience — but the score caps at the target and the 0.6B is far past it, so it collects
the whole 25 points while Llama collects 8.41 for being comfortably readable. Meanwhile
the fit score's capacity arm, which exists precisely to stop this, gives the 0.6B a zero
and still cannot close the gap, because zero is the floor and Llama's own 20.3 is not much
above it.

**This is not caused by my change and it is not fixed by it.** Qwen3-0.6B already
outranked Llama 3.1 8B before this branch; it sat at number two with Gemma above it. What
my change did was remove the thing that was standing on it. That is worth saying plainly:
settling one defect made a second one visible, and the board a person sees today is more
obviously wrong than the board they saw yesterday, in a different place.

I did not touch it. It is a genuine question about the composite — whether quality at 0.35
should be able to lose to a small model maxing speed and context — and answering it means
either the weights or the capacity arm, which is exactly the ground I have just spent
section 2 arguing should not be moved to reorder a specific board. It wants its own
branch, its own argument, and probably a catalog with more than five entries in it.

**Also unresolved, less severe:** Llama 3.1 8B second for reasoning on machine 2 at 55.9,
with a context score of 53.1 because it holds 17K of a 32K request. That looks right and I
mention it only because it is the one place a context score, rather than a speed or a fit
score, is doing the separating, and nobody has audited section 11.2 the way the other
three have now been audited.

---

## 6. The wording I think section 11 should carry

Three revisions to section 11 are now pending and **none of the earlier two has been
placed**, so the specification's section 11 currently describes neither the fit score nor
the speed score as they are implemented. In placement order:

1. **11.3** — the fit score's two arms. Wording is in `fitscore-report.md`, line 291.
2. **11.4** — the log ramp and the reading floor. Wording is in `speedscore-report.md`,
   under "The section 11.4 wording I think the specification should carry".
3. **11.4, continued** — the floor as a test, below. It is written to append to (2), and
   it replaces (2)'s last paragraph, which described a consequence that no longer follows.

### To append to 11.4, replacing its "What the floor costs" paragraph

> **The floor is a test as well as a scale.** A candidate whose speed falls below it is
> **excluded**, with its speed and the floor in the reason, and not ranked last with a
> score of zero.
>
> Zero was not enough on its own, and the reason is worth stating exactly, because it is
> not that the speed term is outvoted. On the reference machine Gemma 3 27B at 2.2 tokens
> per second led a general board at 54.92 over Llama 3.1 8B at 51.34 — and *the speed
> score was already at its rail*. Gemma scored zero, which is the least this section can
> say about anything; no curve on `[floor, target]` can take it lower. Llama earned 36.88,
> which is a fair score for 1.7 times reading speed against a target of 25. Both ends of
> the range were working correctly and the answer was still wrong. That is what places the
> defect outside the score.
>
> **The claim is about a kind of tool, not a degree of quality.** A model that generates
> more slowly than its reader reads is not a worse interactive model; it is a batch one.
> Ranking it second or third tells a person nothing they can act on, and it kept happening:
> on the bundled reference profile, where the ordering was never inverted, Gemma still sat
> at number two of three for general, chat and multimodal. "Generates 2.2 tokens per
> second, below the 6 a person reads at: a batch tool on this machine and not one to sit in
> front of; a smaller model or quantisation would keep up" says what it is, what decided
> it, and what to change — and, because this program excludes rather than filters, the row
> is still on the page with its placement and its speed attached. This is the argument
> section 11.1 already makes about a model whose entry does not claim the use case, and it
> reaches the same answer.
>
> **The comparison is not strict.** A model generating at exactly the reading rate is
> keeping up, by the plain meaning of the words, and stays on the board with a speed score
> of zero. Exclusion is the stronger of the two claims and takes the stricter test, so the
> boundary case falls on the side that keeps a candidate.
>
> **A throughput use case can never be excluded this way.** The rule applies exactly where
> the floor applies, and embedding has no floor: there is no person waiting on those
> tokens, so there is no rate below which the waiting stops being worth it.
>
> **What the floor costs.** On a machine where nothing reaches six tokens per second, the
> ranked half of an interactive board is empty and every candidate appears in the excluded
> half with its speed. That is a real change and not a free one. It is also the true
> answer: the behaviour it replaces was to order such a machine's candidates by quality and
> fit and present the winner as a recommendation, which on a two-core laptop meant leading
> a general board with a model that answers at 1.6 tokens per second.
>
> **The floor is the default, not a law, and it describes a person.** Six tokens per second
> is the middle of a five-to-seven band for somebody reading English prose as it arrives.
> It is the wrong number for a reader who skims, for a language whose tokenizer packs
> differently, and for anybody who starts a generation and walks away. A request should be
> able to carry its own figure — `Needs.min_tps`, defaulting to the floor, set from
> `--min-tps` or the config file — and the exclusion should then name the figure it used.
> That is not implemented.

### Also worth a line in 11.1

> The exclusions are now in three places and a reader should be told where: **11.1**
> excludes a model whose entry does not claim the use case, or which lacks a required
> capability; **11.2** excludes one that cannot hold `--min-context`; **11.4** excludes one
> slower than its reader. All three are exclusions rather than filters — the candidate
> appears with its reason, its placement and whatever else was learned about it — and all
> three name the thing the reader can change.

### `docs/how-it-works.md` §7, the Speed bullet

This is outside the directories I was given, and it now describes the score two revisions
behind. Replacement text, for whoever places the spec:

> - **Speed**: generation tokens per second, scored between two speeds — the rate a person
>   reads at (about six tokens per second) and a target for the use case (chat 30, general
>   25, coding 20, reasoning 15, multimodal 15), on an axis of doublings rather than of
>   tokens per second. Embeddings are scored on prompt throughput instead, with no reading
>   floor, because nobody reads one. A candidate below the floor is excluded with its speed
>   in the reason rather than ranked: it is a batch tool, not a slow interactive one.
>   Coding and reasoning take a deduction when prompt processing is slow.

### `CHANGELOG.md`, under Unreleased → Fixed

Not placed, for the same reason. The previous agent added no entry either, so this covers
only my change:

> - **A model slower than its reader is excluded from an interactive board, not ranked on
>   it.** The speed score already knew that six tokens per second is the rate a person
>   reads at and already scored everything below it zero, and zero was not enough: on the
>   reference machine Gemma 3 27B at 2.2 tokens per second led a general board at 54.92
>   over Llama 3.1 8B at 51.34 with its speed column reading nothing at all. That is not
>   the speed term being outvoted — it is the speed term at its rail, unable to say
>   anything worse about a model that cannot be used, while quality and fit go on being
>   right about a model nobody can wait for. Reweighting only demotes: on the bundled
>   reference profile, where the ordering was never inverted, the same model still sat at
>   number two of three for general, chat and multimodal. So a candidate below the floor is
>   now excluded with its speed and the floor in the reason, keeping its placement and its
>   estimate the way every other exclusion does, and a throughput job with no reader can
>   never be excluded this way. The rule adds no constant: it reuses the reading floor,
>   which was derived from reading rates rather than fitted to a board, and it fires on the
>   machine rather than on the model — the same entry at the same quantisation leads a
>   general board on a machine with a card that holds it.

---

## 7. Concerns

**1. The general board now leads with a 0.6B model on two of the five machines, and I would
not defend that.** Section 5 has the arithmetic. It predates this branch, it is not caused
by it, and removing what was sitting above it has made it the first thing a person sees.
It is the next defect in this chain and it belongs to the weights or the capacity arm —
which means somebody has to make the argument I have just spent section 2 saying should
not be made casually.

**2. There is no way to disagree with six tokens per second.** No `--min-tps`, no config
key, no "nobody is waiting" use case. A batch summarisation user loses their ranked list
and gets an explained one instead. The spec wording above names the shape of the fix —
`Needs.min_tps` defaulting to the floor — and I did not build it: `Needs` is in `models/`
and the flag is in `cli/`, and I was asked to keep to `scoring/`, `quality/` and
`services/`. A half-built override with nothing able to set it would have been worse than
none. **This is the highest-value follow-up** and it is about half an hour across three
files.

**3. The exclusion depends on an estimate nobody has benchmarked.** `speedscore-report.md`
records that `EFF_PP` and `CPU_PP_TFLOPS_PER_CORE` are each fitted to one run on one
machine, and the generation estimate carries its own uncertainty. Until this branch, a
wrong speed estimate cost a candidate some points; now it can remove one from a list.
Gemma at 2.2 tokens per second is far enough below six that no plausible error changes the
answer, but a model estimated at 5.5 would be excluded on a figure that has never been
measured on the machine it is about. Phase 3's `bench` and the `measured` confidence label
are what make this safe, and they are not here yet. **I would not call this rule finished
until a wrong exclusion can be corrected by a measurement.**

**4. The renderer has nothing to say about an empty board.** On machine 5 a person gets an
empty "Recommended" table with its headers, then the full "Not ranked" table. Every fact
they need is there and the reading order is wrong: the answer is "nothing here can keep up
with you", and the program makes them infer it from an absence. A line under an empty
board — naming the fastest candidate and its speed — would fix it. `cli/render_board.py`
and the two dashboards were not mine this session, and one of them is being worked on in
parallel.

**5. `best_quant` now chooses among readable quantisations.** `llamafit plan <model>` with
no quant named ranks the quants and takes the first, and that ranking now excludes the ones
below the floor. When some are readable this is an improvement — it picks the quant that
will keep up. When none are, `rank()` returns an excluded candidate first and the lookup
falls back to catalog order rather than to the best-scoring quant. Every bundled model
publishes exactly one quantisation, so nothing in this repository exercises either path; a
catalog with three quants of a large model on a small machine would. It is a silent
behaviour change in a command that is not `recommend`, and I am noting it rather than
defending it.

**6. `--prefer speed` and `--prefer quality` do not move the boundary, deliberately.** A
preference leans the scales; it does not change what is physically readable. Somebody who
passes `--prefer quality` on machine 5 might reasonably expect to be shown the highest
quality model that will run at all, and gets an empty board instead. I think that is right
— the preference is about trading between candidates, not about what counts as a candidate
— but it is a judgement and somebody could take the other side.

**7. One test in the suite is flaky and it is not mine.**
`test_tui_app.py::test_simulating_a_bigger_card_puts_the_badge_up_and_re_ranks` failed once
in a full run and passed in isolation and on every subsequent full run. It is an async
Textual pilot test; I did not investigate further, and the dashboard is somebody else's
lane this session.

**8. I touched nothing outside `scoring/` and its tests, plus `services/` tests.** No
`quality/` change was needed. The generated `messages.pot` moved because a new translatable
string exists. Everything I would otherwise have written into `docs/`, `CHANGELOG.md`,
`models/` or `cli/` is in section 6 above, ready to place.
