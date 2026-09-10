# The use-case gate: replacing it with the capability it was standing in for

Branch `fix/use-case-gate`, four commits, off `main` at `d4c3342`.

| SHA | Subject |
|---|---|
| `8ed7a85` | `fix(scoring): gate a request on what a model can do, not on what it is for` |
| `bfe6bd0` | `feat(i18n): retire the use-case exclusion's message and translate what replaced it` |
| `1ddfb1d` | `docs(catalog): rest the flagship's general and chat on the model, not on a board` |
| `9b0a161` | `docs: say that a board filters on capabilities and never on use cases` |

Tests: 2,933 passed, 2 skipped, 14 deselected; coverage 96.63 percent against a floor of 85.
`ruff check`, `ruff format --check` and `mypy` clean; all three generators run to a clean
`git diff --exit-code`. Every one of the 37 catalogs reports zero problems against the
regenerated template.

Nothing outside `src/llamafit/web/` was touched, no other branch or worktree, no merge,
rebase or push.

Every board below is the bundled reference profile — RTX 4060 8 GB, 128 GB DDR5 with 100 GB
free, i9-14900KF — through `build_board`, which is the same path `llamafit recommend` takes.
It is deterministic, unlike a live scan, so the before and after columns are comparable to
the second decimal.

---

## 1. What changed

`CAPABILITY_FOR_USE_CASE` in `models/catalog.py`, immediately below the two vocabularies it
bridges:

```python
CAPABILITY_FOR_USE_CASE: Mapping[str, Capability | None] = MappingProxyType(
    {
        "general": None,
        "coding": "coding",
        "reasoning": "thinking",
        "chat": None,
        "multimodal": "vision",
        "embedding": "embeddings",
    }
)

_UNDECIDED = set(get_args(UseCase)) ^ set(CAPABILITY_FOR_USE_CASE)
if _UNDECIDED:  # pragma: no cover - a developer error, caught the moment it is written
    raise RuntimeError(...)
```

It is in `catalog.py` rather than in the scoring that reads it because that is where somebody
adding a seventh use case is standing when the question arises, and the symmetric-difference
check is what turns the question into an answer they have to give: the module refuses to
import until they do. `None` counts as an answer. Without the check an unlisted use case
would quietly require nothing, which is the most permissive result available and the least
likely to be the one intended — exactly the failure mode of a table that can be added to by
forgetting.

`quality/alignment.py` loses `declares_use_case` and gains `required_capability(use_case)`.
`scoring/rank.py`'s first exclusion check is now the capability the job needs, ahead of the
capability the request named for itself.

**Two reasons, not one, because they have different fixes.** A capability the request named
is answered by shortening the request; a capability the job needs is answered by asking for a
different job, or by correcting the entry. Collapsing them into one sentence would have meant
offering at least one reader the fix that does not work:

- `no coding capability, which coding needs; ask for a different use case, or add it to the entry when the model really has it`
- `no vision capability; drop it from the request to see this model` (unchanged)

The job's capability is reported first, being the broader statement about the same thing —
and, where a request names the same capability its use case already implies, the one whose
suggested fix actually clears the exclusion.

---

## 2. Every use case's board, before and after

Reference profile, best quant per model, no other filters.

### general — **the regression, fixed**

| Before | | After | |
|---|---|---|---|
| 1. `llama-3.1-8b-instruct` Q4_K_M | 66.91 | 1. **`qwen3-coder-next`** UD-Q4_K_XL | **82.08** |
| 2. `qwen3.8-flash-next` UD-Q4_K_XL | 66.28 | 2. `llama-3.1-8b-instruct` Q4_K_M | 66.91 |
| 3. `qwen3-0.6b` Q8_0 | 54.00 | 3. `qwen3.8-flash-next` UD-Q4_K_XL | 66.28 |
| — `qwen3-coder-next` — *not a general model; its entry lists coding* | | 4. `qwen3-0.6b` Q8_0 | 54.00 |
| — `gemma-3-27b-it` — 1.6 t/s, below the reading floor | | — `gemma-3-27b-it` — 1.6 t/s, below the reading floor | |

The fastest thing that fits the machine now leads the board it was being kept off. No score
moved; the only change is that a candidate stopped being excluded.

### coding — **unchanged ranking, better reasons**

| Before | | After | |
|---|---|---|---|
| 1. `qwen3-coder-next` UD-Q4_K_XL | 88.45 | 1. `qwen3-coder-next` UD-Q4_K_XL | 88.45 |
| 2. `qwen3.8-flash-next` UD-Q4_K_XL | 72.26 | 2. `qwen3.8-flash-next` UD-Q4_K_XL | 72.26 |
| 3. `qwen3-0.6b` Q8_0 | 54.00 | 3. `qwen3-0.6b` Q8_0 | 54.00 |
| — `llama-3.1-8b-instruct` — *not a coding model; its entry lists general, chat, reasoning* | | — `llama-3.1-8b-instruct` — *no coding capability, which coding needs* | |
| — `gemma-3-27b-it` — *not a coding model; its entry lists general, multimodal, chat* | | — `gemma-3-27b-it` — *no coding capability, which coding needs* | |

This is the board the original rule was written for, and it is identical. Same two models
ranked out, same order among those left, a reason that names an ability instead of a list.

### reasoning — **narrowed by one; see §5**

| Before | | After | |
|---|---|---|---|
| 1. `qwen3.8-flash-next` UD-Q4_K_XL | 73.72 | 1. `qwen3.8-flash-next` UD-Q4_K_XL | 73.72 |
| 2. `llama-3.1-8b-instruct` Q4_K_M | 55.85 | — `llama-3.1-8b-instruct` — *no thinking capability* | |
| — `qwen3-coder-next` — *not a reasoning model* | | — `qwen3-coder-next` — *no thinking capability* | |
| — `qwen3-0.6b` — *not a reasoning model* | | — `qwen3-0.6b` — *no thinking capability* | |
| — `gemma-3-27b-it` — *not a reasoning model* | | — `gemma-3-27b-it` — *no thinking capability* | |

### chat

| Before | | After | |
|---|---|---|---|
| 1. `llama-3.1-8b-instruct` Q4_K_M | 69.36 | 1. **`qwen3-coder-next`** UD-Q4_K_XL | **77.20** |
| 2. `qwen3.8-flash-next` UD-Q4_K_XL | 57.45 | 2. `llama-3.1-8b-instruct` Q4_K_M | 69.36 |
| — `qwen3-coder-next` — *not a chat model* | | 3. `qwen3-0.6b` Q8_0 | 58.75 |
| — `qwen3-0.6b` — *not a chat model* | | 4. `qwen3.8-flash-next` UD-Q4_K_XL | 57.45 |
| — `gemma-3-27b-it` — 1.6 t/s, below the reading floor | | — `gemma-3-27b-it` — 1.6 t/s, below the reading floor | |

### multimodal — **unchanged ranking, better reasons**

| Before | | After | |
|---|---|---|---|
| 1. `qwen3.8-flash-next` UD-Q4_K_XL | 72.84 | 1. `qwen3.8-flash-next` UD-Q4_K_XL | 72.84 |
| — `llama-3.1-8b-instruct` — *not a multimodal model* | | — `llama-3.1-8b-instruct` — *no vision capability* | |
| — `qwen3-coder-next` — *not a multimodal model* | | — `qwen3-coder-next` — *no vision capability* | |
| — `qwen3-0.6b` — *not a multimodal model* | | — `qwen3-0.6b` — *no vision capability* | |
| — `gemma-3-27b-it` — 1.6 t/s, below the reading floor | | — `gemma-3-27b-it` — 1.6 t/s, below the reading floor | |

Gemma is the one model in the catalog that can see. It is excluded here for speed, not for
the request, before and after.

### embedding — **empty before, empty after**

All five excluded, `not a embedding model; its entry lists …` becoming `no embeddings
capability, which embedding needs; …`. The catalog has no model with the `embeddings`
capability, so an empty board is the honest answer; the difference is that the reason now
names the thing a curator would have to add.

Incidentally, the ungrammatical "not a **embedding** model" is gone with the message that
carried it.

---

## 3. Which model is still excluded from which request, and is each one right?

Nine exclusions survive, in four groups.

**Coding: `llama-3.1-8b-instruct` and `gemma-3-27b-it`, for no `coding` capability. Right,
and this is the whole point of the exercise.** Llama 3.1 8B is the model that started this:
it ranked second for a coding request on fit and a capped speed score. It declares neither
the coding use case nor the coding capability, so the outcome that was wanted is preserved
and it now rests on the fact rather than on the wording. Gemma 3 27B likewise.

**Multimodal: `llama-3.1-8b-instruct`, `qwen3-coder-next`, `qwen3-0.6b`, for no `vision`
capability. Right, and not really arguable.** A model with no image encoder cannot answer a
question about an image. This exclusion was already correct before the change; only its
sentence improved.

**Embedding: all five, for no `embeddings` capability. Right.** None of the five is an
embedding model. Before the change they were excluded for the same reason wearing different
clothes.

**Reasoning: `llama-3.1-8b-instruct`, `qwen3-coder-next`, `qwen3-0.6b`, `gemma-3-27b-it`, for
no `thinking` capability. Right on the rule, and it is the one exclusion I want to put in
front of you.** Three of the four were excluded before as well. The new one is Llama 3.1 8B,
which lists `reasoning` in its `use_cases` and does not have `thinking` in its
`capabilities`. Under the rule you specified, reasoning implies thinking, and Llama 3.1 8B is
genuinely not a chain-of-thought model. It is consistent with the rest of the specification:
§11.2 gives reasoning a 32K default "for a long chain of thought", and §11.5 weights its
quality at 0.50 "because thinking tokens are cheap to wait for". Asking for reasoning is
asking for a model that thinks, and the reasoning board is now one model long on this
machine. See §5 for what I would and would not do about that.

Two further exclusions are the reading floor rather than the request: `gemma-3-27b-it` on
general, chat and multimodal, at 1.6 tokens per second. Unrelated to this change and
unchanged by it.

---

## 4. Was the curation change a workaround, and what did I do about it?

**Half of it was, and I removed that half. The entries stay; the argument for them does
not.**

`d4c3342` added `general` and `chat` to `qwen3.8-flash-next` and gave two reasons in the
comment. The second — "the omission was visible from the outside, because a general request
on the reference machine excluded the two strongest models in the catalog and returned the
two weakest" — is a fact about a board the gate had emptied. It is gone with the gate and it
cannot justify anything now. That sentence is a workaround, and I deleted it.

The first — "this is the vendor's flagship instruction-tuned model, not a specialist" — is a
claim about the model, and it survives on its own. Three things persuaded me to keep the
entries rather than revert them:

1. **The catalog was internally inconsistent without them.** Every other generalist in the
   catalog says so in its own entry: `gemma-3-27b-it` is `[general, multimodal, chat]`,
   `llama-3.1-8b-instruct` is `[general, chat, reasoning]`, `qwen3-0.6b` is
   `[general, coding]`. A 125B flagship instruction-tuned MoE with thinking, vision, tools
   and a 262K context, listing neither, was out of step with its peers whether or not any
   rule read the field.
2. **There is a live consumer that has nothing to do with ranking.** `llamafit list
   --use-case general` filters on `use_cases` in `services/catalog.py`, and that filter is
   correct and is staying: there the reader has explicitly asked to browse what entries offer
   themselves for. Omitting `general` hides the strongest general model in the catalog from
   somebody who asked to see general models. That harm exists independently of this branch.
3. **It cannot be a workaround for a board any more, because it changes no board.** The bonus
   is for the primary use case alone, which is still `coding` at position 0. Nothing gates on
   the list. Verified by trimming the entry back to `[coding, reasoning, multimodal]` and
   rebuilding: all five non-empty boards come out identical, model for model and score for
   score.

The test I applied was yours — would a curator with no gate in existence write this? — and
the answer is yes for the fact and no for the evidence sentence. The rewritten comment says
only what is still true, names what does read the field, and repeats the line I would not
want lost: *a specialist stays a specialist; Qwen3-Coder-Next is coding only and keeps it.*

One consequence worth naming: the entry now lists five of the six use cases, which is close
to listing everything. That is defensible for this particular model and it would not be for
most. If it starts happening to a second entry, the field is drifting from emphasis towards
decoration and is worth a rule.

---

## 5. Which tests changed, and was each asserting the rule or something real?

Nine touched. Four were asserting the rule and had to be re-aimed; five were asserting
something real and only needed their expected string or their incidental leader corrected.
Four new tests were added.

### Asserting the rule (re-aimed at the capability)

| Test | Was | Now |
|---|---|---|
| `test_quality.py::test_a_model_declares_the_jobs_it_offers_itself_for` | asserted `declares_use_case(llama, "coding")` is false | **deleted**, replaced by four tests of the mapping and one — `test_the_capability_a_job_asks_for_is_read_off_the_model_not_its_use_cases` — that pins the distinction: Llama lists `reasoning` and lacks both `coding` and `thinking`, and it is the second fact that decides |
| `test_rank.py::test_only_the_models_that_offer_themselves_for_the_job_compete` | ranked list plus "Llama used to place second" | renamed `…_that_can_do_the_job_compete`; same ranked list, and now also asserts `"coding" not in llama.capabilities`, which is the fact doing the work |
| `test_rank.py::test_a_model_that_does_not_offer_itself_for_the_job_is_kept_with_its_reason` | `"general, chat, reasoning" in reason` — the list of jobs | renamed `…_that_cannot_do_the_job…`; asserts the ability is named *and* that both fixes are offered |
| `test_services_recommend.py::test_a_model_built_for_another_job_is_excluded_and_says_so` | `"chat" in reason`, "the use cases its entry does list" | renamed `…_that_cannot_do_the_job…`; asserts `no coding capability` and the fix |

### Asserting something real (string or incidental leader corrected)

| Test | What it is really about | Change |
|---|---|---|
| `test_rank.py::test_an_embedding_request_never_excludes_a_model_for_generating_slowly` | that a throughput job has no reading floor | expected reason `not a embedding model` → `no embeddings capability` |
| `test_rank.py::test_being_too_slow_is_reported_after_everything_the_request_controls` | check ordering: the request before the machine | expected reason updated |
| `test_rank.py::test_what_the_request_asked_for_is_reported_before_what_the_machine_can_do` | check ordering with a candidate failing several ways | now asserts the job's capability is reported and the *named* capability's fix is **not** offered — a sharper assertion than before, since the two reasons are distinguishable |
| `test_render_board.py::test_the_exclusions_carry_the_reason_beside_the_model` | that the reason reaches the rendered table | expected string updated |
| `test_services_recommend.py::test_the_general_board_no_longer_leads_with_a_model_slower_than_its_reader` | the reading floor at the service layer | pinned `rows[0] == "llama-3.1-8b-instruct"`, which is a fact about the catalog rather than about the floor. Now asserts the slow model leads nothing and is off the ranked half, which is what the test is named for |
| `test_rank.py::test_the_same_board_reorders_when_the_request_changes` | that the request decides the order | the old demonstration (chat leads with Llama, coding with the coder) stopped demonstrating anything once both lead with the coder. Re-aimed at general versus chat with the *same four candidates*: Llama at 33.9 t/s is below Flash on general and above it on chat, which is chat's 0.40 speed weight doing visible work, and is a better demonstration than the original |

### New

- `test_rank.py::test_a_model_built_for_one_job_still_competes_for_another` — the regression
  itself: `qwen3-coder-next.use_cases == ["coding"]`, it is not excluded from a general board,
  and it leads it.
- `test_rank.py::test_a_general_or_chat_request_excludes_nobody_for_what_it_asks` — no
  candidate on either board carries a capability reason.
- `test_services_recommend.py::test_a_model_built_for_another_job_still_competes_for_this_one`
  — the same, at the service layer.
- `test_quality.py` — four tests on the mapping: the four implications, the two that imply
  nothing, that the mapping is exhaustive over `UseCase`, that every value is a real
  `Capability`, and that an unknown use case requires nothing rather than everything.

One test I deliberately did **not** restore: `test_the_slow_model_is_not_dropped_from_the_board_it_is_moved_to_the_end`
still asserts `ids[0] != "gemma-3-27b-it"` rather than naming a leader. `d4c3342` softened it
for the right reason and the reason still holds.

---

## 6. The wording section 11.1 should carry now

Replacing the two paragraphs that begin **"A model is excluded when the requested use case is
not in its `use_cases` at all"** and **"The exclusions are in three places"**. The
`alignment_bonus` bullet above them is unchanged and still correct.

> **A model is excluded when it lacks a capability the request needs**, with the reason
> naming the capability. A request names one two ways: outright, in `Needs.capabilities`, and
> by naming a job. A requested use case implies a capability wherever one exists — coding
> implies `coding`, reasoning implies `thinking`, multimodal implies `vision`, embedding
> implies `embeddings` — while `general` and `chat` imply none, because they are the absence
> of a specialisation rather than a specialisation of their own. The table is
> `CAPABILITY_FOR_USE_CASE`, beside `UseCase` and `Capability` in `models/catalog.py`,
> exhaustive over the six and refusing to load until a seventh has been decided about; a use
> case with no entry would silently require nothing, which is the most permissive answer
> available and the least likely to be the one anybody meant.
>
> **The gate is on `capabilities` and never on `use_cases`, and the difference between them
> is the point.** `use_cases` says what a model is *for*, which is a curator's emphasis;
> `capabilities` says what it can *do*, which is a fact about the weights. An earlier
> revision of this section excluded a model whose `use_cases` did not contain the requested
> one, and it answered its own case at the cost of a commoner one. It was written for the
> seeded Llama 3.1 8B, which declares neither the coding use case nor the coding capability
> and ranks second for a coding request on the reference machine, carried there by fit and a
> capped speed score — no weighting fixes that, because the defect is that an unsuitable
> candidate was allowed to compete at all. But it also left Qwen3-Coder-Next, the fastest
> model that fits that machine, off a general board entirely, on the grounds that its entry
> lists coding; and a coding model is a perfectly reasonable thing to hold a general
> conversation with. Gating on emphasis throws away right answers, and it quietly promotes
> every curator's judgement call into a hard filter they did not know they were setting.
> Gating on the capability answers both cases at once: a coding request still excludes Llama
> 3.1 8B, now for the reason that was always the real one, and a general request asks
> Qwen3-Coder-Next for no capability at all, so nothing excludes it and it is ranked on its
> merits.
>
> Two reasons come out of this rather than one, because they have different fixes. "no
> `coding` capability, which coding needs; ask for a different use case, or add it to the
> entry when the model really has it" is answered by changing the job or the entry; "no
> `vision` capability; drop it from the request to see this model" is answered by shortening
> the request. The job's capability is reported first, being the broader statement about the
> same thing. Either way, when a model really can do something its entry does not claim, the
> fix is a one-line catalog change, which is the kind of correction this project wants to be
> easy.
>
> **`use_cases` keeps its jobs and loses only this one.** It still earns `+5` when the
> request names the model's primary use case, so a model built for a task still outranks one
> that merely can do it; and `llamafit list --use-case` still browses on the whole list,
> which is a reader asking to see what an entry offers itself for rather than a board
> deciding who is allowed to compete.
>
> The exclusions are in three places and a reader should be told where: **11.1** excludes a
> model that lacks a capability the request needs, whether the request named it or the use
> case implied it; **11.2** excludes one that cannot hold `--min-context`; **11.4** excludes
> one slower than its reader. All three are exclusions rather than filters — the candidate
> appears with its reason, its placement and whatever else was learned about it — and all
> three name the thing the reader can change.

**One sentence elsewhere refers back and needs a word changed.** Section 11.4, in the
paragraph beginning "**The claim is about a kind of tool, not a degree of quality**", ends:

> This is the argument section 11.1 already makes about a model whose entry does not claim
> the use case, and it reaches the same answer.

which should become:

> This is the argument section 11.1 already makes about a model that lacks a capability the
> request needs, and it reaches the same answer.

I have not touched `docs/specs/` — both of the above are yours to place.

---

## 7. Concerns

**1. The reasoning board is one model long on this machine, and that is new.** Llama 3.1 8B
was ranked second for reasoning before and is now excluded for lacking `thinking`. The rule
is right and the exclusion is right; what it exposes is a catalog entry claiming a job whose
ability it does not have. Two honest answers, and I took neither because neither is mine to
take: either `llama-3.1-8b-instruct` should stop listing `reasoning` (its entry is
over-claiming, and now says so out loud), or the `reasoning` use case means something looser
than chain-of-thought and the implication should be dropped to `None`. I lean strongly to the
first — §11.2's 32K default and §11.5's 0.50 quality weight both describe a thinking model —
but it is a curation decision with a visible board behind it, so it should be yours. It costs
nothing to leave as it is: the exclusion names `thinking`, which is exactly the thing a
curator would change.

**2. A coding model now leads the chat board.** `qwen3-coder-next` at 77.20 over
`llama-3.1-8b-instruct` at 69.36, on a use case that weights speed at 0.40 while the coder
generates at 18.7 tokens per second against Llama's 33.9. It wins on quality, fit and
context. This follows directly from your own argument — a coding model is a fine thing to
talk to — and I believe it is correct, but it is the least intuitive result on any of the six
boards and it is the one somebody will query. If it turns out to be wrong, the fault is in
chat's weights rather than in this rule: 0.40 on speed is not buying much when Llama's 33.9
t/s is already past chat's target of 30 and both candidates sit at or near the rail.

**3. `use_cases` now has no enforcement anywhere in ranking, and unenforced fields rot.** Its
two remaining readers are the `+5` primary-use-case bonus, which only ever looks at index 0,
and the `llamafit list --use-case` browse filter. Nothing checks positions 1..n against
anything. Llama's `reasoning`-without-`thinking` is the first instance of a claim that is now
purely decorative — a catalog validation warning ("this entry lists a use case whose implied
capability it does not have") would catch the next one, and would have caught this one. I did
not add it: it is a new rule, not a correction, and you asked for a correction.

**4. The reason is long.** 137 characters of msgid, against the 80 of the message it replaces, and it
wraps to three lines in the `Not ranked` column at the width `docs/cli.md` shows. It names
both fixes because collapsing them would have meant offering one reader the wrong one. It
could be shortened by dropping "or add it to the entry when the model really has it", at the
cost of the correction path this project wants to keep easy. I chose length.

**5. The `pragma: no cover` on the exhaustiveness guard.** The `raise` cannot execute while
the table is correct, so it is one uncovered statement. Coverage is 96.63 percent and the
pragma matches five existing uses in the codebase, but it is worth knowing that the guard
itself is proved only by `test_every_use_case_has_been_decided_about`, which asserts the same
condition from the outside rather than by triggering the raise.

**6. `docs/cli.md`'s `recommend` example still carries scores from a live machine.** I updated
only the two exclusion reasons and the paragraph about them, and left the ranked rows alone:
their figures (84.8, 23.7, 67.2, 13.5) match neither the reference profile nor this branch,
and were already stale before it. Correcting them means re-capturing the whole block against
a named machine, which is a separate job and one somebody should do deliberately.
