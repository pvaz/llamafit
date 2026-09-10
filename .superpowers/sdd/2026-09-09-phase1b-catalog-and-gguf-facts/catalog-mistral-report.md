# Catalog: Mistral AI's open-weight range

Branch `catalog/cat-mistral`, worktree `llamafit-wt-cat-mistral`. Twelve models in six
family files, all refreshed, the whole suite green.

## Status

Complete. Nothing left half-done; two families I set out to add were dropped for reasons
given below, and both are recoverable if a source appears.

## Commits

| SHA | Subject |
|---|---|
| `f9087b1` | test: make three board assertions independent of catalog size |
| `98fafd0` | feat(catalog): add Mistral Small 3.2 24B and Mistral Small 4 119B |
| `125cd5b` | feat(catalog): add the Ministral 3 family and Ministral 8B 2410 |
| `86ce870` | feat(catalog): add Magistral Small 1.2 |
| `ec5163d` | feat(catalog): add Devstral Small 2 24B Instruct |
| `ea97ea3` | feat(catalog): add Mistral NeMo 12B Instruct |
| `01232f1` | feat(catalog): add Mistral 7B Instruct v0.3 |

## Tests

2925 passed, 2 skipped, 14 deselected, coverage 96.65% (floor 85). `catalog validate` clean
over 10 files, `catalog refresh --check` reports no staleness, ruff check and format clean,
mypy clean over 164 files, `gen_schema.py` and `gen_models_md.py` leave no diff.

## What went in

| id | baseline | licence |
|---|---|---|
| `devstral-small-2-24b-instruct` | 82 | Apache-2.0 |
| `mistral-small-4-119b` | 80 | Apache-2.0 |
| `ministral-3-14b-reasoning` | 77 | Apache-2.0 |
| `magistral-small-1.2` | 75 | Apache-2.0 |
| `ministral-3-14b-instruct` | 75 | Apache-2.0 |
| `mistral-small-3.2-24b-instruct` | 75 | Apache-2.0 |
| `ministral-3-8b-reasoning` | 72 | Apache-2.0 |
| `ministral-3-8b-instruct` | 70 | Apache-2.0 |
| `ministral-3-3b-instruct` | 60 | Apache-2.0 |
| `ministral-8b-instruct-2410` | 58 | **Mistral AI Research License — research use only** |
| `mistral-nemo-12b-instruct` | 58 | Apache-2.0 |
| `mistral-7b-instruct-v0.3` | 52 | Apache-2.0 |

Files: `mistral-small.yaml`, `ministral.yaml`, `magistral.yaml`, `devstral.yaml`,
`mistral-nemo.yaml`, `mistral-7b.yaml`, each with its generated `.facts.json`.

The band the brief asked for is covered from both ends. Twenty to thirty billion:
Devstral Small 2, Magistral Small 1.2 and Mistral Small 3.2, all 24B, plus Ministral 3 14B
which Mistral's own card claims is comparable to the 24B. Seven and under: Ministral 3 3B
and 8B (instruct and reasoning), Mistral 7B v0.3, Ministral 8B 2410. And one model that
sits outside both and is the reason the brief mentioned mixtures of experts: Mistral Small
4, 119B total and 6.5B active, which on the reference machine's own numbers generates at
about 13 tokens per second from system memory and tops the general, chat, reasoning,
multimodal and (behind Qwen3-Coder-Next) coding boards. That is a real result, not an
artefact: a 40 GB two-bit quant that reads 2 GB of weights per token beats an 8B dense
model that reads 5 GB, and it is exactly the case where a parameter count misleads.

## Licences

This was the part worth the care the brief asked for.

Everything Mistral has released since Mistral Small 3 (January 2025) that I catalogued is
Apache-2.0, and each entry links the licence text and says where the claim came from.
`ministral-8b-instruct-2410` is not: it is under the Mistral AI Research License, whose
section 3.2 reads "You shall only use the Mistral Models, Derivatives ... and Outputs for
Research Purposes". It sits in `ministral.yaml` beside `ministral-3-8b-instruct` precisely
because two models called "Ministral 8B" are easy to confuse and only one of them may go
into a product. The file header says so in capitals; so does the entry's licence comment.

Two restrictive licences exist in this range. I could only ship a model under one of them:

- **Mistral AI Research License (MRL)** — shipped, as `ministral-8b-instruct-2410`.
  `mistralai/Mistral-Small-Instruct-2409` (22B, also MRL, and squarely in the target band)
  publishes no benchmark at all on its card, so it could not meet the two-benchmark rule.
- **Mistral AI Non-Production License (MNPL)** — not shipped. See Codestral below.

`license.spdx` for the MRL entry is `mrl-0.1`, following the qwen-community-1.0 precedent
for licences that are not SPDX-registered.

## 1. What I considered and left out

**Codestral 22B v0.1** (MNPL, 22B, in the band, and the obvious `codestral*` file). Its
model card carries no benchmark table, and the announcement at mistral.ai/news/codestral
describes its evaluations — HumanEval, MBPP, CruxEval, RepoBench EM, Spider — in prose
while publishing the numbers only inside charts whose values are not in the page text or
in any accessible image. Two published benchmarks are the floor, so it is out. This is the
one omission I regret, because it was going to be the MNPL example. If someone can extract
those chart values from a primary source, the entry is a fifteen-minute job.

**Mamba-Codestral 7B v0.1** (Apache-2.0, Mamba2, and it *does* publish HumanEval 75.0,
MBPP 68.5, Spider 58.8, CruxEval 57.8). No credible GGUF exists: the only conversions on
the Hub are one repository with zero downloads and a handful of unrelated fine-tunes. A
source nobody uses is a model nobody can install.

**Pixtral 12B 2409** (Apache-2.0, and the vision model the brief named a prefix for). Every
GGUF conversion on the Hub is explicitly text-only — `leafspark/Pixtral-12B-2409-hf-text-only-GGUF`
and its copies. An entry claiming `vision` whose only source cannot deliver it is worse
than no entry, and an entry claiming nothing but text from a vision model is pointless.
Ministral 3 and Mistral Small 3.2 cover the vision slot with working projectors instead.

**Mistral Small Instruct 2409** (22B, MRL) — no published benchmarks, as above.

**Mistral Large 3 675B, Devstral 2 123B, Mistral Medium 3.5 128B, Mistral Small 4 EAGLE and
NVFP4 checkpoints, Leanstral, Shieldstral, Voxtral** — out of band, out of prefix, or both.
Devstral 2 123B and Mistral Medium 3.5 are also `license: other` with no licence name in
their metadata, which would need the LICENSE file read before either could be catalogued.

**Ministral 3 3B Reasoning** — omitted only to keep the file from growing without adding a
distinct choice; the 8B and 14B reasoning entries cover that shape. Its numbers are in the
same table if someone wants it.

## 2. The baseline I am least sure of

`mistral-nemo-12b-instruct` at 58, and the reason is in the entry's header comment: the
"Main Benchmarks" table on the **instruct** card is byte-for-byte the table on the **base**
card. MMLU 68.0, HellaSwag 83.5, TriviaQA 73.8 are pretrained-checkpoint figures that
Mistral republished on the instruct page. Mistral's announcement does publish an
instruction-tuned comparison (Table 2, "Mistral NeMo instruction-tuned model accuracy") but
only as an image whose values are not readable, and no other Mistral card carries a Nemo
Instruct row. So the scores are recorded labelled as base-model scores, following the
`qwen3-0.6b` precedent already in this repository, and the 58 is set from what a
July-2024 12B is worth against the entries around it rather than from those numbers.

What would settle it: readable values from that announcement's Table 2, or any Mistral
comparison table carrying a "Mistral NeMo Instruct" row. Failing that, a measured run.

Runner-up: `mistral-small-4-119b` at 80. Its figures are transcribed from the numeric
labels printed on the vendor's own charts, which is transcription rather than estimation,
but it is one step further from the source than a markdown table and it is the entry now
leading four of five boards on the reference machine. A tech report with the same numbers
in text, or a `measured` block from an actual run, would settle both the number and the
13-tokens-per-second estimate under it.

## 3. Where I judged rather than read

**Ministral 3 Instruct and the `coding` capability.** The Ministral 3 cards publish
LiveCodeBench for the *Reasoning* checkpoints and nothing for code on the *Instruct* ones —
their Instruct table is Arena Hard, WildBench, MATH and MM MTBench. A 14B general model
obviously writes code, but I could not point at a source, so `coding` is on the reasoning
entries and off the instruct ones. The consequence is real and worth flagging: a coding
request excludes Ministral 3 3B/8B/14B Instruct entirely.

**Mistral 7B v0.3 and `coding`.** Here I went the other way and left a capability off
despite having a source for it. Mistral publishes HumanEval 38.4 pass@1 for this exact
checkpoint. That is a factual coding claim, and I still dropped it, because surfacing a
model at 38.4 pass@1 for a coding request is putting it somewhere it will disappoint. This
is the judgement in the file I would most expect a reviewer to overturn.

**Devstral Small 2 and `multilingual`.** Its card's headline table includes "SWE Bench
Multilingual", which counts programming languages. Claiming natural-language multilinguality
from it would have been an invention; the capability is off and the entry says why.

**Devstral Small 2 and `use_cases`.** Kept to `coding` alone, following the
`qwen3-coder-next` precedent, even though the card mentions chat — the chat it means is
chat about code. It has vision, but `multimodal` is deliberately not listed, because the
vision is for screenshots inside a coding task, not for general image work.

**Magistral and Mistral Small 4 and `general`/`chat`.** Both are specialists on paper (a
reasoning model and a mode-switching model) and both are complete general models
underneath. Following the lesson recorded in `qwen3.yaml`, both list `general` and `chat`
so an ordinary request can reach them. Magistral leads with `reasoning`; Mistral Small 4
leads with `general`.

**Ministral 8B 2410's context.** The card documents 128k. The only published GGUF declares
32,768 in its header. I recorded 32,768, because the number has to describe the file a
person downloads — and because a curated value longer than the header is a validation
failure anyway.

## Concerns for whoever integrates this

1. **I edited three tests outside the catalog directory** (`f9087b1`, before any catalog
   commit). They asserted a hard-coded top-row model id, looked for a model among rows the
   default `--limit` truncates, and matched a sentence in a Rich table whose column widths
   move with the longest model id in the catalog. Every one of them fails for any vendor
   that adds a family, so every branch adding one will have hit them. Expect a conflict here
   and keep whichever version asserts the behaviour rather than the catalog; mine asserts
   the reading floor over every ranked row, raises `--limit`, and reads the exclusion
   reason from the JSON board.
2. **`MODELS.md` will conflict with every other branch.** It is generated; resolve by
   running `python scripts/gen_models_md.py` after the merge rather than by hand.
3. **The reference machine's boards have changed shape.** `mistral-small-4-119b` now leads
   general, chat, reasoning and multimodal on the RTX 4060 / 128 GB profile, at
   `UD-Q2_K_XL` for chat and general. A 40 GB two-bit download as the top general
   recommendation is defensible from the numbers and may still be worth a second opinion
   from whoever owns the scoring weights.
4. **No `measured[]` block anywhere in these files.** Nothing here has been run on the
   reference machine; every speed shown is the estimator's.
5. **No `codestral*` or `pixtral*` file exists**, despite both being named as allowed
   prefixes. Reasons in section 1.
