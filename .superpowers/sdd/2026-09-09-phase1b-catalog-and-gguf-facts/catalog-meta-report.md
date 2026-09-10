# Catalog: Meta's Llama, and one community fine-tune

**Status: complete.** Six models added across three files, all refreshed, whole suite green.

Branch `catalog/cat-meta` in `llamafit-wt-cat-meta`. Nothing merged, rebased or pushed.

## Commits

| SHA | Subject |
|---|---|
| `4c181bd` | `test: assert on board reasons without depending on where a cell wrapped` |
| `7152aff` | `feat(catalog): add Llama 3.2 1B, 3.2 3B and 3.3 70B` |
| `2b0bb28` | `feat(catalog): add Llama 4 Scout and Maverick` |
| `74c523f` | `feat(catalog): add Llama 3.3 Nemotron Super 49B v1.5` |

## Tests

2925 passed, 2 skipped, 14 deselected; coverage 96.67% (floor 85). `ruff check`, `ruff
format --check` and `mypy` clean. `catalog validate`: 6 files, no problems. `catalog
refresh --check`: no changes, so the committed facts match the live repositories.
`gen_schema.py` and `gen_models_md.py` leave the tree clean.

## What was added

| id | baseline |
|---|---|
| `llama-3.2-1b-instruct` | 35 |
| `llama-3.2-3b-instruct` | 52 |
| `llama-3.3-70b-instruct` | 75 |
| `llama-4-scout-17b-16e-instruct` | 76 |
| `llama-4-maverick-17b-128e-instruct` | 79 |
| `llama-3.3-nemotron-super-49b-v1.5` | 80 |

Files: `llama3.yaml` (extended), `llama4.yaml` and `llama-nemotron.yaml` (new), each with
its refreshed `.facts.json`. Every file name begins with `llama`, as instructed.

Every architecture I curated by hand was confirmed against the GGUF header the refresh
read: `llama` for the 3.x models, `llama4` for both Llama 4s, `deci` for the Nemotron.
Layer counts matched too (16 / 28 / 80 / 48 / 80).

## The licence question

Meta's licences are not OSI-approved and they are not one document. Llama 3.1, 3.2, 3.3
and 4 each ship their own text, so the entries carry four different slugs — `llama3.1`,
`llama3.2`, `llama3.3`, `llama4` — each pointing at that release's own file in
`meta-llama/llama-models`. All four URLs were fetched and return 200. Each carries an
acceptable-use policy, a naming condition ("Built with Llama", derivatives named
`Llama-<something>`), and the clause withholding the grant from anyone above 700 million
monthly active users on that release's date.

**The fine-tune trap is real and I hit it twice.**

*NVIDIA Nemotron Super 49B v1.5* — the Hugging Face metadata says `license: other,
license_name: nvidia-open-model-license`. True but incomplete. The card's own terms read:
"GOVERNING TERMS: NVIDIA Open Model License. **Additional Information: Llama 3.3 Community
License Agreement. Built with Llama.**" Both apply; a fine-tune cannot shed the terms of
the weights it was built from. I recorded `llama3.3` because that is the half that
actually constrains who may use it, and named the NVIDIA terms in the file beside it,
since the field holds one value. Anyone deploying it needs both documents.

*DavidAU's `Llama3.3-8B-Instruct-Thinking-Claude-4.5-Opus-High-Reasoning`* — declares
`license: apache-2.0`. Its stated base is `allura-forge/Llama-3.3-8B-Instruct`, which
itself declares `license: llama3.3`. A Llama derivative cannot be relicensed to Apache-2.0,
so the metadata is simply wrong. Left out — see below.

## 1. Which models did you consider and left out, and why

**Meta's Muse Glimmer 30B — the important one, and I am flagging it rather than adding
it.** Meta's newest open-weight release is not a Llama at all. On 2026-08-09 the
`meta-models` org (not `meta-llama`) published `Muse-Glimmer-30B`: a 30B dense agentic
model with a perception encoder, **Apache-2.0**, 131k context, knowledge cutoff January
2026, with an official GGUF repository at `meta-models/Muse-Glimmer-30B-GGUF` (389k
downloads) and a `muse-glimmer` architecture already in llama.cpp's arch table, plus a
`dflash` speculative-decoding drafter. Its own card reports SWE-Bench Pro 51.2, AIME 2026
94.7, GPQA Diamond 83.5, MMMU Pro 74, MCP Atlas 75.5 — it would be the strongest model in
this catalog for local agentic work, and it is explicitly designed for consumer hardware.

I did not add it because my family was "Meta's Llama" and I was told to write only to
files whose names begin with `llama`; Muse is a different product line that wants a
`muse.yaml`, and if another agent has it too, a duplicate id across two files fails
`catalog validate` and breaks exactly the merge the file rule protects. **If nobody else
owns it, it is the single biggest gap in the catalog after this branch lands.** The
research above is complete enough to write the entry in one pass.

Left out, with reasons:

- **Llama 3.2 11B / 90B Vision.** llama.cpp has no `mllama` architecture — I checked the
  full arch table in `src/llama-arch.cpp` and it is absent, and the multimodal docs list
  Llama 4 Scout but no Llama 3.2 Vision. The community GGUFs that exist need a fork.
  A model llama.cpp cannot load does not belong in a catalog about llama.cpp.
- **Llama 3.1 70B Instruct.** Same architecture and size as Llama 3.3 70B, strictly worse
  post-training. Listing both would offer a choice with a known wrong answer.
- **Llama 3.1 405B.** At four bits it is roughly 230 GB with no vision or MoE sparsity to
  soften it, and Llama 3.3 70B matches it on most published tasks. Maverick already
  represents the "needs a workstation" tier and does it better.
- **Hermes 3 Llama 3.1 8B** (NousResearch). Licence was establishable — its metadata says
  `llama3` but its base is Llama 3.1 8B, so `llama3.1` is what governs it. Left out on
  merit, not licence: its own card reports Open LLM Leaderboard v2 MMLU-Pro 23.77 and
  IFEval 61.70, which does not beat the Llama 3.1 8B already in the catalog.
- **DavidAU's `Llama3.3-8B-Instruct-Thinking-...`** and the related `allura-forge` 8B.
  Popular (65k downloads on one GGUF conversion) and a genuine 2026 release, but the
  licence chain is broken: the fine-tune claims Apache-2.0 over a base that declares
  `llama3.3`, and Meta never published a Llama 3.3 8B, so the origin of the 8B itself is
  unclear. **I could not establish what governs it, so I left it out**, as instructed.
- **Nemotron-3 Nano 4B / 30B-A3B** (NVIDIA, 2026). Very high traffic, but they are
  NVIDIA's own `nemotron_h` hybrid architecture, not Llama derivatives. Not my family.
- **Abliterated / "heretic" / "uncensored" conversions** of every size. Numerous and
  well-trafficked; no published benchmarks from a primary source and no vendor to hold to
  them.

## 2. The baseline I am least sure of

**`llama-3.3-nemotron-super-49b-v1.5` at 80**, and behind it Maverick at 79.

The problem is that this catalog's scale is compressed at the top: `qwen3.8-flash-next`,
an August 2026 flagship with GPQA Diamond 91.7, sits at 84. Nemotron's GPQA is 71.97. If
84 and 80 are only four points apart, that gap is doing far less work than the capability
difference between the two models warrants, and the same squeeze pushed my Llama 4 and
Llama 3.3 70B scores into a narrow 75–79 band where I was ordering models by judgement as
much as by evidence.

There is also a benchmark-comparability problem specific to Nemotron: its LiveCodeBench
figure is the 24.10–25.02 subset, which is not the same instrument as the LiveCodeBench v6
behind the Qwen entry's 91.9, and its GPQA row is not labelled Diamond while Qwen's is.
I cited them with their exact labels rather than silently aligning them.

What would settle it: one benchmark run by one party across the catalog's models —
Artificial Analysis or a local `llama-bench` plus a fixed eval harness — instead of each
vendor's self-reported table. Failing that, an agreed anchor model at a fixed score that
every new entry is placed against, which the rubric in `docs/catalog.md` currently implies
but does not name.

## 3. Where I judged rather than read

Three places, all recorded as comments in the YAML so the next curator sees the reasoning
rather than just the result.

**`llama-3.2-1b-instruct` — omitted `tools`.** Meta's card advertises tool calling for
both 3.2 sizes and publishes BFCL v2 for each. Read literally, the capability is sourced
and belongs. But the 1B scores 25.7 against the 3B's 67.0, and `capabilities` is a gate:
listing it puts a model that fails three tool calls in four in front of someone who asked
for one that works. I omitted it and kept it on the 3B. This is the judgement I am least
comfortable with, because an omission hides a model just as surely as an invention
misplaces one, and a reasonable curator could go the other way.

**`llama-4-scout` — omitted `coding`.** LiveCodeBench 32.8 is the weakest number on its
card and below what the eighteen-months-older Llama 3.3 70B manages on HumanEval.
Maverick, at 43.4, keeps the capability. The line between them is mine, not Meta's.

**`llama-3.3-nemotron-super-49b-v1.5` — omitted `multilingual`.** The base, Llama 3.3 70B,
is explicitly multilingual and scores MGSM 91.1. NVIDIA declares `en` only for this
checkpoint and publishes no multilingual score. I followed the fine-tuner's narrower claim
rather than inheriting the base's, which is the opposite of what I did for the licence —
deliberately: a licence flows down whether the fine-tuner likes it or not, a capability
does not, because post-training can destroy one.

Also judged, less contentiously: `use_cases` ordering. Llama 3.3 70B, Scout and Maverick
all lead with `general` and carry `chat`, following the correction already recorded in
`qwen3.yaml` where a flagship instruction-tuned model was made invisible to the most
ordinary request there is. Nemotron leads with `reasoning`, which is what its card is
about.

## Concerns

1. **Muse Glimmer 30B is unowned.** See above. Meta's newest, Apache-2.0, purpose-built
   for local use, and outside every file name I was allowed to write.

2. **The Nemotron entry cannot be ranked, and the cause is in the reader, not the entry.**
   `recommend` puts it under "Not ranked" with "This model's file does not say how large
   its key-value cache is per token." The `deci` architecture stores per-layer values where
   every other architecture stores a scalar: `deci.attention.head_count_kv` is an array of
   80 numbers — 8 on the attention layers and **0 on the 31 layers the architecture search
   removed attention from entirely** — and `llamafit/gguf/facts.py` reads it with an
   integer helper, gets `None`, and leaves `kv_bytes_per_token_f16` empty. The cache is
   perfectly computable as the sum over that array times the key and value lengths, and it
   is *smaller* than a uniform 49B's, which is part of why the model is cheaper to run than
   its parameter count suggests. Teaching the reader to sum an array-valued head count
   would make it rankable. I left it in the catalog because the entry is correct and the
   tool is honest about what it does not know; I recorded the diagnosis in the YAML.
   This is a code change outside my lane.

3. **I touched one file outside my lane: `tests/unit/test_cli_board.py`.** Its `flat()`
   helper promised assertions would survive Rich's table wrapping but only collapsed
   whitespace, leaving column borders sitting between the two halves of a wrapped sentence.
   Adding models with longer ids widened the Model column and broke
   `test_a_raised_floor_is_named_under_the_board_and_in_every_reason`. **Every other agent
   adding models with long ids will hit this same test**, so expect a duplicate or
   conflicting fix at merge time; mine strips column separators before collapsing
   whitespace and is committed separately (`4c181bd`) so it can be dropped or taken whole.

4. **`catalog refresh` with repeated `--model` flags persisted only one model.** Running
   `refresh --model a --model b --model c` reported changes for all three but wrote only
   the last one's facts to disk; refreshing each separately worked. I did not investigate
   further and did not file it, but the facts in this branch were all produced one model at
   a time. Worth a look — a curator refreshing a batch would silently ship stale facts.

5. **Llama 4 Scout's context is 10,485,760 tokens** and the GGUF header agrees exactly, so
   `catalog validate` is satisfied. Nothing on a consumer machine will allocate a KV cache
   near that; the figure is the model's, not a recommendation, and any UI that presents it
   as achievable will mislead.

6. **No `measured` blocks.** Nothing here was benchmarked on the reference machine; every
   speed the board shows for these models is the estimator's formula.
