# Catalog: Alibaba Qwen

**Status:** done. Six new entries, three new catalog files, all refreshed, whole suite green.

Branch `catalog/cat-qwen`, worktree `C:\Dev\Projectos Pessoais\2026\llamafit-wt-cat-qwen`.
Nothing outside `src/llamafit/data/catalog/qwen3.{5,6,8}.*` was touched except the two
files every branch has to touch: the generated `MODELS.md`, and one test (see
[What else changed](#what-else-changed-and-why-it-will-conflict)).

## Commits

| SHA | Subject |
|---|---|
| `3286a43` | `test(recommend): assert the general board's leader is readable, not its id` |
| `d7465e3` | `feat(catalog): add the Qwen3.5 range, 2B to 122B-A10B` |
| `2771300` | `feat(catalog): add Qwen3.6-35B-A3B` |
| `e3d1de8` | `feat(catalog): add Qwen3.8-27B` |

`MODELS.md` is regenerated inside each family commit rather than in one at the end, so
every commit on this branch passes `test_models_md_matches_the_catalog` on its own.

## Tests

`2925 passed, 2 skipped, 14 deselected` at 96.65% coverage; `catalog validate` clean over
7 files, `catalog refresh --check` reports no drift, `ruff check`, `ruff format --check`,
`mypy` and `gen_schema.py` / `gen_models_md.py` + `git diff --exit-code` all clean.

## What was added

| id | Params | Class | Baseline | Quants |
|---|---|---|---|---|
| `qwen3.5-2b` | 2B | dense-hybrid | 60 | `UD-Q4_K_XL`, `Q8_0` |
| `qwen3.5-4b` | 4B | dense-hybrid | 76 | `UD-Q4_K_XL`, `Q8_0` |
| `qwen3.5-9b` | 9B | dense-hybrid | 79 | `UD-Q4_K_XL`, `Q8_0` |
| `qwen3.8-27b` | 27B | dense-hybrid | 83 | `UD-Q3_K_XL`, `UD-Q4_K_XL`, `UD-Q6_K_XL` |
| `qwen3.6-35b-a3b` | 35B / 3B | moe-hybrid | 80 | `UD-Q3_K_XL`, `UD-Q4_K_XL`, `UD-Q6_K_XL` |
| `qwen3.5-122b-a10b` | 122B / 10B | moe-hybrid | 81 | `UD-Q2_K_XL`, `UD-Q3_K_XL`, `UD-Q4_K_XL` |

Every one is Apache-2.0 with a `LICENSE` file in the vendor's own repository, fetched and
read rather than assumed. Every one is natively vision-language, so every entry declares a
`mmproj` extra; `qwen3.8-27b` declares its multi-token-prediction weights too.

The ladder a person now meets: 2B and 4B on any card, 9B as the largest dense Qwen3.5 whose
four-bit weights fit 8 GB, 27B dense for a 16–24 GB card, 35B-A3B for a small card with a
lot of system memory, 122B-A10B for a 96 GB machine (three-bit is 57 GB, four-bit is 77 GB,
which is why both are listed).

Checked by eye afterwards: `llamafit list`, `llamafit info` on all six, `llamafit recommend`
for general, coding, reasoning, chat and multimodal, and `llamafit plan qwen3.6-35b-a3b`.
The plan places the Gated DeltaNet recurrent state, the routed experts and the vision
projector separately and comes out "tight" on the reference machine, which is right.
`qwen3.8-27b` and the pre-existing `qwen3.8-flash-next` are both *excluded* from the
reference machine's general board for generating below reading speed, with the reason given
— that is the tool working, not the entries being wrong.

## The three questions

### 1. Which models did I consider and leave out, and why?

**The whole Qwen3-Coder line, and this is the gap in what I delivered.** There is no
Qwen3.5-, Qwen3.6- or Qwen3.8-Coder: the vendor's Hugging Face organisation has shipped no
dedicated coder since Qwen3-Coder-Next, which is already in the catalog. That leaves two
older candidates and I could add neither honestly:

- **Qwen3-Coder-30B-A3B-Instruct** would have been the useful one — a coding model a third
  the size of Qwen3-Coder-Next, for a machine that cannot hold 45 GB. Alibaba has never
  published a benchmark for it. Its model card carries none and defers to the blog; the
  blog scores only the 480B; `QwenLM/Qwen3-Coder` issue #469 is somebody asking for exactly
  these numbers. Two published benchmarks is the bar, and there are zero.
- **Qwen3-Coder-480B-A35B-Instruct** has numbers, but only inside a PNG on the blog — no
  table I can open and quote, and at roughly 270 GB for a four-bit quant it is beyond every
  machine this catalog is for. It also scores *below* Qwen3-Coder-Next on SWE-bench
  Verified, so it would be a row nobody should pick.

Someone with the 30B-A3B on disk and a SWE-bench harness could unblock the first of these
with a `measured` block and a self-published number; I could not.

Also left out, deliberately:

- **Qwen3.5-0.8B** — smaller than useful and the catalog already keeps `qwen3-0.6b` as the
  low-memory smoke test. Its GPQA is 11.9, which is noise.
- **Qwen3.5-27B and Qwen3.6-27B** — identical in shape to `qwen3.8-27b` and behind it on
  every benchmark all three cards publish. Three 27B rows would be completeness at the cost
  of breadth.
- **Qwen3.5-35B-A3B** — same relationship to `qwen3.6-35b-a3b`.
- **Qwen3.5-397B-A17B and Qwen3.8-2.4T-A95B** — 230 GB and roughly 1.3 TB at four bits.
  `qwen3.5-122b-a10b` already serves the largest machine the brief named.
- **The whole Qwen3-VL line (2B/4B/8B/32B/30B-A3B/235B-A22B)** — superseded by its own
  vendor. The Qwen3.5 card's first highlight is that early-fusion multimodal training
  "outperforms Qwen3-VL models across reasoning, coding, agents, and visual understanding",
  and every entry above is natively vision-language, so a separate VL line adds rows that
  are strictly worse at the same sizes.
- **Qwen3-Next-80B-A3B** — September 2025, and behind `qwen3.6-35b-a3b` on MMLU-Pro at more
  than twice the resident weight.
- **Qwen3-ASR, Qwen3-TTS, Qwen3-VL-Embedding/Reranker, Qwen-Drive, Qwen-AgentWorld,
  WebWorld** — real models, but none of them is a chat model llama.cpp serves through
  `llama-server`, and `audio` and `embeddings` capabilities have no consumer here yet.

### 2. Which quality baseline am I least sure of, and what would settle it?

**`qwen3.5-2b` at 60.** Its MMLU-Pro in thinking mode is 66.5, which is *within a point of
`gemma-3-27b-it`'s 67.5 at baseline 74* — a model thirteen times its size. Either the 2B is
worth much more than 60, or MMLU-Pro rewards something a 2B can fake. I scored it down on
the rest of the picture (GPQA 51.6 against Gemma's HumanEval 87.8 and DocVQA 86.6, and no
coding score published at all), but that is a judgement about which benchmark to believe,
not a measurement.

What would settle it: any benchmark both models are scored on by the same evaluator. There
is currently no such number — the two cards share no benchmark whose conditions match.
Failing that, the reference machine can run both at full offload; a head-to-head on twenty
real prompts would say more than either table.

Second least sure: **`qwen3.6-35b-a3b` (80) against `qwen3.5-122b-a10b` (81)**. These two
are inside noise of each other — the 122B leads on MMLU-Pro (86.7 vs 85.2) and long context,
the 35B leads on agentic coding (SWE-bench Verified 73.4 vs 72.0, Terminal-Bench 51.5 vs
49.4). I gave the 122B the point for the higher knowledge scores and 10B active parameters
against 3B. Somebody could reasonably swap them, and the ordering matters because the two
land on very different machines. What would settle it: a benchmark run under one harness;
the two cards' SWE-bench numbers come from different in-house scaffolds a generation apart.

### 3. Which entry's capabilities or use cases did I have to judge rather than read?

**`qwen3.5-2b`, twice, and both ways.**

Its card publishes no coding benchmark at any width — no LiveCodeBench, no HumanEval,
nothing — so I left `coding` off both its capabilities and its use cases. That is a judgement
that an unmeasured ability is not a claim I can make, and it has teeth: the omission hides
the model from every coding request. I think that is right for a 2B whose vendor names
"prototyping, task-specific fine-tuning, and other research or development purposes" as the
intended use, but the pre-existing `qwen3-0.6b` entry claims `coding` on thinner evidence
than I had, so the catalog is now inconsistent about where that line sits.

I also kept `thinking` on it even though it is the one model in this generation that answers
directly by default. The card says it supports both modes and shows the toggle, so the
capability is a fact; the *sampling* block records the non-thinking defaults, because that
is the mode a person actually gets.

**`qwen3.8-27b`'s primary use case** was the other judgement. The vendor's own paper title is
"A New Bar for Coding and Cowork" and its benchmark table leads with coding, which argues for
`coding` first. I made `general` primary anyway, with `coding` second: it is the flagship
dense instruction-tuned model, it answers ordinary requests, and the first entry in
`use_cases` is what earns the alignment bonus. This is the same call the repository already
had to fix once on `qwen3.8-flash-next` — and note that entry still lists `coding` first
while mine lists `general`. Both list `general` and `chat`, so neither is hidden, but the
two entries now disagree about the same generation's primary job. Worth one reviewer's
minute.

## Curation gaps I am leaving behind

- **`llama_cpp.kv_types_allowed` is empty on all six**, with a comment in each file saying
  why. These are Gated DeltaNet hybrids and nothing the vendor or llama.cpp publishes says
  which KV cache types they accept. The loader documents an empty list as "the curator
  recorded nothing", which is exactly the state of knowledge, so this is honest rather than
  lazy — but it means the planner will offer `q8_0` on an architecture nobody has confirmed
  accepts it. The closest relative in the catalog, `qwen3-coder-next` (`qwen3next`), does
  accept it and has a measured run at 262K context to prove it. One `llama-server` run per
  architecture would close this.
- **`llama_cpp.min_build` is unset** on all six for the same reason: I found no source that
  names the build where `qwen35` and `qwen35moe` landed.
- **No `measured` blocks.** I ran nothing; every speed in `recommend` above is the
  estimator's formula.
- **`release_date` is the Hugging Face repository creation date** on all six, stated in a
  comment at the top of each file. It is the only publication date I can verify from a
  primary source. The pre-existing `qwen3.8-flash-next` entry uses 2026-08-27 where its repo
  was created 2026-08-24, so the convention in this repository is the announcement date and
  mine is three days early by that standard. Announcement dates were only available to me
  through secondary sources, which the brief forbids.
- **`family` is `qwen3.5` / `qwen3.6` / `qwen3.8`**, matching the file names and the
  vendor's own generation naming. The pre-existing `qwen3.8-flash-next` carries
  `family: qwen3` in `qwen3.yaml` despite being the same August 2026 generation as
  `qwen3.8-27b`. I did not touch it. `family` is only used for display and search, so
  nothing breaks, but the two files disagree.

## What else changed, and why it will conflict

Two files outside `catalog/qwen3.*`:

- **`MODELS.md`** — generated. Every parallel branch will change it. Resolve by taking
  either side and re-running `python scripts/gen_models_md.py`.
- **`tests/unit/test_services_recommend.py`** —
  `test_the_general_board_no_longer_leads_with_a_model_slower_than_its_reader` asserted
  `board.rows[0].model_id == "llama-3.1-8b-instruct"`. That is not the defect the test
  guards; it is a fact about a five-model catalog, and it fails the moment anybody adds a
  model that is both better and fast enough. **Every other vendor branch will hit this same
  failure**, so expect the same three lines edited on several branches. I replaced the id
  with the property: the leading row's estimated `gen_tps` is at least `READING_TPS`. The
  rest of the case — Gemma 3 27B off the board, the reason string, the speed and placement
  kept on the excluded row — is unchanged, and the companion test that ranks Gemma on a
  roomier machine still passes untouched.
