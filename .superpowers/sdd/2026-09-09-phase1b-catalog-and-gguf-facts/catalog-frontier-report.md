# Catalog: the frontier group

Eighteen models across seven new family files, from a 2.7 GiB four-bit that runs on a
laptop to a 791 GiB four-bit that runs on nothing anybody reading this owns. That spread is
the point of the group: most of these do not fit, and a person deciding what to buy learns
more from a clear "does not fit, and here is what it would take" than from a catalog that
quietly omits everything ambitious.

## Status

Complete. Validate, lint, format, mypy and the generated files are all clean; the catalog
refresh has filled every quant and extra from the GGUF headers over the network, and
`llamafit catalog refresh --check` reports the committed data is current.

## Commits

Base `d4c3342`, branch `catalog/cat-frontier`. One commit per family, plus one that had to
come first.

| SHA | Subject |
|---|---|
| `00f86e2` | test(recommend): assert board behaviour rather than a fixed catalog |
| `eb93bd8` | feat(catalog): add DeepSeek V4 Flash, V4 Pro and the R1 Qwen3-8B distillation |
| `f1710aa` | feat(catalog): add GLM-5.3 and GLM-5.3-Flash |
| `e46d86a` | feat(catalog): add Kimi K3 and Kimi-Linear-48B-A3B-Instruct |
| `f329545` | feat(catalog): add MiniMax-M3 and MiniMax-M2.7 |
| `ae1f73d` | feat(catalog): add gpt-oss-20b and gpt-oss-120b |
| `c8fd8e5` | feat(catalog): add the NVIDIA Nemotron 3 and 3.5 models |
| `3ee9cc5` | feat(catalog): add Cohere's Command A+ and North Mini Code |

The first commit is not a catalog change and needs explaining. Three board tests asserted
which model led a board and that a particular model held a visible row — facts about what
the five-model catalog happened to contain, not about the behaviour they were written for.
Every entry added to the catalog broke them. Each now asserts its own behaviour instead:
that the general board does not lead with a row slower than a reader, that removing the
reading floor stops the floor excluding the slow model, and that a raised floor is named in
every reason it produced. The last moves from the rendered table, which shows only as many
rows as it has room for, to the reasons behind it. **That commit passes unchanged against
the five-model catalog**, which I checked by moving the new files aside and running it, so
the history bisects cleanly and the other agents growing this catalog in parallel get the
fix rather than the same three failures.

## Tests

`pytest --cov -m "not hardware and not network"`: 2925 passed, 2 skipped, 14 deselected,
coverage 96.67% against a floor of 85%.

## The models

| id | baseline | size range (four-bit and around it) | licence |
|---|---|---|---|
| `glm-5.3` | 92 | 222–746 GiB | GLM-5.3 licence |
| `kimi-k3` | 92 | 604–1405 GiB | Kimi K3 licence |
| `deepseek-v4-pro-0813` | 91 | 791–813 GiB | MIT |
| `glm-5.3-flash` | 89 | 95–272 GiB | MIT |
| `deepseek-v4-flash-0731` | 88 | 85–151 GiB | MIT |
| `nemotron-3-ultra-550b-a55b` | 85 | 181–335 GiB | OpenMDW-1.1 |
| `minimax-m3` | 84 | 125–247 GiB | MiniMax Community (non-commercial by default) |
| `minimax-m2.7` | 81 | 61–230 GiB | MiniMax non-commercial |
| `nemotron-3-super-120b-a12b` | 80 | 51–123 GiB | NVIDIA Nemotron Open Model License |
| `gpt-oss-120b` | 78 | 59 GiB | Apache-2.0 |
| `north-mini-code-1.0` | 78 | 10–30 GiB | Apache-2.0 |
| `nemotron-3.5-lightning-30b-a3b` | 76 | 18–36 GiB | OpenMDW-1.1 |
| `command-a-plus-05-2026` | 76 | 71–216 GiB | Apache-2.0 |
| `nemotron-3-nano-30b-a3b` | 72 | 19–31 GiB | NVIDIA Nemotron Open Model License |
| `gpt-oss-20b` | 70 | 11 GiB | Apache-2.0 |
| `kimi-linear-48b-a3b-instruct` | 68 | 15–38 GiB | MIT |
| `deepseek-r1-0528-qwen3-8b` | 62 | 4.7–8.1 GiB | MIT |
| `nemotron-3-nano-4b` | 58 | 2.7–3.9 GiB | NVIDIA Nemotron Open Model License |

Seven of the eighteen have a build under 40 GiB, which is the half of the group that
justifies a person with 128 GB of system memory opening the tool at all. The other half is
there so the answer to "would a bigger machine help" is a number rather than a shrug.

## Licences, because this group is where they stop being boilerplate

Nine distinct licences across eighteen models, six of them bespoke. What each entry records
was read from the licence file or the model card in the repository, never remembered:

- **MIT and Apache-2.0, unconditional**: the three DeepSeek entries, GLM-5.3-Flash,
  Kimi Linear, both gpt-oss models, Command A+ and North Mini Code.
- **MIT-shaped with a revenue threshold**: GLM-5.3 requires a Z.ai security review of a
  Model-as-a-Service operator past ten billion dollars of revenue. Kimi K3 requires a
  separate agreement past twenty million, and interface attribution past a hundred million
  monthly active users; it exempts internal use in as many words.
- **Non-commercial by default**: both MiniMax models, and they are not the same document.
  M3's permits commercial use subject to displaying "Built with MiniMax M3" plus a notice,
  escalating to written authorisation past twenty million dollars of yearly revenue. M2.7's
  prohibits commercial use outright without prior written authorisation. Both permit
  redistribution of derivatives, which is why the GGUF conversions exist and why the entries
  are admissible under the catalog's rule about redistribution.
- **Two different licences inside one vendor's family**: Nemotron 3 Nano 4B, Nano 30B and
  Super 120B are under the NVIDIA Nemotron Open Model License; Ultra 550B and 3.5 Lightning
  are under OpenMDW-1.1, which NVIDIA did not author. Reading one Nemotron card and assuming
  the rest is exactly the mistake this group invites.

Cohere's move to Apache-2.0 for Command A+ and North Mini Code is the one that most needs
checking rather than remembering: the Command series has a history of non-commercial terms,
and `command-a-reasoning-08-2025` is still CC-BY-NC-4.0 and gated.

## Three findings the reviewer should look at

**LlamaFit cannot price MXFP4, and that hides gpt-oss.** `quality/quant_penalty.py` returns
`None` for any name that does not match `I?Q\d`, so `MXFP4` is an unknown quantisation and
the board drops the candidate rather than scoring it. gpt-oss-20b and gpt-oss-120b ship
natively in MXFP4 — it is the precision OpenAI post-trained and evaluated at, not a
conversion — so with only the ggml-org source both models were invisible on every board on
every machine. That is a gap in the scorer, not in the entry, and it will recur: MiniMax,
Nemotron and North all publish `MXFP4_MOE` builds too, and Kimi K3's `TQ1_0`/`TQ2_0` fail
the same way. I did not change the penalty table, because what MXFP4 costs is a decision
for whoever owns section 11.1 and not for a curator. Instead each gpt-oss entry also lists
unsloth's K-quant builds, which have an independent reason to exist — a llama.cpp build
without MXFP4 support cannot load the native file — and Kimi K3 lists `UD-IQ1_M` instead of
`UD-TQ1_0`, with a comment in each file saying why. **The right fix is still to teach the
penalty table about MXFP4 and the ternary formats.**

**One benchmark set was read off a picture, and I want that on the record.**
GLM-5.3-Flash's five benchmarks are not in text anywhere: the model card embeds
`resources/bench_53.png` from the GLM-5 repository and the z.ai blog behind it does not
render without JavaScript. I read the figures from the image and cited the image. As a
check on my own reading, the same chart carries GLM-5.2's bars, and four of its six agree
with the GLM-5.3 model card's text table (Terminal Bench 2.1 81.0, DeepSWE 46.2,
AutomationBench 26.2, HLE with tools 54.7); the two that differ, Agents' Last Exam and
GDPval, differ because the card's table is a later revision of those benchmarks. If Z.ai
publishes a text table for Flash, the entry should move to it.

**Bits per weight is not four for a Mamba-2 hybrid, and that is real.** The refresh reads
6.8 bits per weight for Nemotron 3.5 Lightning's `UD-Q4_K_XL` and 10.3 for its
`UD-Q8_K_XL`. Nothing is wrong: the state-space, convolution and multi-token-prediction
tensors do not quantise and stay wide, so a "four-bit" build of a hybrid is genuinely
fatter than a four-bit transformer. Anyone reviewing the numbers against a mental model of
transformer quantisation will think these entries are broken, and they are not. The
opposite surprise sits in the DeepSeek entries: because DeepSeek trains the experts at FP4,
`UD-Q8_K_XL` is only about four per cent larger than `UD-Q4_K_XL` (161.9 GB against 155.1
GB for Flash), so the eight-bit is nearly free and both are listed.

## The three questions

### 1. Which models did I consider and leave out, and why?

- **`command-a-reasoning-08-2025`** — gated on Hugging Face and CC-BY-NC-4.0. Neither
  freely downloadable nor commercially usable; I could not read its model card at all
  (the raw fetch returns an auth error), so every number would have come from somewhere
  other than a primary source. Said so explicitly in a comment in `command.yaml`.
- **`deepseek-ai/DeepSeek-V4.1-Flash`** — published today, 2026-09-10, with six downloads
  and no GGUF conversion anywhere. Nothing to install and no benchmarks to cite.
- **`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`** — has GGUF builds and real traffic, but
  the vendor labels it experimental and it would need vision verification I did not do.
  Worth a follow-up.
- **`DeepSeek-V3.2`, `GLM-5`/`5.1`/`5.2`, `GLM-4.7-Flash`, `Kimi-K2.5`/`K2.6`/`K2.7-Code`,
  `MiniMax-M2`/`M2.1`/`M2.5`** — superseded within their own families by something I did
  include. Adding a vendor's back catalogue crowds the board without changing what anybody
  should install. GLM-4.7-Flash is the one I would reconsider first: it is smaller than
  GLM-5.3-Flash and still current enough to matter on a small machine.
- **`MiniMax-H3`** — looked like the obvious MiniMax flagship by download count and is not
  a language model at all; every GGUF publisher around it is a diffusion or ComfyUI
  repository. Excluded on inspection, and worth recording as a trap.
- **`Nemotron-3-Nano-Omni-30B-A3B-Reasoning`, `Nemotron-Cascade-2-30B-A3B`,
  `Nemotron-Labs-3-Puzzle-75B-A9B`, `Nemotron-3-Embed-8B`, `nemotron-3.5-asr-streaming`** —
  the Nemotron family is enormous. I took the five that form a clean size ladder for a
  general audience and left the omni, embedding, ASR and research variants, which need
  capability claims (`audio`, `embeddings`) I would want to verify properly rather than
  infer from a name.
- **`CohereLabs/North-Micro-Vision-Instruct` and `North-Small-Translate-1.0`** — both very
  recent (2026-09-08 and 2026-09-10) with no GGUF conversion published yet.
- **`openai/gpt-oss-safeguard-20b` and `-120b`** — classification models for policy
  reasoning rather than assistants. `use_cases` has no honest slot for them and putting
  them under `general` would be exactly the misfiling the brief warns about.
- **`moonshotai/Kimi-VL-A3B-Thinking-2506` and `Moonlight-16B-A3B-Instruct`** — small and
  interesting, but from January 2026 or earlier and clearly superseded by Kimi Linear at a
  similar footprint.

### 2. Which quality baseline am I least sure of, and what would settle it?

**`command-a-plus-05-2026` at 76.** It is the only entry in the group whose benchmarks do
not overlap the ones everything else reports. Cohere's blog gives τ²-Bench Telecom 85,
MMMU 75.1, MathVista 80.6, MMMU Pro 63 and Terminal-Bench Hard 25, and publishes no
SWE-bench, no Terminal-Bench 2.1, no GPQA and no MMLU-Pro. Three of those five are vision
benchmarks that the text-only GGUF cannot reproduce at all. So the number is anchored on a
single comparable figure — Terminal-Bench Hard 25, which is within a point of Nemotron 3
Super's 25.78, a model I scored 80 — and on Cohere's own claim that Command A+ leads its
series. A 218B model with 25B active is far more capable than 76 suggests on the tasks
Cohere optimised for, and far less capable than 76 suggests on agentic coding.

What would settle it: any run of Command A+ on SWE-bench Verified or Terminal-Bench 2.1
from a source that also ran the models around it. Failing that, its Artificial Analysis
Intelligence Index score of 37 placed against the same index for GLM-5.3-Flash and
Nemotron 3 Super would give a common yardstick.

The runner-up is **`kimi-linear-48b-a3b-instruct` at 68**, for the opposite reason: the
numbers are solid but old. They come from the technical report's appendix D, table 9, which
is the released 5.7T-token instruction-tuned checkpoint rather than the paper's 1.4T-token
ablation — I checked this specifically because it is the trap the brief names, and the
paper's main body would have given MMLU-Pro 51.0 and LiveCodeBench v6 26.0 instead of the
correct 72.7 and 45.7. But the checkpoint is from October 2025 and nothing since has
re-benchmarked it against the 2026 field.

### 3. Which entry's capabilities or use cases did I have to judge rather than read?

Four, in descending order of how uncomfortable I am about them.

**`vision`, three times, decided against the model card.** MiniMax-M3, Kimi K3 and Command
A+ are all documented as accepting image input, and Kimi K3's card gives a full column of
vision benchmarks. Kimi K3's entry claims `vision` and declares its projector; the other
two do not, because no vision projector is published alongside their GGUF conversions.
`docs/catalog.md` is explicit that a model declaring vision must declare the projector so
its bytes are charged to a pool, and there is nothing to declare. So those two entries
describe what can be run rather than what the vendor trained, and each says so in its
architecture notes. If a projector is published later, both entries should change.

**`multilingual`, omitted from DeepSeek and GLM.** Both vendors are obviously strong in
Chinese and English and both tag their repositories `en, zh`. Neither publishes a
multilingual benchmark or a supported-language list for the models I added, and two
languages is not what a person means when they filter for multilingual. Every entry that
does claim `multilingual` — the four Nemotron models and Command A+ — has a card that names
six or more languages and reports MMLU-ProX, WMT24++ or SWE-bench Multilingual. It is a
consistent rule, but it is my rule, and it hides DeepSeek and GLM from a multilingual
request.

**`coding` on `command-a-plus-05-2026`.** Cohere positions it as an agentic, multilingual
and reasoning-heavy enterprise model and does not call it a coding model. Terminal-Bench
Hard 25 is a real coding-agent number and comparable to models I did list for coding, so I
included the capability. The gate cuts both ways here — omitting it hides the model from a
coding request, and including it may put it somewhere it disappoints against North Mini
Code at half the size.

**`general` and `chat` on the flagships.** Every big model in this group is benchmarked
almost entirely on agentic coding and tool use, so the temptation is to list `coding` and
stop. That is the mistake this repository already made once with Qwen3.8-Flash-Next: these
are vendors' flagship instruction-tuned models, not specialists, and a general request that
excluded them would return the weakest things in the catalog. So DeepSeek V4 Flash and Pro,
GLM-5.3, GLM-5.3-Flash, Kimi K3, both MiniMax models and both gpt-oss models all carry
`general` and `chat` behind their primary use case. The two genuine specialists keep their
specialism: `north-mini-code-1.0` is `coding` only, exactly as `qwen3-coder-next` is.

I verified this rather than assuming it: `llamafit recommend --use-case general` on the
reference machine now returns Nemotron 3.5 Lightning, Nemotron 3 Nano 30B, Nemotron 3 Nano
4B, MiniMax-M2.7 and DeepSeek-V4-Flash before it reaches Llama 3.1 8B, and
`--use-case coding` leads with Qwen3-Coder-Next and North Mini Code. Nothing I added is
invisible to the request it is for.
