# How LlamaFit works

This page explains where every number on the board comes from. It is written for someone
who wants to check LlamaFit's reasoning, not for someone who wants to change the code; for
that see [architecture.md](architecture.md) and the
[design specification](specs/2026-09-09-llamafit-design.md), which carries the
same formulas with their constants.

## 1. Two facts about running a model locally

**Generation is limited by memory bandwidth, not by compute.** To produce one token the
runtime reads every weight that participates in that token once. A dense 8B model at 4.5 bits
per weight is about 4.5 GB per token; on a card with 270 GB/s that is at most 60 tokens per
second, in practice about 35. A mixture-of-experts model with 3B active parameters reads about
1.7 GB per token even if it has 80B parameters in total, which is why such a model can run at
20 tokens per second with its experts in ordinary DDR5 RAM.

**Memory that does not fit does not fail; it gets slow.** On NVIDIA drivers an allocation
larger than the free VRAM is paged to system RAM over PCIe. llama.cpp starts normally and the
log looks fine. Generation speed drops to a third or worse, and prompt processing drops
further. The only signs are the VRAM figure pinned at the card's total and the speed.

Everything LlamaFit does follows from these two facts: compute the memory need exactly,
keep it under what is free, and predict speed from bandwidth.

## 2. The host

The scan collects what the estimator needs and labels how it got each number:

| Fact | How | Label |
|---|---|---|
| RAM bandwidth | a short in-process copy of a buffer larger than the CPU cache | `measured`; falls back to `estimated` from speed and channel count, which no current source reports, so in practice to `assumed` |
| VRAM total and free | the vendor tool (`nvidia-smi`, `rocm-smi`) | `measured`, or `unknown` |
| VRAM bandwidth and FP16 compute | a bundled table of vendor specifications matched by name | `estimated` |
| PCIe bandwidth | the same table, from the card's slot generation and lanes | `assumed` until a benchmark measures it |
| Performance cores | the OS on Apple Silicon; a table for hybrid Intel parts | `measured` or `estimated` |

Free VRAM means free right now: a browser with hardware acceleration holds half a gigabyte,
a video call another. LlamaFit budgets against what is free at scan time and keeps a reserve
for the desktop.

## 3. The model, exactly

Instead of a size class per model, LlamaFit reads the header of each GGUF file, locally or
over HTTP with range requests that fetch only the header, and derives:

- **bytes of expert weights** (tensor names containing `_exps`), **bytes of every other tensor
  in a block** — attention projections, feed-forward weights and norms alike, which is nearly
  the whole file on a dense model — **bytes of the output head and of the token embedding**,
  and bytes of any table the catalog marks as streamable from disk. Every tensor lands in
  exactly one of these buckets, and they sum to the file's total;
- **KV cache per token**: the key cache and the value cache are sized separately, each from its
  own head dimension (`attention_layers × kv_heads × key_length` and the same with
  `value_length`), each rounded up to a whole number of blocks of its KV type, and then added.
  llama.cpp allocates them as two tensors, so doubling the key cache is not what it does, and
  rounding down would report a cache smaller than the one allocated — the direction that tells
  somebody a model fits when it does not. For hybrid architectures only the full-attention
  layers count; the linear-attention layers keep a small fixed recurrent state instead;
- vocabulary size, layer count, expert count and experts used per token;
- the context length the file itself declares and, where the architecture declares one, the
  sliding attention window. Both are recorded and neither is used yet: the file's context is
  what it permits rather than what the vendor supports, and a sliding window shortens the cache
  on only some layers, which the header does not identify.

These come from the tensor table in the header, so they are exact for the file on disk.

## 4. The memory budget

For a candidate (model, quant) under a placement, at a context `c` and micro-batch `ub`, the
budget lists every allocation llama.cpp will make and which pool it lands in:

| Component | Pool | Size |
|---|---|---|
| attention and other non-expert weights | VRAM for layers on the GPU, else RAM | exact bytes |
| expert weights | RAM when offloaded, else VRAM | exact bytes |
| KV cache | with the attention layers | `kv_bytes_per_token × c` |
| recurrent state | VRAM | small constant |
| compute buffer | VRAM | `base(ub) + k(ub) × ub × c / 1024` |
| vision projector | VRAM when offloaded, RAM otherwise | file bytes plus its compute space |
| runtime overhead | VRAM | about 300 MiB for a CUDA context |
| process overhead | RAM | about 1 GiB |

The compute buffer is the least predictable part; its constants were fitted to measurements
and are replaced by measured values wherever a benchmark has run. Available VRAM is what is
free now minus a reserve; available RAM is what the OS reports as available.

### Fit verdict

Utilisation is `required / available` for each pool. The verdict is decided on the worst pool:

| Utilisation | Verdict | Meaning |
|---|---|---|
| up to 0.65 | **Comfortable** | room for a bigger context or a second model |
| up to 0.85 | **Fits** | the intended configuration runs as planned |
| up to 0.95 | **Tight** | runs, but a browser or a second process can push it over |
| above 0.95 | **Does not fit** | the driver would page or the load would fail |

A placement with attention and KV on the GPU and experts in RAM is judged like any other;
measured speeds there are close to the all-GPU case for the same active parameters. CPU-only
placements never rate above Fits, because they are rarely comfortable to use.

## 5. The placement search

For each candidate the planner tries, in order, and keeps the best that fits:

1. **All on the GPU** at the requested context; then with a quantised KV cache if the
   architecture allows it; then at smaller contexts down to the user's minimum.
2. **Experts in RAM** (mixture-of-experts models): attention, KV and shared experts on the GPU,
   routed experts in RAM; then progressively fewer experts on the CPU while VRAM allows.
3. **Hybrid**: the largest number of whole layers that fit on the GPU, the rest on the CPU.
4. **CPU only.**

Within a placement it also chooses the micro-batch from 2048 down to 256 (a larger micro-batch
makes prompt processing faster but needs a larger compute buffer), decides where the vision
projector lives (GPU if it fits with margin, else CPU, else off), and sets the thread count to
the performance cores.

The result includes a **context tier table**: how much free VRAM each context size needs, so a
launch script can pick the largest tier that fits at the moment it starts.

## 6. Speed

**Generation.** Time per token is the memory traffic in each pool divided by that pool's
effective bandwidth, plus fixed costs:

```
t = bytes_in_vram / (vram_bandwidth × 0.60)
  + bytes_in_ram  / (ram_bandwidth  × 0.70)
  + layers × 0.2 ms + 1 ms
tokens_per_second = 1 / t
```

`bytes_in_ram` for a MoE model with experts offloaded is the expert bytes touched per token,
`expert_bytes × experts_used / experts`, plus whatever KV lives in RAM. The efficiency factors
and overheads are the initial constants; a benchmark on your machine replaces them.

**Prompt processing.** With everything on the GPU it is compute-bound:
`2 × active_parameters × tokens / (fp16_tflops × efficiency)`. With experts in RAM, llama.cpp
streams the experts it needs across PCIe once per micro-batch, so throughput scales with the
micro-batch size and is bounded by PCIe bandwidth. This is why LlamaFit prefers the largest
micro-batch whose compute buffer still fits.

**Confidence.** Every speed carries one label: `measured` (a stored benchmark on this host for
this model, quant and flags, with its date), `calibrated` (formula with factors derived from
this host's measurements), `estimated` (formula with defaults), or `unsupported`.

## 7. Scores

Four scores from 0 to 100:

- **Quality**: the catalog's baseline for the model (set by curators from published benchmarks,
  with sources), minus a penalty for the quantisation (Q8 0, Q6 1, Q5 2, Q4 4, IQ4 6, Q3 10,
  IQ3 12, Q2 20, IQ2 24, IQ1 35; dynamic quants one less than their base), plus a small bonus
  when the model's strengths match the request. A missing required capability excludes the
  candidate.
- **Speed**: generation tokens per second against a target for the use case (chat 30, general
  25, coding 20, reasoning 15, multimodal 15; embeddings are scored on prompt throughput), with
  a deduction when prompt processing is slow for coding and reasoning.
- **Fit**: two questions, scored as the worse of them. *Does it waste the machine?* — the
  model's own weights against every byte of memory the machine has to hold them in: full marks
  at half the machine or more, and about 23 points off for each halving below that. *Is it
  crowded?* — the whole budget against whichever pool is tightest: full marks up to 0.80,
  falling to 40 at 0.98, and 0 above. The weights are read apart from the cache and the buffers
  because on a machine with a small card those dominate the pool and say nothing about how big
  the model is.
- **Context**: how much of the requested context fits, as a percentage.

The composite weights them by use case:

| Use case | Quality | Speed | Fit | Context |
|---|---|---|---|---|
| general | 0.35 | 0.25 | 0.25 | 0.15 |
| coding | 0.40 | 0.20 | 0.20 | 0.20 |
| reasoning | 0.50 | 0.15 | 0.20 | 0.15 |
| chat | 0.25 | 0.40 | 0.25 | 0.10 |
| multimodal | 0.40 | 0.20 | 0.25 | 0.15 |
| embedding | 0.30 | 0.45 | 0.20 | 0.05 |

`--prefer quality` or `--prefer speed` moves 0.10 between the first two columns.

## 8. Explanations

Every row on the board expands into text built from templates, never generated: which pool is
the limit, what the budget contains, why the placement was chosen, which needs it satisfies,
and what would move it up a tier (for example "free 1.2 GB of VRAM to reach 64K context" or
"the Q3 quant would fit entirely in VRAM at an estimated 18 tokens per second"). The CLI prints
the same text with `--explain`; the dashboards show it in the detail pane.

## 9. What changes after a benchmark

`llamafit bench` (phase 3) runs `llama-bench` and a real server at the planned flags, records
generation and prompt throughput and peak VRAM, and compares them with the estimate. From
enough results it fits the efficiency factors, the PCIe bandwidth and the compute-buffer
constants for your machine, and stores them in your hardware profile. Estimates then carry
`calibrated`; the exact configurations you measured carry `measured`. The reference machine's
first results are in [calibration/](calibration/).
