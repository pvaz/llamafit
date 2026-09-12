# Calibration record: reference machine, 2026-09-09

The measurements the estimator's initial constants are fitted to. Everything here was taken
on one machine on one day with one llama.cpp build; the value of the record is in the exact
flags and buffer sizes, which let the numbers be reproduced and compared.

## The machine

| Item | Value |
|---|---|
| CPU | Intel Core i9-14900KF, 8 performance + 16 efficiency cores, 32 threads |
| RAM | 128 GB DDR5-4200, 2 channels (theoretical 67.2 GB/s) |
| GPU | NVIDIA GeForce RTX 4060, 8 GB (8,188 MiB), PCIe 4.0 x8, driver 610.88 |
| OS | Windows 11 (build 26200) |
| llama.cpp | b10867 (f3f1a8f27), CUDA 12.4 release binaries |
| Desktop VRAM use during the runs | 530 to 1,340 MiB depending on open applications |

## Models

| Model | Quant | Size | Parameters | Architecture |
|---|---|---|---|---|
| Qwen3-Coder-Next | Unsloth UD-Q4_K_XL | 49.6 GB | 80B total, 3B active | MoE, hybrid attention, q8_0 KV allowed |
| Qwen3.8-Flash-Next | Unsloth UD-Q4_K_XL | 111.3 GB (4 shards) | 125B total, 6B active, plus a 51B n-gram table streamed from disk | MoE, hybrid attention (48 layers, 512 experts, 10 used, vocabulary 248,320), F16 KV only |

Both run with attention, KV and shared experts on the GPU and routed experts in RAM:
`-ngl 99 --n-cpu-moe 48 --fit off -fa on -t 16 -tb 16 -b 4096`.

## Qwen3-Coder-Next (measured 2026-09-08 and 2026-09-09)

| Test | Result |
|---|---|
| Generation, short context, `llama-bench tg128` | 24.7 tok/s at 16 threads (8 threads 23.8, 24 threads 22.8) |
| Generation at 32K tokens of context | 21.7 tok/s |
| Generation, first request after a cold start | 20.1 tok/s |
| File rewrite with n-gram speculation (`--spec-type ngram-mod --spec-ngram-mod-n-max 4`) | 33 to 35 tok/s, 98.6 percent of drafts accepted |
| Prompt processing, `llama-bench pp2048` by micro-batch | `-ub 512` 118, `1024` 194, `2048` 323, `4096` 315 tok/s |
| Prompt processing, real 11.5K-token request | 225 tok/s |
| Prompt processing, real 1K-token request | 98 tok/s |
| VRAM at 262,144 context with q8_0 KV | 7.8 GB of 8.2 GB (7,770 MiB with a 550 MiB desktop) |
| VRAM at 131,072 context | 6.1 GB |
| `--fit on` (automatic placement) | 11.9 tok/s: rejected |
| `--load-mode none` | no gain, 47 s start, double RAM: rejected |

Implied effective RAM bandwidth for generation: about 1.7 GB of expert weights per token at
24 tokens per second is roughly 41 GB/s, or 0.61 of the theoretical 67.2 GB/s, before
subtracting the per-layer overhead.

## Qwen3.8-Flash-Next: memory by configuration

Buffer sizes from `llama-server -v`. All in MiB on the GPU. "Vision GPU" means the projector
(`mmproj-F16.gguf`, 862 MiB) offloaded; "vision CPU" means `--no-mmproj-offload`.

| Context | `-ub` | Vision | Shared experts | Model | KV | Recurrent | Compute | Total | Free VRAM needed |
|---|---|---|---|---|---|---|---|---|---|
| 32,768 | 2048 | GPU | GPU | 4,606 | 1,056 | 113 | 2,330 + 248 | 8,353 + 862 | more than the card |
| 65,536 | 2048 | GPU | GPU | 4,606 | 2,112 | 113 | 2,586 + 248 | 9,665 + 862 | more than the card |
| 131,072 | 2048 | GPU | GPU | 4,606 | 4,224 | 113 | 4,168 + 248 | 13,359 + 862 | more than the card |
| 65,536 | 512 | GPU | GPU | 4,606 | 2,112 | 113 | 1,922 + 248 | 9,001 + 862 | more than the card |
| 65,536 | 512 | off | CPU | 4,367 | 2,112 | 113 | 1,142 | 7,733 | 7.9 GB |
| 49,152 | 512 | off | CPU | 4,367 | 1,584 | 113 | 1,118 | 7,181 | 7.4 GB |
| 32,768 | 512 | CPU | CPU | 4,367 | 1,056 | 113 | 1,094 | 6,629 | 6.8 GB |
| 32,768 | 1024 | CPU | CPU | 4,367 | 1,056 | 113 | 1,337 | 6,873 | 7.0 GB |
| 40,960 | 1024 | CPU | CPU | 4,367 | 1,320 | 113 | 1,361 | 7,161 | 7.3 GB |
| 65,536 | 1024 | CPU | CPU | 4,367 | 2,112 | 113 | 1,433 | 8,025 | 8.2 GB |

Derived: KV is 33 MiB per 1K tokens (two caches, 24 and 9 MiB per 1K). The compute buffer
at `-ub 1024` grows about 3 MiB per 1K tokens of context; at `-ub 2048` about 8 MiB per 1K up
to 64K and about 25 MiB per 1K beyond. Moving the shared experts to RAM saves 239 MiB. The
projector on the GPU costs its 862 MiB of weights plus 248 MiB of its own compute space plus
about 780 MiB more in the main compute buffer. CUDA context overhead: about 120 MiB beyond
the listed buffers (measured as VRAM used minus buffer total).

## Qwen3.8-Flash-Next: speed by configuration

"Short" is a 24-token prompt with 128 generated tokens on a fresh server; "1K" is a
1,056-token prompt with 64 generated tokens immediately after. The short run is always slower
because the n-gram table pages in from disk during the first request.

| Context | `-ub` | Vision | Shared experts | Fits? | Gen short | Gen 1K | Prompt 1K |
|---|---|---|---|---|---|---|---|
| 131,072 | 2048 | GPU | GPU | paging 6 GB | 5.8 to 6.5 | 6.2 | 6 to 16 |
| 65,536 | 2048 | GPU | GPU | paging 2.3 GB | 10.2 to 11.0 | 12.9 to 14.0 (previous day) | 32 to 33 |
| 32,768 | 2048 | GPU | GPU | paging 1 GB | 10.4 | | |
| 16,384 | 2048 | GPU | GPU | slight paging | 12.8 | 14.5 | 50 |
| 65,536 | 512 | GPU | GPU | paging | 10.5 | 15.1 | 10.5 |
| 65,536 | 512 | off | CPU | slight paging | 9.7 | 13.4 | 28.6 |
| 49,152 | 512 | off | CPU | yes, 0.4 GB spare | 8.3 | 12.3 | 27.1 |
| 32,768 | 512 | CPU | CPU | yes | 8.3 | 13.7 | 24.8 |
| 32,768 | 1024 | CPU | CPU | yes, 0.6 GB spare | 9.9 | 13.6 | 51.5 |
| 40,960 | 1024 | CPU | CPU | yes, 0.4 GB spare | 9.8 | 13.9 | 49.4 |
| 65,536 | 1024 | CPU | CPU | paging 0.4 GB | 10.1 | 13.9 | 31.6 |
| `llama-bench`, tiny context | 2048 | off | GPU | yes | tg128 13.0 | | pp2048 77 |

Tokens per second throughout.

## What the record says

1. **Generation is bandwidth-bound and insensitive to context size** as long as nothing pages:
   13 to 14 tokens per second in every configuration that fits, 6 when 6 GB page.
2. **Prompt processing scales with the micro-batch** when experts are in RAM, because the
   experts stream across PCIe once per micro-batch: 25 at 512, 50 at 1024, 77 at 2048. It
   collapses when the compute buffer pages (10 tokens per second at 512 with the projector
   on the GPU).
3. **Paging is silent.** Every configuration above started without an error. The only signs
   were `nvidia-smi` pinned near 7.9 GB and the speed.
4. **The projector is the expensive option on an 8 GB card**: 1.9 GB, the equivalent of 52K
   tokens of context. On the CPU it costs nothing measurable for text.
5. **The winning configuration** for this card is `-ub 1024 --no-mmproj-offload
   -ot ffn_.*_shexp=CPU` with the context picked from free VRAM: 49,152 needs 7.8 GB free,
   40,960 needs 7.5, 32,768 needs 7.2, 24,576 needs 6.9, 16,384 needs 6.6. Those are the
   flags it *adds* to the base line under "Models" above; the whole command line is
   `-ngl 99 --n-cpu-moe 48 --fit off -fa on -t 16 -tb 16 -b 4096 -ub 1024
   --no-mmproj-offload -ot ffn_.*_shexp=CPU`, and that is what the catalog entry records.
   A run quoted as a delta does not say where its layers went, and the speed estimator
   reads these flags to decide whether a run describes a placement at all.

## Estimator constants derived here

Four runs identify four generation constants. The two below the line were added on
2026-09-10 precisely because two could not: a small dense model is the only way to isolate
a memory pool, since its traffic per token is exactly its block weights and nothing else
competes.

| Run | Traffic per token | Measured | Isolates |
|---|---|---|---|
| Qwen3-Coder-Next UD-Q4_K_XL | 2.36 GB card, 0.916 GB scattered | 23.0 tok/s | scattered reads |
| Qwen3.8-Flash-Next UD-Q4_K_XL | 4.83 GB card, 1.504 GB scattered | 13.9 tok/s | scattered reads |
| Qwen3-0.6B Q8_0, `-ngl 0` | 0.47 GB, contiguous | 78.0 tok/s | system memory, sequential |
| Qwen3-0.6B Q8_0, `-ngl 99` | 0.47 GB, card | 279.5 tok/s | the card |

The same two dense runs were measured processing a prompt, and those two figures are what
the prompt-processing compute constants are fitted to. A dense model held entirely on one
device is the only run in which the compute term is charged to one device alone, which is
what makes each of these isolate a rate rather than a mixture of two.

| Run | Prompt throughput | Isolates |
|---|---|---|
| Qwen3-0.6B Q8_0, `-ngl 99` | 21,734 pp tok/s | the card's share of the compute term |
| Qwen3-0.6B Q8_0, `-ngl 0 -t 8` | 2,924 pp tok/s | the CPU's share of the compute term |

| Constant | Value | From |
|---|---|---|
| System memory, sequential | 0.70 | the two dense runs, with the fixed overhead below |
| System memory, scattered | 0.57 | 0.54 from Coder-Next and 0.59 from Flash-Next, nine percent apart while carrying 64 percent different traffic |
| Card efficiency | 0.67 | the dense run with everything on the card |
| Fixed overhead | about 1 ms per token | the residual common to both dense runs |
| Prompt compute on the card | 0.43 | 21,734 prompt tokens per second with every layer on the card is 26.1 TFLOP/s at the `2 x active_params x ub` the formula charges, against the 60.4 TFLOPS the corrected table rates this card at. One run, and it wants a second machine |
| Prompt compute on the CPU | 0.44 TFLOP/s per performance core | 2,924 prompt tokens per second at `-ngl 0 -t 8` is 3.51 TFLOP/s across eight performance cores. Effective, not a peak, and **not** multiplied by the row above: the card side has a published ceiling to scale and this side does not |
| PCIe effective bandwidth | 4 GB/s | 78 GB of experts per micro-batch at 52 tokens per second with `-ub 1024`. **Known to be wrong**: fitted on the one model whose expert set cannot fit in memory, while the other streams about three times faster from the page cache. Two physical paths sharing one constant |
| KV per 1K tokens, Flash-Next | 33 MiB | the table above. Of this, about 9 MiB is a second cache the file's own header cannot account for |
| Compute buffer base at `-ub` 512, 1024, 2048 | 1,090; 1,240; 2,080 MiB | the table above |
| CUDA context overhead | 120 measured, 300 default | card memory minus the buffer totals. The measured figure is used for this card and the conservative default for every other, because one measurement is not a generalisation |

### Figures this record used to carry, and why they are gone

**There is no per-layer overhead.** It read 0.20 ms, which across 28 layers is 5.6 ms,
while that model's entire measured token is 3.58 ms. The term was not merely too large,
it was impossible, and it capped every model of that depth near 105 tokens per second.

**The scattered efficiency was once given as 0.36 and that was an error.** It was derived
by dividing the expert traffic by the *whole* token time, which already contains the card
traffic and the overhead the formula then adds again; used as a term in a sum it alone
exceeds the measured token, and the estimator built on it came out 42 percent low. An
apparent end-to-end rate and a coefficient inside a sum of terms are not the same
quantity, and the mistake is invisible until something is measured against it.

**A token embedding table is not per-token traffic.** A lookup reads one row. Counting the
whole table put the apparent rate above what the memory bus can physically deliver, which
is how the error above was finally caught.

**The prompt compute term was one term charged wholly to the card, at an efficiency of
0.30.** Both halves of that were wrong and they hid each other. The efficiency was read off
the slope of the `llama-bench` micro-batch sweep under "Qwen3-Coder-Next" above -- 118
tokens per second at `-ub 512`, 194 at 1024, 323 at 2048 -- which is not a clean read of a
compute constant: that model's experts stream across the link, its three points do not lie
on a line to better than 13 percent, and the slope through them implies about 5 TFLOP/s
where the dense run says 26. The sweep stays in this record because it is a real
measurement of that model's prompt throughput; it is simply not the run that isolates this
term. The 0.30 was also divided into a table that rated this card at 15 TFLOPS, which was
its fp32 *shader* rate, and llama.cpp multiplies matrices on the tensor cores: no
efficiency at or below one reaches 26.1 TFLOP/s from 15. The pair cancelled wherever the
streaming term was large enough to hide it, and on a dense model held on the card, where
nothing hides it, the prompt figure came out roughly fourfold low.

**The same term also had no CPU side, and inventing one made it worse.**
`CPU_FP16_TFLOPS_PER_CORE` read 0.05, a fifth of an AVX2 core's fp32 peak arrived at by
argument rather than by measurement, and it was then multiplied by the card's efficiency on
the way out, so the rate the formula spent was a fiftieth of that core's peak and
about twentyfold below the 3.51 TFLOP/s this machine was measured doing. Nothing caught it
because the term only ever fired on a host with no graphics card at all. With an honest
tensor figure in the table, the whole-term-to-the-card reading put Gemma 3 27B -- nine of
sixty-two layers on this card -- near 480 prompt tokens per second, which needs about 22
TFLOP/s from this processor. That is what made the split visible: prompt arithmetic runs
where the weights are, so the term is now divided by the share of the layers each device
holds and the CPU rate is measured rather than argued.
