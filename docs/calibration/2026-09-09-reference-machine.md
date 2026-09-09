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
   40,960 needs 7.5, 32,768 needs 7.2, 24,576 needs 6.9, 16,384 needs 6.6.

## Initial estimator constants derived here

| Constant | Value | From |
|---|---|---|
| RAM efficiency | 0.70 | Coder-Next and Flash-Next generation against theoretical DDR bandwidth, after per-layer overhead |
| VRAM efficiency | 0.60 | published llama-bench results on consumer GPUs; not measured here (the GPU part is small on this machine) |
| PCIe effective bandwidth | 4 GB/s | 78 GB of experts per micro-batch at 52 tokens per second with `-ub 1024` |
| Per-layer overhead | 0.20 ms | the residual between bandwidth-only prediction and measured generation |
| KV per 1K tokens, Flash-Next | 33 MiB | the table above |
| Compute buffer base at `-ub` 512, 1024, 2048 | 1,090; 1,240; 2,080 MiB | the table above |
| CUDA context overhead | 120 to 300 MiB | VRAM used minus buffer totals; 300 is used as a conservative default |
