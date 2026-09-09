# Benchmarking and calibration

Estimates are where LlamaFit starts; measurements on your machine are where it ends up. Phase 3
adds `llamafit bench`, which measures a model exactly as `plan` would run it, stores the result,
shows it next to the estimate, and fits the estimator's constants to your hardware. This page
says what is measured, how to read it, and how to record a measurement by hand until then.

## What `bench` measures

For a model, quant and context (default: the plan's choice):

1. **`llama-bench`** at the planned flags: `pp2048` (prompt processing throughput at a
   2048-token micro-batch) and `tg128` (generation throughput), three repetitions, median.
2. **A real server**: `llama-server` started with the exact preset flags, then three fixed
   requests: a short prompt with a 128-token answer, a 1,000-token prompt with a 64-token
   answer, and a tool-call request. Recorded: prompt tokens per second, generation tokens per
   second, time to first token, and whether the tool call came back well-formed.
3. **Memory**: the budget lines from the server's verbose log (weights, KV, recurrent state,
   compute buffer per device) and peak VRAM from the vendor tool during the run.
4. **Paging check**: VRAM within 3 percent of the card's total during the run combined with
   generation below 60 percent of the estimate is flagged as driver paging, with the tier that
   would avoid it.

Results are stored in `benchmarks.sqlite` in the data directory with the host fingerprint,
llama.cpp build, model, quant, flags and timestamp, and shown in the Benchmarks screen and in
`llamafit info` as `measured` numbers with their date.

## Reading estimate versus measured

```
$ llamafit bench qwen3.8-flash-next
                          estimated   measured   ratio
generation tok/s (tg128)     12.9       13.0     1.01
prompt tok/s (pp2048)        74         77       1.04
generation, 1K prompt        13.4       13.9     1.04
prompt, 1K request           48         49.4     1.03
peak VRAM                    7.2 GB     7.6 GB   1.05
verdict                      Fits       Fits
```

Ratios between 0.8 and 1.25 are normal before calibration. Anything outside is worth a look:
a generation ratio far below 1 with VRAM at the ceiling is paging; a prompt ratio far below 1
usually means the micro-batch or PCIe assumption is off for this machine.

## Calibration

`llamafit bench --calibrate` (or the Benchmarks screen) fits, from every stored result on this
host:

| Factor | What it explains |
|---|---|
| `ram_efficiency` | how much of the theoretical DDR bandwidth generation actually reaches with experts in RAM |
| `vram_efficiency` | the same for VRAM |
| `pcie_effective_gbps` | expert streaming during prompt processing |
| `layer_overhead_ms` | fixed per-layer cost per token |
| compute-buffer constants per micro-batch | the least predictable allocation |

The factors are written to a user hardware profile whose `match` rules fit this machine, and
every later estimate carries `calibrated`. Three or four measured configurations of different
sizes are enough for the fit to be stable; `bench --all` measures every locally installed
model.

## Recording a measurement by hand (before phase 3)

Add an entry to the model's catalog file, or to your custom models file, with the exact flags:

```yaml
measured:
  - profile: rtx4060-8gb-ddr5-128gb
    quant: UD-Q4_K_XL
    context: 32768
    flags: "-ngl 99 --n-cpu-moe 48 -ub 1024 --no-mmproj-offload -ot ffn_.*_shexp=CPU"
    llama_cpp_build: 10867
    gen_tps: 13.6
    pp_tps: 51.5
    peak_vram_gb: 7.0
    date: 2026-09-09
    source: docs/calibration/2026-09-09-reference-machine.md
```

Measure with `llama-bench` for `pp2048` and `tg128`, and with a real request for the prompt
figure; say which is which. The reference machine's record in
[calibration/](calibration/2026-09-09-reference-machine.md) shows the level of detail that
makes a measurement reusable: the buffer sizes from the verbose log, the free VRAM before
the run, and the desktop's own VRAM use.

## Sharing measurements

Not yet. An opt-in upload to a community dataset is on the [roadmap](../ROADMAP.md) after
phase 3, with a hardware picker to browse results from machines like yours. Until then,
pull requests with `measured` entries are welcome; they are reviewed like any catalog change.
