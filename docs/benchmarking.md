# Benchmarking and calibration

Estimates are where LlamaFit starts; measurements on your machine are where it ends up.
`llamafit bench` measures a model exactly as `plan` would run it, stores the result, shows it
next to the estimate, and fits the estimator's constants to your hardware. This page says what
is measured, how to read it, and how to record a measurement by hand.

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
llama.cpp build, model, quant, the settings the run reported about itself, and the timestamp.
`llamafit bench --show` lists them. The Benchmarks screen the design sketches is not built;
`--show` and `--calibrate` are the whole of the interface to the database today.

**A result is refused rather than stored when its conditions are not knowable.** llama.cpp
clamps a context it cannot honour and expands `-ub 512,1024,2048` into a run apiece, so what a
row records is the tool's own report of what it did, checked against the command line it was
given; a disagreement stops the write and names the setting. The host fingerprint includes the
driver version, so a driver update quietly retires the measurements taken before it — the safe
direction, since the label then falls back to `estimated`, which is true.

## Reading estimate versus measured

```
$ llamafit bench qwen3-0.6b --no-store
                 Estimated against measured
┌───────────────────────────┬───────────┬──────────┬───────┐
│ figure                    │ estimated │ measured │ ratio │
├───────────────────────────┼───────────┼──────────┼───────┤
│ prompt (llama-bench)      │  21643.33 │ 18231.12 │  0.84 │
│ generation (llama-bench)  │     41.34 │   279.76 │  6.77 │
│ generation, first request │     41.34 │   254.88 │  6.17 │
│ generation, 1K prompt     │     41.34 │   224.74 │  5.44 │
│ prompt, 1K request        │  21643.33 │ 19126.99 │  0.88 │
│ peak VRAM                 │   6.5 GiB │  5.1 GiB │  0.79 │
└───────────────────────────┴───────────┴──────────┴───────┘
4 figures are outside the 0.8 to 1.25 band an uncalibrated formula is expected to land in.
This configuration was not paging.
Peak VRAM stayed clear of the card's total, so there was nothing for the driver to page.
Nothing was written to the database, so no estimate will be relabelled.
```

One line of that run is left out above: between the paging verdict and the last line the
command prints a sentence quoting the VRAM percentage and the speed ratio, and on a card that
stayed clear there is no speed ratio to quote, so the sentence comes out malformed. That is a
bug in the renderer rather than something to read.

Ratios between 0.8 and 1.25 are normal before calibration, and the command counts the ones
that are not. Anything outside is worth a look: a generation ratio far below 1 with VRAM at
the ceiling is paging; a prompt ratio far below 1 usually means the micro-batch or PCIe
assumption is off for this machine.

**The generation rows of that run are not a like-for-like comparison, and the page keeps them
anyway.** The estimate is the plan's, which sizes this model for 32,768 tokens and so charges
3.5 GiB of key-value cache against the card on every token; `llama-bench`'s `tg128` generates
128 tokens into a cache that is almost empty, and the server questions beside it are barely
longer. The measurement is right, the estimate is right for the context it was made at, and
the ratio between them is measuring the difference in context. A comparison worth trusting has
to estimate at the context the run actually used, and this one does not yet — which is exactly
the kind of thing a sample that only ever showed ratios near 1 would have hidden.

## Calibration

`llamafit bench --calibrate` fits, from every stored result on this host:

| Factor | What it explains | What it needs |
|---|---|---|
| `eff_vram` | how much of the card's bandwidth the decode loop reaches | a run whose traffic is on the card |
| `eff_ram_sequential` | the same for a contiguous read from system memory | a run with dense weights in RAM |
| `eff_ram_scattered` | the same for a routed expert read, which is slower | a run with experts in RAM |
| `fixed_overhead_s` | everything a token costs that is not a byte moved | always in the fit; it is the intercept |
| `eff_pp` and `pcie_effective_gbps` | prompt compute, and what an expert set really streams at | two micro-batches (`bench --sweep`) |
| compute-buffer base and slope per micro-batch | the least predictable allocation | two contexts at that micro-batch |

**Only what the data determines is fitted.** The four generation constants are the four terms
of section 10.1's formula, so each run is one linear equation in them, and a constant is
fitted only when its term appears in some run, when there are at least as many distinct
configurations as free parameters, and when the configurations vary independently enough to
be told apart. Below that, `bench --calibrate` names the constant and says which of the three
it was missing. Two runs cannot pin three constants however carefully they were taken, which
is why the reference machine's own calibration set has four in it and why two of them exist
only to isolate a pool.

**A fit whose answer is impossible is discarded whole.** An efficiency is a fraction of a
bandwidth; a value above one says a pool moved more bytes than it has. Least squares chooses
its parameters against each other, so when one lands outside its own definition the others are
not thereby right, and all of them are refused with the value each reached. That is not
hypothetical: the reference machine's four published runs, solved exactly, put `eff_vram` at
1.62. The shipped constants were chosen as round figures consistent with all four runs, which
is a different and more defensible thing than a solve.

**`layer_overhead_ms` is never fitted.** Section 10.1 removed the per-layer term — for a
28-layer model it exceeded the whole measured token — so a number fitted for it would be
stored, printed, and read by nothing.

A run the paging detector caught is kept as evidence and never fitted against: it measured the
driver moving pages, not the configuration. Neither is the first request after a cold start,
which measures a disk waking up. Three or four measured configurations of different sizes are
enough for the fit to be stable.

The fit is stored in the database beside the runs it came from, and printed with its refusals.
What it is not yet is *applied*: the estimator reads its constants from
`llamafit.constants.speed`, and the hardware profile block that would carry a per-host set has
one `ram_efficiency` where the estimator has two — a contiguous read and a scattered one reach
meaningfully different fractions of the same memory controller, and collapsing them into one
field would be exactly the kind of quietly wrong number this page is about.

**Nor is a stored run read back.** The estimator can be handed measurements and will prefer
them to its own formula, producing section 10.3's `measured` and `calibrated` labels — and
nothing hands it any. `fit`, `recommend` and `plan` hand it none: the catalog's own `measured`
blocks are printed beside the estimate rather than fed into it, and `benchmarks.sqlite` is not
opened at all. So a machine with a hundred runs behind it gets the same `estimated` board as
one with none. Until that wiring exists, what a benchmark buys you is the comparison `bench`
prints: your number beside the estimate, with the ratio between them, which is the thing the
estimate was never going to tell you on its own.

## Recording a measurement by hand

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
