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

**Each measurement is judged against the estimate for its own conditions, not the plan's.**
The formula is run once per row, at the key-value cache that row filled and the micro-batch it
used: 128 tokens for `tg128`, about eleven hundred for the thousand-token request, 2,048 for
`pp2048`. Section 10.1 charges a token for the cache it reads, so an estimate made at the
planned context and a measurement taken at an empty one are two answers to two questions, and
their quotient is not an error. The plan's own figure is printed under the table, quoted and
never given a ratio, because nothing in a benchmark fills 32,768 tokens.

Results are stored in `benchmarks.sqlite` in the data directory with the host fingerprint,
llama.cpp build, model, quant, the settings the run reported about itself, and the timestamp.
`llamafit bench --show` lists them. The Benchmarks screen the design sketches is not built;
`--show` and `--calibrate` are the whole of the interface to the database today.

**A result is refused rather than stored when its conditions are not knowable.** llama.cpp
clamps a context it cannot honour and expands `-ub 512,1024,2048` into a run apiece, so what a
row records is the tool's own report of what it did, checked against the command line it was
given; a disagreement stops the write and names the setting. The host fingerprint includes the
driver version, so a driver update silently retires the measurements taken before it. That
is the safe direction, since the label then falls back to `estimated`, which is true.

## Reading estimate versus measured

```
$ llamafit bench qwen3-0.6b --no-store
                             Estimated against measured
┌───────────────────────────┬─────────┬─────────────┬───────────┬──────────┬───────┐
│ figure                    │ context │ micro-batch │ estimated │ measured │ ratio │
├───────────────────────────┼─────────┼─────────────┼───────────┼──────────┼───────┤
│ prompt (llama-bench)      │   2,048 │       2,048 │  21643.33 │ 18125.33 │  0.84 │
│ generation (llama-bench)  │     128 │       2,048 │    274.00 │   274.83 │  1.00 │
│ generation, first request │     139 │       2,048 │    273.48 │   261.23 │  0.96 │
│ generation, 1K prompt     │   1,184 │       2,048 │    231.79 │   230.37 │  0.99 │
│ prompt, 1K request        │   1,184 │       2,048 │  21643.33 │ 20308.62 │  0.94 │
│ peak VRAM                 │  32,768 │             │   6.5 GiB │  5.2 GiB │  0.80 │
└───────────────────────────┴─────────┴─────────────┴───────────┴──────────┴───────┘
The plan sizes this configuration for 32,768 tokens and estimates 41.34 generated tokens per
second there. Nothing above ran at that context, so that figure is reported and not compared.
This configuration was not paging.
Peak VRAM stayed clear of the card's total, so there was nothing for the driver to page.
Peak VRAM was 65 percent of the card. Generation was not compared with the estimate.
Nothing was written to the database, so no estimate will be relabelled.
```

Ratios between 0.8 and 1.25 are normal before calibration, and the command counts the ones
that are not. Anything outside is worth a look: a generation ratio far below 1 with VRAM at
the ceiling is paging; a prompt ratio far below 1 usually means the micro-batch or PCIe
assumption is off for this machine.

**The first two columns are why the last one means anything.** An earlier version of this
page showed the same run with generation at 6.77 times the estimate, which read as a badly
wrong formula and was nothing of the kind: the estimate in that column was the plan's, made
at 32,768 tokens of key-value cache, and the measurement beside it was `tg128` generating
into a cache that starts empty. Section 10.1 charges `kv_bytes_per_token × working_context`
on every token, so the two were 3.5 GiB per token apart before anything was measured. The
measurement was right, since `docs/calibration/` records the same machine at 279.5 tokens
per second under `tg128`, which the 279.76 in that sample matched to a tenth of a percent.
The estimate was right too, for the context it was made at. The comparison was what was wrong.

Three fixes were available: run `llama-bench` at the planned context with `-d`, estimate at
the context `llama-bench` used, or print both figures and compare neither. The command takes
the second. `-d` needs a build new enough to have the flag and costs a full 32,768-token
prefill per row, and it cannot be done at all for the three fixed server requests without
making them something other than the fixed questions they are; estimating at the measured
depth costs nothing, works on every build, and is how the rest of the project already works.
[The reference machine's record](calibration/2026-09-09-reference-machine.md) gives
"generation, short context, `llama-bench tg128`" and "generation at 32K tokens of context" as
two measurements on two lines. What it gives up is stated rather than hidden: at 128 tokens the
key-value term is a rounding error, so a benchmark checks every term of section 10.1 **except
the growth of the cache**, and the plan's figure for the planned context, the line under the
table, is the one number here that nothing has checked.

The paging block under the table is three lines: the verdict, the reason it was reached, and
the figures behind it. A card that stayed clear of full is cleared on the memory signal
alone, so the speed half of section 16.4 is never reached and there is no ratio to quote; the
third line says so rather than leaving a gap where the number would have been.

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

**A run is one equation only if its recorded conditions are its own.** Each row carries the
traffic the estimator believed it read, at that row's context and micro-batch, and that
record is what the fit does arithmetic on. Sharing one traffic record across a whole
benchmark, which is what happened until the comparison above was fixed, gave the solver
five rows with an identical column, and identical columns cannot separate parameters: on this
machine `eff_vram` and `fixed_overhead_s` were both refused as "not identifiable" from a
sweep that contained exactly the measurements needed to pin them. The same benchmark now fits
both: `eff_vram` at 0.70 against the shipped 0.67, and a fixed overhead of 1.03 ms against
the shipped 1 ms.

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

**`layer_overhead_ms` is never fitted.** Section 10.1 removed the per-layer term, because
for a 28-layer model it exceeded the whole measured token, so a number fitted for it would
be stored, printed, and read by nothing.

A run the paging detector caught is kept as evidence and never fitted against: it measured the
driver moving pages, not the configuration. Neither is the first request after a cold start,
which measures a disk waking up. Three or four measured configurations of different sizes are
enough for the fit to be stable.

The fit is stored in the database beside the runs it came from, and printed with its refusals.
What it is not yet is *applied*: the estimator reads its constants from
`llamafit.constants.speed`, and the hardware profile block that would carry a per-host set has
one `ram_efficiency` where the estimator has two: a contiguous read and a scattered one reach
meaningfully different fractions of the same memory controller, and collapsing them into one
field would be exactly the kind of silently wrong number this page is about.

**Nor is a stored run read back.** The estimator can be handed measurements and will prefer
them to its own formula, producing section 10.3's `measured` and `calibrated` labels, and
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
