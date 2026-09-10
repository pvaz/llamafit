# `llamafit bench`: turning predictions into measurements

Worktree `llamafit-wt-bench`, branch `feat/bench`, on top of `101206f`. Nothing was merged,
rebased, pushed, or touched on another branch or worktree.

| Commit | Subject |
|---|---|
| `4376e4e` | `feat(bench): record, compare and calibrate real llama.cpp runs` |
| `c0e066c` | `feat(cli): add \`llamafit bench\`` |

## What was built

`src/llamafit/bench/`, ten modules, and `src/llamafit/cli/bench_cmd.py`. The one line in
`cli/app.py` is the registration import (plus its `__all__` entry). Two documentation files
were brought up to what shipped: `docs/cli.md` (a test reads the command list out of the
Typer app and demands the page name every command) and `docs/benchmarking.md`, which
described a calibration that writes to a hardware profile — it does not, and the report says
why below.

| Module | What it is |
|---|---|
| `types.py` | `RunConditions`, `RunTraffic`, `BenchRun`, `PagingCheck`, `Calibration`, `Refusal`, `ComparisonRow`, `BenchReport`; the `as_measurement()` bridge into `llamafit.models.catalog.Measured` |
| `fingerprint.py` | host fingerprint, host summary, conditions hash, and `conditions_conflicts` — what a command line asked for against what the tool says it did |
| `parse.py` | `llama-bench -o json`, the markdown table, `llama-server -v` buffer lines, `nvidia-smi` VRAM |
| `store.py` | SQLite at `data/benchmarks.sqlite` (section 14), and the refusals |
| `paging.py` | section 16.4's detector |
| `compare.py` | estimate beside measurement, with the gap |
| `lstsq.py` | a rank-reporting least-squares solver, pure Python (NumPy is optional here) |
| `calibrate.py` | the fit, and the refusal of everything it cannot identify |
| `run.py` | orchestration behind four protocols, with a fake beside each real adapter |
| `cli/bench_cmd.py` | the command and its three renderings |

## Verification

| Check | Result |
|---|---|
| `pytest --cov -m "not hardware and not network"` | **1880 passed**, 1 skipped, 11 deselected; coverage **95.94%** (floor 85%) |
| `ruff check .` | All checks passed |
| `ruff format src tests scripts` then `ruff format --check .` | 234 files already formatted |
| `mypy` (strict) | Success: no issues found in 113 source files |
| `gen_messages.py && gen_schema.py && gen_models_md.py && git diff --exit-code` | clean |

New tests: 132 across `test_bench_parse.py`, `test_bench_store.py`, `test_bench_paging.py`,
`test_bench_calibrate.py`, `test_bench_compare.py`, `test_bench_run.py`,
`test_bench_adapters.py`, `test_cli_bench.py`. None of them runs a benchmark, starts a
process, opens a socket or needs a card. Per-module coverage of the new code: `paging.py`
and `compare.py` 100%, `lstsq.py` 99%, `store.py` 98%, `types.py` 95%, `calibrate.py`,
`fingerprint.py` and `parse.py` 94%, `bench_cmd.py` 92%, `run.py` 89% (what is left there is
mostly `_SubprocessHandle.stop`'s kill path, which needs a wedged process to reach).

The message template grew from 585 to 635 entries; every string a person reads goes through
`_()` or `ngettext`, and the stored keys (`PagingReason`, `RefusalReason`, metric names) are
deliberately **not** translated, because a row written by somebody working in Portuguese is
read later by somebody working in Arabic. The prose is produced at render time by
`paging.reason_text`, `calibrate.refusal_text` and `compare.metric_label`.

---

## 1. What does a stored result record, and could two results ever be confused?

A row is a serialised `BenchRun`. The SQLite columns exist only to find rows; the content is
the model, so a field added to it cannot drift away from a column somebody forgot to add.
The version the row was written under travels with it, and a database written by a newer
LlamaFit is refused whole rather than read half-way.

**The conditions** (`RunConditions`) carry three separate things, and the separation is the
answer to the second half of the question:

- `argv` — exactly what was executed, program first, never normalised or reordered.
- `settings` — the run as the *tool itself reported it*, keyed by llama.cpp's flag names:
  `ngl`, `ub`, `b`, `t`, `ctk`, `ctv`, `fa`, `n-cpu-moe`, `ot`, `c`. It starts from the
  command line and is then overwritten by whatever the tool said about itself, so a value
  llama.cpp chose or clamped replaces the one it was offered.
- `flag_string` — built from `settings`, in section 9.5's order, and this is what a later
  reader compares against a placement.

Plus: host fingerprint and a human-readable host summary, llama.cpp build number and commit,
model id, quant, model file and its size, context, micro-batch, prompt and generation token
counts. The run itself carries the speeds, the time to first token, peak VRAM and the card's
total, the buffer sizes from the server log, the paging verdict, the tool-call result, the
estimate made *before* the run, and a `RunTraffic` record of what the estimator believed the
run would read (bytes per pool and the raw bandwidths they were charged against).

That last one is there so a calibration is arithmetic on recorded facts. Recomputing traffic
months later would mean rebuilding a placement from a stored command line against a catalog
and a set of GGUF facts that have both moved since — the same failure this section is about,
arriving through a different door.

**Could two results be confused?** Three defences, in order of how much work they do.

*The kind is part of the identity.* `llama-bench-tg`, `llama-bench-pp`, `server-short`,
`server-1k`, `server-toolcall`. The reference machine generates 13.0 tokens per second under
`llama-bench tg128` and 13.9 on a real thousand-token request at the same flags, and 8.3 on
the first request after a cold start while the model's data pages in from disk. Three
numbers, one configuration, all true. Nothing ever averages across kinds.

*The conditions hash.* SHA-256 over kind, fingerprint, build, commit, model id, quant, the
model file's basename and size, `flag_string`, context, micro-batch, and the prompt and
generation lengths. Context and micro-batch are hashed separately as well as being inside
`flag_string`, because they are the settings most likely to be swept and a caller that filled
the field without also putting the value in the settings map would otherwise give two
genuinely different runs one hash. (That is not hypothetical — it is exactly what the first
version of the compute-buffer test caught.) What is deliberately *outside* the hash: the
port, the alias, and the directory the model happened to live in. Two people benchmarking the
same file on the same machine from different directories measured the same thing.

*The flags a record hands out are complete, and are the row's own.* This is the specific gap
the brief refers to. `llamafit.speed.estimate._flags_agree` matches a stored benchmark to a
placement by reading `-ngl`, `--n-cpu-moe` and `-ub` out of a recorded command line, and its
rule — correctly — is that *a flag the measurement does not record cannot disagree*. A record
saying `-ngl 99` and nothing about `--n-cpu-moe` therefore matches every offload with that
layer count, including ones it says nothing about. The matcher cannot invent what it was not
told, so the defence has to be on the writing side: `llama_bench_argv` names every setting the
placement carries even where llama-bench would default to it anyway, and `flag_string` renders
all of them. There is nothing left for a placement to differ in silently.

The related trap is the sweep. `bench --sweep` passes `-ub 512,1024,2048` on **one** command
line and gets three rows back, each having run at one of them. A reader parsing `-ub` out of
the raw line would take 512 and file a 2048-token measurement under it. `flag_string` is built
from the row's reported micro-batch and not from `argv`, which is why the two fields exist
separately; `argv` stays as provenance.

**Two things are refused rather than stored.** A run whose report disagrees with its command
line — `conditions_conflicts` compares them, treating a comma list as satisfied by any of its
members, booleans across `on`/`1`/`true`, and numbers numerically; `-ot` and `-fa` are recorded
but never compared, because the tool reports which tensors moved rather than which patterns it
got, and `-fa auto` resolves at load time. And a run on a simulated host: `host_fingerprint`
prefixes those `sim-`, and the store rejects the prefix outright.

**One direction of confusion is left open on purpose.** The fingerprint includes the driver
version, so a driver update retires every measurement taken before it. That is the safe
direction — the label falls back to `estimated`, which is true — but it does mean a user who
updates a driver silently loses their `measured` numbers. The `host_summary` stored beside the
fingerprint is what lets somebody see what changed.

## 2. How many measurements does the calibration need, and what does it do below that?

Section 10.1's generation formula is linear in its four constants, so identifiability is a
question with an arithmetic answer rather than a matter of taste. With `A = 1/eff_vram`,
`B = 1/eff_ram_sequential`, `C = 1/eff_ram_scattered`, `F = fixed_overhead`, each run is

```
1/gen_tps = (device_bytes/device_bw)·A + (sequential_bytes/ram_bw)·B
          + (scattered_bytes/ram_bw)·C + F
```

**The rule is: as many distinct configurations as free parameters, and no fewer.** A constant
is free when its column carries a non-zero somewhere; the constant column is always free. So:

| Fit | Free parameters | Minimum distinct configurations |
|---|---|---|
| Generation, all four pools touched | 4 | 4 |
| Generation, GPU + contiguous RAM only | 3 | 3 |
| Prompt (`eff_pp`, `pcie_effective_gbps`) | 2 | 2 distinct micro-batches |
| Compute buffer, per micro-batch | 2 | 2 distinct contexts at that micro-batch |

Repeats of one configuration are not extra measurements: they are grouped by conditions hash
and reduced to their **median** (not their mean, and pointedly not their best — the fastest of
five runs is a claim about what a machine did once).

**Below that it fits nothing and says so, per constant.** Five refusal reasons, each stored as
a key and rendered as a sentence:

- `no-data` — nothing measured touches this term, so there is no equation it appears in.
- `too-few-measurements` — with the two counts. Two runs, three parameters.
- `not-identifiable` — enough rows, but the columns do not vary independently. Three runs
  where the expert set is always exactly twice the card's traffic vary in one direction, not
  two, and nothing in them says which pool the time went to. The count check passes; the rank
  check catches it. Both exist.
- `unphysical` — the fit produced a number its own definition does not allow.
- `discarded-with-the-fit` — the parameter was plausible on its own and went anyway.

`layer_overhead_ms` is never fitted at any number of measurements: it is refused with
`no-term-in-the-model`, because section 10.1 removed the per-layer term (0.20 ms × 28 layers
is 5.6 ms against a whole measured token of 3.58 ms). A number fitted for it would be stored,
printed, and read by nothing.

**The finding worth reading twice.** A least-squares solution is a single object: its
parameters are chosen together, against each other, to explain the same rows. If one lands
somewhere impossible the others are not thereby correct — they are the numbers chosen to
compensate for an impossible one. So the whole fit is discarded and every parameter is
reported with the value it reached.

This is not defensive theory. **The reference machine's own four published runs, solved
exactly, are inconsistent.** Four equations, four unknowns, one solution, and it is:

| Parameter | Exact solve | Shipped constant |
|---|---|---|
| `eff_vram` | **1.6205** | 0.67 |
| `eff_ram_sequential` | 0.7998 | 0.70 |
| `eff_ram_scattered` | 0.4512 | 0.57 |
| `fixed_overhead_s` | 0.002512 | 0.001 |

An efficiency of 1.62 says the card moved more bytes than it has bytes to move. The two expert
runs disagree in direction — Coder-Next comes in *slower* than the formula predicts (23.0
against 23.7) and Flash-Next *faster* (13.9 against 13.55) — and in an exactly determined
system that disagreement has nowhere to go but into a parameter. `test_bench_calibrate.py`
pins this. It also means the shipped constants were arrived at by judgement (round figures
consistent with all four runs), which is a different and more defensible thing than a solve,
and `constants/speed.py` is right to present them the way it does.

Two consequences follow. An exactly determined fit reports `generation_residual = None` rather
than zero, because a residual of zero by construction is not agreement and a reader is owed
that distinction. And `docs/benchmarking.md`'s advice — three or four measured configurations
of different sizes — is the practical floor: four is the minimum, and a fifth is what gives
the fit a residual to be judged by.

`server-short` is excluded from the generation fit by name: it measures a disk waking up. A
run the paging detector caught is excluded too, from both the fit and from
`store.measurements()` — it is kept in the database as the evidence that the configuration
pages, but it is not a measurement of the speed that configuration runs at.

## 3. What would make the paging detector cry wolf, and what would make it miss?

The signature is a conjunction: peak VRAM ≥ 97% of the card's total **and** measured
generation < 60% of the estimate. Either alone is ordinary — a well-planned configuration is
*supposed* to fill the card, and an estimate can be 40% out for reasons that start with the
estimate being wrong.

**Crying wolf.** Three ways, in decreasing likelihood.

1. *Another process owns the VRAM.* `nvidia-smi --query-gpu=memory.used` reports the whole
   card, not this process. A browser, a compositor or a second model holding 3 GB pushes the
   reading over 97% while the run itself fits, and if the estimate is also optimistic the
   detector fires. This is the realistic false positive. It is not mitigated: reading
   per-process VRAM would need NVML rather than the CSV interface, and the same reading is
   what the budget already reserves against.
2. *The estimate is too high.* The formula is expected to land within 0.8–1.25, and 0.60 sits
   well below anything ordinary error produces — but a model whose active-parameter count the
   catalog overstates, or a machine whose measured RAM bandwidth is optimistic, can predict
   twice the truth. Combined with a card that a good plan deliberately filled to 98%, that is
   a false positive with no innocent-looking signal to distinguish it.
3. *A card that is simply small.* On a 4 GB card, 3% is 120 MiB, which is inside what a
   desktop moves between one sample and the next.

**Missing.** Four ways.

1. *Mild paging.* A configuration 200 MiB over the card pages a little and slows a little.
   The calibration record shows 10.2–11.0 tok/s against 13–14 when 2.3 GB pages, which is a
   ratio near 0.75 — above the threshold, so it is reported as not paging. The threshold is
   set where it is precisely so that ordinary estimate error does not fire it; the cost is
   that mild paging reads as a slightly disappointing configuration.
2. *A pessimistic estimate.* If the formula under-predicts by a third and the run pages by a
   third, the ratio lands near 1.0 and the second half never fires. The detector compares
   against a prediction, so it inherits that prediction's errors — which is also why the
   estimate it uses must be the one from *before* the run. (After calibration this direction
   gets better, not worse.)
3. *No reading at all.* No `nvidia-smi`, an AMD or Apple machine, or a card that answers
   `[N/A]`. That returns `paged=None` with a reason, never `False`, and the command prints
   "Paging could not be checked" — a configuration nobody looked at has not been cleared.
   `parse_used_vram` deliberately returns `None` for `[N/A]` rather than zero, because a zero
   would clear a configuration on the strength of a number that was never taken.
4. *Sampling around requests rather than during them.* Readings are taken before the first
   request and after each one. llama.cpp allocates weights, cache and compute buffers while
   the model loads, so the peak is normally reached before any token is generated — but a
   transient spike inside a request would be missed. Sampling during the run would need a
   second thread; the trade was made deliberately and is documented in `probe_server`.

The one negative the detector gives on a single signal is sound: a card comfortably below the
threshold is cleared without needing the speed half at all, because the driver pages only what
does not fit. That is why `--no-server` still leaves the question open rather than answering
it — `llama-bench` exits before anything can be read, so its rows carry no VRAM reading.

When it does fire, it names the largest rung of the plan's context ladder that the budget did
not already call `too-tight`, so the cure offered is never itself a configuration that pages.

---

## Concerns

1. **The calibration is fitted and stored but not applied.** `ProfileCalibration` in
   `models/hwprofile.py` has one `ram_efficiency` where the estimator has two — a contiguous
   read reaches 0.70 of the memory controller and a scattered one 0.57, and that difference is
   the whole reason `speed/bandwidths.py` reports three figures instead of two. Writing a fit
   into that block would have to pick one, and a silently ambiguous constant is exactly what
   this work exists to prevent. `models/hwprofile.py` is outside this package, so I did not
   change it. A one-field change (`ram_efficiency_sequential` / `ram_efficiency_scattered`,
   or a `ram_efficiency` object) plus a read in `resolve_bandwidths` closes it. Until then
   the route from a benchmark to a relabelled estimate is the stored run itself, via
   `BenchStore.measurements()` → `estimate_speed(measurements=...)`, which is already the
   ladder section 10.3 describes.
2. **Nothing yet calls `BenchStore.measurements()`.** The bridge exists and is tested, but
   `services/plan.py` and `services/recommend.py` pass `measurements=()`, so `plan`, `board`
   and `recommend` still print `estimated`. Wiring it is two lines in `services/plan.py` and
   one in `services/recommend.py` — both outside this package and both likely to collide with
   the other five agents, so I left them. `bench` itself shows the comparison, which is the
   part that cannot be got anywhere else.
3. **Section 16.1 asks for peak VRAM "from the vendor tool during the run"; this samples
   around requests.** See miss (4) above. Adequate for the allocation-time peak that decides
   paging, not for a transient.
4. **`bench --all`** (in the older `docs/cli.md` stub) is not implemented. It would be a loop
   over `llamacpp.local_models`; I left it out rather than ship a flag that runs an unbounded
   number of multi-minute benchmarks without a confirmation design.
5. **The paging thresholds live in `bench/paging.py`, not `constants/`.** They are section
   16.4's own figures and the project's habit is to put constants in `llamafit.constants`;
   `constants/speed.py` is shared with other agents this cycle, so I kept them local with
   their derivation. Worth moving later.
6. **`llama-bench`'s JSON field names vary between builds.** `parse.py` reads defensively and
   a field a build does not emit is left out of `settings` rather than filled from the
   request — but that means an older build produces a less complete record, and the markdown
   fallback less complete still. A record read from the table says nothing about the thread
   count because the default table has no column for it, and that absence is honest rather
   than papered over.
7. **One transient test failure** was seen in a single intermediate run (`1 failed` with no
   name captured) and has not reproduced across five subsequent full runs, including the two
   verification runs on the committed state. I could not identify it; it is worth watching if
   CI shows a flake.
