# Closing the last gap between the specification and the code

Branch `fix/spec-truth`, three commits off `3f38fa8`.

| SHA | Subject |
|---|---|
| `1b0199e` | `feat(cli): wire section 13.1's five substitution flags` |
| `20fae69` | `fix(i18n): give one verdict one word, and drop a message nothing asked for` |
| `7e84c28` | `docs: describe the prompt formula that runs, and the flags that now exist` |

2,950 tests pass (2 skipped, 14 deselected as `hardware`/`network`), coverage 96.63 percent
against a floor of 85. `ruff check`, `ruff format --check`, `mypy` and the three generators
followed by `git diff --exit-code` are all clean on the committed tree.

---

## 1. The five flags

### What was built

`--profile`, `--memory`, `--ram` and `--cpu-cores` are global options on the root callback
and are stored on `CliState` exactly as typed. `cli/common.machine()` is the single place
they become a machine, and it is the only new seam:

```python
def machine(state: CliState, *, measure_bandwidth: bool = True) -> SystemReport:
    scanned: SystemReport | None = None

    def scan_host() -> Host:
        nonlocal scanned
        scanned = scan(measure_bandwidth=measure_bandwidth)
        return scanned.host

    host = resolve_host(scan_host=scan_host, profile=state.profile, ...)
    if scanned is None:
        return SystemReport(host=host, llamacpp=detect_llamacpp(), version=__version__)
    return scanned.model_copy(update={"host": host})
```

The closure is what makes `--profile` keep `resolve_host`'s promise end to end: naming a
profile means the scan is never taken, so a run scoring against somebody else's machine
does not read this one's, and a machine whose graphics driver hangs a probe can still
answer for a profile. llama.cpp is still detected either way, because the binary and the
GGUF files on this disk are real whichever machine the numbers describe — that is what
makes `llamafit --profile NAME system` a superset of `hardware show NAME --as-host` rather
than a duplicate of it.

Commands wired: `system`, `fit`, `recommend`, `plan`, `preset`, and `llamafit` with no
arguments (the non-interactive fallback already ran `recommend` through the same context;
the interactive path now opens the dashboard with the Simulate panel seeded and the badge
already showing). `doctor` **refuses** the four rather than ignoring them; `bench` refuses a
simulated machine as it already did.

Reading an existing global flag first was worth it. `--language` is eager and installs the
translator before Click renders anything; `--json` is stored on `CliState` and read by each
command. The substitution flags follow the second pattern and deliberately not the first:
they need `CliState`, not import-time effect, and `app.py` cannot import `cli/common.py`
because `cli/common.py` imports `CliState` from `app.py`. That is why the two sizes are
stored as the strings the user typed and parsed in `common.py`, rather than parsed in the
callback.

### The thing that must not be lost

`Host` has carried `simulation` and a computed `simulated` since profiles landed, and every
command that prints a host got the marking for free. A **board does not carry a host**. So
`recommend --json`, `fit --json` and `plan --json` had nothing at all in them that
distinguished an answer about a profile from an answer about the reader's own machine: no
host, no heading, no colour. The red line above a table is no use to a script.

`Board`, `FitBoard` and `PlanReport` now carry the same pair, set from the host they were
computed against, with `simulated` a `computed_field` so it cannot drift from `simulation`
and cannot be left out of the serialisation. `"simulated": false` on a scan is part of the
promise, for the reason the `Host` docstring already gives: absence is not evidence.

On the terminal, `render_simulation()` in `cli/render.py` produces one red line and the
three renderers put it above everything, in the two messages `render_host` already used, so
no new wording had to be invented and no translator was asked for the sentence twice.

One heading had to change with it. `fit`'s table title read *Fit on this machine*, one line
under *they are not this machine*; under a simulation it now reads *Fit on the simulated
machine*. That is the only place where the banner was contradicted in its own words.

### `--max-context`

It went on `Needs`, beside `min_context`, and **not** beside the other four. It is not a
host property: a substituted machine changes what the arithmetic runs on, and this changes
the question being asked of it. The web API's own parameter list in section 13.3 puts
`max_context` next to `min_context` for the same reason, and the profiles agent said so
explicitly when it declined to touch it.

It caps four things, through one helper (`placement.modes.context_ceiling`):

- the planner's ceiling and therefore its context ladder;
- `context_tiers`, the rungs a launch script would choose from;
- `max_context_fit`, the largest context reported;
- `requested_context`, the denominator section 11.2's score is measured against.

All four or none. A cap honoured in three of them would print a tier table stopping at
16,384 above a `max_context_fit` of 262,144, or mark every candidate down by half for
failing to reach a length the same request forbade.

`--max-context` below `--min-context` is refused by name. `Needs` refuses the pair with a
model validator so the web and the TUI cannot construct one either; `check_context_ceiling`
exists so the sentence a person reads names the two options they typed (`--max-context` /
`--min-context` on the command line, `max_context` / `min_context` over HTTP) rather than
the two fields they did not.

---

## 2. Section 10.2, and the calibration record

Section 10.2 showed one compute term charged wholly to the graphics card. The code has
split it by layer share since `f965604`, and charges the card's efficiency only to the
card's half. The section now shows the formula that runs, including the third and fourth
terms, and says:

- **`eff_pp` = 0.43**, from the one run in which the compute term is the whole formula —
  Qwen3-0.6B Q8_0 with every layer on the card, 21,734 prompt tokens per second, which at
  the `2 x active_params x ub` the formula charges is 26.1 TFLOP/s against a card the
  corrected table rates at 60.4.
- **`cpu_pp_tflops` = 0.44 TFLOP/s per performance core**, from the same model at
  `-ngl 0 -t 8`: 2,924 prompt tokens per second is 3.51 TFLOP/s across eight performance
  cores. It is effective already and is **not** multiplied by `eff_pp` — the card side has a
  published ceiling to scale and this side does not.
- Why the split exists at all: with an honest tensor figure, Gemma 3 27B (nine of
  sixty-two layers on this card) came out near 480 prompt tokens per second, which needs
  about 22 TFLOP/s from a processor measured at 3.5.
- That `--n-cpu-moe` is deliberately not read as CPU layers, because the streaming term
  already charges those bytes.
- That `eff_pp` disagrees with the only other reading available (the micro-batch sweep's
  slope implies about 5 TFLOP/s where the dense run says 26), that the disagreement is not
  resolved, and that a second machine is what would resolve it.

Section 10.1 had the same disease more quietly: its code block still showed one
`bytes_ram / (ram_bw x eff_ram)` term while the paragraph beneath it explained the two the
code runs. The block now carries both, plus the output head in RAM, which the code charges
and the block never listed.

`docs/calibration/2026-09-09-reference-machine.md` gains a second run table (the two prompt
figures the constants are fitted to) and two rows in the constants table, and its retired
section — already the place where a superseded figure is named, explained and its run kept
— gains two more:

- **`eff_pp` 0.30, and the 15 TFLOPS table it was divided into.** Both halves were wrong and
  they hid each other; 15 was the card's fp32 shader rate and no efficiency at or below one
  reaches 26.1 from it. The `llama-bench` micro-batch sweep it was fitted to stays in the
  record, because it is a real measurement of that model's prompt throughput — it is simply
  not the run that isolates a compute constant.
- **`CPU_FP16_TFLOPS_PER_CORE` 0.05**, which was argued rather than measured and was then
  scaled by `eff_pp` on the way out, so the rate actually spent was about twentyfold below
  what the machine does. Nothing caught it because the term only fired on a host with no
  card.

`docs/how-it-works.md` carried the superseded **generation** formula as well — one
system-memory efficiency and a `layers x 0.2 ms` overhead this project has established is
impossible — and the prompt formula with no CPU side. Both are now the ones in force. One
sample output in `docs/cli.md` still printed the RTX 4060 at 15.0 TFLOPS.

---

## 3. The two English messages

**`roomy` / `Comfortable:`.** Every verdict but this one opens its sentence with the word
that is in the column: `pages` → *Pages:*, `no room` → *No room:*. The sentence now reads
*Roomy: there is room for a longer context or a second model.* The column keeps `roomy`,
because `comfortable` is eleven characters in a table designed for eighty columns. The six
catalogs that had translated the old sentence had all already opened it with their own
column word (*Folgado:*, *Holgado:*, *À l'aise :*, *Reichlich:*, *Comodo:*), so their
wording is unchanged in substance.

**`any`.** Deleted. `app.js` never looked it up, and its source gave a translator no way to
know what it qualified, so every catalog that reached it left it in English. Inventing a use
site would have been inventing a reason for it to exist.

Both entries were removed from every catalog that carried them — only the six near-complete
ones did — rather than left to fall back silently. All thirty-seven now report 0 problems
against the template.

The flags added ten new messages, and rather than leave the six near-complete catalogs
poorer than they were, all ten are translated into `pt_PT`, `pt_BR`, `es`, `fr`, `de` and
`it`, along with the new simulated heading. Those six sit at 1,028–1,029 of 1,034; the six
still-untranslated entries in each are pre-existing (the `--min-tps` family and, in `pt_PT`
only, `Skip the RAM bandwidth measurement.`) and I left them alone.

---

## Answers

### 1. A board computed from a profile, and how a program can tell

```
$ llamafit --profile reference-rtx4060-128gb recommend --use-case coding --limit 4

SIMULATED  these figures come from the hardware profile
reference-rtx4060-128gb; they are not this machine
                          Recommended
+-------------------------------------------------------------+
| # | Model              | Quant      | Score | Gen/s | Fit   |
|---+--------------------+------------+-------+-------+-------|
| 1 | qwen3-coder-next   | UD-Q4_K_XL |  89.6 |  23.9 | tight |
| 2 | qwen3.8-flash-next | UD-Q4_K_XL |  76.2 |  13.6 | tight |
| 3 | qwen3-0.6b         | Q8_0       |  54.0 | 114.6 | tight |
+-------------------------------------------------------------+
Speeds are for 8K tokens of context so every row compares like with like; the
context column is the largest each one holds. Sized and scored for coding at
32K tokens.
Speeds are section 10's formula on its default constants. Nothing has been
benchmarked on this machine yet, so no figure here is a measurement.
Weights: quality 0.40, speed 0.20, fit 0.20, context 0.20.
```

The first two lines are red, and they are printed before the reader meets a figure. The
machine was not probed to produce them.

A program reads the same fact off the document:

```
$ llamafit --json --profile reference-rtx4060-128gb recommend --use-case coding --limit 1
{
  "simulated": true,
  "simulation": {
    "profile": "reference-rtx4060-128gb",
    "path": ".../llamafit/data/profiles/reference-rtx4060-128gb.json",
    "overrides": []
  },
  ...
}
```

`simulated` is a computed field, so it is in every serialisation of every board and every
plan, and it cannot disagree with `simulation`. Without `--profile` the same two keys read
`false` and `null`. With an override instead of a profile they read
`{"profile": null, "overrides": ["gpu_memory"]}` — identifiers a script filters on, not
prose.

**What is *not* in a board's JSON: the substituted values.** `overrides: ["gpu_memory"]`
says the card's memory was replaced; it does not say with what. The `Simulation` docstring
already made that choice for hosts ("the substituted values are in the host's own fields"),
and a board carries no host. `llamafit --memory 24G --json system` under the same flags is
where the figures are. I judged copying a whole `Host` onto three more documents the worse
trade, but it is a real limit and it is the first thing I would revisit.

### 2. What else the specification claims that the code does not do

I went through sections 3 to 14 against the code rather than working from the brief. In
rough order of how much a reader would be misled:

1. **Section 5.2, `llama_cpp.min_build`, is dead.** The catalog model has the field and
   documents it; nothing reads it. No candidate is ever marked *needs newer llama.cpp*, and
   `doctor` never says which build would unlock one. The whole mechanism the section
   describes — "marks candidates … rather than hiding them" — does not exist. This is the
   worst of the set, because a user on an old build gets a recommendation that will not
   load, with nothing said.

2. **Section 13.1's `llamafit info <model>` is missing half its row.** The spec promises
   "facts, quants, **budget on this host per quant**" with `--quant` and `--context`. The
   command takes a model id and nothing else, and prints no budget. It is the only command
   in the table whose notable flags are entirely absent.

3. **Section 14's `config.toml` does not exist.** Nothing in the package imports `tomllib`
   or reads a configuration file. `paths.config_dir` is computed and never used for this.
   Default use case, download directory, llama.cpp directory, weights overrides and theme
   are all specified and all unreachable. `build_board(weight_overrides=...)` is threaded all
   the way through `scoring/rank.py` and no caller ever passes it — the plumbing for the
   feature exists with no tap on the end. (Every other row of section 14's table is
   honoured.)

4. **Section 12.3's counterfactual advice is not built.** "What would move it up a tier (for
   example 'free 1.2 GB of VRAM to reach 64K context' or 'the Q3 quant would fit entirely in
   VRAM at 18 tokens per second')" — `render_explanation` gives the scores, the weights, the
   quality, the budget line by line, the context ladder and the token's time, and never the
   sentence that tells a reader what to *change*. Both example sentences in the spec are
   inventions; nothing resembling either exists.

5. **Section 13.3's `include_too_tight` query parameter does not exist** and is rejected by
   `_no_unknown_parameters` as an unknown parameter. (`max_context`, listed beside it, did
   not exist either until this branch added it.)

6. **Section 5.1 claims local GGUF files are "matched to catalog entries by SHA-256 when
   known, otherwise by file name".** Only the file-name half is implemented; there is no
   SHA-256 of a local model anywhere. The only checksumming in the project verifies the
   llama.cpp archive the installer downloads.

7. **Section 13.2's key bindings are not the TUI's.** The spec gives Board `u` use case,
   `c` capabilities, `l` license; those moved into the Needs form and no such keys exist.
   Needs is `Enter apply, r reset` in the spec and `Ctrl+S` / `Ctrl+R` in the code (`r` and
   `Enter` are taken). The code also adds `A`, `x`, `n`, `y` and `+/-` that the spec does not
   list. Every individual choice looks right; the table is simply out of date.

8. **Section 3.2's package layout is largely fictional now.** `constants.py` is a package;
   `speed/` is `bandwidths.py`, `estimate.py`, `traffic.py` rather than `generation.py`,
   `prompt.py`, `confidence.py`; `gguf/header.py` is `reader.py`; `budget/compute.py` is
   `compute_buffer.py`; `catalog/` has no `schema.py`, `custom.py` or `seed/`; `hardware/`
   is organised by device (`cpu.py`, `gpu.py`, `memory.py`, `disks.py`) rather than by
   operating system; `download/` and `bench/` share no file names with the sketch; and
   `i18n/`, which is thirty-seven catalogs and eleven modules, is not in the tree at all.

9. **Section 4.4's profile JSON is a schema the code refuses.** A file written from the
   spec's example fails validation by name, because the model forbids unknown fields and
   every field in the example was renamed or dropped. I labelled the example in this branch
   rather than rewrite it, because the four deviations were deliberate and are argued in
   `profiles-report.md`; but a reader following the spec would have written a broken file.

10. **Smaller:** section 4.3 calls the PCIe default "12 GB/s effective" while the code
    treats 12 GB/s as the raw link and applies `eff_pcie` 0.33 to get 4; and the spec's
    section 8.3 verdict table still names the long forms (Comfortable, Too Tight, Does Not
    Fit) that no interface prints.

I fixed items 9 and, incidentally, the two stale formula blocks in sections 10.1 and 10.2.
Items 1 through 8 are untouched: each is a feature, not a wording error, and each deserves
its own decision about whether the specification or the code should move.

### 3. Where `--max-context` went, and the argument against it

It is `Needs.max_context`, applied through `placement.modes.context_ceiling` to the
planner's ceiling, the context ladder, `max_context_fit` and `requested_context`.

**The argument against.** `Needs` is *what a person is asking for* — a use case, the
capabilities a model must have, the slowest speed worth having, the shortest context worth
having. Every other field is a property of the job. `max_context` is not: it is a property
of what the person is willing to *pay for*, and the honest place for that kind of constraint
is beside `--max-download`, which is also on `Needs` and is also not a property of the job.
So the objection is not that `Needs` is wrong; it is that `Needs` is now two things, and
nobody has said so.

The sharper version of the objection is about the fourth thing it caps. Capping the
planner, the ladder and the reported maximum is uncontroversial: all three are things the
run offers, and offering a rung the request forbade is offering something nobody wants.
Capping `requested_context` — the score's denominator — is a judgement, and it is
arguable in both directions. I read `--max-context 16384` on a coding request as "16K is all
I want", so scoring every candidate against the use case's 32,768 would mark them all down
by half for failing to reach a length the same request ruled out. Somebody else could read
it as "I can only *run* 16K, but tell me honestly that this job wants 32K", and want the
score to keep saying so. If that reading wins, the fix is one line in
`scoring/context_score.requested_context` and a caption saying which context the score used
— and `Board` already carries `requested_context` and `planned_context` separately, so the
document would not need to change.

A second, weaker objection: `--max-context` is inert on the commands that read no machine
and on `doctor`, where the other four are refused outright. I did not extend the refusal to
it, because every command that plans a context can honour it and refusing it on `doctor`
would have made the rule harder to state, not easier. But "global flag" is doing slightly
different work for the fifth flag than for the other four, and `docs/cli.md` now has to
spend two paragraphs saying so.

---

## Concerns

1. **A board's JSON marks *that* it was simulated, not *what* was substituted.** Discussed
   above. `overrides: ["gpu_memory"]` without the figure is thinner than what a `Host`
   gives, and a script comparing two runs cannot tell 16G from 24G without asking `system`
   under the same flags.

2. **`doctor` refuses the four flags; the catalog commands ignore them.** The rule I settled
   on is "a command that prints machine facts either honours the substitution or refuses
   it", which puts `system` and `preset` on one side and `doctor` and `bench` on the other,
   and leaves `list`, `info`, `catalog`, `hardware`, `llamacpp`, `install` and `serve`
   silently ignoring them. That is defensible and it is documented, but a user who types
   `llamafit --profile fleet info gemma-3-27b-it` expecting a budget for the fleet machine
   gets a model card and no complaint. It would be less surprising once `info` grows the
   budget section 13.1 promises it (finding 2 above), and worse until then.

3. **`preset` writes launch scripts for a simulated machine without saying so in the
   files.** This is the fleet case working as intended — generating scripts from a build
   server for twenty machines none of which is the build server — but the generated README
   stamps the host it was planned against without a word about the host being a profile. The
   terminal says `SIMULATED`; the artefact on disk does not. I did not touch the preset
   renderer because it is phase 2 and it would have meant a new message in every catalog,
   but it is the one place a simulated answer outlives the session that produced it.

4. **`--memory` on a profile keeps the profile's card name.** Inherited, not introduced:
   `override_host` resizes VRAM and leaves the card's name, bandwidth and compute figures
   alone. A "24 GB RTX 4060" is a fiction, and now that both flags can be combined on one
   command line it is easier to reach. Only `simulation.overrides` says which of the two
   questions the reader asked.

5. **The catalogs' six untranslated entries stay untranslated.** Four are the `--min-tps`
   family and one is `Skip the RAM bandwidth measurement.` (in `pt_PT` only). I translated
   the ten messages this branch introduced and left those alone, on the grounds that a
   pre-existing gap is not mine to close silently; but the six catalogs are one message
   further from complete than "nearly complete" suggests.

6. **`Fit on the simulated machine` is the only heading corrected.** Several sentences the
   board and the plan print still say "this machine" — the `does-not-fit` verdict sentence
   ("more memory than this machine has"), the `Card needs` / `On this machine` column
   heading, the measured-elsewhere caption. The red banner says "they are not this machine"
   above all of them and is the correction, but it is a frame the reader has to hold. Fixing
   every one would double a dozen messages across thirty-seven catalogs, and I judged the
   table title — directly under the banner, contradicting it word for word — the only one
   that had to change.

7. **`--max-context` below 16,384 produces an empty context-tier table.** The ladder's
   lowest rung is 16,384, so `--max-context 8192` plans and reports 8,192 correctly and
   prints a tier table with no rows. Nothing lies, but a launch script generated from it has
   no rung to choose. `MIN_CONTEXT_TOKENS` exists to stop the planner halving forever and
   `context_ladder` already has an "explicit request for less" branch; a matching branch for
   the tier table would be the tidy fix and I did not add one.

8. **Two prompt-processing constants, one machine, one run each.** Not introduced here —
   this branch only wrote down what was already true — but section 10.2 now says out loud
   that `eff_pp` and `cpu_pp_tflops` each rest on a single measurement, and that `eff_pp`
   disagrees fivefold with the only independent reading available. Writing it down does not
   make it safer. A second machine is the thing this estimator most needs.
