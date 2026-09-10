# Section 11 placed, and a way to say nobody is waiting

Branch `docs/place-section-11`, worktree `llamafit-wt-spec`, two commits off `main` at `247851d`.

| SHA | Subject |
|---|---|
| `5edb6a2` | `feat(scoring): let a request name the speed below which nothing is ranked` |
| `7992b36` | `docs: place section 11's scores, which described neither one as built` |

```
pytest --cov -m "not hardware and not network"   2,925 passed, 2 skipped, 14 deselected, 96.63 % (floor 85 %)
ruff check .                                     All checks passed
ruff format --check .                            334 files already formatted
mypy                                             Success: no issues found in 164 source files
gen_messages && gen_schema && gen_models_md      git diff --exit-code clean
```

Baseline before the work: 2,908 passed, 96.63 %. Seventeen tests added, none removed, none
changed. `scoring/fit_score.py`, `scoring/speed_score.py` and `scoring/rank.py` are each at
100 percent line and branch coverage.

---

## 1. What was placed

**11.3, the fit score's two arms.** `fitscore-report.md`'s wording, verbatim apart from
paragraph breaks. Checked line by line against `scoring/fit_score.py`: the crowding arm's
100 → 40 → 0 at 0.80 / 0.98, the capacity arm's `w = resident weight bytes / (vram_available
+ ram_available)`, full marks at `w ≥ 0.50`, `WASTE_PER_HALVING = (100 − 70) / log₂(0.5/0.20)
= 22.687`, zero at `0.5 × 2^(−100/22.687) = 0.0236`, which is the report's "about 0.024" and
"about a fortieth". The eight `MODEL_COMPONENTS` lines, the disk pool exclusion and the
`min` of the two arms all match.

**11.4, the log ramp and the reading floor.** `speedscore-report.md`'s wording, with its
last paragraph replaced by `ranking-report.md` §6 as that report instructs, and with one
paragraph of my own at the end for `--min-tps`. Checked against `scoring/speed_score.py`:
the six targets, `READING_TPS = 6.0`, `THROUGHPUT_ONLY = {embedding}` and its straight line,
`speed_ramp`'s `100 × log₂(tps/floor) / log₂(target/floor)`, the non-strict `>=` in
`keeps_up_with_reader`, and the prompt modifier's 10 below 100 and 20 below 40, replacing
rather than stacking.

**11.1** gains `ranking-report.md`'s paragraph naming where the three exclusions live.

**`docs/how-it-works.md` §7**, the Speed bullet, was two revisions behind and is replaced
with the ranking report's text plus a sentence for the new flag. (§7's Fit bullet was
already current — the fit-score branch updated it.)

**`CHANGELOG.md`** gains the two scoring fixes, which both shipped with no entry, and the
flag. **12.1** gains `min_tps`, **13.1** gains `--min-tps`.

---

## 2. Where a report's wording disagreed with the code, and which won

**The one that mattered: `speedscore-report.md`'s closing paragraph, and the code won.**
Its proposed 11.4 ended with "What the floor costs": *every candidate below six tokens per
second now scores zero, so on a machine where nothing reaches six the speed column stops
separating candidates at all and the board is ordered by quality, fit and context.* By the
time `fix/ranking` merged, that consequence no longer follows. A candidate below the floor
is not scored zero and ranked; it is excluded. On such a machine the board is not "ordered
by quality, fit and context" — the ranked half is empty and every candidate is in the
excluded half with its speed. The ranking report saw this, said its §6 block "replaces (2)'s
last paragraph, which describes a consequence that no longer follows", and supplied the
replacement. I placed the replacement and dropped the original. Both reports had to be read
before either could be placed, exactly as the brief warned.

**One I created and am disclosing.** `ranking-report.md` §6 asserts flatly: *"A throughput
use case can never be excluded this way."* With `--min-tps` that is no longer true — an
embedding request that names a figure can exclude on it. I placed the sentence as *"A
throughput use case has no floor of its own, so it is never excluded this way unless the
request names a figure."* The distinction the original was drawing survives: the default
still answers "who is waiting", and embedding still has no reader.

**One where the old specification, not a report, was the odd one out.** 11.3 used to read
"100 for `0.50 ≤ u ≤ 0.80`". `crowding_score` has never had a lower bound — it returns 100
for every utilisation at or below 0.80, including zero. Both the report's wording and the
code agree; the lower bound was a leftover from the revision where one number answered both
questions. It is gone.

**Two numbers that look like a disagreement and are not.** The two reports cite Gemma 3 27B
at 2.1 and at 2.2 tokens per second for what reads like the same board, and the bundled
reference profile gives 1.6 while a live scan today gives 2.2. These are different runs on a
machine whose free card memory moves between them, which the ranking report says in its
preamble. I kept each report's own figure in the paragraph that report wrote, because each
is tied to a specific claim — 8.4 out of 100 goes with 2.1, and 54.92 against 51.34 goes with
2.2 — and rewriting them to one number would invent a measurement nobody took. I added one
clause to 11.4 saying they are two scans of one machine, so the pair does not read as
carelessness.

**Everything else checked out.** The fit report's "a fifth to a quarter of its composite" is
the fit weight's real range (0.20 to 0.25 across the six use cases). Its component list is
the code's `MODEL_COMPONENTS` exactly. The speed report's targets, thresholds and penalties
are the module's constants. The 36.88 it quotes for Llama at 10.16 tok/s is what
`speed_ramp(10.16, 6, 25)` returns.

---

## 3. What a batch user's board looks like now, and how they know

`Needs.min_tps` is a `float | None` with a floor of zero, and `--min-tps` sets it on
`llamafit recommend`. Unset means the use case's own floor, so a request that says nothing
is judged exactly as it was before — there is a test asserting that a silent request and
`--min-tps 6` produce the identical board, ordering and exclusions alike.

The bundled reference profile, general request, before and after:

```
$ llamafit recommend
 #   Model                   Quant    Score   Gen/s
 1   llama-3.1-8b-instruct   Q4_K_M    66.9    33.9
 2   qwen3-0.6b              Q8_0      54.0   114.6
Not ranked: gemma-3-27b-it — generates 1.6 tokens per second, below the 6 a person
reads at: a batch tool on this machine and not one to sit in front of; a smaller
model or quantisation would keep up

$ llamafit recommend --min-tps 0
 #   Model                   Quant    Score   Gen/s
 1   llama-3.1-8b-instruct   Q4_K_M    66.9    33.9
 2   gemma-3-27b-it          Q4_K_M    56.8     1.6
 3   qwen3-0.6b              Q8_0      54.0   114.6
--min-tps 0: nobody is waiting on these tokens, so nothing was excluded for
generating more slowly than a person reads.
```

**They know three ways, and the first is the one that matters.** A caption under the ranked
table, in the same block as the weights and the confidence sentence, on the board itself
rather than in a log. `--min-tps 12` gets the other half of the same caption: *"this board
wanted at least that many tokens per second, and anything slower is excluded below rather
than ranked here."* Second, every row excluded by a figure somebody typed says so in its own
words — *"generates 1.6 tokens per second, below the 12 this request asks for; lower
--min-tps or choose a smaller model or quantisation"* — naming the boundary they moved and
the flag that moved it, rather than the reading rate they overrode. Third, `--json` carries
`needs.min_tps`, so a script can see it too.

**The figure goes into the floor's place for the ramp as well as the test, from one call.**
That is deliberate and it is the only part of this that is more than plumbing. The floor is
what sends everything below it to zero; take it away and Gemma's 1.6 tok/s against a general
target of 25 scores 6.5 rather than nothing, which is the honest thing to say to somebody who
is not sitting in front of it — and it restores the one distinction a batch user actually
cares about, which is that 1.6 and 5.9 tokens per second are nearly four times apart in wall
clock. It is also the embedding rule, arrived at from the other end: no reader, no floor,
straight line to the origin. Reading the floor once means a board can never exclude on one
number while its speed column scores against another.

**It applies to embedding too**, which the ranking report's wording did not anticipate. The
default says who is waiting, which is a fact about the job. A typed figure says what this
person will accept, which is not the program's to overrule, and silently ignoring a flag on
one use case out of six would be worse than either.

**What I did not build.** The web API, the web dashboard form and the TUI Needs pane do not
expose it. The brief said the command line is where it is expressed, and each of the other
three surfaces would need its own answer to "how do they know" — a caption of its own, not
just a field. They keep the default, which is today's behaviour, so nothing there is wrong;
it is unfinished. `docs/web.md`'s parameter table is untouched for the same reason.

---

## 4. What else in the specification is no longer true

Read against the code, not skimmed. In descending order of how badly it would mislead.

**§10.2, the prompt formula, is wrong in the same way 11.3 and 11.4 were.** It gives three
additive terms with the compute term charged wholly to the card and a third term for "when
the CPU computes them instead". The code splits the compute term by the share of transformer
blocks each device holds (`card_share_of_compute`), multiplies only the card's share by
`EFF_PP`, charges the CPU's share to `CPU_PP_TFLOPS_PER_CORE` which is *not* multiplied by
`EFF_PP`, and uses the third term only when there is no card at all. `speedscore-report.md`
§7 flagged this as outside its lane and left the wording for whoever placed the spec. I did
not place it, because unlike 11.3 and 11.4 no agent wrote the replacement and inventing one
would mean re-deriving a formula I have not measured against. **This is the largest remaining
gap and it should be somebody's next task.** `docs/calibration/2026-09-09-reference-machine.md`
has the same problem: its constants table lacks the two prompt runs and the two rates fitted
to them, and lacks the note that the `llama-bench` slope disagrees with the dense run by a
factor of five.

**§13.1's global flags do not exist.** It lists `--profile`, `--memory`, `--ram`,
`--cpu-cores` and `--max-context` as global. The CLI has `--json`, `--verbose`, `--no-color`,
`--language` and `--version`, and nothing else. Simulation is reachable from the web API's
query parameters and the TUI's Simulate panel, which §4.4 and §13.2 describe correctly; it is
only §13.1's line that promises a command-line surface nobody built. A reader would type
`llamafit --profile reference-rtx4060-128gb recommend` — I did — and get "No such option".

**§13.1's command table omits `llamafit launch`,** which ships and which §15 describes. The
table is otherwise accurate.

**§12.1's `Needs` block is half a different object.** `required:` is `capabilities:` in the
model; `preferred:` and `languages:` exist nowhere; `licenses:` and `prefer:` are arguments
to `build_board`, not fields of `Needs`; `max_download_gb` is `max_download_bytes`. I added
`min_tps` in the block's own idiom rather than rewriting it, because rewriting it is a
decision about whether §12.1 describes a request or a data class, and that is not mine to
take in a docs commit.

**§11.2 gives no default context for embedding.** It names five use cases; the code has six,
with embedding taking general's 8K and a docstring explaining why. Harmless, and one line to
fix.

**§12.2's board columns are aspirational.** It lists thirteen columns as though all were
always shown. `_board_columns` drops columns to fit the terminal and shows the confidence
column only when rows disagree about their labels, which is better behaviour than the
sentence describes.

Sections 8, 9, 10.1, 10.3, 11.1, 11.2's formula, 11.5 and 14 I checked and found accurate.
§10.1's four constants (`0.67`, `0.70`, `0.57`, 1 ms) and its per-backend fallback table
(CUDA 250, Metal 150, HIP 200, Vulkan 120, SYCL 100, arm64 80, x86_64 60) are the values in
`constants/speed.py`, and its "no per-layer overhead term" paragraph is still true.

---

## 5. Concerns

**1. The floor is now a knob, and a knob is a thing people turn without reading.** The
argument for excluding a model slower than its reader is that a board must not offer
something nobody can use. `--min-tps 0` turns that off. I think the caption and the reasons
carry it — a batch board says on its face that it is a batch board — but somebody who copies
a command line out of a wiki gets the relaxed rule without ever meeting the argument for the
strict one. The mitigation is that the default is unchanged and the flag has to be typed.

**2. The ramp changing shape with the floor is the part I would most expect to be argued
with.** The alternative is to move only the exclusion and leave the score anchored on six.
That keeps the score comparable between requests, which is a real property: two boards with
different `--min-tps` now have speed columns on different scales, and nothing on the board
says the scale moved except the caption. I chose the coupled version because an uncoupled one
puts two numbers on screen that disagree about what "too slow" means for this request, and
because "no reader, no floor, straight line" is the rule embedding already follows. The test
`test_the_ramp_and_the_test_start_from_the_same_named_figure` is what would fail if somebody
decoupled them.

**3. `--min-tps` and `plan`'s `--target-tps` are different questions with similar names.**
`--target-tps` asks "what context would reach this speed"; `--min-tps` says "below this,
don't show me". They are on different commands, which helps, and `--min-tps` matches
`--min-context` and `--min-fit` in shape, which is why I kept it. It is still two flags with
`tps` in the name meaning different things.

**4. Three interfaces cannot say what the fourth can** (concern in §3 above). The
`llamafit recommend --json` path covers scripting, so nobody is blocked; the dashboard and
the TUI are simply behind.

**5. The exclusion still rests on an estimate nobody has benchmarked.** This is
`ranking-report.md`'s concern 3 and it is unchanged: `EFF_PP` and `CPU_PP_TFLOPS_PER_CORE`
are each fitted to one run on one machine, and a wrong speed estimate can now remove a
candidate rather than cost it points. `--min-tps 0` at least gives a person a way to see what
was removed and judge for themselves, which it did not before, but that is a workaround and
not the fix. The fix is phase 3's `bench`.

**6. Qwen3-0.6B still leads some general boards, and I did not touch it.**
`ranking-report.md` §5 has the arithmetic. It predates all of this and belongs to the weights
or the capacity arm. Placing section 11 does not change it; if anything the placed wording
makes it easier to argue about, because both arms are now written down.

**7. An empty ranked board still has nothing to say for itself.**
`ranking-report.md`'s concern 4: on a machine where nothing keeps up, the reader gets an
empty "Recommended" table and a full "Not ranked" one and has to infer the verdict. The
`--min-tps` caption does not appear there, because `render_board` is not called when there
are no rows. The excluded rows do name the figure, so nothing is hidden, but the reading
order is still wrong.

**8. `messages.pot` moved by four messages and will conflict with the parallel translation
branch.** All four are new English; nothing was removed and no existing translation is reused
for a different sentence. The resolution is to regenerate, not to merge hunks.

**9. `docs/web.md` and the dashboard were left alone deliberately**, so the web API's
parameter table remains complete for what the API actually accepts.
