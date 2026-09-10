# Splitting `constants.py` into a package by area

Worktree `llamafit-wt-constants`, branch `feat/constants`, cut from `main` at `7974e11`.
Nothing was merged, rebased, pushed, or touched on another branch or worktree.

`src/llamafit/constants.py` (515 lines, 44 top-level names -- 39 constants and 2
`NamedTuple` classes counted separately from the docstrings that double each one, which is
where "about ninety declarations" comes from) is now `src/llamafit/constants/`:

| File | Names | Design section |
|---|---|---|
| `__init__.py` | re-exports all 44, `__all__` explicit | -- |
| `budget.py` | 18 (`MIB`, `GIB`, `VRAM_RESERVE_BYTES`, `CUDA_CONTEXT_BYTES`, `MeasuredRuntimeOverhead`, `MEASURED_RUNTIME_OVERHEAD_BYTES`, `PROCESS_OVERHEAD_BYTES`, `PROJECTOR_COMPUTE_BYTES`, `PROJECTOR_MAIN_COMPUTE_FLOOR_BYTES`, `ComputeBufferFit`, `COMPUTE_BUFFER_KNEE_TOKENS`, `COMPUTE_BUFFER_FIT`, `UNACCOUNTED_KV_CACHE_BYTES_PER_1K`, `MIN_BATCH_TOKENS`, `LOGITS_BYTES_PER_TOKEN`, `UTILISATION_COMFORTABLE`, `UTILISATION_FITS`, `UTILISATION_TIGHT`) | 8 |
| `placement.py` | 12 (`CONTEXT_TIERS`, `MICRO_BATCH_LADDER`, `ALL_GPU_LAYERS`, `SHARED_EXPERT_OVERRIDE`, `KV_TYPE_DEFAULT`, `KV_TYPE_QUANTISED`, `MIN_CONTEXT_TOKENS`, `DEFAULT_REQUESTED_CONTEXT`, `MAX_CONTEXT_SEARCH_STEP`, `DEFAULT_SERVER_HOST`, `DEFAULT_SERVER_PORT`, `DEFAULT_PARALLEL_SLOTS`) | 9, 9.5 |
| `speed.py` | 14 (`EFF_VRAM`, `EFF_RAM_SEQUENTIAL`, `EFF_RAM_SCATTERED`, `FIXED_OVERHEAD_S`, `DEFAULT_WORKING_CONTEXT`, `BACKEND_FALLBACK_GBPS`, `CPU_FALLBACK_GBPS`, `CPU_FALLBACK_DEFAULT_GBPS`, `KV_TYPE_BITS`, `EFF_PP`, `EFF_PCIE`, `ASSUMED_PCIE_GBPS`, `CPU_FP16_TFLOPS_PER_CORE`, `ASSUMED_BITS_PER_WEIGHT`) | 10 |

The three area boundaries were exact: the file had no banner in front of the memory
budget (it just started the module), `# Placement planner (design section 9)` at line
215, and `# Speed estimator (design section 10)` at line 332. Everything from `MIB` to
the end of `UTILISATION_TIGHT`'s docstring went to `budget.py`; everything from
`CONTEXT_TIERS` through `DEFAULT_PARALLEL_SLOTS` (including the `llama-server command
line (design section 9.5)` sub-banner, which is `placement.py`'s, not a fourth area) went
to `placement.py`; everything from the "three efficiencies" comment block to
`ASSUMED_BITS_PER_WEIGHT` went to `speed.py`. Every constant kept its docstring, every
inline `#` comment (the four-measurement table, the `# ---` sub-banners inside
`speed.py`) moved with the code around it, and the module docstring's own provenance
paragraphs were distributed to the file whose constants they actually describe -- the
exact-vs-modelled paragraph to `budget.py` (`exact=False` is a `llamafit.budget` concept),
the phase 3 calibration-relabelling paragraph to `speed.py` (`estimated`/`calibrated` are
`llamafit.speed.estimate` labels) -- with the general provenance statement and the reason
for the split now stated in `__init__.py`, the package's own front door. No paragraph was
dropped; the one paragraph that no longer described anything true (the old file arguing
that three files would be worse than one) was rewritten to describe the package instead,
since keeping a docstring that argues against the file structure it sits in would be
worse than editing it.

Checked every consumer before deciding the re-export surface (`grep -rn "from
llamafit.constants import" src tests`): eight source modules import 31 distinct names
between them (`budget/budget.py`, `budget/compute_buffer.py`, `budget/kv.py`,
`budget/projector.py`, `placement/context_tiers.py`, `placement/flags.py`,
`placement/modes.py`, `placement/planner.py`, `speed/bandwidths.py`, `speed/estimate.py`,
`speed/traffic.py`), plus six unit tests. `MeasuredRuntimeOverhead` and
`ComputeBufferFit` are imported by nothing outside `constants.py` itself, but the
`__init__.py` re-exports all 44 names regardless -- the job was to keep the module's
public surface intact, not to shrink it to only what happens to be used today.

## Verification

| Check | Result |
|---|---|
| `pytest --cov -m "not hardware and not network"` | 1508 passed, 1 skipped, 11 deselected; coverage **95.94%** (floor 85%); `constants/__init__.py`, `budget.py`, `placement.py`, `speed.py` each 100% |
| `ruff check .` | All checks passed (one auto-fix applied: `__all__` and the `budget.py` import block needed RUF022's constants-then-classes ordering for the two `NamedTuple` classes) |
| `ruff format src tests scripts` | 169 files left unchanged |
| `ruff format --check .` | 184 files already formatted |
| `mypy` | Success: no issues found in 89 source files |
| Every consumer module (`llamafit.budget.*`, `llamafit.placement.*`, `llamafit.speed.*`) | imports cleanly against the new package |
| Grep for `constants.py` outside the diff | nothing left referencing the old path |

## 1. How did you prove nothing changed value or went missing?

Not by re-reading my own diff. I loaded the pre-change file straight from git
(`git show HEAD:src/llamafit/constants.py`, before this branch's commit) as a module
under a synthetic name, in the same interpreter as the new installed package, and
compared `dir()` of each:

- **Names only in the old module:** `Final`, `NamedTuple`, `annotations` -- the typing
  imports leaking into the single file's own namespace, not constants, and nothing
  imports them from `llamafit.constants` (checked by grep).
- **Names only in the new package:** `budget`, `placement`, `speed` -- the submodule
  objects Python binds onto a package automatically on import, not constants either.
- Every one of the remaining **44 shared names** compared equal by value. For the two
  `NamedTuple` classes (`MeasuredRuntimeOverhead`, `ComputeBufferFit`), a plain `==` on
  the class objects will always read "different" since they're now defined in a
  different module -- that's a property of Python's class identity, not evidence of
  drift -- so those two were compared structurally instead (`_fields` and
  `__annotations__` identical) and then again by the values that actually matter:
  `MEASURED_RUNTIME_OVERHEAD_BYTES` and `COMPUTE_BUFFER_FIT`, the dicts of instances
  built from them, compare equal by value across old and new (namedtuple equality is
  tuple equality, independent of which class produced it).
- `new.__all__` as a set equals the old module's real names (44) exactly.

That script is not checked in; it ran against a scratch copy of the pre-change file and
is reproducible by anyone with `git show` and the two module objects. The full test
suite, run against the new package, additionally pins many of these exact numbers against
the calibration record's measurements -- it would have failed loudly had a digit moved --
and it passed unchanged (1508 passed both before and after, same count).

## 2. What did the bad merge leave behind that you found while moving things?

Nothing. I read the whole 515-line file end to end before slicing it, and re-verified
the slice boundaries against blank-line and banner positions before writing. There is
exactly one `MIB`, one `GIB`, one `EFF_VRAM`, and so on -- the AST-level name count came
out to 44 with no duplicates on the first pass, and the earlier duplication and the
missing docstring/licence header that the job description mentions were apparently
cleaned up by an intervening commit before this branch was cut (`git log` shows several
merge-repair commits on `main` ahead of where `feat/constants` starts). I looked
specifically for two names describing the same quantity under different labels and found
none: every name is referenced by exactly one file elsewhere in the tree, or by none.

## 3. Which constants did you struggle to assign to one area, and where did you put them?

The `llama-server command line (design section 9.5)` trio -- `DEFAULT_SERVER_HOST`,
`DEFAULT_SERVER_PORT`, `DEFAULT_PARALLEL_SLOTS` -- sits under its own sub-banner in the
original file, separate from the rest of the placement constants, and none of the three
is actually consumed by `llamafit.placement.planner` (the search); all three are used
only by `llamafit.placement.flags`, which renders the command line the planner decided
on. It reads almost like a fourth area. I put it in `placement.py` rather than splitting
it out further, for two reasons: section 9.5 is a subsection of section 9 in the design
doc, not a section of its own, and `llamafit.placement.flags` already lives inside the
`llamafit.placement` package alongside the search -- a fourth constants module would
describe a boundary that doesn't exist anywhere else in the tree. `MIN_CONTEXT_TOKENS`
was the other borderline case: it's defined as `CONTEXT_TIERS[0]` and its docstring talks
about the search's own preference order, but it is placement's context floor by every
other measure (name, consumers, section), so it stayed in `placement.py` next to
`CONTEXT_TIERS`.
