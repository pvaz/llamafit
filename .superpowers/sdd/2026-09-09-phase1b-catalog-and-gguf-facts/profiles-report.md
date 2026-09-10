# Hardware profiles: what was built, and what it does not do

Branch `feat/profiles`, worktree `C:\Dev\Projectos Pessoais\2026\llamafit-wt-profiles`.
Nothing was merged, rebased, pushed, or touched outside this worktree.

## Commits

| SHA | Subject |
|---|---|
| `b275c06` | `feat(models): describe a machine in a file, and mark a host that came from one` |
| `0ecd7e0` | `feat(hwprofile): load, validate, match and simulate hardware profiles` |
| `3690af1` | `feat(data): bundle a profile of the machine this project was measured on` |
| `9af4d31` | `feat(cli): llamafit hardware list\|show\|validate\|path` |
| `e9ff847` | `docs(profiles): the format that ships, and how a simulation is marked` |

## Verification

```
pytest --cov -m "not hardware and not network"   1634 passed, 1 skipped, 11 deselected; 96.68% (floor 85)
ruff check .                                     All checks passed
ruff format --check .                            197 files already formatted
mypy                                             Success: no issues found in 94 source files
gen_messages / gen_schema / gen_models_md        git diff --exit-code clean
```

New messages are untranslated in all 37 catalogs, which is correct: the `.pot` grew from
340 to 396 entries and no `.po` was touched.

## What shipped

```
src/llamafit/models/hwprofile.py            HardwareProfile and its sub-models
src/llamafit/models/host.py                 Simulation, Override, Host.simulation, Host.simulated
src/llamafit/hwprofile/loader.py            Problem, LoadedProfile, load, resolve, directories
src/llamafit/hwprofile/validate.py          validate_files: the cross-file checks
src/llamafit/hwprofile/match.py             matches, best_match
src/llamafit/hwprofile/simulate.py          host_from_profile, override_host, resolve_host
src/llamafit/cli/hardware_cmd.py            the hardware command group
src/llamafit/cli/render.py                  the SIMULATED row, render_profiles, render_profile
src/llamafit/data/profiles/                 one bundled profile
src/llamafit/data/schema/hwprofile.schema.json
```

A profile is a `Host` a person can type. `host_from_profile` returns the same
`llamafit.models.host.Host` that `hardware.scan()` returns, so every service downstream
takes a substituted machine exactly as it takes a real one; there is no second type and no
branch anywhere that asks which it got.

---

## 1. How a reader, and a program, tell a simulated result from a real one

**At the data level, on the `Host` itself.** `Host` gained two things:

```python
simulation: Simulation | None = None  # profile name, its file, the overridden pools


@computed_field
@property
def simulated(self) -> bool:
    return self.simulation is not None
```

- **A program** reads `host.simulated`. It is a *computed* field, not a stored one, so it
  cannot drift from `simulation` and cannot be omitted from `model_dump()` or
  `model_dump_json()`. Every command that serialises a host serialises that boolean:
  `llamafit --json system`, `llamafit --json doctor`, `llamafit --json hardware show NAME
  --as-host`, and whatever phase 1C prints. A consumer that never learns the nested shape
  still gets one unmissable flag at the top of the host.
- **A program that wants detail** reads `host.simulation`:
  `{"profile": "reference-rtx4060-128gb", "path": ".../reference-rtx4060-128gb.json",
  "overrides": ["gpu_memory", "ram"]}`. The override names are identifiers, never prose,
  so a script filters on them in any locale. The substituted *values* are in the host's own
  fields, because a simulated machine still has to read as a machine.
- **A reader** gets the first row of the host table, before a single figure, in red:

  ```
  Host
    SIMULATED  these figures come from the hardware profile reference-rtx4060-128gb;
               they are not this machine
    OS         windows Windows-11-10.0.26200-SP0 (x86_64)
    ...
  ```

  Three shapes, each written out whole for the translators: profile, overrides, or both.
  `system memory, CPU cores were overridden` agrees in number through `ngettext`.

The banner lives in `render_host`, not in the commands that print a simulated host. That is
the one function every host goes through; a warning that must be remembered at each call
site is a warning that will be forgotten at one of them. Likewise the mark lives on the
`Host` and not on a report wrapper, because every service takes a `Host` and that is the
only place a mark reaches all of them.

Two properties are enforced by test rather than by convention:

- `override_host(host)` with no override returns the *same object*, still unmarked. A flag
  nobody passed must never turn a scan into a simulation
  (`test_no_override_leaves_a_scan_a_scan`).
- The mark is set in exactly two places, `host_from_profile` and `override_host`, and there
  is no third way to produce a substituted host.

**What this does not claim.** The per-figure labels are a separate axis and stay as they
were. The bundled reference profile reports `bandwidth_source: "measured"`, because that
bandwidth genuinely was measured — on that machine, on 2026-09-09. The host-level
`simulated` flag is what says the machine is not the reader's. A consumer that reads
`bandwidth_source` without reading `simulated` can still misread a profile's measured
figure as a measurement of the machine in front of it; the two fields answer different
questions and both have to be read.

---

## 2. Which profiles were bundled, and where every figure came from

**One: `reference-rtx4060-128gb`.** Nothing else. Bundling an aspirational M3 Max or a
4090 Linux box would mean inventing a memory bandwidth, and an invented bandwidth reads
exactly like a measured one to everything downstream.
`test_only_measured_machines_ship` fails if a second file appears without that decision
being revisited.

| Field | Value | Source |
|---|---|---|
| `os`, `arch` | windows, x86_64 | `tests/fixtures/reference_machine.py` (recorded probe run, 2026-09-09) |
| `os_version` | `Windows-11-10.0.26200-SP0` | the `llamafit system` transcript in `docs/cli.md` |
| `cpu.model`, cores, `isa` | i9-14900KF, 24/32, 8 P-cores, avx2 | recorded probe output (`reference_cpuinfo`, `reference_cores`) |
| `memory.total` / `available` | 128 GiB / 100 GiB | recorded probe output (`reference_vm`) |
| `memory.type`, `speed_mts`, `channels` | DDR5, 4200 MT/s, 2 channels | `docs/calibration/2026-09-09-reference-machine.md` |
| `memory.modules` | 2 | the recorded WMI response in `reference_machine.py` (two 64 GiB modules) |
| `memory.bandwidth_gbps` | 57.0, `measured` | the NumPy read benchmark; `constants.py` ("DDR5 measuring 57 GB/s") and `tests/fixtures/speed.py::REFERENCE_RAM_GBPS` |
| `gpus[0]` name, `vram_total`, `vram_used`, `driver` | RTX 4060, 8188 MiB, 550 MiB, 610.88 | the recorded `nvidia-smi` line in `reference_machine.py` |
| `gpus[0].bandwidth_gbps`, `compute_tflops_fp16` | **absent** | filled at load time from `src/llamafit/data/gpus.json` (272 GB/s, 15 TFLOPS), the same vendor-spec table a live scan uses |
| `backends` | cuda, cpu | the calibration record's CUDA 12.4 build; `docs/cli.md` also lists `rpc`, which is not one of `Host`'s backend names |
| `recorded_at` | 2026-09-09T20:00:00Z | the calibration session's date; matches the estimator fixture's `scanned_at` |
| `calibration` | **absent** | deliberate; see below |

The card's specification figures are deliberately absent from the file, and a test asserts
they are (`test_the_card_specifications_come_from_the_bundled_table_not_the_file`). They are
vendor numbers, sourced once in `gpus.json` and shared by every machine with that card; a
second copy in a profile is a copy that can drift, and `hardware validate` reports any
profile whose typed figure disagrees with the table by more than ten percent.

**No `calibration` block.** The calibration record's "initial estimator constants" table
gives `ram_efficiency 0.70`, `vram_efficiency 0.60`, `pcie_effective 4 GB/s`,
`layer_overhead 0.20 ms`. Two of those have since been superseded in `constants.py` —
`EFF_VRAM` was re-derived to 0.67, and the per-layer overhead is documented there as having
been *wrong* and replaced by a 1 ms fixed intercept. Copying the old table into a bundled
profile would ship a stale duplicate of numbers that nothing in this build reads and nothing
keeps in step. The machine's fitted constants stay in `constants.py`, where they are applied;
`calibration` remains in the schema for `llamafit bench` to write in phase 3, and
`hardware show` prints "this build stores it but does not apply it yet" beside any block
it finds.

**Two in-repo sources disagreed and I had to choose.** `docs/cli.md`'s transcript shows
"4 modules ... 57.3 GB/s"; the recorded WMI fixture says two 64 GiB modules and
`constants.py`/`tests/fixtures/speed.py` derive everything from 57.0. I took the recorded
probe output and the figure the estimator was fitted against (2 modules, 57.0), because the
value of this profile is reproducing the documented *estimates*, and 57.3 was one sample of
a noisy benchmark. Neither figure changes any budget; `modules` is displayed and nothing
derives from it. The profile's `provenance` says both numbers out loud.

**The reproducibility claim is a test, not a promise.**
`tests/unit/test_hwprofile_data.py` compares the host built from the bundled profile against
`tests/fixtures/speed.py::reference_host()` — the machine the estimator's four calibrated
runs are asserted against — field for field, and asserts that
`resolve_bandwidths()` returns the identical `EffectiveBandwidths` for both, with
`assumed is False`. Only `os_version` is excluded (the profile carries the full
`platform.platform()` string, the fixture a shorter form), plus the simulation fields the
fixture cannot have. So a stranger running `llamafit hardware show reference-rtx4060-128gb
--as-host` gets, byte for byte, the machine the documentation's speed numbers were computed
on.

---

## 3. What in the design could not be implemented as written

The task named "section 12", but section 12 of `docs/specs/2026-09-09-llamafit-design.md` is
*Recommendation and explanations*. The hardware profile design is **section 4.4, "Hardware
profiles and simulation"**, plus the `hardware list|show|validate|path` row of 13.1 and the
existing `docs/hardware-profiles.md`. I implemented those. If a different section 12 was
meant, this is the wrong feature and nothing below is the right answer.

### Not implemented

**The global flags are not wired into the CLI.** `--profile`, `--memory`, `--ram`,
`--cpu-cores` are options on the root callback in `cli/app.py`, and the brief says to keep
changes to that file to the single line that registers my command group, because another
agent is editing it. So I built and tested the machinery those flags will call —
`hwprofile.resolve_host(scan_host=..., profile=..., gpu_memory=..., ram=..., cpu_cores=...)`
— and left the flags to whoever adds the commands that use them. `hardware show --as-host`
is the end-to-end path that exists today: it takes a profile through
`host_from_profile` to a rendered and serialised `Host`. Wiring the flags is four
`typer.Option`s and one call to `resolve_host`; `docs/cli.md` says the machinery is here and
the flags arrive with phase 1C.

**`--max-context` is untouched.** It caps the context used for budgets and scores. That is
not a property of the host and does not belong in this module; it belongs with `fit`,
`recommend` and `plan`.

**`calibration` is stored, not applied.** `constants.py` says its values "can be overridden
by a hardware profile or a calibration record", and none of them are yet. Overriding them is
phase 3's job and needs the estimator to carry a `calibrated` confidence label (section 10.3)
that does not exist. Rather than accept a field that silently does nothing, `hardware show`
says on the line that prints it that this build does not apply it.

**`doctor` does not report which profile matched.** `matches()` and `best_match()` are
implemented and tested against the live-scan shape, including a test that the bundled
reference profile recognises the reference machine's own scan and nothing else. Wiring it
into `doctor` would mean editing `services/doctor.py` and `doctor_cmd.py`, and there is
nothing for it to report until phase 3 writes a calibration worth picking up.

### Implemented differently, and why

**`pcie_bandwidth_gbps` was dropped.** Section 4.4's example carries it. `Host` has no field
for it, and `speed.resolve_bandwidths` reads the link's speed from `gpus.json` keyed by card
name. A profile that set it would have been parsed and thrown away — the exact failure mode
the catalog loader exists to prevent. `extra="forbid"` now makes writing it a loud validation
error naming the field.

**Sizes are strings, not `_gb` numbers.** Section 4.4 has `total_ram_gb: 128` and
`gpu_memory_gb: 8`. For a 128 GiB machine `total_ram_gb: 128` has to mean *gibibytes*, while
`gpu_memory_bandwidth_gbps: 272` means decimal gigabytes per second — one suffix, two
meanings, in the same object. And `gpu_memory_gb: 8` rounds away the reference card's actual
8188 MiB, which the calibration record's "7.8 GB of 8.2 GB" depends on. The fields are now
`memory.total`, `gpus[].vram_total` and so on, taking `"128GiB"`, `"8188MiB"`, `"8G"` or a
plain byte count through the existing `units.parse_size` — the same syntax `--memory 24G`
already documents.

**The document is nested and mirrors `Host`.** Section 4.4's example is flat
(`cpu_cores`, `gpu_memory_gb`) and carries no card name, no CPU model, no OS version. A card
name is not cosmetic: it is the key `gputable.enrich_gpu` and
`constants.MEASURED_RUNTIME_OVERHEAD_BYTES` look up, so a profile without one loses its
bandwidth, its compute figure, its PCIe rating and its measured backend overhead. `gpus` is a
list because `Host.gpus` is, and section 4.2 says multi-GPU is recorded.

**`provenance` is required and must be non-empty.** Nothing in section 4.4 asks for it.
Nothing else in the schema can tell a profile of a measured machine from a profile somebody
imagined, and a bundled profile is data this project has to be able to trace. It is weak —
free text nobody can verify — but it is the field a reviewer reads and
`test_every_bundled_profile_says_where_its_figures_came_from` enforces that it says something.

**`memory.bandwidth_source` is required whenever there is a bandwidth.** No default, because
the default would be the label nobody chose. Omitting the bandwidth is fine and produces
`unknown`, which routes to section 10.1's fallbacks and an `assumed` label downstream.

**`hardware show` gained a `--as-host` flag.** Section 13.1's row lists four subcommands and
this adds none: it is a flag on `show`. Without it there is no way to see what a profile
actually becomes, which is where the simulation marking is visible and testable.

**Additional `validate` checks.** Beyond the schema: a name two files both claim, a GPU
figure that contradicts the bundled table by more than ten percent, an Apple-style unified
part described as having its own VRAM pool, and `match` rules that would never recognise the
machine the profile itself describes. These are the analogue of `catalog/validate.py`'s
`context.native` check — things only visible with a second document in hand.

**`Problem` is a new dataclass, not the catalog's.** Same four keys and the same
never-raise contract, but the second field is `profile` rather than `model_id`. Sharing the
catalog's class would have put a `model_id` key in every profile problem's JSON, naming
something that does not exist, and moving it to a neutral module would have churned a symbol
half the codebase imports.

---

## Concerns

1. **`Host` grew two public fields.** `llamafit --json system` and `doctor` now emit
   `"simulation": null` and `"simulated": false` on every host. Additive and, I think,
   correct — a scan asserting it is not a simulation is worth saying — but it is a change to
   output that already shipped, and anything doing an exact-shape comparison will see it.
2. **`render.py` is a shared file.** I added the `SIMULATED` row inside `render_host` plus
   `render_profiles`/`render_profile` and five private helpers. The parallel `fit`/
   `recommend`/`plan` work will very likely touch this file too. `cli/app.py` gained only the
   import-tuple entry and its `__all__` twin, as asked.
3. **The simulation flags are not on the command line yet.** Until phase 1C wires them, a
   user can see a profile as a host but cannot score against one. `resolve_host` is the seam
   and is fully tested; nobody but a test calls it today.
4. **One bundled profile is a thin `hardware list`.** Deliberate, and I would rather it stay
   thin than gain a machine nobody has measured, but it does mean the fleet story
   ("twenty identical machines") is served by a user profile the operator writes, not by
   anything bundled.
5. **`memory.available` on a profile is a moment, not a machine.** The reference profile
   records 100 GiB of 128 GiB free, which is what the scan happened to see. Omitting it means
   "the whole pool", which is optimistic. Neither is a property of the hardware, and a fleet
   profile author has to decide which they mean; the doc says so but the schema cannot.
6. **`docs/cli.md` and the recorded fixtures disagree about the reference machine** (4 modules
   / 57.3 GB/s versus 2 modules / 57.0). I resolved it toward the fixtures and said so in the
   profile's `provenance`, but one of the two documents is wrong and I did not change either.
7. **`--memory` keeps the card's name and specifications.** A "24 GB RTX 4060" is a fiction,
   and its bandwidth, compute and PCIe figures still come from the real card's table row.
   That is the right answer for "my card with more memory" and the wrong one for "a different
   card"; the second is what `--profile` is for, and only `simulation.overrides` says which
   one the reader asked for.
8. **`llamafit hardware validate` checks the bundled profiles plus the user's.** It reports
   `1 file checked` on a machine with no user profiles, which reads oddly next to `catalog
   validate`'s five. Cosmetic.
