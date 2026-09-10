# Hardware profiles

A hardware profile describes a machine well enough for LlamaFit to score models against it
without being on it. Profiles are used to plan a purchase, to choose models for a fleet from a
CI job, to try "what if I had 24 GB of VRAM" from the Simulate screen, and to carry the
calibration factors that benchmarks derive for your own machine.

A profile is a `Host` you can type. That is the whole design: the scan produces a
[`Host`](../src/llamafit/models/host.py), a profile produces the same `Host` from a file, and
nothing downstream is told which it got.

## Simulated is never confused with measured

What downstream *is* told is that the machine is not this one. A host that did not come from
the probes carries a `simulation` object, and `simulated` beside it:

```json
{
  "os": "windows",
  "...": "...",
  "simulation": {
    "profile": "reference-rtx4060-128gb",
    "path": ".../llamafit/data/profiles/reference-rtx4060-128gb.json",
    "overrides": []
  },
  "simulated": true
}
```

`simulated` is computed from `simulation`, so the two cannot drift and neither can be left
out of `--json`. `overrides` names the pools a `--memory`, `--ram` or `--cpu-cores` flag
replaced, as identifiers a script can filter on (`gpu_memory`, `ram`, `cpu_cores`); the
substituted values are in the host's own fields, because a simulated machine still has to
read as a machine. On the terminal the same fact is the first row of the host table, in red,
before a reader has met a single figure.

This is at the data level on purpose. Every service takes a `Host`, so a mark on the `Host`
is the only mark that reaches all of them.

A board and a plan are not hosts, and they are what a script usually reads. So `recommend`,
`fit` and `plan` carry the same two fields on their own documents, set from the host they
were computed against:

```json
{
  "rows": ["..."],
  "simulation": {"profile": "reference-rtx4060-128gb", "path": "...", "overrides": []},
  "simulated": true
}
```

Without them a program reading `llamafit --profile X --json recommend` would have nothing
at all to go on: no host, no heading, no colour. `"simulated": false` on a scan is part of
the same promise, because absence is not evidence.

## Format

JSON, one machine per file, named for the profile inside it (`--profile NAME` looks for
`NAME.json`). The schema is published at
`src/llamafit/data/schema/hwprofile.schema.json`, generated from the pydantic model by
`python scripts/gen_schema.py`; the model is what validates, and the schema is for your
editor and your CI job.

```json
{
  "schema_version": 1,
  "name": "reference-rtx4060-128gb",
  "description": "Reference machine: RTX 4060 8 GB, i9-14900KF, 128 GiB DDR5-4200, Windows 11",
  "provenance": "Recorded probe output in tests/fixtures/reference_machine.py, 2026-09-09; memory type, speed and channels from docs/calibration/2026-09-09-reference-machine.md.",
  "recorded_at": "2026-09-09T20:00:00Z",
  "match": {
    "gpu_name_contains": "RTX 4060",
    "cpu_model_contains": "i9-14900KF",
    "total_ram_min": "100GiB"
  },
  "os": "windows",
  "os_version": "Windows-11-10.0.26200-SP0",
  "arch": "x86_64",
  "cpu": {
    "model": "Intel(R) Core(TM) i9-14900KF",
    "physical_cores": 24,
    "logical_cores": 32,
    "performance_cores": 8,
    "isa": ["avx2"]
  },
  "memory": {
    "total": "128GiB",
    "available": "100GiB",
    "type": "DDR5",
    "speed_mts": 4200,
    "modules": 2,
    "channels": 2,
    "bandwidth_gbps": 57.0,
    "bandwidth_source": "measured"
  },
  "gpus": [
    {
      "vendor": "nvidia",
      "name": "NVIDIA GeForce RTX 4060",
      "vram_total": "8188MiB",
      "vram_used": "550MiB",
      "backend_hint": "cuda",
      "driver": "610.88"
    }
  ],
  "unified_memory": false,
  "backends": ["cuda", "cpu"]
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | `1`. A document from a version this build does not read is refused whole. |
| `name` | yes | identifier used by `--profile`; lowercase letters, digits, hyphens and dots, and the same as the file's stem |
| `description` | no | one line, shown in `hardware list` |
| `provenance` | yes | where every figure in the profile came from, in your own words. See below. |
| `recorded_at` | no | when the machine was in this state; becomes the host's `scanned_at` |
| `match` | no | rules that let a live scan recognise itself: `gpu_name_contains`, `cpu_model_contains`, `total_ram_min`. Every rule given must hold; a profile with no rules is never chosen for you. |
| `os`, `os_version`, `arch` | yes | `windows`, `macos` or `linux`; a version string; `x86_64`, `arm64` or `other` |
| `cpu.model`, `cpu.physical_cores` | yes | |
| `cpu.logical_cores` | no | threads the OS offers; absent means no simultaneous multithreading |
| `cpu.performance_cores`, `cpu.isa` | no | thread choice, and `avx2`/`avx512`/`neon`/`amx` |
| `memory.total` | yes | the system pool |
| `memory.available` | no | free at the moment described; absent means the whole pool |
| `memory.type`, `memory.speed_mts`, `memory.modules`, `memory.channels` | no | for the record; nothing is derived from them |
| `memory.bandwidth_gbps` | no | read bandwidth in GB/s |
| `memory.bandwidth_source` | with a bandwidth | `measured`, `estimated` or `assumed`. Required whenever there is a figure; see below. |
| `gpus[].vendor`, `gpus[].name` | when there is a GPU | `nvidia`, `amd`, `apple`, `intel` or `other`, and the name as a driver reports it |
| `gpus[].vram_total`, `gpus[].vram_used` | no | absent `vram_total` means the card draws on the system pool |
| `gpus[].bandwidth_gbps`, `gpus[].compute_tflops_fp16` | no | **leave these out** unless you measured them; see below |
| `gpus[].backend_hint`, `gpus[].driver` | no | the backend defaults to the vendor's usual one |
| `unified_memory` | no | `true` on Apple silicon and APUs: one pool for everything. A unified machine may not give a card a `vram_total`. |
| `backends` | no | which llama.cpp backends the machine can use, for the record |
| `calibration` | no | factors `llamafit bench` writes; see [benchmarking.md](benchmarking.md) |

### Sizes are written the way the flags are written

`"128GiB"`, `"8188MiB"`, `"8G"`, `"512M"` — the same syntax `--memory 24G` takes, and a plain
number is a count of bytes. Binary suffixes (`GiB`) are powers of 1024, bare and decimal ones
(`G`, `GB`) powers of 1000.

A field called `total_ram_gb` would have had to mean gibibytes to describe a 128 GiB machine
and gigabytes to describe a 272 GB/s link. Bandwidths keep their `_gbps` name and are decimal
GB/s, which is what a vendor's specification sheet states.

### Every figure carries the label of how it was obtained

A profile that states `memory.bandwidth_gbps` must state `memory.bandwidth_source` too:

- `measured` — somebody ran the benchmark on this machine.
- `estimated` — derived from the module type, speed and channel count.
- `assumed` — a plausible number nobody checked.

There is no default, because the default would be the label nobody chose. Leaving the
bandwidth out entirely is fine and is often the honest answer: the host then reports
`unknown`, and the per-architecture fallback in `llamafit/constants.py` stands in, labelled
`assumed`. A worse estimate honestly labelled beats a better one invented.

The same reasoning is why `gpus[].bandwidth_gbps` and `gpus[].compute_tflops_fp16` should
normally be absent. LlamaFit fills them from the bundled GPU table
(`src/llamafit/data/gpus.json`) by matching the card's name, which is where a live scan gets
them too: they are vendor specifications, sourced once and shared by every machine with that
card. Typing them into a profile makes a second copy that can drift, and `hardware validate`
reports one that disagrees with the table by more than ten percent.

### What is not in the schema

Section 4.4 of the design sketched `pcie_bandwidth_gbps`. It is not here: `Host` has nowhere
to put it, and the speed estimator reads the link's bandwidth out of the GPU table by card
name, so a profile that set it would have been quietly ignored. A field nothing can read is a
field that lies.

`calibration` is accepted, validated and stored, and **this build does not apply it**.
`hardware show` says so on the line that prints it. Phase 3 is where those factors start to
count; until then the reference machine's fitted constants live in `llamafit/constants.py`,
where they are applied.

## Where profiles live

Bundled profiles ship in the package (`src/llamafit/data/profiles/`). Only machines this
project has actually measured ship there, because a profile is data and an invented memory
bandwidth reads exactly like a measured one to everything downstream. Today that is one file:

| Profile | Machine | Where its figures come from |
|---|---|---|
| `reference-rtx4060-128gb` | RTX 4060 8 GB, i9-14900KF, 128 GiB DDR5-4200, Windows 11 | The recorded probe output in `tests/fixtures/reference_machine.py` and the measurements in [the calibration record](calibration/2026-09-09-reference-machine.md). Every constant in `llamafit/constants.py` was fitted to this machine and every `measured` block in the catalog was taken on it, so a stranger can reproduce the documentation's numbers without owning it. |

Your own profiles live in the data directory — `llamafit hardware path` prints it, and
`LLAMAFIT_PROFILES` overrides it. A user profile takes precedence over a bundled one with the
same name, so recalibrating the machine LlamaFit ships a profile of keeps the name you already
use.

## Commands

```
llamafit hardware list                       # bundled and user profiles
llamafit hardware show NAME                  # one profile, provenance included
llamafit hardware show NAME --as-host        # the Host it substitutes, marked SIMULATED
llamafit hardware show ./my-machine.json     # a file, by path
llamafit hardware validate [FILE]            # schema and cross-file checks
llamafit hardware path                       # where your own profiles go
```

Beyond the schema, `validate` reports three things a single file cannot see on its own: a name
two files both claim, a GPU figure that contradicts the bundled specification table, and
`match` rules that would never recognise the machine the profile itself describes.

The scoring commands take the profile as a global option, before the command:

```
llamafit --profile reference-rtx4060-128gb recommend
llamafit --profile ./my-machine.json fit
llamafit --profile reference-rtx4060-128gb system      # the Host, with this machine's llama.cpp
```

Naming a profile means this machine is never probed at all: a run scoring against somebody
else's machine has no business reading this one's, and on a machine whose graphics driver
hangs a probe the profile has to work anyway. `system`, `fit`, `recommend`, `plan` and
`preset` honour it; `doctor` refuses it, because every line it prints is a probe that ran
here and a profile has none.

and single-value overrides work on top of the live scan or a profile:

```
llamafit --memory 24G recommend                # pretend the GPU has 24 GB
llamafit --ram 64G --cpu-cores 8 fit
llamafit --profile fleet-node --memory 24G plan qwen3-coder-next
```

An override keeps everything else the scan knows and moves one figure, because somebody asking
"what if my card were bigger" is not asking to have their processor replaced by a guess too.
The arithmetic is written down rather than improvised: a resized card keeps what the desktop
was already using, a resized memory pool keeps what is *in use* rather than what is free, and a
changed core count keeps the machine's threads-per-core ratio. `--memory` on a machine with no
graphics card, or one with a single unified pool, is refused with the reason rather than
guessed at.

## Automatic use

After `llamafit bench` has calibrated a machine (phase 3), the calibration is stored in a user
profile whose `match` rules fit the live scan. Later runs pick it up automatically; `doctor`
shows which profile is in effect and why. A match never overrides the live scan's own figures:
the scan measured this machine and the profile did not.
