# Hardware profiles

A hardware profile describes a machine well enough for LlamaFit to score models against it
without being on it. Profiles are used to plan a purchase, to choose models for a fleet from a
CI job, to try "what if I had 24 GB of VRAM" from the Simulate screen, and to carry the
calibration factors that benchmarks derive for your own machine.

## Format

JSON, one machine per file:

```json
{
  "schema_version": 1,
  "name": "rtx4060-8gb-ddr5-128gb",
  "description": "Reference machine: RTX 4060 8 GB, i9-14900KF, 128 GB DDR5-4200, Windows 11",
  "match": {"gpu_name_contains": "RTX 4060", "total_ram_gb_min": 100},
  "os": "windows",
  "cpu_cores": 24,
  "performance_cores": 8,
  "total_ram_gb": 128,
  "unified_memory": false,
  "ddr_bandwidth_gbps": 60,
  "gpu_memory_gb": 8,
  "gpu_memory_bandwidth_gbps": 272,
  "gpu_compute_tflops_fp16": 15,
  "pcie_bandwidth_gbps": 12,
  "backends": ["cuda"],
  "calibration": {
    "ram_efficiency": 0.70,
    "vram_efficiency": 0.60,
    "pcie_effective_gbps": 4.0,
    "layer_overhead_ms": 0.20,
    "source": "docs/calibration/2026-09-09-reference-machine.md"
  }
}
```

| Field | Required | Meaning |
|---|---|---|
| `schema_version` | yes | `1` |
| `name` | yes | identifier used by `--profile`; lowercase, hyphens |
| `description` | no | shown in `hardware list` |
| `match` | no | rules that let a live scan pick this profile's calibration automatically: `gpu_name_contains`, `cpu_model_contains`, `total_ram_gb_min` |
| `os`, `cpu_cores`, `performance_cores` | no | thread choice |
| `total_ram_gb`, `ddr_bandwidth_gbps` | yes | RAM pool size and bandwidth |
| `unified_memory` | yes | `true` on Apple Silicon and APUs: one pool for everything |
| `gpu_memory_gb`, `gpu_memory_bandwidth_gbps`, `gpu_compute_tflops_fp16`, `pcie_bandwidth_gbps` | when there is a GPU | VRAM pool and the speed model's inputs |
| `backends` | no | which llama.cpp backends the machine can use |
| `calibration` | no | factors written by `llamafit bench`; see [benchmarking.md](benchmarking.md) |

## Where profiles live

Bundled profiles ship in the package (`src/llamafit/data/profiles/`) and cover a few common
shapes: the reference machine above, an Apple M3 Max with 64 GB, a CPU-only 16 GB laptop, an
RTX 4090 Linux box. User profiles live in the data directory
(`llamafit hardware path` prints it) and take precedence over bundled ones with the same name.

## Commands

```
llamafit hardware list                 # bundled and user profiles
llamafit hardware show NAME            # the JSON
llamafit hardware validate FILE        # schema check
llamafit hardware path                 # the user profile directory
llamafit recommend --profile NAME      # score against a profile instead of the live scan
llamafit recommend --profile ./my.json
```

Single-value overrides work on top of the live scan or a profile:

```
llamafit recommend --memory 24G                # pretend the GPU has 24 GB
llamafit recommend --ram 64G --cpu-cores 8
```

The output carries a `simulated: true` flag and the dashboards show a `SIM` badge so a
what-if is never mistaken for the real machine.

## Automatic use

After `llamafit bench` has calibrated a machine, the calibration is stored in a user profile
whose `match` rules fit the live scan. Later runs pick it up automatically; `doctor` shows
which profile is in effect and why.
