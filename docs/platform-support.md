# Platform support

LlamaFit runs wherever Python 3.10 or newer runs: Windows 10 and 11, macOS 12 and later,
current Linux distributions, on x86_64 and arm64. Detection uses the tools each platform
provides; when one is missing the scan continues, and `llamafit doctor` says what was lost
and how to get it back. This page lists every probe, what it needs, and what happens without
it, so a `doctor` warning can always be traced here.

## Probes

| Probe | Platforms | Command | Provides | Without it |
|---|---|---|---|---|
| `cpuinfo` | all | the `py-cpuinfo` library | CPU model, instruction sets (AVX2, AVX-512, AMX, NEON, SVE) | model "unknown", no instruction sets; core counts still come from `psutil` |
| `sysctl-perflevel` | macOS | `sysctl -n hw.perflevel0.physicalcpu` | performance-core count | all physical cores are treated as performance cores |
| `memory-modules` | Windows | PowerShell `Get-CimInstance Win32_PhysicalMemory` | DDR type, speed, channel count, bandwidth estimate | totals only; bandwidth is measured or assumed |
| `memory-modules` | macOS | `system_profiler SPMemoryDataType -json` | memory type | same |
| `memory-modules` | Linux | `dmidecode -t memory` (needs root) | DDR type, speed, channels | same; run `sudo llamafit system` once if you want the facts recorded |
| `nvidia-smi` | Windows, Linux | `nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader,nounits` | NVIDIA name, VRAM total and used, driver | the GPU is listed by name only (from WMI or lspci), VRAM unknown |
| `rocm-smi` | Linux | `rocm-smi --showmeminfo vram --showproductname --json` | AMD name, VRAM total and used | name only |
| `system-profiler` | macOS | `system_profiler SPDisplaysDataType -json` | Apple GPU name and core count | no GPU listed |
| `wmi-video` | Windows | PowerShell `Get-CimInstance Win32_VideoController` | names of all display adapters | vendor tools only |
| `lspci` | Linux | `lspci -nn` | names of all display controllers | vendor tools only |
| `llama-server --version` | all | `llama-server --version` | llama.cpp build number and commit | `bin/VERSION.txt` is read when present |
| `server:<port>` | all | HTTP `GET /health`, `/v1/models`, `/props` on 8080, 8081, 8098 and `LLAMA_SERVER_PORT` | running servers, their model and context | none listed |

The RAM bandwidth measurement is not a probe: it runs in-process and reports `measured` when
it can allocate its buffer (more accurate with NumPy: `pip install llamafit[fast]`),
`estimated` from DDR facts otherwise, and `assumed` (40 GB/s) when nothing better is known.

## Operating systems

| OS | Notes |
|---|---|
| Windows 10 and 11 | PowerShell 5 or 7 must be on `PATH` (it is by default). NVIDIA figures need the driver's `nvidia-smi`, installed with every driver. AMD VRAM is not readable without ROCm, so AMD cards are listed by name; llama.cpp's Vulkan backend still uses them and the budget treats the VRAM size as unknown until you set it in a hardware profile. Long paths and spaces in paths are supported. |
| macOS 12 and later | Apple Silicon reports unified memory: there is no VRAM figure, the whole RAM pool is the budget, and the GPU table supplies bandwidth per chip. Intel Macs with discrete GPUs are listed by name. `system_profiler` can take a few seconds on first use. |
| Linux | `dmidecode` needs root; everything else runs as a user. `pciutils` provides `lspci`. NVIDIA needs the proprietary driver for `nvidia-smi`; AMD needs ROCm for VRAM figures; Intel Arc VRAM is read from sysfs when present. |

## Architectures

`x86_64` and `arm64` are first-class. Other architectures scan, but llama.cpp release binaries
may not exist for them, which `doctor` reports when the installer (phase 2) is asked to fetch
one.

## Backends

LlamaFit reads which backends a llama.cpp build contains from its shipped `ggml-*` libraries:
`cuda`, `hip`, `metal`, `vulkan`, `sycl`, `rpc`, `cpu`. A GPU without a matching backend is a
`doctor` warning with the build to install.

## Terminals

The terminal dashboard needs a terminal with 256 colours and at least 80 columns. Windows
Terminal, PowerShell 7, macOS Terminal, iTerm2 and every common Linux terminal qualify. The
legacy Windows console works for the CLI tables but not for the dashboard.
