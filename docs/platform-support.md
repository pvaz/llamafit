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
| `cpu-cores` | all | `psutil.cpu_count(logical=False)` and `psutil.cpu_count(logical=True)` | physical and logical core counts | counts fall back to one core (physical and logical); thread choice degrades |
| `sysctl-perflevel` | macOS | `sysctl -n hw.perflevel0.physicalcpu` | performance-core count | all physical cores are treated as performance cores |
| `memory-modules` | Windows | PowerShell `Get-CimInstance Win32_PhysicalMemory` (`BankLabel`, `DeviceLocator` included) | DDR type, speed, number of populated modules, and the channel count when the slot labels encode one | totals only; bandwidth is measured or assumed |
| `memory-modules` | macOS | `system_profiler SPMemoryDataType -json` | memory type, number of modules listed | same; the channel count is never reported here |
| `memory-modules` | Linux | `dmidecode -t memory` (needs root) | DDR type, speed, number of populated modules, and the channel count when `Locator`/`Bank Locator` encode one | same; run `sudo llamafit system` once if you want the facts recorded |
| `memory-totals` | all | `psutil.virtual_memory()` | total and available memory | memory budgets cannot be computed until this works |
| `paths` | all | `platformdirs` and the home directory | the downloads directory, whose free space is reported | only the working directory's volume is reported; set `LLAMAFIT_HOME` if there is no home directory |
| `nvidia-smi` | Windows, Linux | `nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader,nounits` | NVIDIA name, VRAM total and used, driver | the GPU is listed by name only (from WMI or lspci), VRAM unknown |
| `rocm-smi` | Linux | `rocm-smi --showmeminfo vram --showproductname --json` | AMD name, VRAM total and used | name only |
| `system-profiler` | macOS | `system_profiler SPDisplaysDataType -json` | Apple GPU name (the core count is not read) | no GPU listed |
| `wmi-video` | Windows | PowerShell `Get-CimInstance Win32_VideoController` | names of all display adapters | vendor tools only |
| `lspci` | Linux | `lspci -nn` | names of all display controllers | vendor tools only |
| `vulkaninfo` | Windows, Linux | `vulkaninfo` (the text output, not `--summary`) | a discrete card's memory size and how much of it the driver will promise | a card no vendor tool sized stays unsized, and nothing is planned on it |
| `llama-server --version` | all | `llama-server --version` | llama.cpp build number and commit | `bin/VERSION.txt` is read when present |
| `server:<port>` | all | HTTP `GET /health` (1 s timeout), then `/v1/models`, `/props` (1.5 s timeout each) on 8080, 8081, 8098 and `LLAMA_SERVER_PORT` | running servers, their model and context | none listed |

The module count is not the channel count: a dual-channel board with four modules reports four
modules, and the module count alone says nothing about channels. On Windows and Linux, the
slot labels the module count comes from (`BankLabel`/`DeviceLocator`, or `dmidecode`'s `Bank
Locator`/`Locator`) usually also encode the channel, either as a letter in forms such as
`ChannelA-DIMM1`, `Channel A Slot 0`, `DIMM_A1`, `DIMM A1`, `A1_DIMM0` or the bare
`CHANNEL A`, or as a controller number in `ControllerN-DIMMx` (the controller number is the
channel on boards that report it this way, one integrated memory controller per channel);
LlamaFit parses whichever it finds and counts the distinct identifiers
across populated modules. An uninformative label such as `BANK 0`, or a form not recognised,
leaves the channel count unknown rather than guessing, so the theoretical
`speed x 8 bytes x channels` estimate only runs once a channel was genuinely parsed out. macOS's
`system_profiler` reports no such labels, so the channel count stays unknown there.

The RAM bandwidth measurement is not a probe: it runs in-process, and it measures *sequential
read* bandwidth specifically, not a copy. llama.cpp streams weights out of RAM during
generation and writes almost nothing back, so a read is what predicts generation speed; a
copy moves each byte twice (read and write, three times on a write-allocate cache), so a copy
figure is not comparable to a read figure and would mislead an estimator calibrated against
real generation speed. With NumPy importable (`pip install llamafit[fast]`), it splits a
buffer across a thread pool sized to the physical core count (capped at 8) and runs a
memory-bound reduction (`.max()`) over the slices concurrently, because a single thread
cannot saturate a multi-channel memory controller; that result is labelled `measured`.
Without NumPy, a single-threaded pure-Python `bytearray` copy is used instead (there is no
faster read-only option in pure Python), scaled by a documented correction factor (3.0x,
`PURE_PYTHON_CORRECTION`) and labelled `estimated`, since it under-reports the real read
figure for two independent reasons at once (single-threaded, and a copy rather than a read)
and the correction is only an approximation. `estimated` is also used for the DDR-facts
calculation above when the in-process measurement is unavailable or implausible, and `assumed`
(40 GB/s) is the last resort when nothing better is known.

A GPU's memory bandwidth and fp16 compute are not measured: they come from a bundled table of
vendor specifications matched by device name, which is why the tables label them `spec`. Only
the VRAM figures next to them are read from this machine.

A failed vendor GPU probe (`nvidia-smi`, `rocm-smi`, `system-profiler`) is hidden by `doctor`
when the machine has GPUs but none from that vendor: a host with only an NVIDIA card is never
told that `rocm-smi` failed. When no GPU was detected at all, every failed probe is reported,
next to the "No GPU detected" warning, because one of them is usually the reason.

`vulkaninfo` is the last source there is for a card's memory, and it runs only when there is a
card no vendor tool sized: on a machine where `nvidia-smi` answered, it is never executed and
never reported. What it reads is the first *device-local* memory heap of each **discrete**
device, which is the same heap llama.cpp's own Vulkan backend takes for that device's memory,
together with the `VK_EXT_memory_budget` figures beside it — `heapBudget` is what the driver
will let a process allocate given everything already on the card, and `heapUsage` is what this
process has taken of it, so `budget - usage` is the free figure. It is labelled `estimated`
rather than `measured`, and printed with `(Vulkan driver)` beside it, because it is the
driver's promise rather than the card's nameplate: on the reference machine it reports 7.77 GiB
where `nvidia-smi` reports 8,188 MiB, and 7.02 GiB free where `nvidia-smi` reports 7,638 MiB.
Conservative is the safe direction for a memory budget to be wrong in.

Integrated GPUs are deliberately skipped. An integrated device's device-local heap *is* system
memory, already counted in the memory total, so reporting it as VRAM would describe a machine
with twice the memory it has and let the planner fill both pools with the same bytes. A
plausible wrong number is worse than no number, so those cards stay unsized. A driver too old
for `VK_EXT_memory_budget` gives a size and no free figure, which `system` prints as
`free unknown` and which is still not enough to plan on.

A card that stays unsized is not silently treated as absent. `system` prints a **Card memory**
row naming it, `doctor` raises *planned around, not planned on*, `recommend` and `fit` carry a
caption under the table, and a CPU-only plan says in its notes that a card is present and was
not used. The figures in all four cases are real; what they are figures *for* is a machine
with one fewer card than the one in front of you.

## Operating systems

| OS | Notes |
|---|---|
| Windows 10 and 11 | PowerShell 5 or 7 must be on `PATH` (it is by default). NVIDIA figures need the driver's `nvidia-smi`, installed with every driver. There is no vendor tool here for AMD (`rocm-smi` is Linux-only) or for Intel (there is none anywhere), so those cards fall back to the `vulkaninfo` probe above; the Vulkan runtime that provides it is installed with every current AMD, Intel and NVIDIA driver. Without it the card is listed by name, and the budget treats its size as unknown until you set it in a hardware profile. Long paths and spaces in paths are supported. |
| macOS 12 and later | Apple Silicon reports unified memory: there is no VRAM figure, the whole RAM pool is the budget, and the GPU table supplies bandwidth per chip. Intel Macs with discrete GPUs are listed by name and nothing sizes them: `vulkaninfo` is not run on macOS, since MoltenVK is a developer installation rather than something a Mac has, and `system_profiler`'s own `spdisplays_vram` is not read yet. `system_profiler` can take a few seconds on first use. |
| Linux | `dmidecode` needs root; everything else runs as a user. `pciutils` provides `lspci`. NVIDIA needs the proprietary driver for `nvidia-smi`; AMD needs ROCm for VRAM figures and Intel has no vendor tool at all, so both fall back to `vulkaninfo`, which `vulkan-tools` provides; without that package the card is listed by name only. |

## Architectures

`x86_64` and `arm64` are first-class. Other architectures scan, but llama.cpp release binaries
may not exist for them, which `doctor` reports when the installer (phase 2) is asked to fetch
one.

## Backends

LlamaFit reads which backends a llama.cpp build contains from its shipped `ggml-*` libraries:
`cuda`, `hip`, `metal`, `vulkan`, `sycl`, `rpc`, `cpu`. A GPU without a matching backend is a
`doctor` warning with the build to install.

## Disks

Disk usage is reported once per distinct volume: a path is identified by its drive letter on
Windows and by its device id (`st_dev`) on macOS and Linux, since every absolute POSIX path
shares the anchor `/` regardless of which filesystem it lives on. Each candidate path is made
absolute first and, if it does not exist, walked up to the nearest existing ancestor before its
usage is read.

## Terminals

The terminal dashboard needs a terminal with 256 colours and at least 80 columns. Windows
Terminal, PowerShell 7, macOS Terminal, iTerm2 and every common Linux terminal qualify. The
legacy Windows console works for the CLI tables but not for the dashboard.
