# Launch presets — report

**Branch:** `feat/presets` · **Worktree:** `llamafit-wt-presets`
**Scope:** points 3 and 4 of section 15 of `docs/specs/2026-09-09-llamafit-design.md`
**Date:** 2026-09-10

---

## What was built

`src/llamafit/presets/`, six modules, plus `src/llamafit/cli/preset_cmd.py` holding both
commands and one registration line in `cli/app.py`.

| Module | What it is |
|---|---|
| `spec.py` | What a script is told, and the one decision it makes: `choose_context(rungs, free_mib, reserve_mib)`. Also the prose both scripts share, so the two files cannot say different things about the same machine. |
| `windows_cmd.py` | `start-<id>.cmd`. Batch quoting, and the reasons for not using delayed expansion or parenthesised blocks. |
| `posix_sh.py` | `start-<id>.sh`. POSIX `sh`, not bash. |
| `models_ini.py` | One `[model]` section for a llama.cpp router — the one artifact that cannot run the ladder, and says so in its own comments. |
| `render.py` | Files on disk: which files, line endings and encoding per file, the checksum stamp, and the refusal to overwrite an edit. |
| `launch.py` | `Launcher` protocol, `SubprocessLauncher`, `FakeLauncher`, the `/health` wait, and the note that lets `--stop` find what was started. |

`llamafit preset <model>` takes `--dir`, `--port`, `--quant`, `--context`, `--force`.
`llamafit launch <model>` takes `--dir`, `--port`, `--timeout`, `--stop`.

### The idea, and where it lives

The script carries no context. It carries the planner's ladder — every rung with the card
memory it needs — reads how much is free at the moment it runs, keeps 256 MiB back for the
desktop, and takes the largest rung that still fits.

Two rules shape the ladder, and both are refusals to be clever:

- **It only steps down.** The top rung is the context the planner chose, and no rung above
  it is offered even when the tier table says one would fit. The planner weighed system
  memory and the user's request; the script can measure one pool. Sizing against the pool
  you can see is how the other one overflows.
- **A rung the machine could not hold as scanned is not on it.** `ContextTier.fits` is
  false for a rung over the card *or* over system memory, and free card memory cannot tell
  those apart. They were unreachable anyway.

`choose_context` is one function on integers. The scripts do not reimplement it — they
unroll it, rung by rung, in their own dialect. That is what lets the rule be tested as a
rule and each script be tested *against* the rule rather than against a second copy of it
(`tests/unit/test_presets_scripts.py`, `_CMD_RUNG` / `_SH_RUNG`).

---

## 1. The Windows script generated for Qwen3-Coder-Next on this machine

Produced by `llamafit preset qwen3-coder-next` on the real machine (RTX 4060 8 GB,
i9-14900KF, 128 GB DDR5, Windows 11, llama.cpp build 10867), 2026-09-10, in full:

```bat
@echo off
rem ==========================================================================
rem  start-qwen3-coder-next.cmd
rem  Launch Qwen3-Coder-Next (UD-Q4_K_XL) with llama.cpp on this machine.
rem
rem  Written by LlamaFit 0.1.0a1 on 2026-09-10,
rem  by:  llamafit preset qwen3-coder-next
rem  Placement: moe-offload, 32768 tokens, llama.cpp build 10867.
rem  Card: NVIDIA GeForce RTX 4060, 8188 MiB total, 7076 MiB free at plan time.
rem
rem  WHY THIS SCRIPT CHOOSES ITS CONTEXT WHEN IT RUNS
rem
rem  The placement above was computed while the card had 7076 MiB free. By the
rem  time you run this, a browser, a game or another model may be holding a
rem  gigabyte of the graphics card. Ask llama.cpp for more of it than is free
rem  and the driver does not refuse the allocation: it pages the overflow into
rem  system memory. The server starts, /health answers, the log looks healthy,
rem  and generation runs at a fraction of its speed -- for weeks, if nobody
rem  thinks to measure it.
rem
rem  So the context is not written in. The ladder below lists each context this
rem  plan can run at and what it costs. The script reads what is free at the
rem  moment it runs, keeps 256 MiB back for everything else, and takes the
rem  largest rung that still fits underneath.
rem
rem  It never climbs above the planned 32768. The planner weighed system memory
rem  and what you asked for as well; this script can measure one pool, and
rem  sizing against the pool you can see is how the other one overflows.
rem
rem  Yours to edit -- that is the point of writing a file rather than printing
rem  a command. LlamaFit will not overwrite this once you have changed it: the
rem  stamp below is a checksum of everything else in the file, and a stamp that
rem  no longer matches makes `llamafit preset qwen3-coder-next` keep your
rem  version and say so.
rem
rem  llamafit-preset-stamp: e64819e3e9e5517859f4647810bf75abe61da5e8168d532d6f82545996fd84d4
rem ==========================================================================

setlocal

rem --- where things are -----------------------------------------------------
set "LLAMA_SERVER=D:\llama.cpp\bin\llama-server.exe"
set "MODEL=D:\llama.cpp\models\Qwen3-Coder-Next\Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

rem --- has this machine changed since 2026-09-10? ---------------------------
rem A model that has moved, or a llama.cpp that is gone, is fatal: there is
rem nothing to run. A different card is not, because the ladder below adapts,
rem but the layer split was chosen for the card named at the top of this file.
if exist "%MODEL%" goto model_found
echo [llamafit] The model file is not where this script expects it:
echo [llamafit]   %MODEL%
echo [llamafit] Fetch it with:      llamafit install model qwen3-coder-next
echo [llamafit] Re-write this with: llamafit preset qwen3-coder-next --force
exit /b 3
:model_found

if exist "%LLAMA_SERVER%" goto server_found
where llama-server.exe >nul 2>nul
if errorlevel 1 goto no_server
rem llama.cpp is not where it was, but PATH still has one; use that and say so.
echo [llamafit] llama.cpp is no longer at the path in this script;
echo [llamafit] using the llama-server.exe found on PATH instead.
set "LLAMA_SERVER=llama-server.exe"
goto server_found
:no_server
echo [llamafit] llama-server was not found, neither here nor on PATH:
echo [llamafit]   %LLAMA_SERVER%
echo [llamafit] Install it with: llamafit install llama.cpp
exit /b 3
:server_found

rem --- how much card memory is free right now? ------------------------------
rem nvidia-smi ships with the NVIDIA driver and answers in MiB. -i pins the card
rem the plan was made for: on a machine with two, the other one's free memory
rem would be a true number about the wrong device.
set "FREE_MIB="
for /f "usebackq tokens=1 delims= " %%f in (`nvidia-smi -i 0 --query-gpu^=memory.free --format^="csv,noheader,nounits" 2^>nul`) do set "FREE_MIB=%%f"
if not defined FREE_MIB goto no_probe
rem Some cards report a field they do not support as [N/A], which is an answer
rem but not a number, and comparing against it would compare against zero.
echo %FREE_MIB%| findstr /r /c:"^[0-9][0-9]*$" >nul
if errorlevel 1 goto no_probe

rem --- is this still the same card? -----------------------------------------
set "TOTAL_MIB="
for /f "usebackq tokens=1 delims= " %%t in (`nvidia-smi -i 0 --query-gpu^=memory.total --format^="csv,noheader,nounits" 2^>nul`) do set "TOTAL_MIB=%%t"
if not defined TOTAL_MIB goto card_checked
if "%TOTAL_MIB%"=="8188" goto card_checked
echo [llamafit] This card reports %TOTAL_MIB% MiB, not the 8188 MiB of
echo [llamafit] NVIDIA GeForce RTX 4060, which this plan was made for. The context below
echo [llamafit] still adapts, but the layer split does not:
echo [llamafit]   llamafit preset qwen3-coder-next --force
:card_checked

rem --- the ladder -----------------------------------------------------------
rem Each rung was costed by the same budget that produced the plan.
rem
rem      context   needs on the card
rem        32768     5549 MiB
rem        24576     5293 MiB
rem        16384     5037 MiB
rem
set /a "USABLE_MIB=%FREE_MIB% - 256"
set "CTX="
if %USABLE_MIB% GEQ 5549 if not defined CTX set "CTX=32768"
if %USABLE_MIB% GEQ 5293 if not defined CTX set "CTX=24576"
if %USABLE_MIB% GEQ 5037 if not defined CTX set "CTX=16384"
if defined CTX goto have_context

echo [llamafit] Only %FREE_MIB% MiB of card memory is free. The smallest
echo [llamafit] configuration in this plan needs 5037 MiB, plus the 256 MiB this
echo [llamafit] script keeps back for everything else.
echo [llamafit] Starting anyway would page into system memory and run at a fraction
echo [llamafit] of the speed, so this script stops here instead.
echo [llamafit] Close whatever is holding the card, or see what fits today:
echo [llamafit]   llamafit plan qwen3-coder-next
exit /b 4

:no_probe
echo [llamafit] Could not read how much card memory is free, so this falls back to
echo [llamafit] the planned 32768 tokens, which fit when this file was written on
echo [llamafit] 2026-09-10.
echo [llamafit] If the card is busier now, the driver will page and generation will
echo [llamafit] be slow. What fits today:
echo [llamafit]   llamafit plan qwen3-coder-next
set "CTX=32768"

:have_context
rem LLAMAFIT_CONTEXT overrules all of the above, for a deliberate experiment.
if defined LLAMAFIT_CONTEXT set "CTX=%LLAMAFIT_CONTEXT%"

rem --- go -------------------------------------------------------------------
echo [llamafit] Qwen3-Coder-Next ^(UD-Q4_K_XL^) at %CTX% tokens on http://127.0.0.1:8080
echo [llamafit] Press Ctrl+C to stop it.

rem Anything you pass to this script is appended to the command below, so you
rem always have the last word:  start-qwen3-coder-next.cmd --verbose
rem
rem The carets are line continuations. A space after one ends the command there,
rem which is the one edit to this block that fails quietly, so watch for it.
"%LLAMA_SERVER%" ^
  -m "%MODEL%" ^
  --alias qwen3-coder-next ^
  --host 127.0.0.1 ^
  --port 8080 ^
  -c %CTX% ^
  -fa on ^
  -ngl 99 ^
  --n-cpu-moe 48 ^
  --fit off ^
  -t 16 ^
  -tb 16 ^
  -b 4096 ^
  -ub 2048 ^
  --temp 1.0 ^
  --top-p 0.95 ^
  --top-k 40 ^
  --jinja ^
  -np 1 ^
  %*
set "RC=%ERRORLEVEL%"
if "%RC%"=="0" goto stopped
echo [llamafit] llama-server exited with code %RC%.
echo [llamafit] If llama.cpp has been upgraded since build 10867, a flag above may no
echo [llamafit] longer be one it accepts. Re-render them with:
echo [llamafit]   llamafit preset qwen3-coder-next --force
:stopped
exit /b %RC%
```

Note the model path: the scan found the file already on `D:` and the command line names the
real one rather than a stand-in for a download that has not happened.

---

## 2. How free card memory is read, and what happens when it cannot be

The rule is: use the tool that already ships with the driver. Anything else is a dependency
a double-clickable script cannot assume.

| Machine | How the script asks | Notes |
|---|---|---|
| NVIDIA, any OS | `nvidia-smi -i N --query-gpu=memory.free --format="csv,noheader,nounits"` | `-i N` pins the device the plan was made for; on a two-card machine the other one's free memory is a true number about the wrong device. |
| AMD on Linux | `/sys/class/drm/card*/device/mem_info_vram_{total,used}`, in bytes | sysfs is always present and readable when amdgpu is loaded; `rocm-smi` is neither, so it is named in a comment and never called. |
| Apple silicon | `vm_stat`, free + inactive + speculative pages × the page size from its header | The graphics memory *is* the system memory. Inactive pages are cache the kernel hands over on demand; leaving them out would understate what is available by most of it. |
| AMD or Intel on Windows, Intel on Linux | nothing | See below. |

Two independent guards on the answer:

- **The value must be a number.** `nvidia-smi` reports an unsupported field as `[N/A]`, on
  standard output, where it looks exactly like a memory figure. The batch file runs it
  through `findstr /r "^[0-9][0-9]*$"`; the shell script through `tr -dc '0-9'`, which turns
  a non-numeric answer into the empty string. A blank answer takes the same path as no
  answer at all.
- **`for /f` needs the format argument quoted.** This cost an afternoon and is worth
  recording: `for /f` hands its back-quoted command to a second `cmd`, whose tokeniser
  splits on commas and equals signs, so an unquoted `--format=csv,noheader,nounits` reaches
  the tool as three arguments and it answers `ERROR: Option noheader is not recognized` — on
  standard output. Without the numeric guard above, `set /a` would have parsed that as zero
  and every run would have stepped to the bottom rung. Quoting the value and escaping the
  equals signs is what was measured to work; the guard is what would have caught it anyway.

**When it cannot be read at all**, there are two different situations and they get two
different answers.

*The probe failed on a card LlamaFit knows how to ask* (nvidia-smi missing, or it answered
with something that is not a number): the script prints, on every run,

```
[llamafit] Could not read how much card memory is free, so this falls back to
[llamafit] the planned 32768 tokens, which fit when this file was written on
[llamafit] 2026-09-10.
[llamafit] If the card is busier now, the driver will page and generation will
[llamafit] be slow. What fits today:
[llamafit]   llamafit plan qwen3-coder-next
```

and starts at the planned context. This is the honest end of a choice between two bad
options. Falling back to the *smallest* rung would leave every machine whose tool failed
once running at the shortest context for ever. Falling back silently to the planned one
would be the baked-in number this whole package exists to remove. So it uses the number that
was true when the plan was made, and says out loud that that is what it did.

*There is no tool to ask* — an AMD or Intel card on Windows, Intel on Linux — the script says
so **once, in its header**, rather than apologising on every run:

```
#  WHY THIS SCRIPT DOES NOT CHOOSE A CONTEXT WHEN IT RUNS
#
#  LlamaFit has no way to read free memory from a intel card on linux: there is
#  no vendor tool a script can call for it here. So the context below is the
#  planned 32768, and that is a number that was true on 2026-09-10 rather than
#  one this script works out for itself.
```

A placement that puts nothing on a card (`cpu` mode) gets a third, shorter note: there is no
free-card-memory figure to read and nothing to step down for.

**When nothing fits.** If even the smallest rung needs more than is free, the script stops
with **exit code 4** and says why. This is the one case where refusing to start is right:
starting would page, and a server at a third of its speed with a healthy-looking log is
precisely the failure a person cannot diagnose for themselves. `LLAMAFIT_CONTEXT` overrules
the ladder for a deliberate experiment.

---

## 3. What happens when the machine has changed

Four things can have changed, and they deliberately do not get the same answer.

| What changed | What the script does | Why |
|---|---|---|
| **Something else is holding the card** | takes a lower rung, and says which | The ladder. This is the ordinary case and needs no drama. |
| **The model file has moved** | prints the path it looked in and exits **3** | There is nothing to run. Guessing at another path would produce a worse error later, from llama.cpp, about a file the user never named. |
| **llama.cpp has moved** | uses the `llama-server` on `PATH` if there is one, saying so; otherwise exits **3** | A moved installation is usually a reinstall, and `PATH` is where a reinstall puts it. Silently using a different binary would be wrong, so it is announced. |
| **A different card** | starts anyway, and says the layer split was chosen for another one | Survivable, *because* the ladder is there: the context adapts. What does not adapt is `-ngl`/`--n-cpu-moe`, which were chosen for the old card's size, so it points at `llamafit preset <id> --force`. Refusing to start would be worse than a working configuration that is not the best one. |

The card check compares **total** card memory, not the name: a name is a string a driver
update can reword, and the total is the figure the layer split was actually chosen against.

**llama.cpp upgraded and a flag no longer accepted.** This one is not checked before the
launch, on purpose. Running `llama-server --version` and parsing a build number on every
start costs a subprocess and a fragile parse, and it would still not tell you whether *this*
flag list is acceptable. Instead the script turns the symptom into an explanation: a non-zero
exit prints

```
[llamafit] llama-server exited with code 1.
[llamafit] If llama.cpp has been upgraded since build 10867, a flag above may no
[llamafit] longer be one it accepts. Re-render them with:
[llamafit]   llamafit preset qwen3-coder-next --force
```

which is the sentence somebody staring at an immediate exit actually needs, at the moment
they need it.

**And the user's own changes.** Every generated file carries `llamafit-preset-stamp:`, a
SHA-256 of the whole file with the stamp itself blanked. A file whose stamp no longer matches
has been edited: LlamaFit keeps it, names it, and says `--force` is what replaces it — and
`--force` reports "overwritten: your edits are gone" rather than letting them go quietly. The
check fails closed: a file with **no** stamp at all counts as edited, because a file that lost
its stamp was either hand-written or hand-cut and both are somebody's work. Files are handled
one at a time, so editing the script does not stop the README being refreshed.

---

## Decisions asked for

### The generated scripts and the translation layer

**Everything LlamaFit prints goes through `gettext`. Nothing LlamaFit writes does.** The
generated files are English, and `scripts/gen_messages.py` extracts nothing from
`llamafit/presets/*` because there is nothing there to extract — 589 messages, unchanged in
count by the package itself; the CLI module's own strings are all in the template.

Three reasons, in order of how hard they are to argue with:

1. **A `.cmd` cannot declare its encoding.** `cmd` reads a batch file in the console's code
   page, and a byte-order mark at the top is printed rather than skipped. A batch file whose
   comments and `echo` lines were Japanese or Portuguese would be mojibake on the machine it
   was written for. This is a technical fact about the format, not a preference.
2. **The artifact outlives the session that made it.** It is shared, pasted into bug reports,
   read by whoever inherits the machine. A script whose language depends on what `--language`
   said one afternoon is a script the next person cannot read.
3. **It is also read by a shell.** Comments are prose, but the `echo` lines are output a user
   may grep, and the labels (`[llamafit]`) are the same in every locale.

The one place the user's own environment intrudes is a path that is not ASCII: that is not
LlamaFit's choice, and `windows_cmd.py` emits `chcp 65001 >nul` with a comment explaining why
in exactly that case. The generated files are otherwise ASCII throughout.

### The copyright header on a generated script

Every source module under `src/llamafit` carries the three-line AGPL notice and
`tests/unit/test_licensing.py` enforces it. The six new modules do.

**A generated script carries a provenance header instead, and deliberately no licence
notice.** A source module is part of LlamaFit; a launch script is *output* — it is the user's
configuration, written to be edited, and stamping somebody's own start script with this
project's licence terms would be a claim over their file that the licence never made. What
the header carries instead is what a licence notice is actually for here: who wrote it, when,
from what plan, on what machine, and the exact command to regenerate it — plus the checksum
that stops LlamaFit taking the file back once they have made it theirs.

The template that produces it is source, is licensed, and is tested for the notice like
everything else.

### The router `models.ini`

Written as `models-<id>.ini`, one section, appended by the user into whatever router config
they keep. It is the **one artifact that cannot run the ladder** — a router starts a model
from a configuration file with no shell in between — so its own comments say so, name the
day the number was true, and give the bottom rung's context and cost so a person sharing the
card with a desktop can choose the smaller one knowingly.

### Line endings, encoding, permissions

Decided once, in `render.py`, and tested on the bytes: `.cmd` is CRLF, UTF-8, no BOM; `.sh`
is LF on every platform (a `.sh` with carriage returns fails at its own shebang — the kernel
looks for an interpreter called `/bin/sh\r`) and gets the execute bits where the platform has
any. The renderers produce `\n` throughout, so every golden file in the repository is plain
text with plain newlines and the translation is tested separately.

---

## Testing

`tests/golden/presets/` holds four files rendered from the real bundled catalog against the
recorded reference machine, compared as text.

The tier logic is tested three ways, deliberately separated:

1. **As a rule.** `choose_context` against a fixed ladder at eight free-memory figures,
   including the boundaries either side of each rung and the case where nothing fits.
2. **As written down.** Each rendered script's thresholds are parsed back out with a regular
   expression and *simulated*, then compared with what `choose_context` says at nine free
   figures. Both dialects are also compared with each other, so they cannot drift.
3. **By running them.** Both scripts were executed for real during development, against a
   fake `llama-server` and a model file whose path holds a space, a per cent sign, an
   ampersand, an exclamation mark, brackets and (on POSIX) an apostrophe:
   `D:\...\100% Mine & Yours!\Qwen3 (v2).gguf` and
   `/tmp/.../it's 100% mine & more/Qwen3 $HOME (v2).gguf`. The path arrived at the program as
   one intact argument in both. Verified paths: top rung taken (32768), step-down taken
   (24576 and 16384), nothing-fits refusal (exit 4), model missing (exit 3), and the POSIX
   script under `sh -n`.

`launch` is tested through `FakeLauncher` and `FakeHttp` with an injected clock, so nothing is
spawned and nothing is dialled. `tests/unit/test_presets_subprocess.py` exercises the real
`SubprocessLauncher` against `python -c` sleepers for the two things a fake cannot prove: that
a note whose start time does not match is **not** acted on (the recycled-pid guard), and that
stopping takes the children with it (the preset runs under `cmd` or `sh`, so `llama-server` is
always somebody's child).

### Verification

```
pytest --cov -m "not hardware and not network"   1903 passed, 1 skipped, coverage 96.47%
ruff check .                                     All checks passed
ruff format --check .                            230 files already formatted
mypy                                             Success: no issues found in 110 source files
gen_messages && gen_schema && gen_models_md && git diff --exit-code   clean
```

Per-module coverage of the new code: `posix_sh` 100%, `models_ini` 100%, `render` 99%,
`windows_cmd` 99%, `spec` 96%, `cli/preset_cmd` 96%, `launch` 94%.

---

## Concerns

1. **The `models.ini` format is a guess at llama.cpp's router.** I could not confirm the exact
   key names a llama.cpp router expects (`model`, `mmproj`, `args` under a `[<id>]` section).
   The shape is what section 15.3 asks for and the values are right; if the router's real
   format differs, only `models_ini.py` and one golden file change. Worth checking against a
   build that has the feature before this is advertised.
2. **A router preset is a baked-in context, by construction.** It is the one thing here that
   can still page silently. The comments say so and name the safer rung, which is the best
   that can be done without a shell in front of the model.
3. **Two cards are handled by pinning one.** The plan is for `Host.primary_gpu` and the script
   asks `nvidia-smi -i <that index>`. A machine whose device order changes between the plan
   and the run would read the wrong device's free memory. The total-memory check catches it
   when the two cards differ in size and not when they are identical — where it also does not
   matter much.
4. **Apple silicon's `vm_stat` arithmetic is unverified on real hardware.** The parse (page
   size from the header, free + inactive + speculative) is right by the manual page but has
   not been run on a Mac; there is no Apple machine here. It is one `awk` block and one
   golden test.
5. **`--stop` needs a note.** A server started by double-clicking the script rather than
   through `llamafit launch` leaves no note and cannot be stopped by `--stop`. That is
   arguably correct — LlamaFit did not start it — but a person may reasonably expect
   otherwise. Discovering it through `llamacpp/server.py`'s port probe is the obvious
   improvement and belongs with the installer work.
6. **The overwrite check is content-based, not timestamp-based.** Re-saving a file unchanged
   in an editor keeps it "unedited"; adding and then removing a comment does too. That is the
   right direction to be wrong in, but it means "you edited it" is really "the bytes differ".
7. **`docs/cli.md` was edited**, which is a shared file. The change replaces the two phase-2
   stubs in place and adds nothing elsewhere, so a merge conflict with another agent should
   be small and obvious.
