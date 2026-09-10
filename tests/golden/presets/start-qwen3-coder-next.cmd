@echo off
rem ==========================================================================
rem  start-qwen3-coder-next.cmd
rem  Launch Qwen3-Coder-Next (UD-Q4_K_XL) with llama.cpp on this machine.
rem
rem  Written by LlamaFit 1.2.3 on 2026-09-10,
rem  by:  llamafit preset qwen3-coder-next
rem  Placement: moe-offload, 32768 tokens, llama.cpp build 10867.
rem  Card: NVIDIA GeForce RTX 4060, 8188 MiB total, 7638 MiB free at plan time.
rem
rem  WHY THIS SCRIPT CHOOSES ITS CONTEXT WHEN IT RUNS
rem
rem  The placement above was computed while the card had 7638 MiB free. By the
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
rem  llamafit-preset-stamp: 0000000000000000000000000000000000000000000000000000000000000000
rem ==========================================================================

setlocal

rem --- where things are -----------------------------------------------------
set "LLAMA_SERVER=D:\llama.cpp\bin\llama-server.exe"
set "MODEL=D:\Models\Qwen3 Coder Next\Qwen3-Coder-Next-UD-Q4_K_XL.gguf"

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
rem        32768     6481 MiB
rem        24576     6225 MiB
rem        16384     5969 MiB
rem
set /a "USABLE_MIB=%FREE_MIB% - 256"
set "CTX="
if %USABLE_MIB% GEQ 6481 if not defined CTX set "CTX=32768"
if %USABLE_MIB% GEQ 6225 if not defined CTX set "CTX=24576"
if %USABLE_MIB% GEQ 5969 if not defined CTX set "CTX=16384"
if defined CTX goto have_context

echo [llamafit] Only %FREE_MIB% MiB of card memory is free. The smallest
echo [llamafit] configuration in this plan needs 5969 MiB, plus the 256 MiB this
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
  --n-cpu-moe 47 ^
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
