#!/bin/sh
# ============================================================================
#  start-qwen3-coder-next.sh
#  Launch Qwen3-Coder-Next (UD-Q4_K_XL) with llama.cpp on this machine.
#
#  Written by LlamaFit 1.2.3 on 2026-09-10,
#  by:  llamafit preset qwen3-coder-next
#  Placement: moe-offload, 32768 tokens, llama.cpp build 10867.
#  Card: NVIDIA GeForce RTX 4060, 8188 MiB total, 7638 MiB free at plan time.
#
#  WHY THIS SCRIPT CHOOSES ITS CONTEXT WHEN IT RUNS
#
#  The placement above was computed while the card had 7638 MiB free. By the
#  time you run this, a browser, a game or another model may be holding a
#  gigabyte of the graphics card. Ask llama.cpp for more of it than is free
#  and the driver does not refuse the allocation: it pages the overflow into
#  system memory. The server starts, /health answers, the log looks healthy,
#  and generation runs at a fraction of its speed -- for weeks, if nobody
#  thinks to measure it.
#
#  So the context is not written in. The ladder below lists each context this
#  plan can run at and what it costs. The script reads what is free at the
#  moment it runs, keeps 256 MiB back for everything else, and takes the
#  largest rung that still fits underneath.
#
#  It never climbs above the planned 32768. The planner weighed system memory
#  and what you asked for as well; this script can measure one pool, and
#  sizing against the pool you can see is how the other one overflows.
#
#  Yours to edit -- that is the point of writing a file rather than printing
#  a command. LlamaFit will not overwrite this once you have changed it: the
#  stamp below is a checksum of everything else in the file, and a stamp that
#  no longer matches makes `llamafit preset qwen3-coder-next` keep your
#  version and say so.
#
#  llamafit-preset-stamp: 0000000000000000000000000000000000000000000000000000000000000000
# ============================================================================

# Unset variables are a bug here, not an empty string: every one of them is a
# path or a memory figure, and continuing with an empty one is how a script
# launches something other than what it was written to launch.
set -u

# --- where things are -------------------------------------------------------
LLAMA_SERVER=/opt/llama.cpp/bin/llama-server
MODEL=/home/pv/models/qwen3-coder-next/Qwen3-Coder-Next-UD-Q4_K_XL.gguf

# --- has this machine changed since 2026-09-10? -----------------------------
# A model that has moved, or a llama.cpp that is gone, is fatal: there is
# nothing to run. A different card is not, because the ladder below adapts,
# but the layer split was chosen for the card named at the top of this file.
if [ ! -f "$MODEL" ]; then
    echo "[llamafit] The model file is not where this script expects it:" >&2
    echo "[llamafit]   $MODEL" >&2
    echo "[llamafit] Fetch it with:      llamafit install model qwen3-coder-next" >&2
    echo "[llamafit] Re-write this with: llamafit preset qwen3-coder-next --force" >&2
    exit 3
fi

if [ ! -x "$LLAMA_SERVER" ]; then
    if command -v llama-server >/dev/null 2>&1; then
        echo "[llamafit] llama.cpp is no longer at the path in this script;" >&2
        echo "[llamafit] using the llama-server found on PATH." >&2
        LLAMA_SERVER=llama-server
    else
        echo "[llamafit] llama-server was not found, neither here nor on PATH:" >&2
        echo "[llamafit]   $LLAMA_SERVER" >&2
        echo "[llamafit] Install it with: llamafit install llama.cpp" >&2
        exit 3
    fi
fi

# --- how much memory is free right now? -------------------------------------
# Prints a whole number of MiB, or nothing at all when it cannot tell. Nothing
# is a real answer and the caller treats it as one: guessing a figure here is
# how a script talks itself into a configuration that pages.
free_mib() {
    # nvidia-smi ships with the NVIDIA driver and answers in MiB. -i pins the
    # card the plan was made for: on a machine with two, the other one's free
    # memory would be a true number about the wrong device. tr keeps only the
    # digits, so the [N/A] some cards report for an unsupported field becomes
    # the empty string rather than a zero to compare against.
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits \
        2>/dev/null | head -n 1 | tr -dc '0-9'
}

FREE_MIB=$(free_mib)

# --- is this still the same card? -------------------------------------------
TOTAL_MIB=$(nvidia-smi -i 0 --query-gpu=memory.total --format=csv,noheader,nounits \
    2>/dev/null | head -n 1 | tr -dc '0-9')
if [ -n "$TOTAL_MIB" ] && [ "$TOTAL_MIB" != "8188" ]; then
    echo "[llamafit] This card reports $TOTAL_MIB MiB, not the 8188 MiB of" >&2
    echo "[llamafit] NVIDIA GeForce RTX 4060, which this plan was made for. The context" >&2
    echo "[llamafit] below still adapts, but the layer split does not:" >&2
    echo "[llamafit]   llamafit preset qwen3-coder-next --force" >&2
fi

# --- the ladder -------------------------------------------------------------
# Each rung was costed by the same budget that produced the plan.
#
#      context   needs on the card
#        32768     6481 MiB
#        24576     6225 MiB
#        16384     5969 MiB
#
CTX=''
if [ -n "$FREE_MIB" ]; then
    USABLE_MIB=$(( FREE_MIB - 256 ))
    if   [ "$USABLE_MIB" -ge 6481 ]; then CTX=32768
    elif [ "$USABLE_MIB" -ge 6225 ]; then CTX=24576
    elif [ "$USABLE_MIB" -ge 5969 ]; then CTX=16384
    fi
    if [ -z "$CTX" ]; then
        echo "[llamafit] Only $FREE_MIB MiB of card memory is free. The smallest" >&2
        echo "[llamafit] configuration in this plan needs 5969 MiB, plus the 256 MiB this" >&2
        echo "[llamafit] script keeps back for everything else." >&2
        echo "[llamafit] Starting anyway would page into system memory and run at a fraction" >&2
        echo "[llamafit] of the speed, so this script stops here instead." >&2
        echo "[llamafit] Close whatever is holding the card, or see what fits today:" >&2
        echo "[llamafit]   llamafit plan qwen3-coder-next" >&2
        exit 4
    fi
else
    echo "[llamafit] Could not read how much card memory is free, so this falls back to" >&2
    echo "[llamafit] the planned 32768 tokens, which fit when this file was written on" >&2
    echo "[llamafit] 2026-09-10." >&2
    echo "[llamafit] If the card is busier now, the driver will page and generation will" >&2
    echo "[llamafit] be slow. What fits today:" >&2
    echo "[llamafit]   llamafit plan qwen3-coder-next" >&2
    CTX=32768
fi

# LLAMAFIT_CONTEXT overrules all of the above, for a deliberate experiment.
if [ -n "${LLAMAFIT_CONTEXT:-}" ]; then
    CTX="$LLAMAFIT_CONTEXT"
fi

# --- go ---------------------------------------------------------------------
echo "[llamafit] Qwen3-Coder-Next (UD-Q4_K_XL) at $CTX tokens on http://127.0.0.1:8080"
echo "[llamafit] Press Ctrl+C to stop it."

# Anything you pass to this script is appended to the command below, so you
# always have the last word:  ./start-qwen3-coder-next.sh --verbose
#
# No exec: the shell stays alive so that a non-zero exit can be explained, which
# is worth one idle process. Ctrl+C reaches llama-server either way, because it
# goes to the whole foreground process group.
"$LLAMA_SERVER" \
    -m "$MODEL" \
    --alias qwen3-coder-next \
    --host 127.0.0.1 \
    --port 8080 \
    -c "$CTX" \
    -fa on \
    -ngl 99 \
    --n-cpu-moe 47 \
    --fit off \
    -t 16 \
    -tb 16 \
    -b 4096 \
    -ub 2048 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 40 \
    --jinja \
    -np 1 \
    "$@"
RC=$?
if [ "$RC" -ne 0 ]; then
    echo "[llamafit] llama-server exited with code $RC." >&2
    echo "[llamafit] If llama.cpp has been upgraded since build 10867, a flag above" >&2
    echo "[llamafit] may no longer be one it accepts. Re-render them with:" >&2
    echo "[llamafit]   llamafit preset qwen3-coder-next --force" >&2
fi
exit "$RC"
