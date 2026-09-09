# The terminal dashboard

`llamafit` with no arguments opens a full-screen dashboard in the terminal. It is built on
Textual, works at 80 columns, and shows exactly the data the CLI prints with `--json`.
The dashboard is part of phase 1D; this page fixes its shape so the work can be reviewed
before it is built.

## Layout

```
┌ LlamaFit ─────────────────────────────────────────────────────────────────────┐
│ RTX 4060 8 GB (7.5 GB free) · 128 GB DDR5 58 GB/s · i9-14900KF 8P · llama.cpp b10867 cuda │
│ [Board] [Needs] [Host] [Plan] [Simulate] [Downloads] [Benchmarks]                │
├────────────────────────────────────────────────────────────────────────────────┤
│ #  Model                     Quant       Mode         Verdict      VRAM  RAM   Ctx   Gen   PP   Q   Score │
│ 1  qwen3-coder-next          UD-Q4_K_XL  experts→RAM  Comfortable  7.2G  48G  262K  23e   322e 86  91    │
│ 2  qwen3.8-flash-next        UD-Q4_K_XL  experts→RAM  Fits         7.2G  60G  40K   14m   50m  92  84    │
│ 3  ...                                                                          │
├────────────────────────────────────────────────────────────────────────────────┤
│ Detail: qwen3.8-flash-next · UD-Q4_K_XL · 111.3 GB · Apache-2.0 · thinking, vision, coding │
│ Budget: weights 4.4 GB VRAM + 60 GB RAM · KV 36 MiB/1K · compute 1.3 GB · reserve 0.25 GB │
│ Why here: quality 92 leads the catalog; context capped at 40K by free VRAM; free 1.2 GB   │
│ (close the browser) to reach 64K.                                                          │
└ / search  f fit  s sort  u use case  c capabilities  l license  a installed  p plan  ? keys ┘
```

Speed cells carry their confidence as a suffix: `m` measured, `c` calibrated, `e` estimated.

## Screens

| Screen | Content | Keys |
|---|---|---|
| **Board** | the ranked table; `Enter` toggles the detail pane; `x` toggles the explanation | `/` search, `f` cycle fit filter (all, runnable, comfortable), `s` cycle sort (score, speed, quality, context, size), `u` use case, `c` capabilities, `l` license, `a` installed only, `q` all quants |
| **Needs** | form: use case, required and preferred capabilities, minimum context, maximum download, allowed licenses, prefer quality or speed; the Board re-ranks as you change it | `Enter` apply, `r` reset |
| **Host** | the scan with every probe and its outcome, llama.cpp status, running servers, `doctor` findings | `R` rescan |
| **Plan** | for the selected row: budget by component, the context tier table, chosen flags, the full command line; copy to clipboard | `p` from the Board, `+` and `-` change context, `v` toggle vision, `y` copy |
| **Simulate** | override VRAM, RAM and cores, or pick a hardware profile; every score recalculates; a `SIM` badge shows in the header until reset | `S`, `Tab` between fields, `Enter` apply, `Ctrl+R` reset |
| **Downloads** (phase 2) | active download with progress and speed, queue, history; change the downloads directory | `D`, `x` delete, `e` edit directory |
| **Benchmarks** (phase 3) | estimated versus measured for every measured configuration; run a benchmark for the selected row | `b`, `Enter` run |

Global keys: `Tab` and `Shift+Tab` move between screens, `?` shows the key map, `t` cycles the
theme, `q` quits.

## Behaviour

- Startup shows the Board within a second using the last scan from the cache, then rescans in
  the background and updates in place; the header says "rescanning" while it does.
- Nothing runs, downloads or changes on the machine from the Board, Needs, Host, Plan or
  Simulate screens. Only Downloads and Benchmarks act, and both ask before they start.
- The dashboard degrades on narrow terminals by hiding the least important columns first
  (RAM, PP, Q), never by truncating the model name.
- Colours are used for verdicts and confidence only, and always paired with text, so the
  screen reads the same without colour.
- Everything the dashboard shows is available from the CLI with `--json`; there is no
  dashboard-only data.
