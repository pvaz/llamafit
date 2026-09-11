# The terminal dashboard

`llamafit` with no arguments opens a full-screen dashboard in the terminal. It is built on
Textual, works at 80 columns, and shows exactly the data the CLI prints with `--json`.
Where there is no terminal to draw one in — a pipe, a redirect, a CI job — it says so on
standard error and prints what `llamafit recommend` would print instead.

## Layout

```
┌ LlamaFit ────────────────────────────────────────────────────────────────────────────┐
│ RTX 4060, 7.2 GiB free of 8.0 GiB · 100 GiB free of 128 GiB RAM · i9-14900KF, 8 co… │
│ [Board] [Needs] [Host] [Plan] [Simulate]                                              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Move with the up and down arrows. Enter says why a row is where it is; p gives the    │
│ command line that runs it.                                                            │
│ Speeds are section 10's formula on its default constants. Nothing has been            │
│ benchmarked on this machine yet, so no figure here is a measurement.                  │
│ 3 of 3 shown, by score, showing every candidate.                                      │
│  #  Model                  Quant       Score  Tok/s  Fit    Runs   Ctx    Have        │
│  1  qwen3-coder-next       UD-Q4_K_XL   91.2   23.4  fits   split  262K   no          │
│  2  qwen3.8-flash-next     UD-Q4_K_XL   84.0   14.1  tight  split   40K   no          │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ Qwen3 Coder Next UD-Q4_K_XL, Alibaba, licensed Apache-2.0; it can: coding, tools      │
│                          Score 91.2                                                   │
│ ┌─────────┬───────┬────────┬──────┐                                                   │
│ │ Part    │ Score │ Weight │ Adds │   … the budget, the ladder and a token's time      │
│ └─────────┴───────┴────────┴──────┘     follow, exactly as `--explain` prints them     │
└ ↑↓ choose  Enter why  p command line  / search  f fit  s sort  a on disk  A all … ───┘
```

Three lines come before the table and none of them is a footnote: what to do, what a speed
is, and what is on the screen. The explanation is open by default — on a terminal it costs
a page per row and sits behind `--explain`; on a screen it costs nothing.

## Screens

| Screen | Content | Keys |
|---|---|---|
| **Board** | the ranked table, with the explanation under it: the four scores and their weights, the quality behind them, the budget line by line with the source of each, the context ladder, and where a token's time goes | `↑↓` choose, `Enter` why, `p` command line, `/` search, `f` cycle the fit filter (every candidate, the ones that run, the ones with room to spare), `s` cycle the sort (score, speed, quality, context, download size), `a` on disk only, `A` all quantisations, `x` hide the explanation, `n` what was not ranked |
| **Needs** | section 12.1's request as a form: use case, required capabilities, minimum context, maximum download, accepted licences, and which way to lean the weights. Each field offers what exists — the six use cases and eight capabilities from the catalog's own types, the licences from its entries | `Ctrl+S` apply, `Ctrl+R` reset |
| **Host** | `system` and `doctor` on one page: the scan, llama.cpp, every probe with its outcome, every finding with its hint, and any catalog file the loader could not read | `R` rescan |
| **Plan** | for the selected row: the memory budget component by component, the context ladder, the flags and the full `llama-server` command line, plus the runs the catalog records for comparison | `+` and `-` move along the ladder, `v` vision on or off, `y` copy the command line |
| **Simulate** | answer for another machine: a bundled or user hardware profile, or an override of VRAM, system memory or core count. `SIMULATED` shows in the header for as long as the figures are not this machine | `Ctrl+S` apply, `Ctrl+R` back to this machine |

Global keys: `Tab` and `Shift+Tab` move between screens, `?` shows the key map, `t` cycles
the theme, `q` quits. Every key is declared once in `llamafit/tui/keys.py`, which is what
the bindings, the bar along the bottom and the `?` screen are all built from, so the bar
cannot advertise a key nothing is bound to.

## Behaviour

- The dashboard draws whatever it already has and scans on a background thread, so the
  screen is never frozen while a probe runs. `R` scans again.
- Nothing runs, downloads or changes on the machine from any of these five screens.
- **No number appears without saying how it was arrived at.** Nothing has been benchmarked
  on any machine yet, so every speed is section 10's formula on default constants, and the
  sentence saying so is a band above the table rather than a caption under it. When rows
  disagree about how their speeds were arrived at, the label becomes a column beside each
  figure, and a terminal too narrow for the pair shows neither.
- A column sort reorders the screen and never the ranking: the `#` column keeps saying where
  `rank` put each row.
- It degrades on narrow terminals by dropping the least important columns first, in a fixed
  priority, and never by truncating a model's name. The four columns that identify a row —
  rank, model, quantisation and score — are never dropped.
- A name wider than the model column folds onto a second line, exactly as it does on the
  command line's board, and the row grows a line to hold it. That is what "never truncated"
  costs, and it is the cheaper of the two: a name cut at the column's edge is not a shorter
  name but a different one, and `nemotron-3.5-lightning-30b-a3b` arriving as
  `nemotron-3.5-lightning-3` is a model nobody can look up.
- Colour is used for verdicts only and always paired with a word, so the screen reads the
  same without it.
- A scan that failed, a catalog that would not load and a substitution the machine cannot
  support each become a band carrying the message and the hint the error was raised with;
  the rest of the screen keeps whatever it still has.
- Everything the dashboard shows comes from `llamafit/services/` — the same calls the CLI
  makes — and most of it is drawn with the CLI's own renderables, so there is no
  dashboard-only data and no second opinion about a number.

## Not yet

**Downloads** (phase 2) and **Benchmarks** (phase 3) are named in section 13.2 and are not
built: there is nothing to download with and nothing to benchmark with yet. Section 12.3's
"what would move it up a tier" is not a sentence any service produces, so the dashboard
shows the context ladder with what each rung costs and leaves the reader to draw the
conclusion, rather than inventing the arithmetic in the interface.
