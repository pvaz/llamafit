# The terminal dashboard

`llamafit` with no arguments opens a full-screen dashboard in the terminal. It is built on
Textual, works at 80 columns, and shows exactly the data the CLI prints with `--json`.
Where there is no terminal to draw one in — a pipe, a redirect, a CI job — it says so on
standard error and prints what `llamafit recommend` would print instead.

## Layout

The Board screen, on the reference machine, in an 88-column terminal:

```
 ⭘                                     LlamaFit
 NVIDIA GeForce RTX 4060, 7.2 GiB free of 8.0 GiB · 94.0 GiB free of 127.8 GiB RAM ·
 Board  Needs  Host  Plan  Simulate
╸━━━━━╺━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Move with the up and down arrows. Enter says why a row is where it is; p gives the
 command line that runs it.
 Speeds are section 10's formula on its default constants. Nothing has been benchmarked
 on this machine yet, so no figure here is a measurement.
 71 of 71 shown, by score, showing every candidate.
 #    Model                         Quant        Score  Tok/s   Fit
 1    qwen3.6-35b-a3b               UD-Q6_K_XL    83.7   24.1   tight               ▅▅
 2    qwen3-coder-next              UD-Q4_K_XL    83.0   23.0   tight
 3    mistral-small-4-119b          UD-Q2_K_XL    82.1   20.8   fits
 4    north-mini-code-1.0           UD-Q4_K_XL    79.3   21.4   tight
 5    nemotron-3-nano-30b-a3b       UD-Q4_K_XL    77.8   22.8   tight
 6    nemotron-3.5-lightning-30b-   UD-Q4_K_XL    77.7   20.0   tight
      a3b
────────────────────────────────────────────────────────────────────────────────────────
Qwen3.6-35B-A3B UD-Q6_K_XL, Alibaba Qwen, licensed Apache-2.0; it can: coding,
thinking, vision, tools, multilingual, long-context                                   ▆▆

1. Qwen3.6-35B-A3B UD-Q6_K_XL
            Score 83.7
┏━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━┓
┃ Part    ┃ Score ┃ Weight ┃ Adds ┃
┡━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━┩
│ quality │  85.0 │   0.35 │ 29.7 │
│ speed   │  97.5 │   0.25 │ 24.4 │
│ fit     │  58.3 │   0.25 │ 14.6 │
 ↑↓ choose   Enter why   p command line   / search   f fit   s sort   S sort back
 o reverse   a on disk   A all quants   e unranked rows   c all columns   x explanation
 n not ranked   ? keys   t theme   q quit
```

Both panes scroll — the marks at the right edge are their scrollbars — so the explanation
under the table continues into the context score, the memory budget, the ladder and where a
token's time goes.

Three lines come before the table and none of them is a footnote: what to do, what a speed
is, and what is on the screen. The explanation is open by default — on a terminal it costs
a page per row and sits behind `--explain`; on a screen it costs nothing.

## Screens

| Screen | Content | Keys |
|---|---|---|
| **Board** | the one list: the ranked table, then the candidates that were not ranked, dimmed, with every figure that was computed for them and the word for the reason where the score would be; and under it the explanation of the row under the cursor: the four scores and their weights, the quality behind them, the budget line by line with the source of each, the context ladder, and where a token's time goes | `↑↓` choose, `Enter` why, `p` command line, `/` filter (see below), `f` cycle the fit filter (every candidate, the ones that run, the ones that fit, the ones with room to spare), `s` next sort key and `S` the previous one (score, speed, quality, context, size, prompt, card, ram, fit, model, quant — the command line's `--sort` words), `o` reverse the order, `a` on disk only, `A` all quantisations, `e` hide or show the unranked rows, `c` every column or the ones the width admits, `x` hide the explanation, `n` jump to the first candidate that was not ranked |
| **Needs** | section 12.1's request as a form: use case, required capabilities, minimum context, maximum download, the slowest generation worth having (the command line's `--min-tps`), accepted licences, and which way to lean the weights. Each field offers what exists — the six use cases and eight capabilities from the catalog's own types, the licences from its entries | `Ctrl+S` apply, `Ctrl+R` reset |
| **Host** | `system` and `doctor` on one page: the scan, llama.cpp, every probe with its outcome, every finding with its hint, and any catalog file the loader could not read | `R` rescan |
| **Plan** | for the selected row: the memory budget component by component, the context ladder, the flags and the full `llama-server` command line, plus the runs the catalog records for comparison | `+` and `-` move along the ladder, `v` vision on or off, `y` copy the command line |
| **Simulate** | answer for another machine: a bundled or user hardware profile, or an override of VRAM, system memory or core count. `SIMULATED` shows in the header for as long as the figures are not this machine | `Ctrl+S` apply, `Ctrl+R` back to this machine |

Global keys: `Tab` and `Shift+Tab` move between screens, `?` shows the key map, `t` cycles
the theme, `q` quits. Every key is declared once in `llamafit/tui/keys.py`, which is what
the bindings, the bar along the bottom and the `?` screen are all built from, so the bar
cannot advertise a key nothing is bound to.

The `/` box takes the web page's column filters as text, several separated by spaces:
`qwen` (part of an id, a name or a quantisation), `fit>=fits`, `speed>=20`, `size<=30G`,
`card<=6G`, `ram<=32G`, `ctx>=32K`, `quality>=70`, `runs=gpu`, `have`. Enter applies them,
Enter on an empty box clears them, and a term the box cannot read is refused by name with
nothing applied. `f` and `a` set `fit>=` and `have` in the same model, so the box shows what
they set and the state line reads the same whichever way a filter was set. A view filter
changes what is drawn and never what is ranked: the request lives on the Needs screen.

## Behaviour

- The dashboard draws whatever it already has and scans on a background thread, so the
  screen is never frozen while a probe runs. `R` scans again.
- Nothing runs, downloads or changes on the machine from any of these five screens.
- **No number appears without saying how it was arrived at.** Nothing the dashboard draws is
  fed a benchmark, so every speed is section 10's formula on default constants, and the
  sentence saying so is a band above the table rather than a caption under it. When rows
  disagree about how their speeds were arrived at, the label becomes a column beside each
  figure, and a terminal too narrow for the pair shows neither.
- A column sort reorders the screen and never the ranking: the `#` column keeps saying where
  `rank` put each row.
- It degrades on narrow terminals by dropping the least important columns first, in a fixed
  priority, and never by truncating a model's name. The four columns that identify a row —
  rank, model, quantisation and score — are never dropped. The chooser is the command line's
  own (`board_columns` in `llamafit/cli/render_board.py`), with the same budget measured from
  the catalog and the labels in force, so the dashboard and `llamafit recommend` draw the same
  columns at the same width — `docs/cli.md` tabulates them — and a test draws both to check.
  `c` overrides it with every column, and the table then scrolls sideways under `←` and `→`.
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

**Downloads** and **Benchmarks** are named in section 13.2 and are not built. The commands
behind them are: `llamafit install model`, `llamafit install llama.cpp` and `llamafit bench`
all ship, and none of them has a screen here — which is also why nothing on these five
screens downloads, runs or changes anything. Section 12.3's "what would move it up a tier" is
not a sentence any service produces, so the dashboard shows the context ladder with what each
rung costs and leaves the reader to draw the conclusion, rather than inventing the arithmetic
in the interface.
