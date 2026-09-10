# The web dashboard — what was built, and what it cost

Branch `feat/web`, worktree `llamafit-wt-web`. Three commits:

| SHA | Subject |
|---|---|
| `46136e1` | build: add the optional web extra, with its rows in the dependency table |
| `9874cd8` | feat(web): the dashboard and its JSON API, on this machine only |
| `25421a4` | docs: the dashboard, the API it reads, and what it does about languages |

Verification, all from the worktree's own `.venv`:

```
pytest --cov -m "not hardware and not network"   1857 passed, 1 skipped, 11 deselected; coverage 96.53%
ruff check .                                     All checks passed
ruff format --check .                            225 files already formatted
mypy                                             no issues in 108 source files
gen_messages && gen_schema && gen_models_md      git diff --exit-code clean
```

Per-file coverage of the new code: `web/api.py` 99%, `web/server.py` 100%,
`web/strings.py` 100%, `cli/serve_cmd.py` 100%.

I also ran the real thing: started `llamafit serve --port 8791` on this machine and drove
the page in a browser. Every panel renders, the board sorts, a row expands into its score,
speed, budget and context ladder, the Plan panel produces a pasteable command line, and
Simulate with `memory=24G` moves `llama-3.1-8b-instruct` to the top at 31 tokens per second
with 97K of context and puts the `SIMULATED` badge in the header. The only console error was
a missing favicon, which is now an inline SVG data URI.

## What shipped

`src/llamafit/web/`:

| File | Lines | What it is |
|---|---|---|
| `api.py` | 796 | the endpoints, the query dependency, the error mapping, the static mount |
| `strings.py` | 376 | every word the page draws, built through gettext |
| `server.py` | 187 | the loopback guard, the warning, `uvicorn` |
| `static/app.js` | 875 | the whole page |
| `static/styles.css` | 340 | |
| `static/index.html` | 175 | |

Plus `src/llamafit/cli/serve_cmd.py` (96 lines) and one line added to each of the import
tuple and `__all__` in `cli/app.py` — nothing else outside the package was touched except
docs and the changelog.

Endpoints: `GET /health`, and under `/api/v1`: `ui`, `system`, `POST scan`, `doctor`,
`models`, `models/top`, `models/{id}`, `POST plan`, `profiles`, `catalog/schema`,
`catalog/problems`. FastAPI's own `/api/v1/docs` comes along with it.

Every endpoint calls a service and returns `model.model_dump_json(by_alias=True)` — the
identical call the commands make for `--json`. Two tests run the command through Typer's
runner and the endpoint through the test client and assert the parsed documents are equal,
for the board and for a plan. `by_alias` is not cosmetic: a budget line's size is `bytes_`
in Python and `bytes` in the document, and an API that dropped the aliases would publish a
second spelling of every model in the project.

Nothing in `web/` computes a byte, a token or a score.

### Where I reused rather than rewrote

- Verdict, run-mode, confidence, pool, budget-component and source labels are **imported**
  from `cli/render_board.py`, which already made them public. The page cannot call
  `too-tight` anything but what the terminal calls it.
- Capability names are the one table the terminal keeps private, so `web/strings.py` writes
  it again using the identical `pgettext` context and messages — one catalog entry serves
  both — and a test compares the two tables entry by entry.
- `use_case`, `require`, `prefer` and `max_download` are validated by `cli/common.py`'s own
  checkers, so a bad value is refused in the same words. One consequence to know about: the
  message for a bad `use_case` names `--use-case`, the flag, not the parameter. I left it
  that way rather than add a 38-language string to say the same thing differently.
- `choose_quant` for the plan endpoint comes from `cli/plan_cmd.py`.
- The catalog-problem sentence is built by the same `ngettext` call `cli/common.py` makes,
  because plural rules are not something JavaScript can be handed a count and asked to
  guess at.

## The three questions

### 1. What does the page do about languages, and what does a reader who does not read English get?

**The page contains no English at all.** Not as a slogan — as a test. `index.html` has one
untranslatable word in its body (`LlamaFit`), and `test_web_static.py` strips the tags and
fails if any other word longer than two letters survives. Every label is a `data-t="key"`
attribute or a `t("key")` call, and both are filled from `GET /api/v1/ui`.

That payload is built in Python, at request time, by the same `_()` and `pgettext()` calls
every other message in LlamaFit goes through. So `scripts/gen_messages.py` extracts the
dashboard's strings from `web/strings.py` exactly as it extracts the command line's, they
land in `messages.pot` with the rest, and a translator who fills in a `.po` file translates
the dashboard without knowing it exists. There is no second catalog format, no second
extractor, and nothing compiled.

I considered the three alternatives and rejected each: an English-only page abandons most
of the project's readers; a JavaScript catalog is a second set of strings that will drift
from the first; compiling the catalogs into JS at build time needs a build step, and goal
one says there isn't one.

Where possible the page asks for a message the terminal already asks for — the same literal
string, so both read one catalog entry. Of the 105 label entries the page needs, **44 reuse
an entry the command line already had and 61 are new**; the template went from 549 messages
to 610.

**What a reader who does not read English actually gets, today, honestly:** the page in
their language *to the extent their catalog is filled in*, message by message, with English
as the per-message fallback — the same deal as every other part of LlamaFit. That is not
the same as "translated". Only `pt_PT` has substantial content and even it is about half
done and flagged as not reviewed by a native speaker, so of the page's 75 chrome strings
roughly ten come out in Portuguese today and the rest fall back. What the reader *does* get
in full, in every language with a catalog, is the part that matters most: the sentences the
services produce — every exclusion reason, every placement note, every budget-line note,
every `doctor` finding — because those are already translated wherever the terminal's are,
and the page shows them verbatim rather than re-writing them. The screenshot I took on this
machine (system locale `pt_PT`) shows *Máquina*, *Memória*, *Conclusões*, *Instalado*,
*Notas*, *programação*, and numbers written `76,4` and `3 750`.

Two more things follow the language:

- **Number punctuation.** The page draws byte counts and token counts because the API sends
  integers, not pictures of them. `GET /api/v1/ui` carries the group separator, the decimal
  separator and the word for an unknown size, all from `llamafit.units`, and `app.js`
  applies them. The arithmetic — which unit, one decimal — is the one thing repeated from
  Python; a test pins the unit ladder and four representative sizes so a change to how
  LlamaFit writes a size fails a test that names `app.js`.
- **Direction.** `ui.direction` is `rtl` for Arabic, Hebrew and Urdu and the page sets
  `document.documentElement.dir` from it; the CSS uses logical properties
  (`margin-inline-start`, `text-align: start`) throughout, so the layout mirrors. I have not
  had a right-to-left reader look at it, and the bidi isolation the terminal applies to
  identifiers inside sentences is not applied here — the browser does its own bidi
  algorithm, which is usually right and occasionally is not. See concerns.

**One language per process.** The server speaks whatever `--language`, `LLAMAFIT_LANGUAGE`
or the system locale chose at start-up. `Accept-Language` is ignored. The translator is
process-global state and handlers run in a thread pool, so per-request switching would be a
race; for a server whose entire design is one person at one keyboard, honouring the
language they started it in is also simply the right answer.

### 2. What is reachable from another machine on the network, under what settings, and what would that expose?

**By default: nothing.** The server binds `127.0.0.1`.

More usefully — the default is not the guard. `llamafit.web.serve()` **refuses** a
non-loopback address with a `ConfigError` unless the caller passes `allow_remote=True` as a
separate argument, and `serve_cmd.py` sets that only from `host is not None`, i.e. only when
`--host` was actually typed. So an address arriving from a config file, an environment
variable or a default nobody re-read cannot become a remote bind; somebody has to write the
consent down. `is_loopback` accepts `127.0.0.0/8`, `::1` and the literal name `localhost`,
and treats any other hostname as remote — it will not do a DNS lookup, because a lookup
that decides whether a security guard applies is a guard somebody else's resolver controls.

When `--host 0.0.0.0` *is* typed, the server starts, and it prints this first, in red, to
stderr:

> 0.0.0.0:8765 is not this machine only. Anyone who can reach this computer on that port can
> read everything the dashboard shows — your hardware, your disks, your file paths and the
> models you have downloaded. There is no password, because LlamaFit is meant for the person
> sitting at the keyboard. Press Ctrl+C now if you did not mean this.

That is accurate. What an unauthenticated caller on the network would then get:

- **`/api/v1/system` and `/api/v1/doctor`** — the full host scan: CPU model, core counts,
  memory size and DDR type, GPU model and driver version, **every disk path with its free
  space** (on this machine that includes `C:\Dev\Projectos Pessoais\2026\llamafit-wt-web`),
  the llama.cpp install path, and the paths of every GGUF file on disk. That is the most
  sensitive thing here: it is a partial map of the user's filesystem and a fingerprint of
  their machine.
- **`/api/v1/models*` and `/api/v1/plan`** — the catalog with per-host scoring, and command
  lines that embed local file paths.
- **`POST /api/v1/scan`** — re-runs the probes. It is the one unauthenticated write-shaped
  verb: it spawns `nvidia-smi`/`wmic`-class probes and measures memory bandwidth. It takes
  arguments from nobody and cannot be pointed anywhere, but it is a few seconds of CPU per
  call and there is no rate limit, so it is a trivial nuisance-level denial of service on a
  server somebody has deliberately exposed.
- Nothing writes to disk, downloads anything, or runs a model. There is no endpoint that
  takes a path.

Three deliberate hardenings, all of which hold even on a loopback bind:

1. **The `Host` header is checked** (`TrustedHostMiddleware`, allowing the loopback names
   plus whatever `--host` bound to). Without this, "bound to loopback" would not mean what a
   reader thinks: any website could resolve a hostname of its own to `127.0.0.1` and read
   this API through the browser of anyone who visited, because the browser would consider it
   same-origin with that site. DNS rebinding is the one route by which a loopback bind leaks
   to the internet, and it is the attack a local-only dashboard most needs to think about. A
   test asserts a request addressed to `evil.example` gets a 400.
2. **No CORS headers.** A page on another origin may send a request; it may not read the
   reply.
3. **`profile` takes a name, never a path.** `--profile` on the command line happily takes a
   file path, because whoever types it already owns the filesystem. Over HTTP that would be
   an arbitrary-file-read-shaped primitive, so the query parameter refuses anything
   containing `/`, `\` or `..`, or ending `.json`, and points at `GET /api/v1/profiles` for
   the names. Tested.

A fourth, smaller one: `app.js` builds the page with `textContent` and never `innerHTML`, so
a model id, a note or a file path off somebody's disk cannot become markup. Tested by
grepping the file for `innerHTML`, `insertAdjacentHTML` and `eval`.

### 3. How much JavaScript is there, and would a contributor be able to change it?

**875 lines**, of which 77 are comments, in 43 top-level functions — about 18 lines each.
No dependencies, no build, no bundler, no framework, no transpiler. The whole page is
`index.html` (175 lines, entirely structural), `styles.css` (340) and `app.js`. A test fails
if the three together pass 1,500 lines, with the message "split it or cut it"; they are at
1,390, so the next person to add a panel has to make room rather than pile on.

Whether a contributor could change it: I think yes, and here is the shape of the argument
rather than an assertion. The file is four flat sections with rules that do not vary —
words and numbers, DOM and transport, tables, panels. There is one global (`ui`, the payload
from the server) and one mutable object (`asked`, the current request). There are exactly
three primitives to learn: `el(tag, attrs, children)` builds an element, `fill(target,
children)` replaces its contents, and `table(columns, rows)` builds a table from two plain
lists. Every panel is then a function that maps a JSON document onto those three. Adding a
column to the board is one entry in `BOARD_COLUMNS` and one line in `boardCells`. Adding a
panel is a `<section>`, a `<button data-panel>`, and a render function.

The two things a newcomer would trip over, and which are commented in the file:

- **You cannot type a word into it.** Every string is `t("key")`, and the key has to exist
  in `web/strings.py` or the test that scrapes `app.js` and `index.html` for keys fails.
  That is the price of the page being translated, and it is the right price, but it is the
  first surprise.
- **There is no test for the JavaScript itself,** because there is no Node and there is not
  going to be one. What I could check from Python, I did: every string key, column heading
  and label table the page asks for is served; the byte-unit ladder matches `units.py`; the
  assets carry the licence notice; none of them reaches a host. A logic error inside a
  render function would not be caught by any of it. That is the honest cost of the no-build
  rule, and it is why I drove the real page in a browser before committing.

## The licence notice for HTML, CSS and JavaScript

`tests/unit/test_licensing.py` walks `*.py` and would never see these files, so I made the
equivalent explicit: **the same three lines every Python module carries, in each language's
own comment syntax** — an HTML comment at the top of `index.html`, a `/* */` block at the
top of `styles.css`, `//` lines at the top of `app.js`. `tests/unit/test_web_static.py`
checks all three, the way the licensing test checks the modules. The reason is the licensing
test's own: a file that loses its notice is a file somebody will later copy out of the
project and believe is unencumbered, and a single-file HTML page is more likely to be copied
out than a Python module, not less. I did not extend `test_licensing.py` itself, to stay out
of a file five other agents may be touching; the check lives with the assets it is about.

The AGPL's section 13 is answered in the page footer: a visible link to the source, next to
a line saying the dashboard is served from your own machine and sends nothing anywhere.
`docs/web.md` says in as many words that somebody serving a modified version should point
that link at *their* source.

## Concerns and what I left undone

1. **The explanation is missing four sentences.** `--explain` prints the score table and
   then four sentences saying what each part was measured against ("18 tokens per second
   against the 15 this use case asks for"). Those live in `render_board.py` as private
   functions that call into `scoring` to recompute a target and a utilisation. Importing a
   private name across modules is poor, and reimplementing them in JavaScript is exactly
   what the brief forbids, so the page shows the score table with its parts, weights and
   "Adds" column, the speed breakdown, the budget and the ladder — the bulk of `--explain`,
   but not all of it. The clean fix is to make those four sentence builders public in
   `render_board.py`, which is another agent's file this week.
2. **Right-to-left has not been read by anyone who reads right to left.** The direction is
   set, the CSS is logical-property throughout, and the layout mirrors; but the terminal
   applies explicit bidi isolation to identifiers inside sentences and the page leaves that
   to the browser. A path or a flag inside an Arabic sentence may render with its
   punctuation in the wrong place. `for_display`/`isolate` insert control characters that
   would work in HTML too; I did not add them because I could not check the result.
3. **61 new messages for 37 languages.** Reusing the terminal's literals kept it to 61
   rather than 105, but it is still 61 entries every translator now owes, and the
   catalog-completeness test will warn about them for every language until somebody does.
4. **`sort` reorders without renumbering.** A caller who asks for `sort=speed` gets rows in
   speed order, each still carrying the rank the composite score gave it. I think that is
   the honest reading — the ranking is section 11's, not the interface's — but it is a
   decision somebody could disagree with, and it is documented in `docs/web.md`.
5. **Parameters in `docs/web.md` that I did not implement.** The pre-implementation draft
   listed `min_fit`, `include_does_not_fit`, `max_context` and `sort=date`. `min_fit` and
   `perfect` belong to `build_fit_board`, which is `llamafit fit`'s service and not the
   board's; `max_context` is a global CLI flag that does not exist yet; `date` is not on a
   board row. Rather than implement approximations of them in the interface I removed them
   from the table, and any of them arriving in a query string now gets a 400 naming it. The
   surface doc now describes what ships.
6. **`web/api.py` imports `llamafit.cli.app` for its side effect.** `cli/common.py` reads
   `CliState` out of `cli/app.py`, whose last act is to import every command module, each of
   which reads `cli/common.py` back. Entered from the command line the cycle resolves;
   entered from `web/api.py` — which is what a test does — `common` is half-built when a
   command asks it for a name. Loading the Typer application first makes the two entrances
   equivalent. It works and it is commented, but it is a symptom of `common` depending on
   `app`, which is worth untangling at some point.
7. **`POST /api/v1/scan` has no rate limit.** Harmless on loopback; a nuisance on a server
   somebody deliberately exposed. I did not add one because adding a throttle to a
   single-user tool is a feature nobody asked for, but it is the one endpoint that costs
   real work per call.
8. **The release workflow's packaged-data check does not know about `web/static/`.** It
   asserts the wheel carries the catalog, profiles, schema, translations and GPU table. The
   static files are read through `packaged_dir`, so a wheel without them gives a clear
   "reinstall LlamaFit" error rather than a stack trace — but the check that would catch it
   at build time needs one more entry in `.github/workflows/release.yml`, and that file is
   outside this package.
