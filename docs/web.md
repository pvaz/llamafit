# The web dashboard and the JSON API

`llamafit serve` starts a small web server on your machine. It serves a dashboard for the
browser and a JSON API that the dashboard, scripts and other tools can call.

```
llamafit serve                     # http://127.0.0.1:8765
llamafit serve --port 9000 --open  # another port, open the browser
```

The server is an optional extra, because somebody who only ever uses the command line
should not have to install one:

```
pip install "llamafit[web]"
```

Without it, `llamafit serve` says so and gives you that line; every other command works as
before.

## Safety

- **It binds to `127.0.0.1`.** Not by default — by refusal. `llamafit.web.serve()` will not
  bind to anything else unless the caller says separately that it means to, and the command
  line says that only when you actually typed `--host`. So the address cannot drift outwards
  through a config file or a default nobody re-read.
- **Passing `--host` with another address prints what it exposes and then does it.** There is
  no authentication and no password: the server reads your hardware, your disks and your file
  paths, and anyone who can reach that port can read all of it. It is your decision, made in
  front of a warning that says what the decision is.
- **The `Host` header is checked.** Requests are answered only when they are addressed to a
  loopback name, or to the address `--host` bound to. Without that check any website in the
  world could point a hostname of its own at `127.0.0.1` and read this API through the browser
  of anyone who visited it — DNS rebinding, which is the one way a loopback bind leaks.
- **No CORS headers.** A page on another origin may send a request and may not read the reply.
- **A profile is named, never pathed.** `--profile` on the command line takes a file path,
  because whoever types it already owns the filesystem. Over HTTP the `profile` parameter takes
  a profile's *name* and refuses anything shaped like a path, so a request cannot ask the
  server to open a file.
- **An unknown query parameter is an error.** `use-case` for `use_case` gets a 400 naming it,
  not a confident answer to a question you did not ask.
- **Nothing is sent anywhere.** The page fetches no font, no script and no stylesheet from the
  internet; it works with the network unplugged. The only outbound requests are the ones the
  command line would make for the same operation (Hugging Face metadata, downloads).

## The dashboard

The same five panels as the [terminal dashboard](tui.md), laid out for a browser: **Board**
with the ranked table, each row expanding into the score, the speed, the memory budget and
the context ladder behind it; **Needs** as a form that re-ranks the board; **Host** with the
machine, the llama.cpp installation and every finding `doctor` reports; **Plan** with the
budget, the tiers and a copyable command line; **Simulate** with a hardware profile or an
override of VRAM, RAM or cores, and a `SIMULATED` badge everywhere while one is in force.
Phase 2 adds Downloads with progress; phase 3 adds Benchmarks.

It is plain HTML, CSS and JavaScript served from the package. There is no build step, no
package manager and no framework: it calls the API below and renders the JSON.

Nothing has been benchmarked on any machine yet, so every speed on the page is a computed
estimate. The page says so, once, under the header, in the same sentence the command line
prints under the same table.

### The page's own language

The dashboard speaks the language LlamaFit was started in — `--language`, then
`LLAMAFIT_LANGUAGE`, then the system locale, exactly as the command line chooses it. It does
not read `Accept-Language`, and it is one language per process: this is a server for the
person at the keyboard, not a multi-user site.

The page holds no words of its own. `GET /api/v1/ui` returns every label it draws, built on
the server by the same gettext calls the rest of LlamaFit uses, so a translator who fills in
a `.po` file translates the dashboard without knowing it exists — and a message they have
not reached yet falls back to English, message by message, the way it does anywhere else.
The words for verdicts, run modes, confidence labels, memory pools and budget components are
the terminal's own, imported rather than rewritten, so the two interfaces cannot call the
same thing by different names. [translations.md](translations.md) is the whole story.

## The API

All responses are JSON documents of the same models the CLI prints with `--json` — the
identical serialisation call, aliases included — so the two never disagree. Errors return
`{"error": {"message": ..., "hint": ..., "command": ...}}` with status 400 (user input), 404
(unknown model or profile) or 503 (environment problem: a probe that could not run, llama.cpp
missing, an installation missing its own data).

### Phase 1D

| Method and path | Returns |
|---|---|
| `GET /health` | `{"status": "ok", "version": "..."}` |
| `GET /api/v1/ui` | the language, the reading direction, the number punctuation and every word the page draws |
| `GET /api/v1/system` | `SystemReport`: host and llama.cpp status from the cached scan |
| `POST /api/v1/scan` | rescans and returns the new `SystemReport` |
| `GET /api/v1/doctor` | `Diagnosis` |
| `GET /api/v1/models` | `Board`: every quantisation of every model, planned and scored |
| `GET /api/v1/models/top` | `Board`: the best quant per model, ranked |
| `GET /api/v1/models/{id}` | `ModelDetail`: one model with its quants and the facts read from their headers |
| `POST /api/v1/plan` | body `{"model": id, "quant": name?, "context": n?, "ub": n?, "vision": bool?, "target_tps": n?, "profile": name?, "memory": size?, "ram": size?, "cpu_cores": n?}` → `PlanReport` |
| `GET /api/v1/profiles` | bundled and user hardware profiles, in the shape `hardware list --json` prints |
| `GET /api/v1/catalog/schema` | the JSON schema of catalog entries |
| `GET /api/v1/catalog/problems` | what was wrong with any catalog file, with the sentence to show |

`GET /api/v1/docs` is FastAPI's own interactive documentation for the same surface.

Query parameters for `/api/v1/models` and `/api/v1/models/top`. They mirror the options
`llamafit recommend` takes, and a value one of them refuses is refused here in the same
words:

| Parameter | Values |
|---|---|
| `use_case` | `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding` |
| `require` | capability names, repeatable |
| `prefer` | `balanced`, `quality`, `speed` |
| `license` | SPDX identifiers, repeatable |
| `min_context` | integer, tokens |
| `max_download` | a size such as `40G` or `7.5GiB`; a bare number is bytes |
| `all_quants` | `true` or `false` (`/api/v1/models` is always every quant) |
| `vision` | `false` to plan without a vision projector |
| `limit` | integer; how many ranked rows to keep |
| `sort` | `score` (default), `speed`, `quality`, `context`, `size`. It reorders rows the board already produced; each row keeps the rank the composite score gave it |
| `profile` | a bundled or user profile's **name** |
| `memory`, `ram` | sizes to pretend the card and the machine have |
| `cpu_cores` | physical cores to pretend the processor has |

A candidate that did not qualify is never dropped: it comes back in `excluded` with the
reason, exactly as `recommend` prints it.

### Phase 2

| Method and path | Effect |
|---|---|
| `POST /api/v1/install/llamacpp` | start an installation; returns a job id |
| `POST /api/v1/install/model` | start a download; returns a job id |
| `GET /api/v1/jobs/{id}` | job state and progress |
| `GET /api/v1/jobs/{id}/events` | server-sent events with progress until the job ends |
| `POST /api/v1/preset` | write launch scripts for a plan; returns the paths |
| `POST /api/v1/launch`, `POST /api/v1/launch/{id}/stop` | start and stop a server LlamaFit manages |

### Phase 3

| Method and path | Effect |
|---|---|
| `POST /api/v1/bench` | start a benchmark job for a model, quant and context |
| `GET /api/v1/bench/results` | stored results with estimated versus measured |
| `GET /api/v1/bench/calibration` | the calibration factors in use for this host |

## Using the API from a script

```python
import httpx

api = "http://127.0.0.1:8765/api/v1"
board = httpx.get(f"{api}/models/top", params={"use_case": "coding", "min_context": 32768}).json()
for row in board["rows"][:3]:
    print(row["model_id"], row["quant"], row["candidate"]["speed"]["gen_tps"])
```

## The licence

LlamaFit is AGPL-3.0-or-later, and section 13 is the reason it is that rather than the GPL:
somebody running a modified version as something people reach over a network has to offer
them its source. The dashboard's footer links to it, which is where that offer belongs — and
if you modify LlamaFit and serve it, point that link at *your* source.
