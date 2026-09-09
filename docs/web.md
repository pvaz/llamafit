# The web dashboard and the JSON API

`llamafit serve` starts a small web server on your machine. It serves a dashboard for the
browser and a JSON API that the dashboard, scripts and other tools can call. Both are part of
phase 1D; the API surface is fixed here so it can be reviewed before it is built.

```
llamafit serve                     # http://127.0.0.1:8765
llamafit serve --port 9000 --open  # another port, open the browser
```

## Safety

- The server binds to `127.0.0.1`. Passing `--host` with another address prints a warning and
  is your decision; there is no authentication, because the server is meant for the person
  sitting at the machine.
- The API reads state and starts the same operations the CLI offers. It cannot run arbitrary
  commands, download to arbitrary paths, or change files outside LlamaFit's directories.
- Nothing is sent anywhere. The only outbound requests are the ones the CLI would make for the
  same operation (Hugging Face metadata, downloads).

## The dashboard

The same five panels as the [terminal dashboard](tui.md), laid out for a browser: Board with
the ranked table and a detail drawer, Needs as a form that re-ranks live, Host with the probe
list and findings, Plan with the budget, context tiers and copyable command line, Simulate
with overrides and profile choice. Phase 2 adds Downloads with progress; phase 3 adds
Benchmarks with estimate-versus-measured charts.

The dashboard is plain HTML, CSS and JavaScript served from the package. There is no build
step and no framework; it calls the API below and renders the JSON.

## The API

All responses are JSON documents of the same models the CLI prints with `--json`, so the two
never disagree. Errors return `{"error": {"message": ..., "hint": ..., "command": ...}}` with
status 400 (user input), 404 (unknown model or profile) or 503 (environment problem such as
llama.cpp missing for an operation that needs it).

### Phase 1D

| Method and path | Returns |
|---|---|
| `GET /health` | `{"status": "ok", "version": "..."}` |
| `GET /api/v1/system` | `SystemReport`: host and llama.cpp status from the cached scan |
| `POST /api/v1/scan` | rescans and returns the new `SystemReport` |
| `GET /api/v1/doctor` | `Diagnosis` |
| `GET /api/v1/models` | catalog entries with per-host scoring; query parameters below |
| `GET /api/v1/models/top` | the board: the best quant per model, ranked |
| `GET /api/v1/models/{id}` | one model with facts, quants and per-quant budgets |
| `POST /api/v1/plan` | body `{"model": id, "quant": name?, "context": n?, "ub": n?, "vision": bool?}` → `Plan` |
| `GET /api/v1/profiles` | bundled and user hardware profiles |
| `GET /api/v1/catalog/schema` | the JSON schema of catalog entries |

Query parameters for `/api/v1/models` and `/api/v1/models/top`:

| Parameter | Values |
|---|---|
| `use_case` | `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding` |
| `require`, `prefer` | capability names, repeatable |
| `min_context`, `max_download`, `max_context` | integers (bytes for `max_download`) |
| `license` | SPDX identifiers, repeatable |
| `min_fit` | `comfortable`, `fits`, `tight` |
| `include_does_not_fit` | `true` or `false` |
| `all_quants` | `true` or `false` |
| `sort` | `score`, `speed`, `quality`, `context`, `size`, `date` |
| `limit` | integer |
| `profile` | a profile name or path; `memory`, `ram`, `cpu_cores` for single overrides |

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
    print(row["model"]["id"], row["quant"], row["verdict"], row["speed"]["generation_tps"])
```
