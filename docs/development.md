# Development

## Set up

```
git clone https://github.com/pvaz/llamafit.git
cd llamafit
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate           # macOS, Linux
pip install -e ".[dev,fast]"
pytest
```

Python 3.10 or newer. The `dev` extra brings pytest, coverage, ruff and mypy; `fast` brings
NumPy for the bandwidth measurement.

## The checks CI runs

```
ruff check . && ruff format --check .
mypy
pytest --cov -m "not hardware"
llamafit catalog validate            # once the catalog exists (phase 1B)
```

CI runs them on Ubuntu, macOS and Windows with Python 3.10 and 3.13. Tests marked
`hardware` need a real machine (they measure bandwidth or run vendor tools); run them locally
with `pytest -m hardware`. Coverage must stay at or above 85 percent.

## How detection is tested without hardware

Every external command runs through `llamafit.hardware.runner.Runner`. Tests pass a
`FakeRunner` whose responses are recorded outputs, keyed by program name or full command line.
The reference machine (RTX 4060 8 GB, i9-14900KF, 128 GB DDR5, Windows 11) lives in
`tests/fixtures/reference_machine.py`.

To add a machine:

1. Run each command from the probe table in [platform-support.md](platform-support.md) and
   save its exact output.
2. Add a fixture module with a `runner()` function returning a `FakeRunner`, plus the
   `cpuinfo` and virtual-memory values.
3. Add a test in `tests/unit/test_scan.py` asserting the `Host` LlamaFit builds from it.

`scripts/record_fixtures.py` (phase 1A) runs the commands and writes the module for you.

## Layout

`src/llamafit/` follows [architecture.md](architecture.md): core modules are pure and typed,
services compose them, interfaces render. Anything that touches the OS goes behind a small
protocol (`Runner`, `HttpClient`) so it can be faked.

Files stay small and single-purpose. When one grows past a few hundred lines, split it.

## Dependencies

Runtime dependencies, with the reason each earns its place:

| Package | Why |
|---|---|
| `typer` | the CLI, with completion and help for free |
| `rich` | tables and colour in the terminal; Typer uses it too |
| `textual` | the terminal dashboard (optional extra `tui`) |
| `fastapi`, `uvicorn` | the JSON API and static dashboard (optional extra `web`) |
| `pydantic` | every data model, validation, JSON in and out |
| `pyyaml` | the catalog |
| `platformdirs` | the right directories on each OS |
| `psutil` | memory totals and core counts everywhere |
| `py-cpuinfo` | CPU model and instruction sets everywhere |
| `httpx` | Hugging Face metadata, running-server discovery, downloads |
| `numpy` (optional) | a faster, more accurate memory-bandwidth measurement |

Adding a dependency means adding a row here with a reason, in the same pull request.

## Style

Enforced: ruff (line length 100, import order, pydocstyle Google convention), mypy strict.
Conventions beyond the tools:

- Public functions have docstrings that say what the function is for and what it returns.
- Detection code catches broadly and says so in a comment; everything else catches
  specifically.
- No `print` outside `cli/`, `tui/`, `web/`.
- No number reaches the user without a source label.
- Messages to the user are short sentences with a subject and a verb, and hints are actions.

## Commits and branches

Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`), the
reason in the body. Work happens on branches and lands on `main` through pull requests once
CI is green. `main` is always releasable.

## Releasing

1. Move the *Unreleased* section of `CHANGELOG.md` under the new version and date.
2. Bump `version` in `pyproject.toml`.
3. Tag `vX.Y.Z` and push the tag; the release workflow builds the wheel and source
   distribution and publishes them to PyPI through trusted publishing.
4. Create the GitHub release from the tag with the changelog section as its notes.

Versions follow semantic versioning. Before 1.0, a minor version may change interfaces and the
changelog says so.
