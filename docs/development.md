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
pytest --cov -m "not hardware and not network"
llamafit catalog validate            # local: the bundled catalog and your custom models file
```

`.github/workflows/ci.yml` runs the first three on Ubuntu, macOS and Windows with Python 3.10
and 3.13. Tests marked `hardware` need a real machine (they measure bandwidth or run vendor
tools) and tests marked `network` reach the internet; CI skips both, so run them locally with
`pytest -m hardware` and `pytest -m network`. Coverage must stay at or above 85 percent.

On Ubuntu with Python 3.13 the same workflow adds one more step: it regenerates the catalog
JSON schema and `MODELS.md` and fails if either differs from what is committed. When it does,
run `python scripts/gen_schema.py` and `python scripts/gen_models_md.py` and commit what they
write. `llamafit catalog validate` is not a CI step of its own; the test suite loads the
bundled catalog and fails on any problem it finds.

`.github/workflows/catalog.yml` is the other workflow. Once a week it runs `llamafit catalog
refresh --check`, which exits non-zero when a repository has published something the committed
facts files do not record. It needs the network, so it never runs on a pull request: a flaky
connection must not fail somebody's unrelated change. It opens no issue and no pull request —
a red run is the notification, and a `catalog refresh` is the answer.

`.github/dependabot.yml` is not a workflow but belongs with them. `.github/workflows/release.yml`
pins every action it uses to a commit rather than to a moving tag, because those actions handle
the token that publishes under this project's name on PyPI and the bytes that go with it; the
file's header comment explains where that line is drawn and why. An exact pin gives up automatic
updates, so Dependabot opens a weekly pull request when a pinned commit moves and rewrites the
version in the trailing comment. Read the release notes for what changed, let CI run on it, and
merge — but never resolve one of those pull requests by replacing a pin with a moving tag.

## How detection is tested without hardware

Every external command runs through `llamafit.hardware.runner.Runner`. Tests pass a
`FakeRunner` whose responses are recorded outputs, keyed by program name or full command line.
The reference machine (RTX 4060 8 GB, i9-14900KF, 128 GB DDR5, Windows 11) lives in
`tests/fixtures/reference_machine.py`.

To add a machine:

1. Run each command from the probe table in [platform-support.md](platform-support.md) and
   save its exact output.
2. Add a fixture module with a `runner()` function returning a `FakeRunner`, plus `cpuinfo()`,
   `vm()` and `cores()` returning the recorded values.
3. Add a test in `tests/unit/test_scan.py` asserting the `Host` LlamaFit builds from it. Pass
   every provider, `cores_provider=cores` included: a scan that reads the running machine's
   core counts asserts numbers that hold only on the machine the fixture was recorded on.

`scripts/record_fixtures.py` (phase 1A) runs the commands and writes the module for you,
`cores()` included.

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
| `pydantic` | every data model, validation, JSON in and out |
| `platformdirs` | the right directories on each OS |
| `psutil` | memory totals and core counts everywhere |
| `py-cpuinfo` | CPU model and instruction sets everywhere |
| `httpx` | Hugging Face metadata, running-server discovery, downloads |
| `pyyaml` | the catalog: reading the curated `<family>.yaml` files, and printing one entry back with `catalog show --yaml` |
| `numpy` | a faster, more accurate memory-bandwidth measurement; optional extra `fast`, and part of `dev` so CI exercises it |
| `textual` | the terminal dashboard `llamafit` opens with no arguments; a hard dependency and not an extra, because the dashboard is the first thing a newcomer meets and an interface that may or may not be installed cannot be that |

Planned, not installed and not yet declared in `pyproject.toml`; each arrives with the phase
that needs it, together with the extra it belongs to:

| Package | Why | Phase |
|---|---|---|
| `fastapi`, `uvicorn` | the JSON API and static dashboard | 1D |

Adding a dependency means adding a row here with a reason, in the same pull request.

## Messages and translations

Messages to the user go through a small gettext layer in `src/llamafit/i18n/`, with GNU
`.po` catalogs under `src/llamafit/data/locale/` and `scripts/gen_messages.py` to regenerate
the template. [translations.md](translations.md) documents the format, how a language is
chosen, and what a translator needs to know.

The machinery is in place; no interface message goes through it yet, so `llamafit` is English
everywhere today. Read that page before you write a message that will one day be translated:
anything built at import time needs `lazy_gettext` rather than `_()`, and the extractor
refuses a message that is not a literal string.

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

`.github/workflows/release.yml` does the release. A person moves a changelog section, bumps a
number and pushes a tag; nothing else is done by hand, and nothing is uploaded from a laptop.

### What a person does

1. Move the *Unreleased* section of `CHANGELOG.md` under a `## [X.Y.Z] - YYYY-MM-DD` heading
   and add its link reference at the foot of the file.
2. Bump `version` in `pyproject.toml` to the same number.
3. Commit both and land them on `main` through a pull request as usual.
4. Tag that commit and push the tag:

```
git tag -a v0.1.0 -m "llamafit 0.1.0"
git push origin v0.1.0
```

Versions follow semantic versioning. Before 1.0, a minor version may change interfaces and the
changelog says so. The workflow watches tags shaped `v<digits>.<digits>.<digits>` with anything
after them, so `v0.1.0a1` and `v1.0.0rc1` release too, and a pre-release version is marked as a
pre-release on GitHub.

### What the workflow does

1. Runs the pull-request checks — it *calls* `ci.yml` rather than copying its steps, so a
   release never tests less than a pull request does.
2. Refuses to go on unless the tag and the `version` in `pyproject.toml` are the same version.
   Publishing 0.2.0 from a tag that says 0.1.0 cannot be undone: a version number on PyPI can
   never be reused, not even after deleting the file.
3. Builds a wheel and a source distribution and runs `twine check --strict` on both.
4. Checks that the wheel carries every packaged data directory — the catalog, the generated
   facts, the JSON schema, the GPU table and the translations — and that it carries no test,
   script or document. Each of those is read through `importlib.resources`, so a wheel missing
   one installs perfectly and fails only when a user reaches the command that needs it.
5. Installs that wheel into a clean virtual environment outside the checkout, with no extras,
   and runs `llamafit list`, `llamafit info`, `llamafit catalog validate` and a non-English
   language resolution from it. The test suite runs against the source tree and cannot see a
   packaging mistake; this step can.
6. Reads the tag's section out of `CHANGELOG.md` with `scripts/changelog_section.py` *before*
   anything is uploaded, so a tag with no changelog section fails while failing is still free.
7. Publishes to PyPI, then creates the GitHub release from the tag with that changelog section
   as its notes and both files attached.

There is no API token and no secret in the repository. PyPI verifies the workflow's identity
through GitHub with OpenID Connect — *trusted publishing* — which is why the publishing jobs
ask for the `id-token: write` permission and name a deployment environment. PyPI's side of the
configuration names this repository, this workflow file and that environment, and refuses a
token minted for anything else. The environments are `pypi` and `testpypi`.

### Rehearsing on TestPyPI

PyPI has a separate test instance, and the same workflow publishes there when it is started by
hand instead of by a tag: **Actions → Release → Run workflow**. It runs the same checks, the
same build and the same verification, and makes no GitHub release. Uploading a version TestPyPI
already has is skipped rather than treated as a failure, so a rehearsal can be repeated; the
first rehearsal of a given version is the one that proves the upload.

Install the rehearsed build the way a user would, from a directory that is not the checkout:

```
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ llamafit
```

The extra index is needed because the dependencies live on the real PyPI, not on TestPyPI.
