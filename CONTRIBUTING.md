# Contributing to LlamaFit

Thank you for helping people run open models on their own machines. This page says what kind
of help is most useful, how the project is organised, and what a pull request needs to be
merged quickly.

## Ways to help

| Contribution | Where | What makes it mergeable |
|---|---|---|
| A model that is missing from the catalog, or a wrong number | `src/llamafit/data/catalog/` | a primary source for every fact; `llamafit catalog validate` passes |
| Your machine's probe outputs, so the scan is tested on hardware the maintainers do not own | `tests/fixtures/` | recorded exactly as `docs/platform-support.md` lists the commands |
| A GPU that the specification table does not know | `src/llamafit/data/gpus.json` | vendor-published bandwidth and FP16 figures with a link in the pull request |
| A benchmark on your hardware (phase 3) | `llamafit bench --share` when it exists | run with the preset LlamaFit generated, not hand-edited flags |
| Bug reports | issues | the output of `llamafit --json doctor` |
| Documentation | `docs/` | matches the code; the docs tests enforce command and probe names |
| Code | `src/`, `tests/` | see below |

## Before you start on code

Open an issue first for anything larger than a fix, and say you are working on it. The
design specification in `docs/superpowers/specs/` explains why things are the way they are;
if your change disagrees with it, the specification is what to discuss, not the code.

Work happens in phases with written plans in `docs/superpowers/plans/`. Each plan is a list of
small tasks with tests; picking a task from the current plan is the easiest way in.

## Setting up

```
git clone https://github.com/pvaz/llamafit.git
cd llamafit
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
source .venv/bin/activate           # macOS, Linux
pip install -e ".[dev,fast]"
pytest
```

More in [`docs/development.md`](docs/development.md).

## What a pull request needs

1. **Green checks.** `ruff check . && ruff format --check . && mypy && pytest` locally, and
   the CI matrix (Ubuntu, macOS, Windows; Python 3.10 and 3.13) on the pull request.
2. **Tests for behaviour you change.** Detection code is tested with recorded command output,
   never by running vendor tools in CI. Estimator changes are tested against the golden host
   profiles and the calibration measurements.
3. **Documentation in the same pull request.** A new flag appears in `docs/cli.md`; a new probe
   appears in `docs/platform-support.md`; a user-visible change gets a line under
   *Unreleased* in `CHANGELOG.md`.
4. **One topic.** A catalog fix and a refactor are two pull requests.
5. **Sources.** Any number in data files or documentation has a link. "I remember" is not a
   source.

## Style

Most of it is enforced by ruff and mypy in strict mode. Beyond that:

- Docstrings on every public function, Google style, saying what the function is for and what
  it returns, not restating the signature.
- No bare `except`. Detection code catches broadly on purpose, and says so in a comment.
- No `print` outside the interface packages (`cli`, `tui`, `web`). Services return data.
- No number without a source label in data structures that reach the user.
- Small files with one responsibility. If a module grows past a few hundred lines, split it.
- Sentences in documentation and messages are short and say who does what. The user reads
  hints when something has just failed; make them actionable.

## Commit messages

Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`, with a scope
when it helps (`feat(catalog): ...`). The body says why. A pull request is squashed on merge
unless its commits are individually meaningful.

## Reviews

Maintainers try to answer within a few days. A review asks for changes when something is
unclear or untested, never as a matter of taste when the style tools are satisfied. If a
review seems wrong, say so; the design specification is the tie-breaker.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Be kind, be specific,
assume good faith.
