# Project review and CI recovery — 2026-09-14

## Scope and baseline

Reviewed the local checkout at `2fcace0a00f8f1f0dc0d5f8e36412e227e7a3fea`,
the GitHub workflows and recent runs, application architecture, tests, packaging,
and the CLI, terminal and browser interfaces. The checkout was clean at the start.
Changes described below are local until a new commit is published and checked by CI.

## Why CI failed

The [failed main run](https://github.com/pvaz/llamafit/actions/runs/34705527810)
failed only on Windows with Python 3.10. The other five jobs passed. The failing test was
`test_the_back_button_does_what_the_back_key_does` in `test_tui_screens.py`:
`textual.css.query.NoMatches: No nodes match '#tabs' on Screen(id='_default')`.
That job reported 3,256 passing tests, one failure, two skips and 96.73% coverage.
Its test step took 51 minutes and 50 seconds.

The exception occurred while leaving `app.run_test()`. Textual marks the application
as stopped and removes widgets before closing its message queue. A pending tab activation
then reached `LlamaFitApp.on_tabbed_content_tab_activated`, which tried to redraw widgets
that had already been removed. The issue was an application lifecycle race, not a lint,
dependency installation or coverage failure.

The [release run for the same commit](https://github.com/pvaz/llamafit/actions/runs/34706190911)
passed, including its reusable CI checks. This explains why the failure could disappear
on another execution without the underlying race being corrected.

The fix stops handling tab activation once `is_running` is false. A regression test
delivers a real tab activation after removal of the tab widget during shutdown. It failed
with the same exception before the fix and passed afterwards. No retry, arbitrary sleep
or suppression of `NoMatches` was added.

## Additional corrections

- Commands displayed for copying now have a shared shell renderer. The CLI, TUI and web
  page use the same text; the JSON response retains its original `command` argument list
  and adds `command_text` and `command_shell`. This addresses paths containing spaces,
  apostrophes and shell metacharacters. Preset scripts keep their existing renderers.
- Explicit loopback aliases such as `127.0.1.5` are included in the web server's trusted
  hosts. Previously the socket accepted the address but the HTTP middleware rejected it.
  An IPv4 wildcard bind now accepts a finite list of local interface addresses, while
  unrelated Host headers remain rejected.
- The range download transport sends Hugging Face credentials only to the exact HTTPS
  Hugging Face origin. Tests cover non-HF URLs supplied to the lower-level reader,
  lookalike domains, nonstandard ports, userinfo and redirects. Normal catalog downloads
  already construct Hugging Face URLs; the defect was the transport's broader URL input.
  No real credentials were used in the regression tests and no actual leak was observed.
- POSIX PATH persistence quotes the installation directory before writing a shell profile,
  preventing command substitution and preserving spaces and apostrophes in paths.
- Progress-rendering tests now specify console height as well as width. This makes their
  assertions independent of the `TERM=dumb` environment used by headless test workers.
- The roadmap now records standalone binaries as shipped since 0.1.1.
- The local development environment's old pip was upgraded after dependency auditing
  identified advisories. This is an environment repair, not a runtime dependency change.

## Validation

The final run used Windows and Python 3.13.1 with stable source files:
**3,301 passed, two skipped, zero failures; 96.71% coverage**, above the repository's
85% gate. The suite finished in 297.69 seconds with eight local test workers.
This duration is not directly comparable to the serial GitHub runner.

```powershell
python -m pytest -n 8 --cov --cov-report=xml --durations=15 `
  -m 'not hardware and not network' -p no:cacheprovider `
  --basetemp=.superpowers/pytest-stable-20260914 `
  --junitxml=.superpowers/test-results-final.xml
```

`pytest-xdist` was installed only in the local development environment; CI configuration
and project dependencies were not changed. The run emitted 16 dependency deprecation
warnings from the Starlette test client and AnyIO, with no application test failures.
The slowest test was the full-catalog sort contract check (120.83 seconds).

- Ruff lint and formatting: passed (330 files checked for formatting).
- Mypy: passed (166 source files).
- Generated catalog/hardware schemas, model documentation and 1,153-message translation
  template: current.
- Wheel and source distribution: built successfully; `twine check --strict` passed.
- Dependency audit: 88 packages checked, no known vulnerabilities reported;
  `pip check` found no broken requirements.
- Command quoting: native PowerShell 5.1 argument round trip passed. POSIX profile
  execution through Git's `sh` preserved literal paths without running substitutions;
  applying the same apostrophe-containing path twice did not duplicate the entry.

The deterministic lifecycle regression and the original TUI screen tests passed together.
Browser checks used a local server with the project's recorded machine fixture and its
bundled catalog: initial load, model filtering, row details, plan generation and hardware
simulation, plus the displayed shell-specific command and successful clipboard copying.
This exercises the actual JavaScript and HTTP interface without downloading
models or launching an inference server.

Wheel smoke checks run outside the checkout, from a fresh environment without the web
extra: version, JSON model list, catalog validation, hardware profile validation, JSON
model information and Portuguese output. Imports are asserted to come from the installed
wheel's `site-packages`, not the editable checkout.

## Priorities for the next iteration

| Priority | Work | Why it matters | Acceptance evidence |
| --- | --- | --- | --- |
| P1 | Add browser E2E checks to CI | Static source checks and FastAPI TestClient do not execute the dashboard's JavaScript. | Load, filter, plan, simulate, reset and error recovery in a real browser, with no console errors. |
| P1 | Apply stored calibration to estimates | Calibration is saved but recommendations still use the shipped constants. This is the main gap between measuring and improving the answer. | Match calibration to hardware/backend/build; show provenance; demonstrate holdout prediction accuracy and fallback when no compatible fit exists. |
| P1 | Prove install → launch → bench on clean machines | Binary packaging checks are useful but do not prove the complete user journey. The roadmap explicitly says this proof is outstanding. | Recorded Windows, Linux and macOS runs, a small permitted test model, checksums, health check, benchmark and cleanup. |
| P1 | Complete IPv6 web bind-address handling | IPv4 wildcard and loopback alias handling are corrected; IPv6 needs explicit middleware and browser coverage. | Requests using configured IPv6 addresses pass; unrelated Host headers still fail. Never solve this by accepting every hostname. |
| P2 | Reduce expensive repeated fixture work | Many rendering and TUI tests rebuild the full catalog and ranking. The failed CI job took almost 52 minutes. | Profile slow tests; use small catalogs for interface-only assertions; retain full-catalog integration tests and coverage. Compare measured CI durations. |
| P2 | Test CLI/API parity for non-default options | The default plan is covered, but boundary values can differ; CLI accepts target TPS zero while the API requires a positive value. | A shared table of context, batch, target speed, vision and simulation cases produces equivalent answers or equivalent rejections. |
| P2 | Add continuous dependency security auditing | Dependabot update PRs exist, but the alerts API reported that vulnerability alerts are disabled. | Enable alerts in repository settings and add an audited dependency check with a documented response process. |
| P2 | Separate interface parsing from services | `web/api.py` imports CLI helpers and explicitly loads the CLI application to work around an import cycle. | Shared validation and selection live below the interfaces; CLI and API contract tests remain green. |
| P2 | Bound expensive web operations | The intentionally exposed LAN mode has no authentication; repeated scan requests can consume local resources. | Define same-origin protections and scan concurrency/rate limits without breaking local CLI clients. |
| P3 | Improve recovery and accessibility coverage | The dashboard works, but keyboard access, narrow screens and failed refreshes need browser assertions. | Keyboard-only main journey, labelled controls, meaningful error states and responsive checks. |

The open [Actions update PR #8](https://github.com/pvaz/llamafit/pull/8) passed all six CI
jobs when reviewed. Review and merge it through the normal protected-branch process.
The main branch is protected; no branch settings, credentials or releases were changed.

## Limits of the conclusion

Passing automated checks does not establish prediction accuracy on every machine or prove
all supported hardware. Network/hardware-marked tests and a real model install/inference
journey require separate runs. New code must still pass the remote matrix after publication;
green runs on the previous commit do not validate local changes.

Windows PowerShell 5.1 still has native argument-passing limitations for empty arguments
and embedded double quotes. The copied-command regression covers actual planned arguments,
paths and tensor patterns; scripts and programmatic clients should retain the argv list or
the existing preset renderers for more general command construction.
