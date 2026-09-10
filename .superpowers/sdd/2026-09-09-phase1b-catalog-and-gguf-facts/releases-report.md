# `llamafit install llama.cpp` — report

Branch `feat/releases`, off `101206f`. Two commits:

- `5ec736c` feat(llamacpp): install published llama.cpp releases
- `e78c55d` feat(cli): add llamafit install llama.cpp

Files: `src/llamafit/llamacpp/releases.py`, `src/llamafit/llamacpp/install.py`,
`src/llamafit/cli/install_cmd.py`, one registration line in `src/llamafit/cli/app.py`,
`tests/fixtures/llamacpp_release.py`, `tests/unit/test_llamacpp_releases.py`,
`tests/unit/test_llamacpp_install.py`, `tests/unit/test_cli_install.py`, plus `docs/cli.md`,
`CHANGELOG.md` and the regenerated `messages.pot`.

---

## 1. What does a user see before anything is written to their disk?

A table, then a question. Nothing else has happened by then: `plan_install` reads the release
listing and looks at the target directory, and writes not one byte. Below is a real
`llamafit install llama.cpp --dry-run` on the reference machine, pinned to English:

```
Release             b10892 (build 10892)
Backend             cuda
Archive             llama-b10892-bin-win-cuda-13.3-x64.zip
Also                cudart-llama-bin-win-cuda-13.3-x64.zip
Download            515.6 MiB
Checksum            published, checked before anything is unpacked
Install into        C:\Users\pvaz\.llamafit\llama.cpp\bin
Directory           will be created
Free space (cache)  352.8 GiB on C:\Users\pvaz\AppData\Local\llamafit\Cache\llamacpp, 515.6 MiB needed
Free space          352.8 GiB on C:\Users\pvaz\.llamafit\llama.cpp, 1.5 GiB needed
Install this? [Y/n]
```

Every row is there because a person might say no to it. The `Also` row is the CUDA runtime
archive, which is a second half-gigabyte they did not ask for and should not discover as a
surprise. The `Checksum` row goes yellow and reads *none published* when there is nothing to
verify against, and in that state the install refuses to proceed at all without
`--allow-unverified`. The `Directory` row is the ownership answer in one sentence — *will be
created*, *exists and is empty*, *holds b10880, installed by LlamaFit; it will be replaced*, or
the refusal — and it is red in the last case. Free space is shown per volume, because the
download cache and the install directory are not always the same disk, and the row that is short
goes red.

A resumed download says so in the `Download` row: `12.4 MiB (503.2 MiB already on disk)`.

`--dry-run` stops after the table. So does `--json` without `--yes`: a machine-readable run
should not be the one that stops at a prompt nobody is watching, so it prints the plan with
`"applied": false` and exits 0. Declining at the prompt prints *Nothing was written.* and exits 1.

Afterwards the report says where it landed, which backends were read back out of the directory,
and how to remove it — the undo, unasked.

## 2. How do you know a directory is yours, and what happens when it is not?

By a file LlamaFit itself wrote: `.llamafit-install.json` in the install root, whose `tool` field
reads `llamafit`. Nothing else counts. `inspect_target` returns one of four answers:

| Ownership | What was found | Install proceeds |
|---|---|---|
| `absent` | the directory does not exist | yes |
| `empty` | it exists and holds nothing | yes |
| `ours` | it carries a readable marker naming LlamaFit | yes, replacing it |
| `foreign` | anything else | no |

`foreign` covers four cases and each gets its own sentence: a directory with contents and no
marker (*"… already holds 7 files that LlamaFit did not install."*), a marker that will not parse,
a marker naming another tool, and a *file* where the directory should be. An unreadable marker is
deliberately not treated as ours: the whole point of the marker is that it is *proof*, and proof
is what licenses deleting what is there.

When it is not ours the command stops before downloading anything, with a hint naming the two
ways forward: `--force` to replace it, `--dir` to install elsewhere. The refusal is checked twice
— once when the plan is built, so the table shows it in red before the prompt, and again inside
`install_release`, so the guarantee does not depend on any caller remembering to look.

The marker also records what was installed: tag, build, commit, asset name, its SHA-256, the
backend asked for, the OS and architecture, and the backends actually detected afterwards. That
is what lets the next run say *"holds b10880, installed by LlamaFit"* rather than *"holds
something"*.

## 3. Which asset would you install on this Windows machine, and on a Mac, and on a Linux box with an AMD card?

From release `b10892`, the listing recorded in `tests/fixtures/llamacpp_release.py` on 2026-09-10:

| Machine | Backend | Archive |
|---|---|---|
| This Windows box (RTX 4060, x86_64, driver 610.88) | `cuda` | `llama-b10892-bin-win-cuda-13.3-x64.zip` **plus** `cudart-llama-bin-win-cuda-13.3-x64.zip` |
| The same box on an older driver (say 551) | `cuda` | `llama-b10892-bin-win-cuda-12.4-x64.zip` plus its own `cudart-...-12.4-...` |
| Apple silicon Mac | `metal` | `llama-b10892-bin-macos-arm64.tar.gz` |
| Intel Mac | `cpu` | `llama-b10892-bin-macos-x64.tar.gz` |
| Linux, AMD card | `hip` | `llama-b10892-bin-ubuntu-rocm-10.0-x64.tar.gz` |
| Linux, NVIDIA card | `vulkan` | `llama-b10892-bin-ubuntu-vulkan-x64.tar.gz` |
| Linux, no GPU | `cpu` | `llama-b10892-bin-ubuntu-x64.tar.gz` |

Three of those rows deserve a note.

**The Mac row has no Metal archive in it** because there is not one. Metal is compiled into the
plain macOS arm64 build, so asking for `metal` selects the archive with no backend token at all —
the same rule that selects a CPU build.

**The Linux AMD row is ROCm**, not Vulkan. The specification says "rocm when published", and it is
published now (`ubuntu-rocm-10.0-x64`). The preference list asks for it first and falls through to
Vulkan on its own on a release that does not publish it, so the answer follows the release rather
than a constant in the code.

**The Linux NVIDIA row is Vulkan** because llama.cpp publishes no Linux CUDA archive at all. That
is the fallback chain doing its job rather than an error.

The Windows CUDA row is the one that changed during the work. See below.

---

## What running against the real release page found

The suite has one `network`-marked test, `test_the_selector_still_finds_a_build_in_the_release_llama_cpp_publishes_today`.
It is the only check a recorded fixture cannot make — that the names have not moved — and it is
excluded from CI by the project's marker. Running it once, deliberately, found three things, all
of which would have shipped as silent failures:

1. **`/releases/latest` is the wrong endpoint.** Every llama.cpp build release is marked a
   prerelease, and GitHub's "latest" skips prereleases: it answers with the repository's
   semantic-version tag `v0.4.0`, whose only attachment is a seven-byte text file naming the
   current nightly. An installer built on it downloads nothing and cannot say why. `latest()` now
   lists `/releases?per_page=30` and takes the highest `bNNNN` tag, and `parse_build` is anchored
   so `v0.4.0` is not mistaken for one.

2. **The OpenVINO archive would have been installed for anyone asking for CPU.** A CPU build is
   chosen by the *absence* of every accelerator token, and `llama-b10892-bin-ubuntu-openvino-2026.3.1-x64.tar.gz`
   is marked by nothing but the word `openvino`. With the token missing from
   `ACCELERATOR_TOKENS`, it sorted ahead of the plain `...-ubuntu-x64.tar.gz` on the alphabet
   alone. Added, along with `webgpu` and `zdnn`, and the constant now carries a docstring saying
   why the list has to be complete rather than merely correct.

3. **The newest CUDA is not the right CUDA.** A release publishes archives for two or three CUDA
   majors at once — 12.4, 13.3 and 13.4 in `b10892`. A CUDA 13 build does not run *slowly* on a
   driver below 580; it does not start, and the message names a missing library rather than an old
   driver. The card's driver version is now read (it is already in `Gpu.driver`), archives the
   driver cannot load are filtered out, and the newest of what remains is taken. Where the driver
   is unknown the *oldest* published CUDA is taken instead, which is the most compatible — the same
   principle that already preferred an AVX2 build to an AVX512 one.

The fixture was replaced with the real recorded listing (names, sizes and published SHA-256
digests, unedited) once these were understood. Two earlier namings the project has used are kept
beside it, so the selector is held to reading all three; the names have moved twice already.

Also worth recording: macOS and Linux ship as `.tar.gz` now, not `.zip`. The extractor handles
both, with the same escape checks on every member and, where the interpreter has it, Python's own
`data` filter asked for as well.

## The rest of the design, briefly

**Verify before extracting.** Bytes go to `<archive>.part`. The checksum is computed on that file,
and only a file that matches is renamed to the archive's real name. Everything downstream may
therefore assume that a file under the archive name has been verified. A file that fails is
deleted, not kept: keeping it would mean every later run resumes onto bytes already known to be
wrong and fails identically with nothing saying why.

**Resume rather than restart.** `_fetch_into` continues from `part.stat().st_size`, and the fetcher
reports whether the server honoured the range. A server that answers a ranged request with the
whole file is detected and the partial file discarded rather than appended to — appending would
build a file that is part duplicate and would fail its checksum with no explanation. Three attempts,
each continuing from what is on disk. The archive cache is a real cache, not a temporary directory,
so "run it again" continues rather than restarts even across processes.

**Ask before changing PATH.** `plan_path_change` computes the change and its undo together and
applies nothing. The command prints the description, then the undo, then asks. On Windows it is the
`HKCU\Environment` value — user scope, no administrator, reversible by the same person — written
through the injected `Runner`; the undo names the exact `setx` line and the GUI path. On POSIX it is
a block labelled `# added by llamafit` appended to the profile the user's shell actually reads, and
the undo says to delete those two lines. `--add-to-path` alone still asks; only `--add-to-path
--yes` skips the question.

**Swap, don't dribble.** The new `bin` is assembled whole in a staging directory *inside* the
install root, so the final move is a rename on one volume. The old `bin` is renamed aside first and
deleted only once the new one has arrived; a failed move puts it back. A test proves a file in the
old directory survives a swap that fails.

**What is installed is what is detected.** The managed directory is `~/.llamafit/llama.cpp`, whose
`bin` is literally the first entry of `detect.well_known_dirs` — a test asserts that identity rather
than restating the path. `VERSION.txt` is written in the exact form `detect.parse_version` reads, and
a test round-trips it through that function. After the swap the backends are read back with
`detect.detect_backends`, recorded in the marker, printed, and compared against the backend asked
for: a CUDA install whose `ggml-cuda` libraries are missing is reported at once rather than at first
use.

## Testing

Injection follows the two patterns already in the project. `ReleaseClient` and `Fetcher` are
Protocols with a real implementation and a fake, the way `catalog/hf.py` has `HfClient` /
`HttpHfClient` / `FakeHfClient`; the command runner for the Windows registry is the project's
existing `Runner` / `FakeRunner`. The real `HttpReleaseClient` and `HttpFetcher` are exercised
through `httpx.MockTransport`, so the actual header-building, status handling and range logic are
covered without a socket. Nothing in the suite reaches the network or writes outside `tmp_path`.

Failure paths tested: a checksum that does not match (and the file deleted), a truncated body, a
connection dropped mid-stream and resumed from the right offset, a server that ignores `Range`,
a `.part` file longer than the asset, retries exhausted, no published checksum, a full disk
(`ENOSPC`), a volume without room (refused before any download), a directory holding somebody
else's build, a marker that will not parse, a marker naming another tool, a file where the
directory should be, a zip and a tar member that would escape the directory, a tar symlink, a
corrupt archive, an archive with no `llama-server`, a failed swap, a registry write that is denied,
and a backend named on the command line that the release does not publish.

`FakeFetcher` carries the failures rather than the tests faking them: `stop_after` drops the
connection once and then clears itself, which is what makes a resume testable at all, and
`honours_range=False` is the proxy that ignores ranges.

## Verification

```
pytest --cov -m "not hardware and not network"   1926 passed, 2 skipped, 12 deselected — 96% total
ruff check .                                     All checks passed
ruff format --check .                            221 files already formatted
mypy                                             Success: no issues found in 105 source files
gen_messages / gen_schema / gen_models_md        no diff
```

New-module coverage: `releases.py` 97%, `install.py` 95%, `install_cmd.py` 96%. The one skip is the
POSIX executable-bit test, skipped on Windows. The `network` test passes when run deliberately.

`messages.pot` gained 96 messages, 549 to 645, untranslated in all 37 catalogs, which is correct.

The dry run above is also, incidentally, the proof that the number formatting goes through the
translation layer like everything else: on this machine, whose interface language is Portuguese,
the same command prints `515,6 MiB`.

## Concerns and things left

- **`ARCHIVE_EXPANSION = 3.0`** is a rounded-up guess at how much larger the unpacked tree is than
  the archive. It is used only for the space check and errs generous, so it can refuse an install
  that would in fact have fitted on a nearly-full disk. Reading the zip's own uncompressed sizes
  would be exact for zips and is not possible for a stream, so it stays a constant for now.
- **`CUDA_DRIVER_FLOOR`** is three numbers from NVIDIA's compatibility table and will need a line
  added when CUDA 14 appears. An unknown CUDA major is deliberately allowed through rather than
  refused, so a stale constant cannot be what blocks a working build — but it does mean a genuinely
  incompatible future build could be selected on an old driver.
- **`--backend` cannot name a CUDA version.** `--backend cuda` gets whatever the driver rule
  chooses. If someone needs the other one, only `--dir` plus a manual unzip gets them there today.
- **The `hip` backend name** is what the rest of LlamaFit calls it (`models/host.Backend`), while
  llama.cpp now names the archive `rocm`. The selector reads both tokens; the flag a user types is
  `--backend hip`, which is slightly off from what they will see downloaded. Worth reconsidering
  when `Backend` is next touched — it is not this module's type to change.
- **Windows `bin` swap while a server is running** will fail with a sharing violation, and the
  rollback puts the old directory back. The message is the operating system's, which says a file is
  in use but not that they should stop `llama-server` first. A check for a running server before the
  swap belongs with `launch`, which knows about them.
- **No `install model`** — the `install` group's help says llama.cpp "and later models", and
  `docs/cli.md` documents the model subcommand as still to come. That is another agent's work.
- **`llamafit doctor`** is not yet aware that an install exists; it will find one because the
  managed directory is a well-known directory, but it does not offer `install llama.cpp` as the fix
  when llama.cpp is missing. One line in `services/doctor.py`, deliberately not taken here to stay
  out of another agent's module.
