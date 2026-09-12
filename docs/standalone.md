# The standalone binary

One file, no Python, no `pip`, no virtual environment. Download it, run it, delete it when
you are done. This page is about that file: which one to take, how to check it is the one
this project built, what your operating system will say about it the first time, and what
it contains that the `pip` install does not, and the other way round.

If you already have a Python, `pip install llamafit` is still the better way in, and the
[README](../README.md#install) says why in two lines. Everything here is for the case where
acquiring a language runtime to answer "will this model run on my machine?" is a worse
trade than downloading 25 MB.

## Which file

Every release attaches one binary per platform, on the
[releases page](https://github.com/pvaz/llamafit/releases).

| Your machine | File |
|---|---|
| Windows, Intel or AMD | `llamafit-<version>-windows-x86_64.exe` |
| macOS, Apple silicon (M1 and later) | `llamafit-<version>-macos-arm64.tar.gz` |
| macOS, Intel | `llamafit-<version>-macos-x86_64.tar.gz` |
| Linux, Intel or AMD | `llamafit-<version>-linux-x86_64.tar.gz` |
| Linux, ARM (Raspberry Pi 4 and later, Ampere, Graviton) | `llamafit-<version>-linux-aarch64.tar.gz` |

The Linux builds are made on Ubuntu 22.04 and need glibc 2.35 or newer, which covers
Debian 12, Ubuntu 22.04, RHEL 9, Fedora 36 and everything released after them. On an older
distribution, or on Alpine and anything else built on musl, use `pip install llamafit`
instead: the binary will not start, and it will say so in the unhelpful way a dynamic
linker says things.

There is no 32-bit build and there is not going to be one.

## Running it

**Windows.** Put the `.exe` somewhere you can find it, open a terminal there (PowerShell,
Windows Terminal or `cmd`), and run it:

```
.\llamafit-0.1.1-windows-x86_64.exe recommend
```

Double-clicking it opens a window that closes again immediately. That is not a fault: it is
a command-line program, it ran with no arguments, and it printed its help to a window that
had nothing left to do. Rename it to `llamafit.exe` and put it somewhere on your `PATH` if
you would rather type less.

**macOS and Linux.** Unpack it and run it:

```
tar -xzf llamafit-0.1.1-macos-arm64.tar.gz
./llamafit recommend
```

The archive keeps the execute bit, so there is no `chmod` step. To have it on your `PATH`:

```
mkdir -p ~/.local/bin && mv llamafit ~/.local/bin/
```

## Checking what you downloaded

Each release carries a `SHA256SUMS.txt` listing every binary, and each binary also has its
own `.sha256` file beside it, so checking one download does not mean fetching a list of
five. In the directory you downloaded into:

```
sha256sum -c llamafit-0.1.1-linux-x86_64.tar.gz.sha256     # Linux
shasum -a 256 -c llamafit-0.1.1-macos-arm64.tar.gz.sha256  # macOS
```

```powershell
Get-FileHash .\llamafit-0.1.1-windows-x86_64.exe -Algorithm SHA256   # Windows
```

and compare with the line in `SHA256SUMS.txt`.

This proves the file arrived intact and unmodified from the release page. It is not a
signature and does not claim to be one: anyone who could replace the binary on that page
could replace the checksum next to it. What it is good for is the ordinary case: a
truncated download, a proxy that rewrote something, a mirror you are not sure about.

The binaries are built by
[`.github/workflows/binaries.yml`](../.github/workflows/binaries.yml) from the tagged
commit, on GitHub's runners, with no credential and nothing uploaded by hand. That file and
[`packaging/llamafit.spec`](../packaging/llamafit.spec) are the whole recipe, and you can
run them yourself.

## What your operating system will say

Neither binary is code-signed. A Windows certificate and an Apple Developer ID are annual
fees and an identity check, which is a reasonable thing for a project to pay for once
people are relying on it and not a reasonable thing to pretend has already happened. So the
first run comes with a warning, and it is worth knowing what each one means.

**Windows SmartScreen** shows *"Windows protected your PC"* with a **Don't run** button.
Click **More info**, then **Run anyway**. The message means Microsoft has not seen this
file often enough to have an opinion about it: the message is about the file's
popularity and its signature, not about its contents.

**macOS Gatekeeper** refuses with *"cannot be opened because the developer cannot be
verified"*. Either right-click the binary in Finder and choose **Open**, which offers a
button the double-click does not, or clear the quarantine flag the browser attached:

```
xattr -d com.apple.quarantine ./llamafit
```

**Antivirus software** occasionally flags PyInstaller binaries in general, because
"executable that unpacks itself and then runs Python" describes a lot of malware and a
handful of legitimate tools. There is no compression in these builds, because UPX is off
on purpose for exactly this reason, but a heuristic is a heuristic. If yours quarantines the
file, `pip install llamafit` is the way around it, and a report to your vendor is the way
to fix it for the next person.

None of this is advice to ignore security warnings in general. It is a description of what
these particular ones are reacting to, so that you can decide.

## What is inside it

The interpreter, every dependency, and all of LlamaFit's own data: the model catalog, the
generated GGUF facts, the hardware profiles, the JSON schemas, the GPU table and all 37
message catalogs. Every command works offline exactly as it does after a `pip install`, and
`--language` reaches the same 38 languages.

The web dashboard is included, so `llamafit serve` works. NumPy is not, and that is the one
visible difference: the memory-bandwidth measurement falls back to its pure-Python method,
which is slightly less accurate. LlamaFit tells you which method it used, so you are never
reading a number without knowing where it came from. NumPy and its dependencies measured
10.6 MB on a 25 MB binary, unpacked again on every run, to sharpen one figure;
`pip install "llamafit[fast]"` is there for anyone who wants that trade the other way
round.

## One thing it cannot be told

LlamaFit falls back to English when the stream it is writing to cannot carry the script
you asked for, and says which encoding refused which language. For a `pip install` the fix
is `PYTHONUTF8=1`, which is what the message suggests. **That does not work for the
binary**, and the message says it anyway: PyInstaller starts the interpreter from an
isolated configuration, which reads no `PYTHON*` variable at all, so neither
`PYTHONUTF8` nor `PYTHONIOENCODING` reaches it.

In practice this is narrower than it sounds. On a real terminal it never comes up: Python
on Windows has written console output as UTF-8 since 3.6, whatever the code page and
whatever UTF-8 mode says, so every script draws correctly. It appears only when you
redirect output into a file or a pipe on Windows, where the encoding comes from the ANSI
code page instead. The setting that does fix that one is Windows' own: Region →
Administrative → *Beta: Use Unicode UTF-8 for worldwide language support*, which makes the
ANSI code page UTF-8. `chcp 65001` does not, because it changes the console's code page
and not that one.

Two smaller things follow from it being one file:

- **It unpacks itself on every run.** Expect roughly a second before the first output, more
  on a cold cache or a slow disk. A `pip` install does not pay this.
- **There is no update command.** Download the new file when there is one.

## It shares its state with the installed version

Settings, caches, benchmark records and downloaded models live in the usual per-user
directories (`%LOCALAPPDATA%\llamafit` on Windows, `~/Library/Application Support/llamafit`
on macOS, the XDG directories on Linux), and the binary uses the same ones. Start with the
binary, install with `pip` later, and everything you measured or configured is still there.
`LLAMAFIT_HOME` overrides all of it and puts config, cache, logs and models under one
directory, which is what you want on a USB stick or a machine you are only borrowing.

## Building one yourself

```
pip install -e ".[web]" pyinstaller
pyinstaller packaging/llamafit.spec
```

The result lands in `dist/`. PyInstaller is not a cross-compiler: you get a binary for the
machine you built it on and no other. [`packaging/llamafit.spec`](../packaging/llamafit.spec)
explains each of its choices, including the ones that look like they could be removed.
