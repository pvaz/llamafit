# Model downloader — `src/llamafit/download/` and `llamafit install model`

Branch `feat/downloads`, worktree `C:\Dev\Projectos Pessoais\2026\llamafit-wt-downloads`.
Built from point 2 of section 15 of `docs/specs/2026-09-09-llamafit-design.md`.

| | |
|---|---|
| Commits | `ffa09c1` feat(download): parallel, resumable, checksum-verified model downloader<br>`01dad42` feat(cli): llamafit install model, and the downloads history |
| Tests | 1913 passed, 1 skipped, 13 deselected; coverage 96.33% (floor 85%) |
| Lint / format / types | ruff check clean, ruff format clean, mypy strict clean on 114 files |
| Generators | `gen_messages.py`, `gen_schema.py`, `gen_models_md.py` leave `git diff --exit-code` clean |
| Files touched outside the package | `src/llamafit/cli/app.py` (one import line plus its `__all__` entry), `docs/cli.md`, `CHANGELOG.md`, the generated `messages.pot` |

Both commits are individually green: the first was verified with the CLI half held
back (1894 passed, template regenerated for the package alone).

---

## The three questions

### 1. What does a user see before the first byte is written?

Everything needed to decide whether to spend the next two hours on it, built from the
catalog alone — no socket is opened to produce it, so `--dry-run` works offline.

```
$ llamafit install model qwen3-coder-next --dry-run
Qwen3-Coder-Next UD-Q4_K_XL from unsloth/Qwen3-Coder-Next-GGUF
About to download
┌──────────────────────────────────┬─────────┬──────────┬──────────┐
│ File                             │ Role    │     Size │ Status   │
├──────────────────────────────────┼─────────┼──────────┼──────────┤
│ Qwen3-Coder-Next-UD-Q4_K_XL.gguf │ weights │ 46.2 GiB │ to fetch │
└──────────────────────────────────┴─────────┴──────────┴──────────┘
1 file, 46.2 GiB in all, 46.2 GiB still to fetch.
Into ...\models\qwen3-coder-next, which has 352.9 GiB free; 306.7 GiB would be left.
```

That is a real run against the bundled catalog. The model's display name, the
quantisation, the publishing repository; one row per file with its role (`weights`,
`mmproj`, `draft`) and whether it is already here; the total; what is still to fetch;
the destination; the free space and what would be left. A quant the catalog holds no
checksum for adds a yellow line naming the files.

Then, in a terminal, `Start the download? [y/N]`. Outside a terminal it does not ask —
there is nobody there to answer, and whoever wrote the script asked by running it.
`--yes` skips it. `--json --dry-run` gives the same content as a document, including
each file's URL, target path, checksum and `enough_space`.

Two refusals happen *before* the prompt, so nobody is invited to confirm a download
that is about to be turned down: the missing-checksum refusal, and the disk check.

During the run: a Rich progress bar per file — name, bar, percent, `4.7 GiB of
61.2 GiB`, rate, time left — with columns coming off from the right as the window
narrows and the bar dropped below sixty cells. A pipe or a log file gets one line per
file instead, because nothing can redraw a line that has already been written. Every
byte figure goes through `llamafit.units.format_bytes`, so a Portuguese reader is told
`46,2 GiB`; a test asserts that.

At the end: `2 files, 46.2 GiB, checked against the catalog.` and where they are.

### 2. How did you prove resume works, rather than assert it?

Three ways, of increasing realism.

**A server that refuses to serve what is already here.**
`test_a_resumed_transfer_asks_only_for_what_is_missing` runs a download until the fake
hub sets the stop event after four chunks, asserts the sidecar records exactly
`[0, 1, 2, 3]`, then runs it again against a second hub configured with
`forbidden_below=4 * CHUNK` — that hub *raises* if it is asked for a byte the first run
already wrote. The download completes, only ranges 4–9 are requested, and the finished
file's SHA-256 equals the original blob's. It cannot pass by re-fetching quietly.

**A chunk cut off mid-stream.**
`test_a_chunk_cut_off_mid_stream_is_not_recorded_and_is_fetched_again` truncates one
chunk's body halfway. That chunk is not recorded, the progress it claimed is taken back
(a negative delta the test asserts on), range `(0, 1023)` is requested exactly twice,
and the file still verifies. The bar's position at the moment verification begins is
exactly the file size — it never claimed bytes that were not on disk.

**A real process, hard-killed, over a real socket.**
Not a committed test — it needs a server — but run during development and worth
recording. A throwaway HTTP server on `127.0.0.1:8791` served a 40 MiB random file with
genuine `Range` support, logging every range it was asked for. The shipping
`download_file` ran against it through the shipping `HttpRangeReader`, throttled to
2 MiB/s, and the process was killed with `timeout 6` — SIGKILL, no clean shutdown, no
handler:

```
exit=124 (killed midway)
f.gguf.part            41943040 bytes
f.gguf.part.state      {"...","size":41943040,"chunk_size":2097152,"done":[0,1,2,3]}
ranges served in run 1: 0-0, 0-2097151, 2097152-4194303, ... 14680064-16777215
```

Nine ranges had been requested; four chunks were recorded. The record was *behind* the
file by the four chunks that were still in flight, which is the invariant — behind
costs a re-fetch, ahead would be a hole. The second run:

```
{"fetched": 33554432, "resumed": true, "verified": true,
 "sha256": "a5a39a...1972"}
lowest byte asked for: 0-0 (the size probe), then 8388608-10485759
```

32 MiB fetched of 40 MiB, nothing below 8 MiB requested except the one-byte size probe,
and the digest matches what the server generated. The scratch server and driver were
deleted; the port is confirmed closed.

**Why a chunk index and not a byte offset.** Sixteen workers fill a part file with
holes, so its length says nothing about what has arrived. The sidecar
(`<name>.gguf.part.state`) records the URL, the total size, the chunk size and the set
of completed chunk indices, and it is written under the same lock that adds the index —
a real bug found by the first test run on Windows, where two workers racing on the one
temporary file gets the loser a sharing violation rather than a merged result. Any
disagreement between the record and the run (different URL, size, chunk size, missing
part file, unreadable JSON, future schema) throws the record away and starts clean:
costing bandwidth is always correct, and a file made of two different downloads is not.

### 3. What happens when a checksum does not match, and what is left on disk?

The part file and its record are both deleted, and nothing is left that a later run
could resume into. Nothing is ever moved to the final name.

The order is: all chunks arrive → `check_size` against the size the server declared
(cheap, and it names truncation as truncation rather than reading out two hexadecimal
strings) → `sha256_of` the part file → compare with the catalog. Only then `os.replace`
onto the final name. A failure at either check calls `PartFile.discard()`, which unlinks
`<name>.gguf.part` and `<name>.gguf.part.state`, and raises `ChecksumError`.

The user sees:

```
Qwen3-Coder-Next-UD-Q4_K_XL.gguf does not match the checksum the catalog holds.
Hint: Expected 3f2a…  but the file hashes to 91bc… . The bad copy has been
deleted; run the command again to fetch it.
```

Exit code 1. No manifest is written, so nothing believes the model is installed. Files
of the same model that *did* verify stay where they are — they are correct, and the next
run skips them.

Deleting rather than keeping is the point: a kept part file would be resumed into
forever, and running the command again would produce the same wrong file faster.

Tested at three levels: `test_a_checksum_that_does_not_match_leaves_nothing_behind`
(engine — asserts target, part and state are all absent, and that the hint names both
digests), `test_a_shard_that_fails_its_checksum_leaves_no_trace_of_itself` (install —
also asserts the good shard survives and no manifest is written), and
`test_a_checksum_that_does_not_match_stops_the_command` (CLI).

---

## The parallelism: the two things to be careful about

### A server that rate-limits

Backing off in time alone is not enough — the same sixteen workers arrive together
after the delay. So there are two responses, and the second is the one that matters.

1. **Honour the delay.** A 429 or 503 raises `RateLimitedError` carrying the server's
   `Retry-After` when it sent a numeric one. That wins over our own backoff; otherwise
   the delay doubles per attempt, capped at 60 s, multiplied by jitter in
   `[0.5, 1.0)` — without the jitter, workers all refused at the same moment come back
   at the same moment.
2. **Actually reduce the pressure.** `Governor` caps requests in flight and, on a rate
   limit, *retires a permit permanently*: the worker gives its permit up instead of
   returning it, so the ceiling comes down by one and stays down. It never climbs back
   during the run — a server that said 429 once will say it again, and creeping back up
   to sixteen to find out is the rudeness this exists to stop. The floor is one, so a
   server that rate-limits everything still gets its files fetched, one request at a
   time, rather than the command failing. The user is told:
   `the server asked for fewer connections; using 3.`

After `max_attempts` (5 by default) the file fails with the server's own status in the
message, and everything fetched stays for the next run. Only 408, 425, 429, 500, 502,
503 and 504 are retried; 401/403 says the repository is gated and names `HF_TOKEN`, 404
says to run `catalog refresh`, and neither is retried.

Tested: `test_a_rate_limited_server_is_waited_for_and_given_fewer_connections`,
`test_a_server_that_only_ever_rate_limits_gives_up_saying_so`, the `Governor` unit
tests (a permit is retired only when it is given back; the ceiling stops at one).

### A user who wants their connection back

Three answers.

- **`--workers` defaults to 8, not 16.** The specification says 16 to 32 and `--workers`
  reaches all of it, capped at 32. But eight is the default because the machine is
  somebody's own and the line is shared with whoever else is in the house. Sixteen
  streams from a content network take every bit of a domestic connection, and the person
  who typed the command did not ask for their video call to stop working. Someone who
  wants the line saturated can say so; someone who does not should not have to find out
  that they should have.
- **`--limit-rate 5M`** is a single leaky bucket shared by every worker, so the figure
  is the figure whatever the concurrency is. A test runs four threads through one
  limiter on an injected clock and asserts they wait 1 s, 2 s and 3 s between them — a
  per-worker limiter would have let all four straight through. The reservation is made
  under the lock and the waiting done outside it, so a waiting worker does not hold up
  one that is writing to disk. It takes the same size parser (`8G`, `7.5GiB`, `512M`) as
  every other size option, and rejects `fast` by name.
- **Ctrl+C gives it back immediately and cleanly.** The handler sets the event the
  workers already watch instead of raising `KeyboardInterrupt` into whichever thread
  happened to be running — that would leave the pool half torn down and the record
  possibly a chunk behind the file. Workers stop between 64 KiB pieces, the record and
  the file agree, and the last line is: *Stopped. What arrived is kept; run `llamafit
  install model <id>` again to carry on from there.* A second press restores Python's
  own handler and raises, because somebody pressing it twice means it now.

One more restraint worth naming: **files are fetched one at a time**, with the workers
spent on the chunks of the file in hand. A model's shards are tens of gigabytes each;
there is nothing to gain from four half-finished at once, and something real to lose,
since a run stopped halfway would then have four part files instead of one.

---

## The rest of what was built

**A split model is one thing.** `build_plan` takes the quant's shards, plus every extra
the entry declares (vision projector, draft weights), under one total. The manifest
`llamafit-install.json` is written only when *all* of them have arrived and been
checked; its presence is what "this model is installed" means, and a directory holding
three shards of four does not have one. `test_a_model_that_loses_a_shard_writes_no_manifest_and_keeps_the_rest`
holds that.

**A shard's size is not invented.** The catalog records one total per quant and nothing
per shard. Rather than apportion the total and pretend — which would be the size the
part file was created at — a shard's `size` is left `None` and the engine learns it from
the server's `Content-Range` via a one-byte probe before allocating. For a file the
catalog *does* size (a single-file quant, an extra), a mismatch fails immediately with
"the repository has been re-uploaded since the catalog was refreshed", rather than after
a 46 GiB download that could only end at the checksum. The manifest is also where a
shard's real size is recorded, so a later run can tell a finished shard from a file that
merely has the right name.

**Nothing is fetched that cannot be checked.** A quant with no checksums is refused,
naming `catalog refresh`. `--allow-unverified` is the way past it and says out loud what
it gives up; files fetched that way come back `verified: false`.

**A server that stops honouring ranges.** A 200 where a partial was asked for is caught
before a byte is written — writing a full body into a chunk's slot would corrupt the
file in a way only the checksum catches, hours later. The file's part is discarded and
it restarts as one sequential stream, with a note to the user. Two tests: one where it
happens after three chunks, one where the server never serves a range at all.

**Testing.** Nothing in the suite reaches the network or writes outside `tmp_path`. The
transport is a one-method protocol (`RangeReader.open`), and every test drives the real
`HttpRangeReader` through `httpx.MockTransport` — so redirect following, header
construction and status handling are the shipping code. `FakeHub` redirects
`huggingface.co/...` to a CDN URL by default and the injected client is left at its
default (no redirect following), so a reader that relied on the client's setting would
fail. Failure paths covered: checksum mismatch, truncated body, transfer interrupted and
resumed, full disk (before and at allocation), a server that stops accepting ranges
halfway, rate limiting, gated repository, 404, a stale catalog size, an unreadable or
disagreeing resume record, an unreadable manifest, a history that cannot be written.

Two tests carry the `network` marker (`tests/unit/test_download_real_hub.py`) and are
deselected by default: they check that the real hub honours a range *through* its
redirect (206, not 200) and that the size it reports matches the catalog's. That is the
one thing no mock can check, and the reason `gguf/source.py` carries the same warning.

**Translation and units.** Every user-visible string goes through `_()`/`ngettext`;
`scripts/gen_messages.py` regenerated the template (596 → 631 messages). Every byte
count goes through `llamafit.units.format_bytes`, every identifier inside a sentence
through `isolate()` and every finished line through `for_display()`, so the output reads
correctly in Arabic and Hebrew.

---

## Concerns

1. **`--recheck` on a large model is expensive and it does not say so.** Re-hashing
   46 GiB takes minutes; the progress bar moves, but nothing warns before starting.
2. **`fsync` is not called.** Chunks are flushed, not fsynced, so the guarantee is
   against a process crash rather than a power cut. Sync per 32 MiB chunk would be a
   real throughput cost for a failure mode the checksum catches anyway — but a power cut
   mid-download can leave the record ahead of the file, and the next run would resume
   over a hole. It would fail the checksum rather than produce a bad model, but the user
   pays for a whole re-download to find out.
3. **A file that is already on disk is trusted by size, not by hash.** That is the right
   default (hashing 46 GiB on every run is not free) and `--recheck` exists, but it means
   a file corrupted after installation is not noticed by `install`.
4. **The one-byte probe costs a request per file.** Negligible for a 46 GiB model,
   noticeable for a model with many small extras. It buys the shard's real size and
   early detection of a stale catalog, which I judged worth it.
5. **`install llama.cpp` (section 15.1) is not built** — the `install` Typer group is
   created in `install_cmd.py`, so whoever builds it should add a command to the existing
   group rather than a second one.
6. **`--limit-rate` and `--workers` are not persisted in `config.toml`.** Section 14 lists
   config for the download directory; a user on a metered line has to pass the flags every
   time. Worth doing when the config module lands.
7. **The parallel-write path is exercised on Windows only.** Each worker opens its own
   handle and seeks to its own non-overlapping range; POSIX behaves the same way, but CI
   is what will confirm it on Linux and macOS.
