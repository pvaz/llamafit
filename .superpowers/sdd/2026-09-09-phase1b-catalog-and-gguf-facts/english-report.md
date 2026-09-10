# The English round: what changed, what did not, and what it cost the catalogs

Branch `feat/english`, four commits off `main`. Nothing was merged, rebased or pushed, and
no other branch or worktree was touched.

| | |
|---|---|
| `8244e6f` | feat(i18n): whole sentences and counted forms in the rendered tables |
| `0194b78` | feat(i18n): counted forms in the GGUF reader and a spelled-out timeout unit |
| `bda2a23` | feat(i18n): translate the model capability names |
| `44e837e` | docs: the two message-shape rules a translator cannot work around |

Verification: 1038 passed, 1 skipped, 11 deselected; coverage 92.65% (floor 85%). `ruff
check`, `ruff format --check` and `mypy` clean. The three generators rewrite nothing
(`git diff --exit-code` passes). Every one of the thirty-seven catalogs reports **0
problems** from `check_po.py`.

The template went from 248 to 271 messages. Thirty of the thirty-seven catalogs now
translate 247 of them, seven translate 239, and `pt_PT` translates 246 because it
deliberately leaves `Skip the RAM bandwidth measurement.` empty to prove the fallback
works. That drop is the honest outcome, not a regression: twenty-four English sentences
genuinely became different sentences, and a translation of the old one attached to the new
one would be a lie with a green build behind it.

---

## 1. Which messages changed, and is the old translation still valid?

### Carried across — the English text is unchanged, so the translation still stands

Adding a `msgctxt` does not change what the sentence says; it changes what the program
looks the entry up by. For every one of these the translation was moved to the new key in
all thirty-seven catalogs, so nothing regressed in any language and no translator has to
retype a word. What they gain is a second slot to answer differently if their language
wants one.

| Was | Is now | Note |
|---|---|---|
| `ID` | `msgctxt "column heading"` | |
| `Params` | `msgctxt "column heading"` | |
| `Quality*` | `msgctxt "column heading"` / `Quality` | the `*` is a footnote marker and is added in code now; every catalog's translation was carried with the trailing `*` stripped |
| `Context` | `msgctxt "column heading"` **and** `msgctxt "table row label"` | one entry became two; both start life with the translation the single entry had |
| `Capabilities` | `msgctxt "column heading"` **and** `msgctxt "table row label"` | same |
| `isa unknown` | `msgctxt "CPU instruction sets"` | |
| `type unknown` | `msgctxt "memory type"` | |
| `VRAM unknown` | `msgctxt "GPU VRAM"` | |
| `unknown build` | `msgctxt "llama.cpp build"` | |
| `unknown model` | `msgctxt "model name"` | |
| `unknown error` | `msgctxt "probe error"` | |
| `no specs` | `msgctxt "GPU specifications"` | |
| `no server` | `msgctxt "probe result"` | joins `ok` and `failed`, which already had it |
| `not read yet` | `msgctxt "GGUF facts"` | |
| `unspecified method` | `msgctxt "context extension method"` | |

One more changed without changing its message at all: `%(count)d/%(used)d expert` now
selects its plural form on `n_expert_used` rather than on `n_expert`. The `msgid` is
untouched, both forms are the ones the translator already wrote, and only the choice
between them moved — to the number the noun actually stands next to, which is the one
Russian and Polish agreement follows.

These survived untouched and are still translated everywhere: `%(count)d module`,
`%(count)d channel`, `%(count)d layer`, `%(count)d expert`, `%(count)d performance core`,
`%(speed)d MT/s`, `%(specs)s (spec)`, `driver %(driver)s`, `%(tokens)s tokens native`.
Keeping them was worth designing around: `tests/fixtures/messages.py` exercises the
machinery against `%(count)d module` in the real `pt_PT` catalog, and the two-count
problem was solved by *adding* counted entries beside them rather than by replacing them.

### Orphaned — the English is a different sentence, so the translation was dropped

Twenty-four entries are empty in all thirty-seven catalogs and fall back to English until
somebody translates them. Nine old entries were deleted outright.

**The context line** (`render_model_facts`). `, %(tokens)s extended via %(method)s` opened
with a comma and continued the message above it. Gone; the two shapes are now
`%(tokens)s tokens native` (kept, still translated) and
`%(tokens)s tokens native, %(extended)s extended via %(method)s` (new, empty).

**The host GPU line.** The specification phrase and the driver were joined with a
hard-coded `", "`. The four shapes are written out; three reuse existing entries and
`%(specs)s (spec), driver %(driver)s` is new.

**The memory line.** `", ".join(...)` of three separately translated fragments became one
message per shape: `%(type)s at %(speed)d MT/s`, `%(kind)s, %(modules)s across
%(channels)s`, `%(kind)s, %(modules)s`, `%(kind)s, %(channels)s`, `%(modules)s across
%(channels)s` — five new entries. The counted phrases inside them are the untouched
`%(count)d module` and `%(count)d channel`, so each noun still agrees with its own number.

**The quant facts summary.** The same `", ".join` shape, not reported by anyone but the
identical defect: `%(arch)s, %(layers)s`, `%(arch)s, %(experts)s` and `%(arch)s,
%(layers)s, %(experts)s` are new.

**Cores and threads.** `%(physical)d cores / %(logical)d threads` carried two numbers and
was not a plural entry at all. Deleted. In its place: `%(count)d core` and `%(count)d
thread` as counted entries, set into `%(cores)s / %(threads)s` or `%(cores)s /
%(threads)s, %(performance)s`. The rendered English is character-for-character what it was
("24 cores / 32 threads, 8 performance cores"), so no test moved and no reader will notice
anything but the language.

**The two captions.** `Quality is the editorial baseline…` and `Params is total/active…`
spelled their column's English word out a second time, with nothing keeping the two
entries in step. They take `%(heading)s` now and are handed the heading itself, so the
heading is the single source. Both are new entries.

**The parameters row.** `%(total)sB total, %(active)sB active` welded a `B` to each
placeholder while the list table asked `billions_suffix()` for the same mark — so a
language that answered that entry got its own suffix in one table and the English one in
the other. Now `%(total)s total, %(active)s active`, with the suffix supplied by
`billions_suffix()`. New entry.

**The timeout.** `%(program)s: timed out after %(seconds)ss` deleted; `%(program)s: timed
out after %(seconds)s second(s)` is a counted entry, and the number goes through
`localise_number` like every other figure LlamaFit prints. A duration can be fractional, so
the form is selected on the nearest whole second while the exact number is shown.

**The GGUF reader**, four messages, all deleted and replaced:

- `truncated GGUF header: wanted %(wanted)d byte(s) at offset %(offset)d, got %(got)d` —
  counted; `%(offset)d` and `%(got)d` have no noun beside them.
- `%(count)d GGUF file(s) of the %(total)d given carries/carry no '%(key)s' key…` —
  counted on the number the noun stands next to; `%(total)d` is followed by *given*, a
  participle, not by a noun that has to agree with it.
- `incomplete shard set: the model declares %(shards)s and %(arrived)d arrived…` — the
  shard count is built as its own counted phrase (`%(count)d shard`, new) and set in;
  `%(arrived)d` is followed by a verb.
- `incomplete shard set: the model declares %(tensors)s and its %(shards)s hold %(held)d
  between them` — two counted phrases (`%(count)d tensor`, new, and `%(count)d shard`)
  each agreeing with its own number, and `%(held)d` followed by *between them*.

**The capability names**, eight new entries under `msgctxt "model capability"`: `coding`,
`thinking`, `vision`, `tools`, `multilingual`, `long-context`, `embeddings`, `audio`.

### Which languages I filled for the capability names

**Filled (30):** `bg` `ca` `cs` `da` `de` `el` `es` `fi` `fr` `hr` `hu` `id` `it` `ja`
`ko` `ms` `nb` `nl` `pl` `pt_BR` `pt_PT` `ro` `ru` `sr` `sv` `tr` `uk` `vi` `zh_CN`
`zh_TW`.

Each reuses that catalog's own word for *context* in `long-context`, so the two rows agree
with each other (`μεγάλο πλαίσιο` in Greek because that catalog says `Πλαίσιο`, `uzun
bağlam` in Turkish because it says `Bağlam`). `pt_PT` was checked against the
Brazilian-word list the test suite enforces.

**Left empty (7):** `ar` `bn` `he` `hi` `ta` `th` `ur`. I am not confident enough in the
register these words take in a scanned table column in those languages, and the two RTL
catalogs mix scripts in a way I would not want to guess at. They fall back to English,
which a reader can look up; a wrong word in a column they scan is worse.

---

## 2. What I decided not to fix

**The `list` command's help epilog quotes a column heading in English prose.** "…is sorted
by the Quality column shown…" is exactly the caption defect, and it is the one instance I
could not fix the same way: the epilog is a `lazy_gettext` built while the module is
imported, before a language has been chosen, so there is no heading to pass in as a value.
A translator does translate the whole sentence, so the words are reachable; what stays
unenforced is that they keep it in step with the heading entry. Fixing it properly means
teaching `LazyString` to interpolate at render time, which is a change to the i18n layer
and not to an English message.

**Lists of identifiers are still joined with `", "` in Python.** `llamacpp.backends`,
`model.use_cases`, `cpu.isa`, `_VALID_CAPABILITIES` in the invalid-value hints, the
suggested model ids, the model ids in a refresh warning. These are variable-length lists,
and "one whole message per shape" cannot express a list of unknown length; that needs
either a translatable list separator (the `pgettext_literal` mechanism the number
separators already use) or ICU-style list formatting. Both are a design decision about the
i18n layer rather than an English fix, and none of the six reports named them. Noted
below as something to decide, not something I quietly left broken.

**Indic digit grouping.** `docs/translations.md` already documents that LlamaFit groups in
threes everywhere, so Hindi and Urdu see `3,276,800` and not `32,76,800`. Out of scope and
already written down.

**Use-case names.** `general`, `coding`, `reasoning`, `chat`, `multimodal`, `embedding`
reach every language in English for exactly the reason the capability names did. The brief
named capabilities, so that is what I changed. See below.

---

## 3. What I found that nobody reported

- **`_facts_summary` had the same `", ".join` defect as the memory line.** Fixed; three
  new entries. Leaving a known-identical bug in the same file after fixing its siblings
  would have put it straight back in the next round of reports.
- **The parameters row welded its `B`** while the list table used `billions_suffix()`. Two
  tables, one table row, two different suffixes for a language that answered the entry.
  Fixed.
- **`unknown error`** in `services/doctor.py` was a bare fragment sitting next to entries
  that already carry a context. Given one.
- **`no specs`, `no server`, `not read yet`, `unspecified method`** are the same class as
  the reported `X unknown` group without matching its shape. Given contexts; their
  translations were carried, so this cost nothing.
- **The truncated-header message counts bytes** the way the reported messages count files,
  shards and tensors. Counted now.
- **Use-case names are still English enum values** in the Use cases row, for the same
  reason the capability names were. Same fix would apply; not in this round's brief.
- **The capabilities row and column still join translated words with a hard-coded `", "`**,
  and so does the `+N` marker in the column. Translating the names made this visible: it is
  now a list of *words in the reader's language* behind a separator no catalog can reach,
  which is a stronger case for a translatable list separator than the identifier lists are.
- **Translating the capability names costs column width.** The budget measures with
  `cell_len` and is correct, but German `Bildverstehen` and `Langer Kontext` are much wider
  than `vision` and `long-context`, so a German reader at eighty columns sees fewer
  capability names before the `+N` marker than an English one does. That is the right
  trade — a name they can read beats a name they cannot — but it is a real change and
  worth knowing about.
- **Arabic carries six plural forms**, so each new counted entry costs `ar` six empty
  `msgstr[n]` slots rather than two. Its untranslated count is the same as everyone
  else's; the work behind it is not.

---

## What a reviewer should look at first

`src/llamafit/cli/render.py` — the shape of `_memory_details`, `_gpu_specs` and `_cores` is
the pattern the whole change argues for, and if that pattern is wrong then a lot of this is
wrong with it. Then `docs/translations.md`, which now writes the two rules down so the next
person does not have to rediscover them from six independent bug reports.
