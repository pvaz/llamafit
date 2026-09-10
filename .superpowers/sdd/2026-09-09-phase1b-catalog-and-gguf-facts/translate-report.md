# Catching six catalogs up: what was translated, what was not, and what the English cost

Branch `fix/translate`, seven commits off `main`: one per language, and this report.
Nothing was merged, rebased or pushed, and no other branch or worktree was touched.
The only code or data that changed is `src/llamafit/data/locale/`.

| | |
|---|---|
| `c61d6c9` | `feat(i18n): translate the new messages into European Portuguese` |
| `f110ff5` | `feat(i18n): translate the new messages into Spanish` |
| `c99c720` | `feat(i18n): translate the new messages into French` |
| `4a62357` | `feat(i18n): translate the new messages into German` |
| `320c4cf` | `feat(i18n): translate the new messages into Italian` |
| `1dedb04` | `feat(i18n): translate the new messages into Brazilian Portuguese` |

Verification: 2885 passed, 2 skipped, 14 deselected; coverage 96.61% (floor 85%). `ruff
check`, `ruff format --check` and `mypy` all clean. `check_po.py` reports **0 problems**
on each of the six, and the translated count went up in every one of them:

| Catalog | Was | Is | Of |
|---|---|---|---|
| `pt_PT.po` | 244 | **1018** | 1020 |
| `es.po` | 245 | **1019** | 1020 |
| `fr.po` | 245 | **1019** | 1020 |
| `de.po` | 245 | **1019** | 1020 |
| `it.po` | 245 | **1019** | 1020 |
| `pt_BR.po` | 245 | **1019** | 1020 |

774 messages were translated in each of the six. The other thirty-one catalogs are
untouched and fall back to English, which is what the brief asked for and what an
unreviewed guess into a language nobody here reads would not have been.

Every catalog still says at the top that no native speaker has read it. That line was not
edited in any of the six, and `docs/translations.md` still records all thirty-seven as
**not yet** reviewed, because that is still true.

## What the files look like now

Each of the six was rewritten in the template's order, the way `msgmerge` would leave it:
the 269 entries that were already there keep their translations and their translator
comments byte for byte, the 751 the template has and the catalog did not are added in
place, and the whole file now holds all 1020 template entries rather than only the ones
somebody had reached. That is why the diffs are large — between 3,100 and 3,300 changed
lines each — and why almost none of it is a changed translation. A script asserted the
invariant directly: every entry the committed file translated says the same thing in it,
and every `#` comment line the committed file carried is still in it. Zero differences on
all six.

Line endings stayed LF, there is no byte order mark, and the no-break space European
Portuguese and French use as a thousands separator survived the rewrite (checked as a
byte, not by eye).

---

## 1. What was left untranslated, and why

**`any` — left in English in all six.** `src/llamafit/web/strings.py:345` registers it as
`app.any`, and nothing in `src/llamafit/web/static/app.js` ever asks for that key. Its
sibling `app.none` is used, at `app.js:335`, for a table with no rows. `any` is not, so
the source does not say what it qualifies, and every one of these six languages has to
know: *cualquiera* is not *cualquier*, *qualquer* has to agree with something, and French
and Italian have to pick a gender before they can write the word at all. A wrong agreement
in a dropdown is worse than the English word, so the entry is empty and falls back.

The fix is in the code, not the catalog: either the page starts using the key, in which
case its use site settles the question, or it gets a `msgctxt` naming the field it labels,
or it is deleted. Any of the three makes it translatable in one line.

**`Skip the RAM bandwidth measurement.` in `pt_PT` only — left as it was found.** That
entry is deliberately empty, with a comment above it saying so, and
`tests/unit/test_i18n_translator.py` proves the English fallback through it. Filling it in
would have broken the suite for the right reason. It stays empty; the other five catalogs
already translated it in an earlier round and were not touched.

**Everything else was translated.** For each of the 774 I could reach a use site that
settled the reading — `_status_text` for *already here* and *to fetch*, `render_calibration`
for the bare `from` column, `_download_plan_table` for `Role`, `_rungs_table` for `Card
needs`. Where the reading needed a decision rather than a lookup, the entry carries a `#`
comment saying which decision was taken: ten of them in every one of the six, plus one
in German and one in European Portuguese. Section 4 lists them.

---

## 2. Which of the new English is hard to translate, and why

The earlier round's report (`english-report.md`) named two rules a translator cannot work
around: a fragment joined in Python, and a word that has to stay in step with another
entry nothing links it to. Both are back in the new text.

### A column named in prose, with nothing keeping it in step with the heading

Four of the new messages name a column by its heading in the middle of a sentence:

- `the How column says which each row is` (`tui/summary.py`, `cli/render_board.py`)
- `compares the free memory it sees against the card column` (`cli/render_board.py:394`)
- `the context column is the largest each one holds` (three messages, `board_cmd.py:231`
  and `render_board.py:889`/`899`)

The headings themselves are separate entries — `pgettext("column heading", "How")`,
`"Card"`, `"Ctx"` — and nothing makes the two agree. This is exactly the caption defect the
English round fixed by handing `%(heading)s` in as a value, and these four did not get that
treatment. I translated each sentence with the same word I had used for its heading, so
they agree today in all six languages; nothing but that care keeps them agreeing tomorrow.
`the context column` is already out of step in the English itself, where the heading reads
`Ctx`.

### A verdict word that the sentence beside it does not repeat

`verdict_label("comfortable")` is `roomy`; `verdict_sentence("comfortable")` opens
`Comfortable:`. Two entries, two different English words for one state, and a translator
has to guess which one the reader will match against. The other four pairs do agree
(`fits`/`Fits:`, `tight`/`Tight:`, `pages`/`Pages:`, `no room`/`No room:`), which makes the
odd one out read like an oversight rather than a choice. I made the sentence repeat the
column word in every language — `folgado`/`Folgado:`, `holgado`/`Holgado:`,
`à l'aise`/`À l'aise :`, `reichlich`/`Reichlich:`, `comodo`/`Comodo:` — because the reader
meets the short form in the table first. That is a deliberate departure from the English,
and it is the one place where the six catalogs say something the English does not.

### A list joined in Python

`_simulation_note` in `cli/render.py:182` builds `fields` with `", ".join(...)` out of
three *translated* labels — VRAM, system memory, CPU cores — and interpolates the joined
string into a counted message. The separator is in the code where no catalog can reach it,
and Japanese wants `、` and Arabic `، `. It is the same defect `docs/translations.md`
writes up, one file away from the `_memory_details` and `_from_table` shapes that are held
up as the fix. Nothing a translator can do; noted, not worked around.

### Abbreviations cut to English word-boundaries

`Gen/s`, `PP/s`, `gen tok/s`, `prompt tok/s`, `Ctx`, `Qual`, `#`. A column three
characters wide is an English abbreviation of an English word, and the shortening does not
survive into a language whose word starts differently. I left `Gen/s`, `PP/s`, `gen tok/s`
and `prompt tok/s` as they are in all six, with a comment saying so, because they are what
`llama-bench` itself prints and a reader meets them in its output before meeting our table.
`Ctx` I left as well. `Qual` becomes `Qual.` in five and `Cal.` in Spanish; `#` stays `#`,
though French would ordinarily write `N°` and German `Nr.`.

### Two elliptical English headings

`Have` (a verb with no subject) and `Why not` (an elliptical question) are grammatical in
an English table and not in any of the six. `Have` became *on disk* everywhere, which is
what the cell actually answers and matches the key-bar label beside it. `Why not` became a
noun: *Motivo* / *Raison* / *Grund*.

### Two bare words with no context

`none` is read in two unrelated places with no `msgctxt` to tell them apart: the
`Backends found in it:` line of `llamafit install llama.cpp`, and the web dashboard's word
for a table with no rows. One `msgstr` has to serve both, which is the case
`docs/translations.md` says a context exists for. It happens to work in these six — the
form the Backends line wants also reads as *none at all* on its own — but that is luck,
not design, and a language that inflects harder would be stuck. There is a `#`
comment on the entry in all six saying so. `any` is the same class and is the one I did not
translate.

### Prose that is genuinely hard, and was translated anyway

- `for one of those this term is that much too slow` (`speed/estimate.py:277`). The
  `that much` refers back to *about three times* two clauses earlier. Rendered as *exactly
  that much too slow* in Portuguese and Italian, *d'autant trop lent* in French, *genau um
  so viel zu langsam* in German. Worth a second reader.
- `it has only not been checked` (`bench/paging.py:161`) and *cleared of paging* — the
  legal sense of *cleared*. Rendered with *ilibada* / *scagionata* / *blanchie* /
  *freigesprochen*, which keeps the register.
- `llama-server did not become healthy` (`bench/run.py:917`). *Healthy* here is the
  `/health` endpoint, not a metaphor. Rendered as *never became operational* rather than a
  literal *healthy*, which loses the link to the endpoint name; the two neighbouring
  messages do mention `/health`, so the reader can still make it.
- `pass it to mean it` (`web/server.py:139`). Rendered as *pass it deliberately*.

---

## 3. Did the new text assume it would only ever be read in English?

Four places, in ascending order of how much it matters.

**A trail through the Windows user interface, given in English.**
`llamacpp/install.py:1258` says `remove the entry in System > About > Advanced system
settings > Environment Variables`. Those are the words English Windows shows. A reader on
Portuguese Windows is looking for *Definições avançadas do sistema*, and on German Windows
for *Erweiterte Systemeinstellungen*. I translated the trail into each language's own
Windows wording, with a `#` comment saying why, because the sentence is a set of
directions and directions that name the wrong buttons are worse than useless. This is the
one message where I changed a string the English clearly meant as literal, and a reviewer
should agree with it or say so.

**`Ctrl+C`, twice** (`cli/serve_cmd.py:84`, `web/server.py:75`). German keyboards label
that key `Strg`. Both German messages say `Strg+C`, with a comment; the other four
languages label it `Ctrl` and keep it. Same reasoning: the reader is being told which key
to press.

**A double hyphen used as a dash.** `web/server.py:75` writes `-- your hardware, your
disks`. In every language that renders as a stray flag prefix in the middle of a sentence,
and `docs/translations.md` already explains at length that a leading hyphen has no
direction of its own and reorders inside a right-to-left paragraph. I kept the `--` so the
six catalogs match the English exactly; an em dash in the source would be better for
everyone, the three RTL catalogs most of all. Note that the neighbouring
`verdict_sentence("too-tight")` in `render_board.py` already uses a real em dash, so the
project is not consistent with itself here.

**A section number and a phase number as a citation.** `section 10.1`, `section 10.2`,
`section 10`, `Section 10.2` and `phase 3` appear inside six of the new messages. They point
at a design document that exists only in English. The words around them translate; what
they cite does not. Nothing to do in a catalog, but a reader who follows the pointer lands
somewhere they may not be able to read.

**One outright defect, not a language assumption but found the same way.**
`board_cmd.py:68` — `Capability the model must have; a model without it is excluded ` —
ends in a space where every other option's help on that command ends in a full stop. All
six translations end in the full stop, with a comment on the entry saying the English does
not. Fixing the English is a one-character change that will orphan six translations, so it
should happen in a round that regenerates the template.

---

## 4. Decisions a reviewer should check first

Each of these is written into the catalogs as a `#` comment above the entry, so nobody has
to come back to this file to find them.

| Entry | Decision |
|---|---|
| `gen tok/s`, `prompt tok/s`, `Gen/s`, `PP/s` | left as `llama-bench` prints them, in all six |
| `flags` / `Flags` | left as the command line's own word, in all six |
| `Chat completions` | left as the OpenAI-compatible API's own name for the endpoint |
| `llama.cpp` (a web label) | left as the project spells itself |
| `B` (parameter count) | untouched; it was already answered and the reason is already in the file |
| ` · ` (dashboard separator) | the same middle dot, spaces and all, in all six |
| `none` | one word serving two places that want a context |
| `Capability the model must have…` | the missing full stop is supplied |
| `balanced`, `quality`, `speed`, `comfortable`, `fits`, `tight` | kept English inside help text: they are the values the option takes |
| Windows settings trail | given in each language's own Windows wording |
| `Strg+C` (German only) | the key as a German keyboard labels it |

Two more that are not in the files:

**European Portuguese cannot use `arquivo` for an archive.**
`tests/unit/test_i18n_catalogs.py` reserves that word as the Brazilian spelling of
*ficheiro*, and rightly — but *arquivo* is also the ordinary European word for a `.zip`,
which is what nine of the installer's new messages are about. Using it would have made one
word carry two meanings in the same file and failed the variety test at the same time, so
`pt_PT` calls the release asset a *pacote* throughout, matching `pt_BR`. There is a comment
on the `Archive` row saying so. Anyone tempted to "correct" it back will break the build.

**The board's short verdict column is called *space*, not *fit*.** `Fit`, the `fit` score
part and `Fit on this machine` all became *espaço* / *espacio* / *place* / *Platz* /
*spazio*, because the five values under that heading — roomy, fits, tight, pages, no room —
are all about room, and none of these languages has a one-word noun for *fit* that a reader
would match to them. It is the one heading where I chose a different metaphor from the
English rather than a different word.

---

## What is still wrong and was left alone

- The thirty-one other catalogs are 775 messages behind. That is deliberate.
- `%(fields)s` is still a Python-joined list (section 2).
- Four sentences still name a column heading in prose without being handed it (section 2).
- `roomy` and `Comfortable:` still disagree in the English (section 2).
- `any` is still an unused, contextless entry in `web/strings.py` (section 1).
- The `--` in `web/server.py:75` is still a double hyphen (section 3).
- The trailing space in `board_cmd.py:68` is still there (section 3).

None of these was fixed here because the brief was the catalogs and only the catalogs, and
because four of the seven change a `msgid`, which orphans work already done in
thirty-seven files. They belong in a round that regenerates the template.
