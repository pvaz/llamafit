# Translations

LlamaFit's messages are written in English. Any of them can be translated, and the
translation is read at runtime from a plain text file in this repository.

The format is GNU gettext, the one every translation tool already speaks. There is no new
dependency and no compiled `.mo` file: the `.po` file a translator edits is the exact file
the program reads.

| Where | What |
|---|---|
| `src/llamafit/data/locale/messages.pot` | the template: every message, translated by nobody |
| `src/llamafit/data/locale/<language>.po` | one catalog per language, for example `pt_PT.po` |
| `scripts/gen_messages.py` | reads the sources and rewrites the template |
| `llamafit.i18n` | the two functions the code calls, and the language choice |

## What LlamaFit speaks today

| Language | Catalog |
|---|---|
| English | none needed; it is the language the messages are written in |
| Portuguese (Portugal) | `pt_PT.po` |

## How the language is chosen

Most explicit first:

1. the `--language` option;
2. the `LLAMAFIT_LANGUAGE` environment variable;
3. the operating system's locale;
4. English.

Both `pt` and `pt_PT` are accepted. A request without a region takes the most specific
catalog for that language, so `pt` gets `pt_PT`. A request with a region takes the
region-less catalog when there is one, so `pt_PT` would get `pt`. One region never stands
in for another: `pt_BR` gets English, not `pt_PT`, because European and Brazilian
Portuguese are different translations and quietly serving the wrong one is worse than
serving English.

Asking for a language LlamaFit does not have falls back to English **and says so**, once.
An operating system set to a language LlamaFit does not have simply gets English: that is
not somebody asking, and repeating it on every run would be noise.

Where the operating system's locale comes from differs by platform. On Linux and macOS it
is `LANGUAGE` (a colon-separated list of preferences), then `LC_ALL`, `LC_MESSAGES` and
`LANG`. On Windows those are normally unset, so LlamaFit asks for the user interface
language through `GetUserDefaultUILanguage` and turns the identifier into a tag with the
standard library's table; `locale.getlocale()` is no use there, because it answers
`Portuguese_Portugal`, which nothing can resolve.

## Adding a language

1. Regenerate the template so it matches the code:

   ```
   python scripts/gen_messages.py
   ```

2. Copy it to your language's file. The name is the tag, `ll` or `ll_CC`, with the
   language lower-case and the region upper-case:

   ```
   cp src/llamafit/data/locale/messages.pot src/llamafit/data/locale/fr_FR.po
   ```

3. Fill in the header. `Language`, `Language-Team` and `Plural-Forms` are all required,
   and `Content-Type` must say UTF-8:

   ```
   "Content-Type: text/plain; charset=UTF-8\n"
   "Language: fr_FR\n"
   "Language-Team: French (France)\n"
   "Plural-Forms: nplurals=2; plural=(n > 1);\n"
   ```

   The plural rule for your language is the one your locale already publishes; the GNU
   gettext manual lists them. It decides which `msgstr[n]` a count selects, and English's
   rule is wrong for most languages: French puts zero in the singular, Polish has three
   forms, Japanese has one.

4. Translate. Leave a `msgstr` empty rather than guessing: an empty translation falls back
   to the English message, so an unfinished catalog degrades to English and never shows a
   blank line.

5. Run the tests. `tests/unit/test_i18n_catalogs.py` checks every catalog: it must parse,
   it must declare its plural rule, and it must not translate a message the template does
   not have. A message it is *missing* is a warning, not a failure, so you can land a
   translation that is still in progress.

## What a translator needs to know

- **Placeholders keep their names.** `%(count)d module` may become
  `%(count)d módulo`, and the pieces may be reordered, but `%(count)d` itself must survive
  exactly. A placeholder that is dropped or renamed is a crash, not a typo.
- **Plurals are not a suffix.** Use `msgstr[0]`, `msgstr[1]`, and as many forms as your
  `Plural-Forms` header declares. Do not write `1 module(s)`.
- **The file is UTF-8**, always, on every platform. Save it as UTF-8 without a byte order
  mark.
- **Strings can be split over adjacent lines** for readability; the reader joins them with
  nothing in between, so keep the trailing spaces where they belong.
- **Escapes** are the usual ones: `\\`, `\"`, `\n`, `\t`, `\r`, `\a`, `\b`, `\f`, `\v`.
  Anything else is refused with the line number.
- **`msgctxt` is not supported.** Nothing here writes one, so a catalog with one is
  reported rather than half-read.
- **Style follows the English.** A message is a short sentence with a subject and a verb; a
  hint is an action the reader can take. Command names, flags, file paths, `llama.cpp`,
  `numpy` and the like are not translated.
- **Keep the meaning, not the word order.** If your language wants the number at the end of
  the sentence, put it at the end.

## The rule about machine translation

**An unreviewed machine translation is not accepted.** A catalog is only added when
somebody who reads the language has read every line of it. This is the same rule the
project applies to every other number and fact it ships: nothing goes in that nobody has
checked. A partly finished catalog that a person has read is welcome; a complete one that
nobody has is not.

## Running the extractor

```
python scripts/gen_messages.py
```

It reads the syntax tree of every source, finds each call to `_()` and `ngettext()`, and
rewrites `messages.pot`. Give it paths to read something else.

It refuses to write anything when a call passes something that is not a literal string:

```
src/llamafit/cli/render.py:41: argument 1 is not a literal string, so it cannot be
extracted or translated
```

That is a bug in the calling code, not in the extractor. A message built at runtime can
never reach a translator, so it has to become a literal with a placeholder:

```python
# no: the extractor cannot see this, and no translator can ever fix it
_(f"{count} modules")

# yes
ngettext("%(count)d module", "%(count)d modules", count) % {"count": count}
```

`tests/unit/test_i18n_catalogs.py` fails when the committed template is out of date, so
regenerate it in the same commit as the code that changed a message.

The translation machinery under `src/llamafit/i18n/` is not extracted from itself. Its own
messages stay in English on purpose: a sentence saying LlamaFit does not speak a language
cannot be written in the language it does not speak.

## Calling the two functions

```python
from llamafit.i18n import _, ngettext

_("No GPU detected")
ngettext("%(count)d module", "%(count)d modules", count) % {"count": count}
```

Both take literal strings so the extractor can find them. Use named placeholders
(`%(count)d`) rather than positional ones, so a translator can move them.

Translation happens when the function is called, not when the module is imported. A
message stored in a module-level constant is translated once, in whatever language was
active at import time, which is usually English. Call the function where the message is
used.

An interface chooses the language once, at start-up, and says so when the request could
not be met:

```python
from llamafit.i18n import set_language

choice = set_language(language_option)
if choice.notice:
    console.print(choice.notice)
    if choice.hint:
        console.print(choice.hint)
```
