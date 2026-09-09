# Translations

LlamaFit's messages are written in English. Any of them can be translated, and the
translation is read at runtime from a plain text file in this repository.

> **Status: the interface goes through the layer.** Every string `llamafit` prints is
> wrapped, `--language` is a global option, and the language is chosen once at start-up from
> the option, the environment, the operating system's locale, then English. What is *not* a
> claim is that any given language is finished or reviewed: the table below says which
> catalogs exist and which have had a second reader.
>
> Three kinds of text stay English on purpose, and each says why where it lives: the
> translation layer's own messages (`src/llamafit/i18n/`, below), the per-field diagnostics
> `catalog validate` and `catalog refresh` print about a YAML file, and identifiers — command
> names, flags, paths, backend names, capability and use-case ids, and the catalog's own
> data.

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

| Language | Catalog | Reviewed by a native speaker |
|---|---|---|
| English | none needed; the messages are written in it | — |
| Portuguese (Portugal) | `pt_PT.po` | **not yet** |

`pt_PT.po` was written alongside the machinery that reads it and grew with the sweep that
wrapped the interface, and it has had no second reader. By the standard this page sets
below, that is not enough, and the file says so at the top. It ships because a first
catalog is what makes everything else testable, not because it has met the bar. If you
read European Portuguese, going through it line by line is the most useful contribution
you can make here.

## How the language is chosen

This is what `set_language` does, and the command-line interface calls it once at
start-up, from the `--language` callback. The option is eager, so it is read before Click
renders anything and `--language pt_PT --help` comes out in Portuguese too. Most explicit
first:

1. the `--language` option;
2. the `LLAMAFIT_LANGUAGE` environment variable;
3. the operating system's locale;
4. English.

Both `pt` and `pt_PT` are accepted. A request without a region takes the most specific
catalog for that language, so `pt` gets `pt_PT`. A request with a region takes the
region-less catalog when there is one, so `pt_PT` would get `pt`.

One region's catalog will stand in for another's when that is all there is: `pt_BR` gets
`pt_PT`. European and Brazilian Portuguese are not the same translation, but the
difference is a long way short of the difference between Portuguese and English, so
LlamaFit serves what it has and **says once** which variety the reader is getting:

```
LlamaFit has no pt_BR translation, so it is using the Portuguese (Portugal) one.
Contribute a pt_BR catalog: docs/translations.md says how.
```

The name in that sentence comes from the catalog's own `Language-Team` header, which is
why the header is required.

Asking for a language LlamaFit does not have at all falls back to English **and says so**,
once. An operating system set to such a language simply gets English with no remark: that
is not somebody asking, and repeating it on every run would be noise. A substitution is
different, and is announced whatever asked for it — including the operating system —
because the reader deserves to know why some of the wording looks foreign.

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

   Spell it another way — `fr-FR.po`, `FR_fr.po` — and LlamaFit still finds it, because it
   offers a language and then opens it by the same rule. Use the canonical name anyway: a
   catalog that ships has to be named for its tag exactly, and the test suite says so.

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

   `Plural-Forms` is required in the strong sense: a catalog without one is **refused**
   with the line number, rather than quietly given English's rule. Inheriting English's
   rule is the one mistake that produces no error at all — a three-form language would
   simply pick the wrong form, and its reader would meet real words in the wrong grammar
   with nothing anywhere to say why.

   `Language` has to agree with the file's name, and a `fr_FR.po` whose header says
   `Language: de` is **refused** too. Nothing downstream would notice otherwise: the tag
   comes from the file's name, so the catalog would be installed as French and would then
   answer in German with no error anywhere. Spelling is forgiven — `fr-FR`, `FR_fr` and
   `fr_FR.UTF-8` all mean `fr_FR` — and a catalog that has not filled the header in yet is
   still read, because saying nothing is not the same as saying something false.

4. Translate. Leave a `msgstr` empty rather than guessing: an empty translation falls back
   to the English message, so an unfinished catalog degrades to English and never shows a
   blank line. A `msgstr` holding only spaces, tabs or newlines counts as empty too, and
   the completeness check still lists it as a message your language needs.

5. Run the tests. `tests/unit/test_i18n_catalogs.py` checks every catalog: it must parse,
   it must declare its plural rule and the right language, it must not translate a message
   the template does not have, and nothing in it may be a translation the reader had to
   drop. A message it is *missing* is a warning, not a failure, so you can land a
   translation that is still in progress.

   Your own catalog, in your own copy, is held to the same rules without the build: the
   reader drops what it cannot use, shows the English there, and tells you how many and
   where. Running the tests is how you turn that into a failure instead.

## What a translator needs to know

- **Placeholders keep their names.** `%(count)d module` may become
  `%(count)d módulo`, and the pieces may be reordered, but `%(count)d` itself must survive
  exactly. A translation whose placeholders do not match its message's is **not used** —
  LlamaFit shows the English instead and says so once at start-up, and `--verbose` names
  each one in the log. For a catalog that ships, the same mismatch fails the build.
- **A literal percent sign is written `%%`**, and **placeholders are named, never
  positional**. A bare `%s` looks like it works and does not: filled from a dictionary it
  puts the dictionary itself into the sentence rather than the number.
- **Plurals are not a suffix.** Use `msgstr[0]`, `msgstr[1]`, and as many forms as your
  `Plural-Forms` header declares. Do not write `1 module(s)`.
- **The file is UTF-8**, always, on every platform. A byte order mark at the start is
  read and ignored, so a Windows editor that adds one has not broken anything;
  without one is still the tidier file.
- **Strings can be split over adjacent lines** for readability; the reader joins them with
  nothing in between, so keep the trailing spaces where they belong.
- **Escapes** are the usual ones: `\\`, `\"`, `\n`, `\t`, `\r`, `\a`, `\b`, `\f`, `\v`.
  Anything else is refused with the line number.
- **A `msgctxt` says where the message is used**, and it changes what you translate. See
  below.
- **A `#.` line above an entry is a note from the code**, written for you by whoever wrote
  the call. It says what the message alone cannot: that an entry is punctuation rather than
  prose, say. Read those first; there are not many.
- **Style follows the English.** A message is a short sentence with a subject and a verb; a
  hint is an action the reader can take. Command names, flags, file paths, `llama.cpp`,
  `numpy` and the like are not translated.
- **Keep the meaning, not the word order.** If your language wants the number at the end of
  the sentence, put it at the end.

### Three entries are punctuation, not words

Numbers are written the English way and then repunctuated from the catalog, so a language
says how it writes a number by translating three entries rather than by anyone shipping a
locale database:

| Context | English | What to write |
|---|---|---|
| `thousands separator` | `,` | What goes between groups of three digits: `32,768`, `32.768` |
| `decimal separator` | `.` | What goes before the fraction: `127.8`, `127,8` |
| `parameter count` | `B` | The abbreviation for a thousand million, as in `27B` |

This is not cosmetic. English writes a model's context as `32,768`; a reader whose language
groups with a point reads that as a fraction and is told the model holds thirty-two tokens.

Two warnings the `#.` notes repeat. **Do not translate the word *billion*:** the long and
short scales disagree about what one is, so write the abbreviation your readers expect for
a thousand million. And **an empty translation means untranslated**, so it falls back to the
English character rather than to nothing.

There is a limitation here, and it is ours rather than yours: a translation holding only a
space counts as empty, so a language that groups digits with a space — French, Russian,
Swedish, Polish and others — cannot say so yet and silently gets the English comma. Please
open an issue rather than working around it; the fix belongs in the reader.

### When a message has a context

Some entries carry a `msgctxt` line above the `msgid`:

```
msgctxt "GPU"
msgid "none detected"
msgstr "nenhuma detetada"

msgctxt "backends"
msgid "none detected"
msgstr "nenhum detetado"
```

A context appears when the same English words are used in two places that your language
may not translate the same way. Here `none detected` is the value of the GPU row, where
the Portuguese noun is feminine, and of the Backends row, where it is masculine. One
`msgstr` could only ever be right in one of the two, so each row asks its own question.

What the context tells you is **where the message is read**, nothing more. It is never
shown to the user, it is never translated, and you must never change it: the context and
the `msgid` together are what the program looks the entry up by, so an edited context
means the program finds nothing and shows English.

Two entries with the same `msgid` and different contexts are two different entries, and
so are an entry with a context and an entry without one. If your language uses the same
words in both, write the same words in both — that is a fine answer, and English does
exactly that. Leave one empty and that row falls back to English while the other does not,
which is the one outcome to avoid.

The extractor writes the context; you never add one. If you find a message where a
context would help and there is none, say so in an issue: the fix belongs in the code
that asks for the message, and it needs to be made once for every language.

## The rule about machine translation

**An unreviewed machine translation is not accepted.** A catalog counts as reviewed when
somebody who reads the language has read every line of it. This is the same rule the
project applies to every other number and fact it ships: nothing is presented as checked
that nobody has checked. A partly finished catalog that a person has read is welcome; a
complete one that nobody has is not.

The rule is about the *claim*, not only about the file. `pt_PT.po` is in the repository
and is not yet reviewed — and it says so, at the top of the file and in the table above,
because the failure this rule exists to prevent is a confident claim with nothing behind
it. Ship what you have, label it honestly, and ask for a reader.

## Running the extractor

```
python scripts/gen_messages.py
```

It reads the syntax tree of every source, finds each call to `_()`, `ngettext()`,
`pgettext()`, `npgettext()` and their four `lazy_` counterparts, and rewrites
`messages.pot`. Give it paths to read something else.

### Leaving a note for the translator

A comment block touching a call, whose first line opens with `Translators:`, is copied into
the template as `#.` lines:

```python
# Translators: this is punctuation, not prose. It is what your language puts
# between a number's groups of three digits: English writes 32,768.
return pgettext("thousands separator", ",")
```

Only a marked block is copied, so a comment written for whoever maintains the code stays in
the code. A blank line, or any code, ends the block. Use it when the message alone cannot
tell a translator what to do — and prefer fixing the message when it can.

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

## Calling the functions

```python
from llamafit.i18n import _, ngettext

_("No GPU detected")
ngettext("%(count)d module", "%(count)d modules", count) % {"count": count}
```

All of them take literal strings so the extractor can find them. Use named placeholders
(`%(count)d`) rather than positional ones, so a translator can move them.

### A word two rows share needs a context

Wrapping the same English word in two places gives a translator one entry to answer for
both. That is fine when the two really are the same sentence, and wrong the moment a
language inflects: `none detected` is the GPU row's value, where the Portuguese noun is
feminine, and the Backends row's, where it is masculine, and one `msgstr` cannot be right
in both.

`pgettext` and `npgettext` take a context first — a short word naming where the message is
read, written for the translator and never shown to a user:

```python
from llamafit.i18n import npgettext, pgettext

pgettext("GPU", "none detected")
pgettext("backends", "none detected")
npgettext("GPU", "%(count)d device", "%(count)d devices", count) % {"count": count}
```

`lazy_pgettext` and `lazy_npgettext` are the deferred pair, for anything built at import
time.

The context is part of what identifies the message, so the two calls above are two
entries in the template and two questions to the translator. Name the row, the column or
the screen, not the grammar: a translator needs to know *where the words are read*, and
which languages inflect for what is their business, not ours.

Reach for a context whenever an English word is short enough that two places could share
it — a value in a table, a status, a unit. Do not wrap the same bare word twice without
one and hope.

### Anything built at import time needs the deferred pair

`_()` translates at the moment it is called. A module-level constant, a dictionary of
hints, or an argument to a decorator is built while the module is imported — before any
interface has chosen a language — so `_()` there freezes the message in English for the
life of the process, silently. Nothing raises, the interface simply comes out half
translated, and nobody goes looking.

`lazy_gettext` and `lazy_ngettext` defer the lookup to the moment the text is rendered:

```python
from llamafit.i18n import lazy_gettext

# built at import time, translated when it is shown
PROBE_HINTS = {
    "nvidia-smi": lazy_gettext("Install or repair the NVIDIA driver."),
}
```

The result is a `LazyString`, not a `str`. It renders, formats, compares, sorts,
concatenates, indexes and forwards every string method to the text it currently resolves
to, so it works in an f-string, in `%` formatting, and anywhere a string method is called
on it. It deliberately is **not** a `str` subclass: one of those would carry a frozen
English buffer that C-level fast paths such as `str.join` would use in preference to any
override, producing another silent English message. A separate type raises `TypeError`
instead, at the call site — where wrapping it in `str()` is the right fix anyway, because
that call site is the render moment:

```python
", ".join(str(hint) for hint in hints)
```

The extractor reads `lazy_gettext` and `lazy_ngettext` exactly like the eager pair, so a
deferred message reaches the template like any other.

### At a Typer boundary

A deferred message works as a Typer `help=` string: the decorator runs at import time, the
language is chosen afterwards, and the help still comes out translated.
`tests/unit/test_i18n_lazy.py` asserts that rather than assuming it, so the suite says so
if a future Typer or Click stops rendering help late.

Typer declares `help` as `str | None`, though, so mypy rejects a `LazyString` there. Say
what you mean at the call site:

```python
from typing import cast

no_measure: bool = typer.Option(
    False,
    "--no-measure",
    # Typer only ever renders this, so a lazy message is safe here; it is not a str,
    # and the cast is what says so out loud.
    help=cast(str, lazy_gettext("Skip the RAM bandwidth measurement.")),
)
```

A command's help text is the exception nothing can fix: Typer takes it from the function's
docstring, and a docstring is a literal, not a call, so no wrapper can reach it. A command
whose help has to be translatable needs an explicit `help=` argument on the decorator.

An interface chooses the language once, at start-up, and says so when the request could
not be met:

```python
from rich.text import Text

from llamafit.i18n import set_language

choice = set_language(language_option)
if choice.notice:
    console.print(Text(choice.notice))
    if choice.hint:
        console.print(Text(choice.hint))
```

`Text(...)`, not the bare string. A substitution notice quotes the catalog's own
`Language-Team` header, which is data read from a file, and `console.print` parses square
brackets as Rich markup: a catalog whose team read `Portuguese [Brazil]` would raise at
start-up, before the program had done anything. Wrapping the sentence in `Text` says it is
text and not markup, which is the rule everywhere in this project that prints a value it
did not write itself.
