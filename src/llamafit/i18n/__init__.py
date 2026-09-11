# LlamaFit. Copyright (C) 2026 Paulo Vaz.
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of LlamaFit; see LICENSE for the full terms and the warranty disclaimer.
"""Translation: choose the user's language and read its catalog at runtime.

Call sites use two functions::

    from llamafit.i18n import _, ngettext

    _("No GPU detected")
    ngettext("%(count)d module", "%(count)d modules", n) % {"count": n}

the same two with a context, for an English word that two places translate differently —
*none detected* is feminine in the GPU row and masculine in the Backends row, so one
Portuguese string cannot be right in both::

    from llamafit.i18n import npgettext, pgettext

    pgettext("GPU", "none detected")
    npgettext("GPU", "%(count)d device", "%(count)d devices", n) % {"count": n}

and a deferred set for anything built while a module is imported, which happens before a
language has been chosen::

    from llamafit.i18n import lazy_gettext

    PROBE_HINTS = {"nvidia-smi": lazy_gettext("Install or repair the NVIDIA driver.")}

An interface chooses the language once at start-up, and prints the notice as text rather
than as markup, because it carries a name read from a file::

    choice = set_language(language_option)
    if choice.notice:
        console.print(Text(choice.notice))

A fifth for the few entries that hold punctuation rather than prose, where a translation
of nothing but whitespace is the answer and not the absence of one -- the space French
puts between a number's groups of three digits::

    from llamafit.i18n import pgettext_literal

    pgettext_literal("thousands separator", ",")

Three of the languages are written right to left, and a Latin identifier inside one of
their sentences needs a directional isolate around it or a reader is shown ``verbose--``
where ``--verbose`` was meant. That belongs to whoever draws the screen, not to a
translator, so nothing about it appears in a catalog::

    from llamafit.i18n import for_display, isolate

    Text(for_display(_("%(path)s: %(free)s free") % {"path": isolate(path), ...}))

``llamafit.i18n.bidi`` has the whole story, and every function in it returns its input
unchanged for a left-to-right language.

None of that looks at where the words are going, and a file or a pipe on Windows takes
the system code page, which has no Japanese and no direction mark in it. An interface
that writes to a stream makes the stream forgiving first, and tells the choice what the
stream can write, so a language the stream cannot carry is refused with a reason rather
than raised over half-way down a table::

    from llamafit.i18n import encoding_of, tolerate

    tolerate(sys.stdout)
    choice = set_language(language_option, encoding=encoding_of(sys.stdout))

``llamafit.i18n.encoding`` argues both halves of that.

The catalogs are GNU gettext ``.po`` files under ``llamafit/data/locale/``, read as text.
Nothing is compiled and no binary is committed, so a translator edits the same file the
program reads. Every message that is missing, or left empty, falls back to English, and
so does a blank one for every lookup but ``pgettext_literal``.
"""

from __future__ import annotations

from llamafit.i18n.bidi import (
    FIRST_STRONG_ISOLATE,
    POP_DIRECTIONAL_ISOLATE,
    RIGHT_TO_LEFT_MARK,
    RTL_LANGUAGES,
    for_display,
    is_rtl,
    isolate,
    isolate_identifiers,
    mirror_justify,
    reading_order,
)
from llamafit.i18n.catalogs import (
    CATALOG_SUFFIX,
    TEMPLATE_NAME,
    available_languages,
    catalog_dir,
    catalog_path,
    load_language,
)
from llamafit.i18n.detect import FixedLocale, LocaleProvider, SystemLocale, windows_ui_language
from llamafit.i18n.encoding import (
    ERROR_HANDLER,
    can_write,
    clear_replacements,
    encoding_of,
    replacements,
    tolerate,
)
from llamafit.i18n.lazy import (
    LazyString,
    lazy_gettext,
    lazy_ngettext,
    lazy_npgettext,
    lazy_pgettext,
)
from llamafit.i18n.plurals import (
    DEFAULT_PLURAL_FORMS,
    PluralFormsError,
    PluralRule,
    parse_plural_forms,
)
from llamafit.i18n.po import (
    Message,
    MessageKey,
    PoCatalog,
    PoSyntaxError,
    parse_po,
    placeholders,
    read_po,
)
from llamafit.i18n.select import (
    LANGUAGE_ENV_VAR,
    LanguageChoice,
    resolve_language,
    substitution_notice,
)
from llamafit.i18n.tags import SOURCE_LANGUAGE, is_substitution, match, normalise
from llamafit.i18n.translator import (
    CatalogTranslator,
    EnglishTranslator,
    Translator,
    _,
    current_language,
    get_translator,
    gettext,
    ngettext,
    npgettext,
    pgettext,
    pgettext_literal,
    reset,
    set_language,
    set_translator,
)

__all__ = [
    "CATALOG_SUFFIX",
    "DEFAULT_PLURAL_FORMS",
    "ERROR_HANDLER",
    "FIRST_STRONG_ISOLATE",
    "LANGUAGE_ENV_VAR",
    "POP_DIRECTIONAL_ISOLATE",
    "RIGHT_TO_LEFT_MARK",
    "RTL_LANGUAGES",
    "SOURCE_LANGUAGE",
    "TEMPLATE_NAME",
    "CatalogTranslator",
    "EnglishTranslator",
    "FixedLocale",
    "LanguageChoice",
    "LazyString",
    "LocaleProvider",
    "Message",
    "MessageKey",
    "PluralFormsError",
    "PluralRule",
    "PoCatalog",
    "PoSyntaxError",
    "SystemLocale",
    "Translator",
    "_",
    "available_languages",
    "can_write",
    "catalog_dir",
    "catalog_path",
    "clear_replacements",
    "current_language",
    "encoding_of",
    "for_display",
    "get_translator",
    "gettext",
    "is_rtl",
    "is_substitution",
    "isolate",
    "isolate_identifiers",
    "lazy_gettext",
    "lazy_ngettext",
    "lazy_npgettext",
    "lazy_pgettext",
    "load_language",
    "match",
    "mirror_justify",
    "ngettext",
    "normalise",
    "npgettext",
    "parse_plural_forms",
    "parse_po",
    "pgettext",
    "pgettext_literal",
    "placeholders",
    "read_po",
    "reading_order",
    "replacements",
    "reset",
    "resolve_language",
    "set_language",
    "set_translator",
    "substitution_notice",
    "tolerate",
    "windows_ui_language",
]
