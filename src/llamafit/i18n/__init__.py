"""Translation: choose the user's language and read its catalog at runtime.

Call sites use two functions and nothing else::

    from llamafit.i18n import _, ngettext

    _("No GPU detected")
    ngettext("%(count)d module", "%(count)d modules", n) % {"count": n}

An interface chooses the language once at start-up::

    choice = set_language(language_option)
    if choice.notice:
        console.print(choice.notice)

The catalogs are GNU gettext ``.po`` files under ``llamafit/data/locale/``, read as text.
Nothing is compiled and no binary is committed, so a translator edits the same file the
program reads. Every message that is missing or left empty falls back to English.
"""

from __future__ import annotations

from llamafit.i18n.catalogs import (
    CATALOG_SUFFIX,
    TEMPLATE_NAME,
    available_languages,
    catalog_dir,
    catalog_path,
    load_language,
)
from llamafit.i18n.detect import FixedLocale, LocaleProvider, SystemLocale, windows_ui_language
from llamafit.i18n.plurals import (
    DEFAULT_PLURAL_FORMS,
    PluralFormsError,
    PluralRule,
    parse_plural_forms,
)
from llamafit.i18n.po import Message, PoCatalog, PoSyntaxError, parse_po, read_po
from llamafit.i18n.select import LANGUAGE_ENV_VAR, LanguageChoice, resolve_language
from llamafit.i18n.tags import SOURCE_LANGUAGE, match, normalise
from llamafit.i18n.translator import (
    CatalogTranslator,
    EnglishTranslator,
    Translator,
    _,
    current_language,
    get_translator,
    gettext,
    ngettext,
    reset,
    set_language,
    set_translator,
)

__all__ = [
    "CATALOG_SUFFIX",
    "DEFAULT_PLURAL_FORMS",
    "LANGUAGE_ENV_VAR",
    "SOURCE_LANGUAGE",
    "TEMPLATE_NAME",
    "CatalogTranslator",
    "EnglishTranslator",
    "FixedLocale",
    "LanguageChoice",
    "LocaleProvider",
    "Message",
    "PluralFormsError",
    "PluralRule",
    "PoCatalog",
    "PoSyntaxError",
    "SystemLocale",
    "Translator",
    "_",
    "available_languages",
    "catalog_dir",
    "catalog_path",
    "current_language",
    "get_translator",
    "gettext",
    "load_language",
    "match",
    "ngettext",
    "normalise",
    "parse_plural_forms",
    "parse_po",
    "read_po",
    "reset",
    "resolve_language",
    "set_language",
    "set_translator",
    "windows_ui_language",
]
