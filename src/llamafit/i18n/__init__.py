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

The catalogs are GNU gettext ``.po`` files under ``llamafit/data/locale/``, read as text.
Nothing is compiled and no binary is committed, so a translator edits the same file the
program reads. Every message that is missing, or left empty or blank, falls back to
English.
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
    "catalog_dir",
    "catalog_path",
    "current_language",
    "get_translator",
    "gettext",
    "is_substitution",
    "lazy_gettext",
    "lazy_ngettext",
    "lazy_npgettext",
    "lazy_pgettext",
    "load_language",
    "match",
    "ngettext",
    "normalise",
    "npgettext",
    "parse_plural_forms",
    "parse_po",
    "pgettext",
    "placeholders",
    "read_po",
    "reset",
    "resolve_language",
    "set_language",
    "set_translator",
    "substitution_notice",
    "windows_ui_language",
]
