"""The deferred form: a message defined before a language is chosen must still obey it."""

from collections.abc import Iterator
from pathlib import Path

import pytest

from llamafit.i18n import translator
from llamafit.i18n.lazy import (
    LazyString,
    lazy_gettext,
    lazy_ngettext,
    lazy_npgettext,
    lazy_pgettext,
)
from llamafit.i18n.po import parse_po
from llamafit.i18n.translator import CatalogTranslator, _, set_language
from tests.fixtures import messages

CATALOG = """
msgid ""
msgstr ""
"Language: pt_PT\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"

msgid "No GPU detected"
msgstr "Nenhuma GPU detetada"

msgid "If a GPU is present, check that its driver tools are installed."
msgstr "Se existir uma GPU, verifique se as ferramentas do controlador estao instaladas."

msgid "%(count)d module"
msgid_plural "%(count)d modules"
msgstr[0] "%(count)d modulo"
msgstr[1] "%(count)d modulos"
"""

CONTEXT_CATALOG = (
    CATALOG
    + """
msgctxt "GPU"
msgid "none detected"
msgstr "nenhuma detetada"

msgctxt "GPU"
msgid "%(count)d device"
msgid_plural "%(count)d devices"
msgstr[0] "%(count)d placa"
msgstr[1] "%(count)d placas"
"""
)


@pytest.fixture(autouse=True)
def _english_again() -> Iterator[None]:
    translator.reset()
    yield
    translator.reset()


def _portuguese() -> None:
    translator.set_translator(CatalogTranslator("pt_PT", parse_po(CATALOG)))


def test_a_message_defined_before_the_language_is_chosen_still_obeys_it() -> None:
    # This is the trap, in the order it actually springs: define first, choose after.
    hint = lazy_gettext("No GPU detected")
    assert str(hint) == "No GPU detected"
    _portuguese()
    assert str(hint) == "Nenhuma GPU detetada"


def test_the_eager_form_is_the_trap_the_lazy_one_avoids() -> None:
    eager = _("No GPU detected")
    lazily = lazy_gettext("No GPU detected")
    _portuguese()
    assert eager == "No GPU detected", "an eager call is frozen at the moment it was made"
    assert str(lazily) == "Nenhuma GPU detetada"


def test_a_module_level_table_built_at_import_time_follows_the_language(tmp_path: Path) -> None:
    # tests/fixtures/messages.py builds these when it is imported, long before now.
    assert str(messages.GPU_SUMMARY) == "No GPU detected"
    assert str(messages.PROBE_HINTS["nvidia-smi"]).startswith("If a GPU is present")
    (tmp_path / "pt_PT.po").write_text(CATALOG, encoding="utf-8")
    set_language("pt_PT", env={}, available=("en", "pt_PT"), directory=tmp_path)
    assert str(messages.GPU_SUMMARY) == "Nenhuma GPU detetada"
    assert str(messages.PROBE_HINTS["nvidia-smi"]).startswith("Se existir uma GPU")
    assert str(messages.MODULE_COUNT) == "%(count)d modulos"


def test_a_lazy_plural_picks_the_form_its_count_selects() -> None:
    one = lazy_ngettext("%(count)d module", "%(count)d modules", 1)
    many = lazy_ngettext("%(count)d module", "%(count)d modules", 5)
    _portuguese()
    assert str(one) == "%(count)d modulo"
    assert str(many) == "%(count)d modulos"


def test_it_renders_inside_an_f_string_and_a_format_call() -> None:
    lazily = lazy_gettext("No GPU detected")
    _portuguese()
    assert f"{lazily}" == "Nenhuma GPU detetada"
    assert f"{lazily:>25}" == "     Nenhuma GPU detetada"
    assert "{}".format(lazily) == "Nenhuma GPU detetada"  # noqa: UP032


def test_it_compares_and_sorts_as_the_text_it_resolves_to() -> None:
    lazily = lazy_gettext("No GPU detected")
    assert lazily == "No GPU detected"
    assert lazily != "Nenhuma GPU detetada"
    assert lazily == lazy_gettext("No GPU detected")
    assert (lazily == 7) is False
    assert hash(lazily) == hash("No GPU detected")
    assert sorted([lazy_gettext("no server"), lazy_gettext("none detected")])[0] == "no server"
    assert lazy_gettext("a") < "b"
    assert lazy_gettext("b") > "a"
    assert lazy_gettext("a") <= "a"
    assert lazy_gettext("a") >= "a"


def test_it_measures_indexes_iterates_and_searches_as_a_string() -> None:
    lazily = lazy_gettext("No GPU detected")
    assert len(lazily) == len("No GPU detected")
    assert lazily[0] == "N"
    assert lazily[:2] == "No"
    assert "".join(iter(lazily)) == "No GPU detected"
    assert "GPU" in lazily
    assert bool(lazily) is True
    assert bool(lazy_gettext("")) is False


def test_it_concatenates_and_fills_placeholders() -> None:
    assert lazy_gettext("No GPU detected") + "!" == "No GPU detected!"
    assert ">> " + lazy_gettext("No GPU detected") == ">> No GPU detected"
    counted = lazy_ngettext("%(count)d module", "%(count)d modules", 2)
    assert counted % {"count": 2} == "2 modules"


def test_every_string_method_goes_through_the_current_translation() -> None:
    lazily = lazy_gettext("No GPU detected")
    assert lazily.upper() == "NO GPU DETECTED"
    _portuguese()
    assert lazily.upper() == "NENHUMA GPU DETETADA"
    assert lazily.split(" ")[0] == "Nenhuma"
    assert lazily.expandtabs() == "Nenhuma GPU detetada"


def test_it_keeps_the_english_message_for_the_extractor_to_read() -> None:
    lazily = lazy_gettext("No GPU detected")
    _portuguese()
    assert lazily.message == "No GPU detected"
    assert repr(lazily) == "LazyString('No GPU detected' -> 'Nenhuma GPU detetada')"


def test_it_is_not_a_string_so_a_silent_english_leak_is_impossible() -> None:
    # A str subclass would carry a frozen English buffer that str.join would use in
    # preference to any override, and the message would come out English with no error.
    # A separate type raises instead, at the call site where str() belongs.
    lazily = lazy_gettext("No GPU detected")
    assert not isinstance(lazily, str)
    assert isinstance(lazily, LazyString)
    with pytest.raises(TypeError):
        ", ".join([lazily])  # type: ignore[list-item]
    _portuguese()
    assert ", ".join([str(lazily)]) == "Nenhuma GPU detetada"


def test_an_unknown_attribute_still_fails_as_a_string_would() -> None:
    with pytest.raises(AttributeError):
        lazy_gettext("No GPU detected").no_such_method()


def test_a_typer_help_string_built_by_the_decorator_still_follows_the_language() -> None:
    # The decorator runs at import time, so this is the case the eager form loses. It is
    # asserted rather than assumed: Click could normalise help text when the decorator
    # runs, which would render the message to English there and then. It does not, and
    # this test is what says so if that ever changes.
    import typer
    from typer.testing import CliRunner

    app = typer.Typer()
    help_text = lazy_gettext("If a GPU is present, check that its driver tools are installed.")

    @app.command()
    def demo(check: bool = typer.Option(False, "--check", help=help_text)) -> None:
        """A command."""

    _portuguese()
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Se existir uma GPU" in result.output


def test_a_command_docstring_cannot_be_deferred_at_all() -> None:
    # Typer takes a command's help from its docstring, and a docstring is a literal, not
    # a call: no wrapper can reach it. A command whose help must be translatable needs an
    # explicit help= argument. This is a limit of the approach, recorded so the sweep
    # meets it here rather than in review.
    import typer
    from typer.testing import CliRunner

    app = typer.Typer()

    @app.command()
    def demo() -> None:
        """No GPU detected"""

    _portuguese()
    result = CliRunner().invoke(app, ["--help"])
    assert "No GPU detected" in result.output
    assert "Nenhuma GPU detetada" not in result.output


def test_a_deferred_contextual_message_obeys_the_language_chosen_after_it() -> None:
    row = lazy_pgettext("GPU", "none detected")
    assert str(row) == "none detected"
    translator.set_translator(CatalogTranslator("pt_PT", parse_po(CONTEXT_CATALOG)))
    assert str(row) == "nenhuma detetada"
    assert row.message == "none detected"


def test_a_deferred_contextual_plural_picks_the_form_its_count_selects() -> None:
    one = lazy_npgettext("GPU", "%(count)d device", "%(count)d devices", 1)
    many = lazy_npgettext("GPU", "%(count)d device", "%(count)d devices", 5)
    translator.set_translator(CatalogTranslator("pt_PT", parse_po(CONTEXT_CATALOG)))
    assert str(one) == "%(count)d placa"
    assert str(many) == "%(count)d placas"


def test_a_deferred_table_built_at_import_time_keeps_its_context() -> None:
    assert str(messages.GPU_ROW_EMPTY) == "none detected"
    translator.set_translator(CatalogTranslator("pt_PT", parse_po(CONTEXT_CATALOG)))
    assert str(messages.GPU_ROW_EMPTY) == "nenhuma detetada"
