"""What a stream that cannot write a character, or a language, is made to do instead.

Every test that writes does so through a real ``TextIOWrapper`` with a narrow encoding, so
the bytes are the bytes a file or a pipe would receive. An in-memory runner buffer has no
encoding to fail at, which is how the crash these tests guard went unnoticed.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import pytest

from llamafit.i18n import (
    ERROR_HANDLER,
    can_write,
    clear_replacements,
    current_language,
    encoding_of,
    replacements,
    set_language,
    tolerate,
    translator,
)

JAPANESE_CATALOG = """
msgid ""
msgstr ""
"Language: ja\\n"
"Language-Team: Japanese\\n"
"Plural-Forms: nplurals=1; plural=0;\\n"

msgid "No GPU detected"
msgstr "GPU が検出されません"

msgid "Run again with --verbose for the full traceback."
msgstr "完全なトレースバックを見るには、--verbose を付けて実行し直してください。"
"""

PORTUGUESE_CATALOG = """
msgid ""
msgstr ""
"Language: pt_PT\\n"
"Language-Team: Portuguese (Portugal)\\n"
"Plural-Forms: nplurals=2; plural=(n != 1);\\n"

msgid "No GPU detected"
msgstr "Não foi detetada nenhuma GPU"
"""


@pytest.fixture(autouse=True)
def _clean_slate() -> Iterator[None]:
    translator.reset()
    clear_replacements()
    yield
    translator.reset()
    clear_replacements()


def narrow_stream(encoding: str) -> io.TextIOWrapper:
    """A text stream that fails on the first character ``encoding`` lacks, as a pipe does."""
    return io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")


def written(stream: io.TextIOWrapper) -> str:
    stream.flush()
    buffer = stream.buffer
    assert isinstance(buffer, io.BytesIO)
    return buffer.getvalue().decode(stream.encoding)


# --- a character the stream cannot write ----------------------------------------------


def test_a_narrow_stream_raises_before_it_is_made_forgiving() -> None:
    stream = narrow_stream("cp1252")
    with pytest.raises(UnicodeEncodeError):
        stream.write("予期しないエラー")


def test_a_visible_character_becomes_a_question_mark_and_is_counted() -> None:
    stream = narrow_stream("cp1252")
    tolerate(stream)
    stream.write("model 予期 done")
    assert written(stream) == "model ?? done"
    assert replacements() == 2


def test_a_direction_mark_the_page_lacks_is_dropped_and_not_counted() -> None:
    # The isolates bidi.py puts around a Latin identifier are outside even the Arabic code
    # page. They carry no text, so nothing is lost by leaving them out, and a ? in their
    # place would be a visible mark where an invisible one was meant. The right-to-left
    # mark is in that page, at 0xFE, and goes through like any other character.
    stream = narrow_stream("cp1256")
    tolerate(stream)
    stream.write("‏خطأ ⁨--verbose⁩.")
    assert written(stream) == "‏خطأ --verbose."
    assert replacements() == 0


def test_a_space_of_an_unusual_width_becomes_an_ordinary_space() -> None:
    # The narrow no-break space CLDR gives French for grouping digits: a space, not a ?.
    stream = narrow_stream("cp1252")
    tolerate(stream)
    stream.write("1\u202f234")
    assert written(stream) == "1 234"
    assert replacements() == 0


def test_a_latin_letter_with_a_lost_diacritic_is_a_hole_the_reader_fills() -> None:
    stream = narrow_stream("cp1250")  # Central European: has á and ç, lacks ã and õ
    tolerate(stream)
    stream.write("não há configurações")
    assert written(stream) == "n?o há configuraç?es"
    assert replacements() == 2


def test_the_count_can_be_cleared() -> None:
    stream = narrow_stream("ascii")
    tolerate(stream)
    stream.write("é")
    assert replacements() == 1
    clear_replacements()
    assert replacements() == 0


def test_a_decoding_error_is_not_this_handlers_business() -> None:
    with pytest.raises(UnicodeDecodeError):
        b"\xff".decode("utf-8", ERROR_HANDLER)


def test_a_stream_that_cannot_be_reconfigured_is_left_alone() -> None:
    tolerate(None)
    tolerate(object())
    tolerate(io.BytesIO())
    closed = narrow_stream("cp1252")
    closed.close()
    tolerate(closed)


def test_the_encoding_is_read_off_the_stream_or_not_at_all() -> None:
    assert encoding_of(narrow_stream("cp1252")) == "cp1252"
    assert encoding_of(io.BytesIO()) is None
    assert encoding_of(None) is None


# --- a language the stream cannot write ----------------------------------------------


def test_a_script_the_encoding_lacks_cannot_be_written() -> None:
    assert can_write("予期しないエラー", "cp1252") is False
    assert can_write("خطأ غير متوقع", "cp1252") is False
    assert can_write("Неожиданная ошибка", "cp1252") is False


def test_a_script_the_encoding_has_can_be_written_marks_or_no_marks() -> None:
    assert can_write("予期しないエラー", "cp932") is True
    assert can_write("خطأ ⁨--verbose⁩", "cp1256") is True
    assert can_write("Неожиданная ошибка", "cp1251") is True
    assert can_write("予期しないエラー", "utf-8") is True


def test_a_latin_language_is_always_written_holes_and_all() -> None:
    # Missing diacritics make holes a reader fills; they never take the language away.
    assert can_write("não há configurações", "cp1250") is True
    assert can_write("não há configurações", "ascii") is True
    assert can_write("Việt Nam có phần mềm", "cp1252") is True


def test_latin_identifiers_inside_a_translation_do_not_rescue_it() -> None:
    # A catalog is dense with llama.cpp, GGUF, VRAM and --verbose. Those are writable
    # anywhere and they are not the language; only the letters of its own script decide.
    assert can_write("llama.cpp GGUF VRAM --verbose PYTHONUTF8 予期", "cp1252") is False


def test_a_stray_character_does_not_take_a_language_away() -> None:
    # More kept than lost: a Traditional Chinese sentence on the Japanese code page loses
    # a character or two and keeps the language. Half is where the answer flips.
    assert can_write("這是一個測試句子", "cp932") is True


def test_an_encoding_python_does_not_know_is_trusted() -> None:
    assert can_write("予期しないエラー", "no-such-encoding") is True


def test_a_catalog_the_stream_cannot_write_is_refused_with_the_reason_and_the_fix(
    tmp_path: Path,
) -> None:
    (tmp_path / "ja.po").write_text(JAPANESE_CATALOG, encoding="utf-8")
    choice = set_language(
        "ja", env={}, available=("en", "ja"), directory=tmp_path, encoding="cp1252"
    )
    assert current_language() == "en"
    assert choice.language == "en"
    assert choice.requested == "ja"
    assert choice.honoured is False
    assert choice.notice is not None
    assert "cp1252" in choice.notice
    assert "Japanese" in choice.notice, "the language is named the way its catalog names it"
    assert "English" in choice.notice
    assert choice.hint is not None
    assert "PYTHONUTF8=1" in choice.hint
    # The one sentence that has to be readable on a stream that cannot write the
    # language is written in the one language every encoding can write.
    assert choice.notice.encode("ascii")
    assert choice.hint.encode("ascii")


def test_the_refusal_is_said_once_per_process(tmp_path: Path) -> None:
    (tmp_path / "ja.po").write_text(JAPANESE_CATALOG, encoding="utf-8")
    first = set_language(
        "ja", env={}, available=("en", "ja"), directory=tmp_path, encoding="cp1252"
    )
    again = set_language(
        "ja", env={}, available=("en", "ja"), directory=tmp_path, encoding="cp1252"
    )
    assert first.notice is not None
    assert again.notice is None
    assert again.hint is None
    assert current_language() == "en"


def test_a_stream_that_can_write_the_catalog_gets_it(tmp_path: Path) -> None:
    (tmp_path / "ja.po").write_text(JAPANESE_CATALOG, encoding="utf-8")
    for encoding in ("utf-8", "cp932", "UTF-16"):
        translator.reset()
        choice = set_language(
            "ja", env={}, available=("en", "ja"), directory=tmp_path, encoding=encoding
        )
        assert current_language() == "ja", encoding
        assert choice.notice is None, encoding


def test_a_latin_catalog_keeps_its_language_on_a_narrow_stream(tmp_path: Path) -> None:
    (tmp_path / "pt_PT.po").write_text(PORTUGUESE_CATALOG, encoding="utf-8")
    choice = set_language(
        "pt_PT", env={}, available=("en", "pt_PT"), directory=tmp_path, encoding="cp1250"
    )
    assert current_language() == "pt_PT"
    assert choice.notice is None


def test_no_encoding_means_no_question(tmp_path: Path) -> None:
    # The dashboard draws its own screen and a test has no stream: neither names one,
    # and neither is refused anything.
    (tmp_path / "ja.po").write_text(JAPANESE_CATALOG, encoding="utf-8")
    choice = set_language("ja", env={}, available=("en", "ja"), directory=tmp_path)
    assert current_language() == "ja"
    assert choice.notice is None
