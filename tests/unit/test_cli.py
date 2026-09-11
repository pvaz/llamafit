import io
import json
import logging
import sys
import unicodedata
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.models import ParameterInfo
from typer.testing import CliRunner
from typer.utils import get_params_from_function

from llamafit.cli.app import app
from llamafit.errors import NotInstalledError
from llamafit.i18n import (
    LANGUAGE_ENV_VAR,
    LazyString,
    PoCatalog,
    available_languages,
    can_write,
    clear_replacements,
    current_language,
    is_rtl,
    load_language,
    replacements,
)
from llamafit.models import Cpu, Disk, Gpu, Host, LlamaCpp, Memory, SystemReport

runner = CliRunner()


def fake_report() -> SystemReport:
    host = Host(
        os="windows",
        os_version="11",
        arch="x86_64",
        cpu=Cpu(
            model="Intel i9-14900KF",
            physical_cores=24,
            logical_cores=32,
            performance_cores=8,
            isa=["avx2", "avx512"],
        ),
        memory=Memory(
            total_bytes=128 * 1024**3,
            available_bytes=100 * 1024**3,
            type="DDR5",
            speed_mts=4200,
            modules=4,
            channels=2,
            bandwidth_gbps=67.2,
            bandwidth_source="estimated",
        ),
        gpus=[
            Gpu(
                index=0,
                vendor="nvidia",
                name="NVIDIA GeForce RTX 4060",
                vram_total_bytes=8188 * 1024**2,
                vram_used_bytes=550 * 1024**2,
                bandwidth_gbps=272,
                compute_tflops_fp16=60.4,
                backend_hint="cuda",
                driver="610.88",
            )
        ],
        disks=[Disk(path="D:\\", free_bytes=355 * 1024**3, total_bytes=1024**4)],
        scanned_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )
    llamacpp = LlamaCpp(
        installed=True,
        path="D:/llama.cpp/bin",
        build=10867,
        commit="f3f1a8f27",
        backends=["cuda", "rpc", "cpu"],
    )
    return SystemReport(host=host, llamacpp=llamacpp, version="0.1.0a1")


@pytest.fixture(autouse=True)
def patch_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **kwargs: fake_report())
    monkeypatch.setattr("llamafit.cli.doctor_cmd.scan_system", lambda **kwargs: fake_report())


def test_system_table_mentions_gpu_and_memory() -> None:
    result = runner.invoke(app, ["system"])
    assert result.exit_code == 0, result.output
    # Rich wraps the table to the terminal width, which can split "2 channels" across
    # lines; normalise whitespace so the assertion does not depend on where that lands.
    output = " ".join(result.output.split())
    assert "RTX 4060" in output
    assert "DDR5" in output
    assert "4 modules" in output
    assert "2 channels" in output
    assert "67.2" in output
    assert "b10867" in result.output
    assert "D:\\" in result.output
    assert "free of" in result.output


def test_system_json_is_the_report() -> None:
    result = runner.invoke(app, ["--json", "system"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["host"]["gpus"][0]["name"] == "NVIDIA GeForce RTX 4060"
    assert data["llamacpp"]["build"] == 10867
    assert data["version"] == "0.1.0a1"


def test_doctor_lists_findings_and_exit_code_zero_when_no_error() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "llama.cpp installed" in result.output


def test_doctor_json_has_findings() -> None:
    result = runner.invoke(app, ["--json", "doctor"])
    data = json.loads(result.output)
    assert isinstance(data["findings"], list)
    assert data["findings"][0]["level"] in {"ok", "warn", "error"}


def test_external_text_with_square_brackets_is_not_parsed_as_markup() -> None:
    from rich.console import Console

    from llamafit.cli.render import render_findings, render_probes
    from llamafit.models import Probe
    from llamafit.services.doctor import Finding

    hostile = "bad path C:/x[/y]/z"
    console = Console(width=200, no_color=True)
    with console.capture() as capture:
        console.print(
            render_probes([Probe(name="wmi-video", ok=False, duration_ms=3, error=hostile)])
        )
        console.print(
            render_findings([Finding(level="warn", title=hostile, detail=hostile, hint=hostile)])
        )
    output = capture.get()
    assert output.count(hostile) == 4


def test_a_port_with_no_server_is_not_shown_as_a_failure() -> None:
    from rich.console import Console

    from llamafit.cli.render import render_probes
    from llamafit.models import Probe

    console = Console(width=200, no_color=True)
    with console.capture() as capture:
        console.print(
            render_probes(
                [
                    Probe(name="server:8080", ok=False, duration_ms=1, error="no llama-server"),
                    Probe(name="lspci", ok=False, duration_ms=1, error="lspci: not found"),
                ]
            )
        )
    output = capture.get()
    assert "no server" in output
    assert "failed lspci: not found" in output
    assert output.count("failed") == 1


def test_main_prints_an_unexpected_error_containing_markup_verbatim(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import llamafit.cli.app as app_module

    def boom(**kwargs: object) -> None:
        raise ValueError("bad path C:/x[/y]/z")

    monkeypatch.setattr(app_module, "app", boom)
    monkeypatch.setattr(sys, "argv", ["llamafit", "system"])
    with pytest.raises(SystemExit):
        app_module.main()
    assert "bad path C:/x[/y]/z" in capsys.readouterr().err


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "llamafit" in result.output


def test_main_maps_known_errors_to_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import llamafit.cli.app as app_module

    def boom(**kwargs: object) -> None:
        raise NotInstalledError("llama.cpp not found", hint="Install it")

    monkeypatch.setattr(app_module, "app", boom)
    monkeypatch.setattr(sys, "argv", ["llamafit", "system"])
    with pytest.raises(SystemExit) as exit_info:
        app_module.main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "llama.cpp not found" in err and "Install it" in err


def test_main_reports_unexpected_errors_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import llamafit.cli.app as app_module

    def boom(**kwargs: object) -> None:
        raise ValueError("kaboom")

    monkeypatch.setattr(app_module, "app", boom)
    monkeypatch.setattr(sys, "argv", ["llamafit", "system"])
    with pytest.raises(SystemExit) as exit_info:
        app_module.main()
    assert exit_info.value.code == 1
    assert "kaboom" in capsys.readouterr().err


def test_verbose_enables_file_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import llamafit.cli.app as app_module

    def quiet(**kwargs: object) -> None:
        return None

    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path))
    monkeypatch.setattr(app_module, "app", quiet)
    monkeypatch.setattr(sys, "argv", ["llamafit", "--verbose", "system"])
    logger = logging.getLogger("llamafit")
    try:
        app_module.main()
        assert (tmp_path / "log" / "llamafit.log").exists()
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


def test_get_logger_is_a_child_of_the_package_logger() -> None:
    from llamafit.logging import get_logger

    assert get_logger("hardware.gpu").name == "llamafit.hardware.gpu"


# --- the language option ------------------------------------------------------------
#
# Nothing below names a shipped catalog or counts them. Every catalog that lands adds a
# language to the set, and a test that hard-codes the set fails on somebody else's
# contribution, which teaches the next contributor to edit the assertion instead of
# reading it -- and the next real failure gets edited away with it.

NO_SUCH_LANGUAGE = "zz"
"""A request that can never be met: ``zz`` is not an ISO 639 language."""


def _a_shipped_language() -> str:
    """Any language that ships and is not the source language."""
    return next(tag for tag in available_languages() if tag != "en")


def _a_region_that_will_never_ship() -> str:
    """A region of a shipped language that nobody can contribute.

    ``ZZ`` is user-assigned in ISO 3166, so this stays a substitution however many
    catalogs land; naming a real region would stop testing substitution the day that
    region's catalog arrived and the request began to be met exactly.

    The language has to be one whose own catalog names a region. A request with a region
    served by a region-less catalog is the documented exact-enough match rather than a
    substitution, and has nothing to announce.
    """
    regioned = next((tag for tag in available_languages() if "_" in tag), None)
    if regioned is None:
        pytest.skip("no shipped catalog names a region, so no substitution can happen")
    return f"{regioned.split('_')[0]}_ZZ"


def _deferred(candidate: object) -> str:
    """The English of a help message, read off the application rather than retyped here.

    A test that retyped a help sentence would freeze a real call site, and improving the
    sentence would break the build; the application already holds every one of them.
    """
    assert isinstance(candidate, LazyString), candidate
    return candidate.message


def _parameter_help(callback: Callable[..., object]) -> list[str]:
    """Every deferred option and argument help on a command, in English."""
    return [
        _deferred(meta.default.help)
        for meta in get_params_from_function(callback).values()
        if isinstance(meta.default, ParameterInfo) and isinstance(meta.default.help, LazyString)
    ]


def _command_help(name: str) -> tuple[str, list[str]]:
    """A subcommand's description and every deferred parameter help, in English."""
    info = next(info for info in app.registered_commands if info.name == name)
    assert info.callback is not None
    return _deferred(info.help), _parameter_help(info.callback)


def _root_help() -> tuple[str, list[str]]:
    """The root description and every deferred global option's help, in English."""
    assert app.registered_callback is not None and app.registered_callback.callback is not None
    return _deferred(app.info.help), _parameter_help(app.registered_callback.callback)


def _a_language_that_translates(*messages: str) -> tuple[str, PoCatalog]:
    """Any shipped language whose catalog translates every one of ``messages``."""
    for tag in available_languages():
        if tag == "en":
            continue
        catalog = load_language(tag)
        if all(catalog.gettext(message) != message for message in messages):
            return tag, catalog
    pytest.skip("no shipped catalog translates the help this test reads")


def _words(text: str) -> str:
    """Text as one line of single-spaced words, box edges and all wrapping gone.

    Rich draws the help in panels whose edges land inside a sentence wherever it wraps;
    ``│`` on a terminal that has it, ``|`` on one that does not.
    """
    return " ".join(text.replace("│", " ").replace("|", " ").split())


def _run_main(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], *, encoding: str = "utf-8"
) -> tuple[int, str, str]:
    """Run the real entry point with real encoded streams, as a file or a pipe would give it.

    Both streams are ``strict``, which is narrower than Python's own stderr: a report
    that cannot be encoded has to be handled by the program, not excused by the stream.
    """
    import llamafit.cli.app as app_module

    stdout = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors="strict")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "argv", ["llamafit", "--no-color", *argv])
    monkeypatch.setenv("COLUMNS", "200")
    clear_replacements()
    code = 0
    try:
        app_module.main()
    except SystemExit as exit_:
        code = exit_.code if isinstance(exit_.code, int) else 0
    stdout.flush()
    stderr.flush()
    assert isinstance(stdout.buffer, io.BytesIO) and isinstance(stderr.buffer, io.BytesIO)
    return (
        code,
        stdout.buffer.getvalue().decode(encoding),
        stderr.buffer.getvalue().decode(encoding),
    )


@pytest.fixture
def _no_leftover_count() -> Iterator[None]:
    clear_replacements()
    yield
    clear_replacements()


# The runner builds the command tree and only then parses --language, which is an order a
# real run never has and the order every runner.invoke below has. Both have to render the
# whole screen in the language asked for: the description, every option, every argument.
# The earlier version of this test checked only which language had been installed.


def test_the_whole_root_help_screen_is_in_the_language_asked_for() -> None:
    description, options = _root_help()
    language, catalog = _a_language_that_translates(description, options[0])
    result = runner.invoke(app, ["--language", language, "--help"])
    assert result.exit_code == 0, result.output
    screen = _words(result.output)
    assert _words(catalog.gettext(description)) in screen
    for message in options:
        assert _words(catalog.gettext(message)) in screen, message
    assert current_language() == language


@pytest.mark.parametrize("command", ["list", "info"])
def test_the_whole_subcommand_help_screen_is_in_the_language_asked_for(command: str) -> None:
    # `list` has options and no argument; `info` has an argument and no option. Between
    # them every kind of help line Typer draws is covered.
    description, parameters = _command_help(command)
    language, catalog = _a_language_that_translates(description, *parameters)
    result = runner.invoke(app, ["--language", language, command, "--help"])
    assert result.exit_code == 0, result.output
    screen = _words(result.output)
    assert _words(catalog.gettext(description)) in screen
    for message in parameters:
        assert _words(catalog.gettext(message)) in screen, message


def test_main_chooses_the_language_before_click_can_draw_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # --help before --language: Click processes eager options in the order given, so the
    # help screen is drawn before the --language callback runs. main has chosen by then.
    description, options = _root_help()
    language, catalog = _a_language_that_translates(description, options[0])
    code, out, _ = _run_main(monkeypatch, ["--help", "--language", language])
    assert code == 0
    assert _words(catalog.gettext(description)) in _words(out)
    assert _words(catalog.gettext(options[0])) in _words(out)


def test_the_environment_variable_reaches_the_help_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    description, options = _root_help()
    language, catalog = _a_language_that_translates(description, options[0])
    monkeypatch.setenv(LANGUAGE_ENV_VAR, language)
    code, out, _ = _run_main(monkeypatch, ["--help"])
    assert code == 0
    assert _words(catalog.gettext(description)) in _words(out)
    assert _words(catalog.gettext(options[0])) in _words(out)


def test_the_system_locale_reaches_the_help_screen(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing asked: the operating system's answer, read from the environment the way
    # SystemLocale reads it before it asks Windows.
    description, options = _root_help()
    language, catalog = _a_language_that_translates(description, options[0])
    monkeypatch.delenv(LANGUAGE_ENV_VAR)
    monkeypatch.setenv("LANGUAGE", language)
    code, out, _ = _run_main(monkeypatch, ["--help"])
    assert code == 0
    assert _words(catalog.gettext(description)) in _words(out)
    assert _words(catalog.gettext(options[0])) in _words(out)


def test_main_says_a_refused_language_once_not_once_per_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # main reads --language from the raw arguments and the callback reads it from Click;
    # two readers, one notice.
    code, _, err = _run_main(monkeypatch, ["--language", NO_SUCH_LANGUAGE, "--version"])
    assert code == 0
    assert err.count(f"LlamaFit does not speak {NO_SUCH_LANGUAGE}") == 1


def test_a_language_llamafit_does_not_have_lists_the_ones_it_does() -> None:
    # What is worth pinning is that the refusal names the languages that exist and that
    # English is one of them, not that there happen to be two of them today.
    result = runner.invoke(app, ["--language", NO_SUCH_LANGUAGE, "--version"])
    assert result.exit_code == 0, "an unknown language falls back, it does not fail bare"
    assert current_language() == "en"
    offered = available_languages()
    assert "en" in offered, "English is always available, catalog or no catalog"
    # Flattened, because the notice wraps to the console and a long list wraps with it.
    flattened = " ".join(result.output.split())
    assert f"LlamaFit does not speak {NO_SUCH_LANGUAGE}" in flattened
    for language in offered:
        assert language in flattened, f"{language} ships but the refusal does not name it"


def test_another_regions_catalog_says_which_variety_the_reader_is_getting() -> None:
    requested = _a_region_that_will_never_ship()
    result = runner.invoke(app, ["--language", requested, "--version"])
    assert result.exit_code == 0
    assert current_language() != "en", "a region with no catalog is served by its language"
    # The variety is named from the serving catalog's own Language-Team header, so this
    # asks that catalog rather than knowing the name itself.
    team = load_language(current_language()).headers["Language-Team"]
    flattened = " ".join(result.output.split())
    assert f"LlamaFit has no {requested} translation" in flattened
    assert team in flattened
    assert f"Contribute a {requested} catalog" in flattened


def test_the_notice_never_lands_in_the_json_on_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **kwargs: fake_report())
    requested = _a_region_that_will_never_ship()
    result = runner.invoke(app, ["--language", requested, "--json", "system"])
    assert result.exit_code == 0
    # Whatever the runner merged into `output`, the JSON document itself has to parse.
    assert json.loads(result.stdout)["version"]


def test_a_language_the_environment_asks_for_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    # No arguments at all, so nothing eager can exit before the language is chosen:
    # Click runs the eager options the command line named first, and --version is one.
    language = _a_shipped_language()
    monkeypatch.setenv("LLAMAFIT_LANGUAGE", language)
    result = runner.invoke(app, [])
    assert result.exit_code == 0
    assert current_language() == language


# --- a stream that cannot carry the language ------------------------------------------
#
# Every test here writes through a real TextIOWrapper with a narrow encoding, which is what
# `llamafit --language ja list > out.txt` writes through on a Western Windows. The runner's
# in-memory buffer has no encoding to fail at, which is why none of this was noticed.


def _everything_said_in(tag: str) -> str:
    catalog = load_language(tag)
    return "\n".join(form for message in catalog.messages.values() for form in message.translations)


def _a_language_beyond(encoding: str) -> str:
    """Any shipped language whose own script ``encoding`` cannot write."""
    for tag in available_languages():
        if tag != "en" and not can_write(_everything_said_in(tag), encoding):
            return tag
    pytest.skip(f"every shipped catalog can be written as {encoding}")


def _a_right_to_left_language_within(encoding: str) -> str:
    """Any shipped right-to-left language whose script ``encoding`` writes."""
    for tag in available_languages():
        if tag != "en" and is_rtl(tag) and can_write(_everything_said_in(tag), encoding):
            return tag
    pytest.skip(f"no shipped right-to-left catalog can be written as {encoding}")


def _a_latin_language_with_a_letter_missing_from(encoding: str) -> str:
    """Any shipped Latin-script language with at least one letter ``encoding`` lacks."""
    for tag in available_languages():
        if tag == "en":
            continue
        text = _everything_said_in(tag)
        if can_write(text, encoding) and text.encode(encoding, "replace").decode(encoding) != text:
            return tag
    pytest.skip(f"every shipped Latin catalog can be written whole as {encoding}")


@pytest.mark.usefixtures("_no_leftover_count")
def test_a_language_the_output_cannot_carry_falls_back_to_english_and_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    language = _a_language_beyond("cp1252")
    code, out, err = _run_main(monkeypatch, ["--language", language, "system"], encoding="cp1252")
    assert code == 0, err
    assert "RTX 4060" in out, "the table still came out"
    assert "Traceback" not in err
    assert "cp1252" in err and "English" in err, err
    assert "PYTHONUTF8=1" in err, "the fix is named"
    assert current_language() == "en"


@pytest.mark.usefixtures("_no_leftover_count")
def test_a_right_to_left_language_survives_a_code_page_without_direction_marks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Arabic code page writes Arabic, and the right-to-left mark, and not the isolates
    # bidi.py puts around a Latin identifier. The letters and the mark it has go through,
    # the isolates are left out, and nothing is counted as lost, because nothing a reader
    # could see was.
    language = _a_right_to_left_language_within("cp1256")
    code, out, err = _run_main(monkeypatch, ["--language", language, "system"], encoding="cp1256")
    assert code == 0, err
    assert any(unicodedata.bidirectional(char) in {"R", "AL"} for char in out), "spoken"
    assert "⁨" not in out and "⁩" not in out
    assert replacements() == 0
    assert "PYTHONUTF8" not in err


@pytest.mark.usefixtures("_no_leftover_count")
def test_a_latin_language_keeps_its_language_and_is_told_what_it_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    language = _a_latin_language_with_a_letter_missing_from("cp1250")
    code, out, err = _run_main(monkeypatch, ["--language", language, "system"], encoding="cp1250")
    assert code == 0, err
    assert current_language() == language, "a hole in a Latin word does not cost the language"
    assert "?" in out
    assert replacements() > 0
    assert "cp1250" in err and "PYTHONUTF8=1" in err, err


def test_the_error_report_itself_survives_a_stream_that_cannot_write_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worst version of this bug: a crash while reporting that something could not be
    # printed. The stream is strict, so this is the program handling it, not Python.
    import llamafit.cli.app as app_module

    def boom(**kwargs: object) -> None:
        raise NotInstalledError("llama.cpp が見つかりません", hint="D:\\モデル を確認")

    monkeypatch.setattr(app_module, "app", boom)
    code, _, err = _run_main(monkeypatch, ["system"], encoding="cp1252")
    assert code == 2
    assert "llama.cpp" in err and "?" in err
    assert "Traceback" not in err

def test_a_global_option_after_the_command_says_where_it_goes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``llamafit recommend --json`` is the form people try first, and it does not work.

    The options that describe the run rather than the command belong to ``llamafit``
    itself, so they come before it. Click says only that the option does not exist, which
    is true where it was typed and misleading everywhere else: a reader who has just seen
    ``--json`` documented concludes the documentation is wrong.

    Driven through ``main`` rather than the runner because the hint is printed where
    Click's own exit is caught, which is a thing only ``main`` does.
    """
    from llamafit.cli.app import main

    monkeypatch.setattr(sys, "argv", ["llamafit", "recommend", "--json", "--limit", "1"])
    with pytest.raises(SystemExit) as exit_:
        main()
    assert exit_.value.code == 2
    printed = " ".join(capsys.readouterr().err.split())
    assert "--json is an option of llamafit itself" in printed
    assert "llamafit --json recommend --limit 1" in printed


def test_a_command_that_simply_fails_gets_no_option_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Silence when the run failed for any other reason, or the hint becomes noise."""
    from llamafit.cli.app import main

    monkeypatch.setattr(sys, "argv", ["llamafit", "recommend", "--nonsense"])
    with pytest.raises(SystemExit):
        main()
    assert "option of llamafit itself" not in capsys.readouterr().err
