"""A wheel missing its packaged data must explain itself, not raise from importlib.

Every one of these directories is read through ``importlib.resources``, so a wheel built
without one installs perfectly and fails only when a command reaches it. These tests
manufacture that failure, because a guard nobody has seen fire is a guard nobody knows
works.
"""

import sys
from collections.abc import Iterator
from importlib import resources

import pytest

from llamafit.catalog.loader import bundled_catalog_dir
from llamafit.data import packaged_dir, packaged_text
from llamafit.errors import LlamaFitError, PackagedDataError
from llamafit.hardware import gputable
from llamafit.i18n.catalogs import catalog_dir


@pytest.fixture
def no_packaged_data(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Make every ``llamafit.data`` lookup fail the way a wheel without it would."""
    real = resources.files

    def files(anchor: object) -> object:
        if isinstance(anchor, str) and anchor.startswith("llamafit.data"):
            raise ModuleNotFoundError(f"No module named {anchor!r}")
        return real(anchor)

    monkeypatch.setattr(resources, "files", files)
    gputable._table.cache_clear()
    yield
    gputable._table.cache_clear()


def test_the_real_directories_resolve() -> None:
    assert bundled_catalog_dir().is_dir()
    assert catalog_dir().is_dir()
    assert packaged_text("llamafit.data", "gpus.json", what="the GPU table").startswith("{")


@pytest.mark.parametrize(
    ("resolve", "expected"),
    [
        (bundled_catalog_dir, "its model catalog (llamafit/data/catalog)"),
        (catalog_dir, "its translation catalogs (llamafit/data/locale)"),
    ],
)
def test_a_missing_directory_is_explained(
    no_packaged_data: None, resolve: object, expected: str
) -> None:
    with pytest.raises(PackagedDataError) as info:
        resolve()  # type: ignore[operator]
    rendered = info.value.render()
    assert expected in rendered
    assert "pip install --force-reinstall llamafit" in rendered
    assert isinstance(info.value, LlamaFitError)


def test_a_missing_file_is_explained(no_packaged_data: None) -> None:
    with pytest.raises(PackagedDataError) as info:
        gputable.lookup_gpu("NVIDIA GeForce RTX 4060")
    assert "its GPU specification table (llamafit/data/gpus.json)" in info.value.render()


def test_a_file_that_is_simply_absent_is_explained() -> None:
    with pytest.raises(PackagedDataError, match=r"llamafit/data/absent\.json"):
        packaged_text("llamafit.data", "absent.json", what="a file that never shipped")


def test_a_directory_that_shipped_empty_is_explained() -> None:
    """The catalog read from an empty directory would be an empty catalog and no error."""
    with pytest.raises(PackagedDataError, match=r"llamafit/data/catalog"):
        packaged_dir("llamafit.data.catalog", what="its model catalog", contains="*.nothing")


def test_an_empty_translation_directory_is_not_an_error() -> None:
    """With no catalog to read, every message falls back to English; that is the design."""
    assert packaged_dir("llamafit.data.locale", what="its translation catalogs").is_dir()


def test_the_command_line_prints_it_instead_of_a_traceback(
    no_packaged_data: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import llamafit.cli.app as app_module

    monkeypatch.setattr(sys, "argv", ["llamafit", "list"])
    with pytest.raises(SystemExit) as exit_info:
        app_module.main()
    err = capsys.readouterr().err
    # 2, not 1: an installation missing its own data is an environment problem, and a
    # script has to be able to tell that from "your catalog file has a typo".
    assert exit_info.value.code == 2
    assert "This llamafit installation is missing its model catalog" in err
    assert "pip install --force-reinstall llamafit" in err
    assert "Traceback" not in err
