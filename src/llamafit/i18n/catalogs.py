"""Where the translation catalogs live, which ones exist, and how one is read.

The catalogs ship inside the package as ``llamafit/data/locale/<tag>.po``, addressed
through ``importlib.resources`` the same way the model catalog and the GPU table are.
They are plain text in the repository, read at runtime; nothing here compiles anything
and no binary is committed.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from llamafit.i18n.po import PoCatalog, read_po
from llamafit.i18n.tags import SOURCE_LANGUAGE, normalise

CATALOG_SUFFIX = ".po"
TEMPLATE_NAME = "messages.pot"
"""The extracted template, which lists every message but translates none."""


def catalog_dir() -> Path:
    """The packaged directory holding the ``.po`` catalogs and the template."""
    return Path(str(resources.files("llamafit.data.locale")))


def catalog_path(language: str, *, directory: Path | None = None) -> Path:
    """The file a language's catalog would live in, whether or not it exists."""
    folder = catalog_dir() if directory is None else directory
    return folder / f"{language}{CATALOG_SUFFIX}"


def available_languages(*, directory: Path | None = None) -> tuple[str, ...]:
    """List the languages LlamaFit can speak, sorted, English always included.

    English is the source language and ships no catalog, so it is listed without one.
    A file whose name is not a language tag is ignored rather than reported: the
    completeness check in the test suite is what holds the catalogs to their shape.

    Returns:
        The language tags, for example ``("en", "pt_PT")``.
    """
    folder = catalog_dir() if directory is None else directory
    languages = {SOURCE_LANGUAGE}
    if folder.is_dir():
        for file in folder.glob(f"*{CATALOG_SUFFIX}"):
            tag = normalise(file.stem)
            if tag is not None:
                languages.add(tag)
    return tuple(sorted(languages))


def load_language(language: str, *, directory: Path | None = None) -> PoCatalog:
    """Read one language's catalog from disk.

    The file must be named for the language exactly as :func:`available_languages`
    reports it, so ``pt_PT`` lives in ``pt_PT.po``.

    Returns:
        The parsed catalog.

    Raises:
        PoSyntaxError: If the catalog is malformed or is not valid UTF-8.
        OSError: If the file is missing or cannot be read.
    """
    return read_po(catalog_path(language, directory=directory))
