"""Where the translation catalogs live, which ones exist, and how one is read.

The catalogs ship inside the package as ``llamafit/data/locale/<tag>.po``, addressed
through ``importlib.resources`` the same way the model catalog and the GPU table are.
They are plain text in the repository, read at runtime; nothing here compiles anything
and no binary is committed.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from llamafit.errors import ConfigError
from llamafit.i18n.po import PoCatalog, read_po
from llamafit.i18n.tags import SOURCE_LANGUAGE, normalise

CATALOG_SUFFIX = ".po"
TEMPLATE_NAME = "messages.pot"
"""The extracted template, which lists every message but translates none."""


def catalog_dir() -> Path:
    """The packaged directory holding the ``.po`` catalogs and the template."""
    return Path(str(resources.files("llamafit.data.locale")))


def catalog_path(language: str, *, directory: Path | None = None) -> Path:
    """The file a language's catalog is read from, whether or not it exists.

    The name is the tag exactly as :func:`available_languages` reports it, so ``pt_PT``
    lives in ``pt_PT.po``, and that name wins whenever such a file is there.

    A file whose name spells the same tag another way — ``pt-PT.po``, ``PT_pt.po`` — is
    found too, because :func:`available_languages` normalises a file's name before
    offering it and a language that is offered has to be openable. Listing one spelling
    and opening another is how a catalog got offered to a user and then refused with a
    message about a missing file, which reads as though the file were not there at all.

    Returns:
        The file to read. When nothing matches, the canonical name, so a reader who has to
        be told the catalog is missing is told which name was expected.
    """
    folder = catalog_dir() if directory is None else directory
    canonical = folder / f"{language}{CATALOG_SUFFIX}"
    wanted = normalise(language)
    if canonical.is_file() or wanted is None or not folder.is_dir():
        return canonical
    spelt_otherwise = sorted(
        file for file in folder.glob(f"*{CATALOG_SUFFIX}") if normalise(file.stem) == wanted
    )
    return spelt_otherwise[0] if spelt_otherwise else canonical


def available_languages(*, directory: Path | None = None) -> tuple[str, ...]:
    """List the languages LlamaFit can speak, sorted, English always included.

    English is the source language and ships no catalog, so it is listed without one.
    A file whose name is not a language tag is ignored rather than reported: the
    completeness check in the test suite is what holds the catalogs to their shape.

    A name is normalised before it is offered, so ``pt-BR.po`` is offered as ``pt_BR``.
    :func:`catalog_path` looks for the same spellings, so everything listed here can be
    opened.

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
    """Read one language's catalog from disk, refusing one that is not that language.

    The file is named for the language exactly as :func:`available_languages` reports it,
    so ``pt_PT`` lives in ``pt_PT.po``.

    A file whose ``Language`` header names some other language is refused. Nothing else
    would notice: the tag comes from the file's name, so the catalog would be installed
    as the language that was asked for and would then answer in the language it actually
    holds, with no error anywhere. A reader who asked for Portuguese would be reading
    German and be told nothing. The test suite checks this for every catalog LlamaFit
    ships, and a catalog somebody writes themselves never goes near the test suite, which
    is the whole reason the check belongs here as well.

    A catalog that declares no ``Language`` at all is still read: a translation in
    progress may not have filled the header in yet, and saying nothing is not the same as
    saying something false.

    Returns:
        The parsed catalog.

    Raises:
        ConfigError: If the catalog's ``Language`` header names another language, or if
            the catalog is malformed or is not valid UTF-8.
        OSError: If the file is missing or cannot be read.
    """
    path = catalog_path(language, directory=directory)
    catalog = read_po(path)
    declared = catalog.language.strip()
    if declared and normalise(declared) != normalise(language):
        raise ConfigError(
            f"{path}: this file is read as {language}, but its Language header says "
            f"{declared!r}, so it would answer in the wrong language",
            hint=f"Rename the file to {declared}{CATALOG_SUFFIX} if the header is right, "
            f"or set the header to {language} if the name is.",
        )
    return catalog
