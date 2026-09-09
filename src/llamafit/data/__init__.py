"""Bundled data, and how the package finds it once it is installed.

Four things ship inside the package and are read through ``importlib.resources``: the
model catalog with the facts generated beside it (``data/catalog/``), the GPU
specification table (``data/gpus.json``), the translation catalogs (``data/locale/``)
and the generated JSON schema (``data/schema/``, which this program never reads; it is
published for other people's tooling — ``docs/catalog.md`` says what for).

A wheel built without one of them installs perfectly and fails only when a user reaches
the command that needs it. Left alone that surfaces as a ``ModuleNotFoundError`` out of
``importlib``, or — worse, for the catalog — as an empty table and no error at all.
Every reader goes through :func:`packaged_dir` or :func:`packaged_text` instead, so the
failure says what is missing and that reinstalling is the fix.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from llamafit.errors import PackagedDataError

__all__ = ["packaged_dir", "packaged_text"]


def _missing(location: str, what: str) -> PackagedDataError:
    """Build the error for a packaged file or directory that did not ship."""
    return PackagedDataError(
        f"This llamafit installation is missing {what} ({location}).",
        hint=(
            "The installed package is incomplete, which is nothing you can fix by editing "
            "a file. Reinstall it: pip install --force-reinstall llamafit"
        ),
    )


def packaged_dir(package: str, *, what: str, contains: str | None = None) -> Path:
    """Return the directory a packaged data subpackage lives in.

    Args:
        package: The subpackage to resolve, such as ``llamafit.data.locale``.
        what: What that directory holds, named as a user would recognise it.
        contains: A glob at least one file must match, for a directory whose contents
            are not optional. A wheel can lose a directory's files while keeping its
            ``__init__.py``, and the catalog read from an empty directory is an empty
            catalog: no error, no models, no explanation. Leave it ``None`` where an
            empty directory degrades sensibly instead, as a missing translation does.

    Returns:
        The directory the installed package reads that data from.

    Raises:
        PackagedDataError: The directory did not ship in this installation, or shipped
            with nothing in it that ``contains`` matches.
    """
    location = package.replace(".", "/")
    try:
        found = resources.files(package)
    except (ModuleNotFoundError, FileNotFoundError) as exc:
        raise _missing(location, what) from exc
    path = Path(str(found))
    if not path.is_dir():
        raise _missing(location, what)
    if contains is not None and next(path.glob(contains), None) is None:
        raise _missing(location, what)
    return path


def packaged_text(package: str, name: str, *, what: str) -> str:
    """Return the text of one packaged data file.

    Args:
        package: The subpackage the file sits in, such as ``llamafit.data``.
        name: The file's name, such as ``gpus.json``.
        what: What the file holds, named as a user would recognise it.

    Returns:
        The file's contents, decoded as UTF-8.

    Raises:
        PackagedDataError: The file did not ship in this installation.
    """
    location = f"{package.replace('.', '/')}/{name}"
    try:
        return resources.files(package).joinpath(name).read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError, IsADirectoryError) as exc:
        raise _missing(location, what) from exc
