"""Where LlamaFit keeps its files, following each platform's conventions."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from platformdirs import PlatformDirs
from pydantic import BaseModel

APP_NAME = "llamafit"


class AppPaths(BaseModel):
    """Directories used by LlamaFit; all absolute, none created until needed."""

    config_dir: Path
    data_dir: Path
    cache_dir: Path
    log_dir: Path
    downloads_dir: Path


def get_paths(env: Mapping[str, str] | None = None) -> AppPaths:
    """Resolve the application directories.

    Setting ``LLAMAFIT_HOME`` puts everything under one directory, which is what
    tests and portable installations want. Otherwise the platform conventions apply
    (``%LOCALAPPDATA%`` on Windows, ``~/Library/Application Support`` on macOS, the
    XDG directories on Linux) and downloads default to ``~/llamafit/models``.
    """
    env = os.environ if env is None else env
    home = env.get("LLAMAFIT_HOME")
    if home:
        root = Path(home).expanduser().resolve()
        return AppPaths(
            config_dir=root / "config",
            data_dir=root / "data",
            cache_dir=root / "cache",
            log_dir=root / "log",
            downloads_dir=root / "models",
        )
    dirs = PlatformDirs(APP_NAME, appauthor=False)
    return AppPaths(
        config_dir=Path(dirs.user_config_dir),
        data_dir=Path(dirs.user_data_dir),
        cache_dir=Path(dirs.user_cache_dir),
        log_dir=Path(dirs.user_log_dir),
        downloads_dir=Path.home() / "llamafit" / "models",
    )
