from pathlib import Path

from llamafit.paths import get_paths


def test_llamafit_home_overrides_everything(tmp_path: Path) -> None:
    paths = get_paths({"LLAMAFIT_HOME": str(tmp_path)})
    assert paths.config_dir == tmp_path / "config"
    assert paths.data_dir == tmp_path / "data"
    assert paths.cache_dir == tmp_path / "cache"
    assert paths.log_dir == tmp_path / "log"
    assert paths.downloads_dir == tmp_path / "models"


def test_default_paths_are_absolute_and_named_llamafit() -> None:
    paths = get_paths({})
    for p in (paths.config_dir, paths.data_dir, paths.cache_dir, paths.log_dir):
        assert p.is_absolute()
        assert "llamafit" in str(p).lower()
