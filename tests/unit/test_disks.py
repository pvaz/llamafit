from pathlib import Path

import pytest

from llamafit.hardware.disks import detect_disks


def test_detect_disks_dedupes_mounts_and_skips_missing(tmp_path: Path) -> None:
    disks = detect_disks([tmp_path, tmp_path / "sub", tmp_path / "missing" / "xyz"])
    assert len(disks) == 1
    assert disks[0].free_bytes > 0 and disks[0].total_bytes >= disks[0].free_bytes


def test_distinct_volumes_give_distinct_disks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import llamafit.hardware.disks as disks_module

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setattr(disks_module, "_mount_key", lambda path: "vol-a" if path == a else "vol-b")
    disks = detect_disks([a, b, a])
    assert [d.path for d in disks] == [str(a), str(b)]
