from pathlib import Path

from llamafit.hardware.disks import detect_disks


def test_detect_disks_dedupes_mounts_and_skips_missing(tmp_path: Path) -> None:
    disks = detect_disks([tmp_path, tmp_path / "sub", Path("/definitely/missing/xyz")])
    assert len(disks) == 1
    assert disks[0].free_bytes > 0 and disks[0].total_bytes >= disks[0].free_bytes
