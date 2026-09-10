"""Hashing a file, and refusing one that is not the file the catalog described."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from llamafit.download.errors import ChecksumError, DownloadCancelledError, DownloadError
from llamafit.download.verify import check_sha256, check_size, sha256_of


def test_the_digest_is_the_ordinary_sha256_of_the_file(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    data = bytes(range(256)) * 40
    path.write_bytes(data)
    assert sha256_of(path, block_bytes=97) == hashlib.sha256(data).hexdigest()


def test_an_empty_file_hashes_to_the_empty_digest(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"")
    assert sha256_of(path) == hashlib.sha256(b"").hexdigest()


def test_hashing_reports_its_progress_because_it_takes_minutes_on_a_real_model(
    tmp_path: Path,
) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 1000)
    seen: list[int] = []
    sha256_of(path, block_bytes=250, on_bytes=seen.append)
    assert seen == [250, 250, 250, 250]


def test_hashing_stops_when_the_user_does(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 1000)
    with pytest.raises(DownloadCancelledError):
        sha256_of(path, block_bytes=100, should_stop=lambda: True)


def test_a_file_that_cannot_be_read_is_a_download_error(tmp_path: Path) -> None:
    with pytest.raises(DownloadError) as caught:
        sha256_of(tmp_path / "missing.bin")
    assert "could not read" in caught.value.message


def test_a_short_file_is_named_as_truncated_rather_than_as_a_digest_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 10)
    with pytest.raises(ChecksumError) as caught:
        check_size(path, 20, name="f.bin")
    assert "10 bytes" in caught.value.message
    assert caught.value.hint is not None
    assert "cut short" in caught.value.hint


def test_a_file_of_the_right_length_passes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 10)
    check_size(path, 10, name="f.bin")


def test_measuring_a_file_that_is_not_there_is_a_download_error(tmp_path: Path) -> None:
    with pytest.raises(DownloadError):
        check_size(tmp_path / "missing.bin", 10, name="missing.bin")


def test_a_matching_checksum_comes_back(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert check_sha256(path, digest.upper(), name="f.bin") == digest


def test_a_mismatch_names_both_numbers_and_says_the_copy_is_gone(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    with pytest.raises(ChecksumError) as caught:
        check_sha256(path, "b" * 64, name="f.bin")
    assert "f.bin" in caught.value.message
    assert caught.value.hint is not None
    assert "b" * 64 in caught.value.hint
    assert hashlib.sha256(b"hello").hexdigest() in caught.value.hint


def test_a_digest_already_computed_is_not_computed_twice(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    # The file is gone, so anything that reads it would fail; passing the digest must not.
    path.unlink()
    assert check_sha256(path, digest, name="f.bin", actual=digest) == digest
