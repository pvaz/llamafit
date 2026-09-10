"""What a download is before it starts: the files, the total, and the disk arithmetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llamafit.catalog.hf import HttpHfClient
from llamafit.download.errors import DiskSpaceError, DownloadError
from llamafit.download.plan import (
    DISK_HEADROOM_BYTES,
    MANIFEST_NAME,
    build_plan,
    check_disk_space,
    hugging_face_url,
    read_manifest,
    refuse_unverifiable,
)
from llamafit.models.catalog import CatalogModel, Extra, ModelSource, Quant
from llamafit.units import format_bytes
from tests.unit.test_models_catalog import minimal

MEGABYTE = 1024**2
"""File sizes here are small on purpose: a test that truncates a real file to four
gigabytes spends a minute of somebody's day zero-filling NTFS, and proves nothing that
four megabytes does not."""


def _model(**source_overrides: object) -> CatalogModel:
    base: dict[str, object] = {
        "repo": "acme/model-gguf",
        "trust": "unsloth",
        "quants": [
            Quant(
                name="Q4_K_M",
                files=["model-Q4_K_M.gguf"],
                bytes=4 * MEGABYTE,
                sha256=["a" * 64],
            )
        ],
    }
    base.update(source_overrides)
    return minimal(id="acme-model", name="Acme Model", sources=[ModelSource(**base)])  # type: ignore[arg-type]


def _split_model() -> CatalogModel:
    return _model(
        quants=[
            Quant(
                name="Q8_0",
                files=[f"model-0000{index}-of-00003.gguf" for index in (1, 2, 3)],
                bytes=90 * MEGABYTE,
                sha256=["a" * 64, "b" * 64, "c" * 64],
            )
        ]
    )


# --- the URL -----------------------------------------------------------------


def test_the_download_url_is_the_one_the_catalog_client_builds() -> None:
    # Written in two places, so a test says out loud that they must agree.
    built = HttpHfClient.file_url(
        HttpHfClient.__new__(HttpHfClient), "acme/model-gguf", "model-Q4_K_M.gguf"
    )
    assert hugging_face_url("acme/model-gguf", "model-Q4_K_M.gguf") == built


# --- building a plan ---------------------------------------------------------


def test_a_single_file_quant_carries_its_size_and_checksum(tmp_path: Path) -> None:
    plan = build_plan(_model(), directory=tmp_path)
    assert plan.model_id == "acme-model"
    assert plan.quant == "Q4_K_M"
    assert plan.repo == "acme/model-gguf"
    assert len(plan.files) == 1
    file = plan.files[0]
    assert file.name == "model-Q4_K_M.gguf"
    assert file.size == 4 * MEGABYTE
    assert file.sha256 == "a" * 64
    assert file.target == tmp_path / "model-Q4_K_M.gguf"
    assert plan.total_bytes == 4 * MEGABYTE
    assert plan.unverifiable == ()


def test_a_split_quant_is_one_thing_with_one_total_and_no_invented_shard_sizes(
    tmp_path: Path,
) -> None:
    plan = build_plan(_split_model(), directory=tmp_path)
    assert [file.name for file in plan.files] == [
        "model-00001-of-00003.gguf",
        "model-00002-of-00003.gguf",
        "model-00003-of-00003.gguf",
    ]
    assert plan.total_bytes == 90 * MEGABYTE
    assert [file.size for file in plan.files] == [None, None, None]
    assert [file.sha256 for file in plan.files] == ["a" * 64, "b" * 64, "c" * 64]


def test_a_directory_inside_the_repository_becomes_a_flat_local_name(tmp_path: Path) -> None:
    model = _model(
        quants=[Quant(name="Q4_K_M", files=["UD-Q4_K_XL/model.gguf"], bytes=100, sha256=["a" * 64])]
    )
    plan = build_plan(model, directory=tmp_path)
    assert plan.files[0].name == "model.gguf"
    assert plan.files[0].repo_path == "UD-Q4_K_XL/model.gguf"
    assert plan.files[0].url.endswith("/UD-Q4_K_XL/model.gguf")


def test_the_extras_a_model_declares_come_too(tmp_path: Path) -> None:
    model = _model(
        extras=[
            Extra(role="mmproj", file="mmproj-F16.gguf", bytes=1 * MEGABYTE, sha256="d" * 64),
            Extra(role="draft", file="draft-Q4.gguf", bytes=2 * MEGABYTE, sha256="e" * 64),
        ]
    )
    plan = build_plan(model, directory=tmp_path)
    assert [file.role for file in plan.files] == ["weights", "mmproj", "draft"]
    assert plan.total_bytes == 7 * MEGABYTE


def test_no_extras_leaves_them_out_of_the_total_too(tmp_path: Path) -> None:
    model = _model(
        extras=[Extra(role="mmproj", file="mmproj-F16.gguf", bytes=MEGABYTE, sha256="d" * 64)]
    )
    plan = build_plan(model, directory=tmp_path, with_extras=False)
    assert len(plan.files) == 1
    assert plan.total_bytes == 4 * MEGABYTE


def test_an_extra_the_catalog_has_not_sized_is_left_out_rather_than_fetched_blind(
    tmp_path: Path,
) -> None:
    model = _model(extras=[Extra(role="mmproj", file="mmproj-F16.gguf")])
    plan = build_plan(model, directory=tmp_path)
    assert [file.name for file in plan.files] == ["model-Q4_K_M.gguf"]


def test_a_quant_is_matched_without_regard_to_case(tmp_path: Path) -> None:
    assert build_plan(_model(), quant_name="q4_k_m", directory=tmp_path).quant == "Q4_K_M"


def test_a_quant_that_does_not_exist_names_the_ones_that_do(tmp_path: Path) -> None:
    with pytest.raises(DownloadError) as caught:
        build_plan(_model(), quant_name="Q2_K", directory=tmp_path)
    assert "Q2_K" in caught.value.message
    assert caught.value.hint is not None
    assert "Q4_K_M" in caught.value.hint


def test_a_quant_with_no_file_list_says_to_refresh(tmp_path: Path) -> None:
    model = _model(quants=[Quant(name="Q4_K_M", bytes=100)])
    with pytest.raises(DownloadError) as caught:
        build_plan(model, directory=tmp_path)
    assert caught.value.hint is not None
    assert "catalog refresh" in caught.value.hint


def test_a_quant_with_no_size_says_to_refresh(tmp_path: Path) -> None:
    model = _model(quants=[Quant(name="Q4_K_M", files=["model.gguf"], sha256=["a" * 64])])
    with pytest.raises(DownloadError) as caught:
        build_plan(model, directory=tmp_path)
    assert "how big" in caught.value.message


def test_a_model_with_only_a_local_source_cannot_be_installed(tmp_path: Path) -> None:
    model = minimal(sources=[ModelSource(kind="local", path="/models/x.gguf")])
    with pytest.raises(DownloadError) as caught:
        build_plan(model, directory=tmp_path)
    assert "no downloadable source" in caught.value.message


def test_the_default_directory_is_the_downloads_directory_and_the_model_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLAMAFIT_HOME", str(tmp_path))
    plan = build_plan(_model())
    assert plan.directory == tmp_path / "models" / "acme-model"


# --- checksums ---------------------------------------------------------------


def test_a_quant_with_no_checksums_is_refused_by_name(tmp_path: Path) -> None:
    model = _model(quants=[Quant(name="Q4_K_M", files=["model.gguf"], bytes=100)])
    plan = build_plan(model, directory=tmp_path)
    assert plan.unverifiable == ("model.gguf",)
    with pytest.raises(DownloadError) as caught:
        refuse_unverifiable(plan)
    assert "model.gguf" in caught.value.message
    assert caught.value.hint is not None
    assert "--allow-unverified" in caught.value.hint


def test_checksums_that_do_not_pair_with_the_files_are_treated_as_absent(tmp_path: Path) -> None:
    model = _model(
        quants=[
            Quant(
                name="Q4_K_M",
                files=["a.gguf", "b.gguf"],
                bytes=100,
                sha256=[],
            )
        ]
    )
    plan = build_plan(model, directory=tmp_path)
    assert plan.unverifiable == ("a.gguf", "b.gguf")


def test_a_plan_with_every_checksum_passes_the_refusal(tmp_path: Path) -> None:
    refuse_unverifiable(build_plan(_model(), directory=tmp_path))


# --- what is already here ----------------------------------------------------


def test_a_file_of_the_recorded_size_counts_as_here(tmp_path: Path) -> None:
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"\0" * (4 * MEGABYTE // MEGABYTE))
    plan = build_plan(_model(), directory=tmp_path)
    assert plan.missing_bytes == 4 * MEGABYTE  # the wrong size is not "here"

    big = tmp_path / "model-Q4_K_M.gguf"
    big.write_bytes(b"")
    with big.open("r+b") as handle:
        handle.truncate(4 * MEGABYTE)
    plan = build_plan(_model(), directory=tmp_path)
    assert plan.missing_bytes == 0
    assert plan.missing == ()


def test_a_shard_is_recognised_only_from_a_manifest_a_previous_install_wrote(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model-00001-of-00003.gguf"
    shard.write_bytes(bytes(30 * MEGABYTE))
    assert build_plan(_split_model(), directory=tmp_path).missing_bytes == 90 * MEGABYTE

    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps({"files": [{"name": shard.name, "bytes": 30 * MEGABYTE}]}), encoding="utf-8"
    )
    assert build_plan(_split_model(), directory=tmp_path).missing_bytes == 60 * MEGABYTE


def test_installed_needs_the_manifest_and_every_file(tmp_path: Path) -> None:
    plan = build_plan(_model(), directory=tmp_path)
    assert plan.is_installed() is False
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(bytes(4 * MEGABYTE))
    assert plan.is_installed() is False
    (tmp_path / MANIFEST_NAME).write_text("{}", encoding="utf-8")
    assert build_plan(_model(), directory=tmp_path).is_installed() is True


@pytest.mark.parametrize("text", ["{not json", "[]", '{"files": {}}', '{"files": [1, 2]}'])
def test_a_manifest_that_will_not_read_is_treated_as_absent(tmp_path: Path, text: str) -> None:
    (tmp_path / MANIFEST_NAME).write_text(text, encoding="utf-8")
    assert read_manifest(tmp_path) == {}


def test_a_missing_manifest_is_no_manifest(tmp_path: Path) -> None:
    assert read_manifest(tmp_path / "nowhere") == {}


# --- the disk ----------------------------------------------------------------


def test_a_volume_with_room_passes(tmp_path: Path) -> None:
    plan = build_plan(_model(), directory=tmp_path)
    check = check_disk_space(plan, free_bytes=DISK_HEADROOM_BYTES + 10 * MEGABYTE)
    assert check.needed == 4 * MEGABYTE
    assert check.ok is True
    assert check.short_by == 0
    check.raise_if_short()


def test_a_volume_that_would_fill_up_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    plan = build_plan(_model(), directory=tmp_path)
    check = check_disk_space(plan, free_bytes=4 * MEGABYTE + 1)
    assert check.ok is False
    assert check.short_by == DISK_HEADROOM_BYTES - 1
    with pytest.raises(DiskSpaceError) as caught:
        check.raise_if_short()
    assert format_bytes(4 * MEGABYTE) in caught.value.message
    assert caught.value.hint is not None
    assert "--dir" in caught.value.hint


def test_the_headroom_can_be_named_by_the_caller(tmp_path: Path) -> None:
    plan = build_plan(_model(), directory=tmp_path)
    assert check_disk_space(plan, free_bytes=4 * MEGABYTE, headroom=0).ok is True


def test_the_volume_measured_is_the_nearest_directory_that_exists(tmp_path: Path) -> None:
    plan = build_plan(_model(), directory=tmp_path / "not" / "there" / "yet")
    assert check_disk_space(plan).free > 0


def test_a_volume_that_cannot_be_measured_is_a_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_path: object) -> object:
        raise OSError("no such device")

    monkeypatch.setattr("llamafit.download.plan.shutil.disk_usage", broken)
    plan = build_plan(_model(), directory=tmp_path)
    with pytest.raises(DownloadError) as caught:
        check_disk_space(plan)
    assert "free space" in caught.value.message
