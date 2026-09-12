"""Which published archive belongs on which machine, and how the release is read.

Nothing here touches the network: the GitHub client is exercised through an
``httpx.MockTransport`` and everything else reads a recorded asset listing.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from llamafit.errors import NetworkError
from llamafit.llamacpp.releases import (
    FakeReleaseClient,
    HttpReleaseClient,
    Release,
    ReleaseAsset,
    backend_preference,
    companion_assets,
    cuda_runs_on,
    cuda_version,
    driver_version,
    is_runtime_archive,
    parse_build,
    resolve_backend,
    select_asset,
    tokens,
)
from llamafit.models.host import Arch, Backend, OsName, Vendor
from tests.fixtures import llamacpp_release as fixture


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------------------
# Reading names
# --------------------------------------------------------------------------------------


def test_names_are_split_into_tokens_not_matched_as_substrings() -> None:
    assert tokens("llama-b10892-bin-win-cuda-12.4-x64.zip") == [
        "llama",
        "b10892",
        "bin",
        "win",
        "cuda",
        "12",
        "4",
        "x64",
        "zip",
    ]


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("b10892", 10892),
        ("b4600", 4600),
        ("6100", 6100),
        ("master", None),
        ("b12", None),
        # The repository's other tag scheme: not a build, and the reason `latest` reads
        # the listing rather than GitHub's own "latest" endpoint.
        ("v0.4.0", None),
    ],
)
def test_the_build_number_is_read_out_of_the_tag(tag: str, expected: int | None) -> None:
    assert parse_build(tag) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("llama-b10892-bin-win-cuda-12.4-x64.zip", (12, 4)),
        ("llama-b4600-bin-win-cuda-cu12.4-x64.zip", (12, 4)),
        ("llama-b10892-bin-win-vulkan-x64.zip", ()),
        ("cudart-llama-bin-win-cuda-13.3-x64.zip", (13, 3)),
    ],
)
def test_the_cuda_version_is_read_from_either_spelling(
    name: str, expected: tuple[int, ...]
) -> None:
    assert cuda_version(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("llama-b10892-bin-win-cuda-12.4-x64.zip", True),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", False),
        ("llama-b10892-xcframework.zip", False),
        ("llama-b10892-bin-ubuntu-x64.tar.gz", True),
        ("llama-b10892-ui.tar.gz", False),
        ("llama-b10892-bin-win-cuda-12.4-x64.exe", False),
    ],
)
def test_only_the_binary_archives_count_as_runtime_archives(name: str, expected: bool) -> None:
    assert is_runtime_archive(name) is expected


# --------------------------------------------------------------------------------------
# Choosing the archive
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("os_name", "arch", "backend", "expected"),
    [
        ("windows", "x86_64", "cuda", "llama-b10892-bin-win-cuda-12.4-x64.zip"),
        ("windows", "x86_64", "vulkan", "llama-b10892-bin-win-vulkan-x64.zip"),
        ("windows", "x86_64", "cpu", "llama-b10892-bin-win-cpu-x64.zip"),
        ("windows", "arm64", "cpu", "llama-b10892-bin-win-cpu-arm64.zip"),
        ("windows", "arm64", "cuda", "llama-b10892-bin-win-cuda-13.4-arm64.zip"),
        ("windows", "x86_64", "hip", "llama-b10892-bin-win-rocm-10.0-x64.zip"),
        ("windows", "x86_64", "sycl", "llama-b10892-bin-win-sycl-x64.zip"),
        ("macos", "arm64", "metal", "llama-b10892-bin-macos-arm64.tar.gz"),
        ("macos", "x86_64", "cpu", "llama-b10892-bin-macos-x64.tar.gz"),
        ("linux", "x86_64", "vulkan", "llama-b10892-bin-ubuntu-vulkan-x64.tar.gz"),
        ("linux", "x86_64", "cpu", "llama-b10892-bin-ubuntu-x64.tar.gz"),
        ("linux", "x86_64", "hip", "llama-b10892-bin-ubuntu-rocm-10.0-x64.tar.gz"),
        ("linux", "arm64", "vulkan", "llama-b10892-bin-ubuntu-vulkan-arm64.tar.gz"),
    ],
)
def test_every_row_of_the_published_matrix_selects_its_own_archive(
    os_name: OsName, arch: Arch, backend: Backend, expected: str
) -> None:
    chosen = select_asset(fixture.ASSETS, os_name=os_name, arch=arch, backend=backend)
    assert chosen is not None
    assert chosen.name == expected


def test_the_plain_linux_archive_is_the_cpu_one_even_without_a_cpu_token() -> None:
    """The plain ubuntu tarball carries no backend word; absence is what marks it."""
    chosen = select_asset(fixture.ASSETS, os_name="linux", arch="x86_64", backend="cpu")
    assert chosen is not None
    assert chosen.name == "llama-b10892-bin-ubuntu-x64.tar.gz"


def test_the_openvino_archive_is_not_handed_to_somebody_who_asked_for_cpu() -> None:
    """It carries no word but ``openvino``, and it sorts before ``x64`` on the alphabet."""
    chosen = select_asset(fixture.ASSETS, os_name="linux", arch="x86_64", backend="cpu")
    assert chosen is not None
    assert "openvino" not in chosen.name


def test_neither_the_android_nor_the_s390x_archive_is_ever_chosen() -> None:
    for os_name in ("linux", "macos", "windows"):
        for arch in ("x86_64", "arm64"):
            chosen = select_asset(fixture.ASSETS, os_name=os_name, arch=arch, backend="cpu")
            if chosen is not None:
                assert "android" not in chosen.name
                assert "s390x" not in chosen.name


def test_the_web_ui_bundle_and_the_xcframework_are_not_builds() -> None:
    names = {asset.name for asset in fixture.ASSETS}
    assert "llama-b10892-ui.tar.gz" in names
    assert not is_runtime_archive("llama-b10892-ui.tar.gz")
    assert not is_runtime_archive("llama-b10892-xcframework.zip")


def test_without_a_known_driver_the_most_compatible_cuda_archive_is_taken() -> None:
    """A CUDA 13 build does not start on a driver below 580; it is not merely slower."""
    chosen = select_asset(fixture.ASSETS, os_name="windows", arch="x86_64", backend="cuda")
    assert chosen is not None
    assert cuda_version(chosen.name) == (12, 4)


def test_a_driver_new_enough_for_cuda_13_gets_the_cuda_13_archive() -> None:
    chosen = select_asset(
        fixture.ASSETS, os_name="windows", arch="x86_64", backend="cuda", driver="610.88"
    )
    assert chosen is not None
    assert cuda_version(chosen.name) == (13, 3)


def test_a_driver_too_old_for_cuda_13_is_never_given_it() -> None:
    chosen = select_asset(
        fixture.ASSETS, os_name="windows", arch="x86_64", backend="cuda", driver="551.23"
    )
    assert chosen is not None
    assert cuda_version(chosen.name) == (12, 4)


def test_a_driver_too_old_for_every_published_cuda_gets_no_cuda_build_at_all() -> None:
    assert (
        select_asset(
            fixture.ASSETS, os_name="windows", arch="x86_64", backend="cuda", driver="470.10"
        )
        is None
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [("610.88", 610.88), ("570.86.10", 570.86), ("", None), (None, None), ("n/a", None)],
)
def test_a_driver_string_is_read_as_a_number_or_as_nothing(
    text: str | None, expected: float | None
) -> None:
    assert driver_version(text) == expected


def test_a_cuda_major_nobody_has_a_floor_for_is_allowed_through() -> None:
    """A constant that has not been updated must not be what refuses a working build."""
    assert cuda_runs_on("llama-b1-bin-win-cuda-99.0-x64.zip", 610.88)


def test_a_release_that_publishes_nothing_for_the_machine_selects_nothing() -> None:
    assert select_asset(fixture.ASSETS, os_name="macos", arch="arm64", backend="cuda") is None
    assert select_asset(fixture.ASSETS, os_name="linux", arch="x86_64", backend="cuda") is None


def test_the_earlier_naming_still_selects() -> None:
    """``win-hip-radeon`` before it became ``win-rocm``, and zips before tarballs."""
    chosen = select_asset(fixture.EARLIER_ASSETS, os_name="windows", arch="x86_64", backend="hip")
    assert chosen is not None
    assert chosen.name == "llama-b6100-bin-win-hip-radeon-x64.zip"


def test_the_oldest_naming_still_selects_and_prefers_avx2_over_avx512() -> None:
    """A CPU build that cannot start on the machine is worse than a slower one."""
    chosen = select_asset(fixture.OLD_STYLE_ASSETS, os_name="windows", arch="x86_64", backend="cpu")
    assert chosen is not None
    assert chosen.name == "llama-b4600-bin-win-avx2-x64.zip"


def test_the_older_cuda_spelling_is_still_found() -> None:
    chosen = select_asset(
        fixture.OLD_STYLE_ASSETS, os_name="windows", arch="x86_64", backend="cuda"
    )
    assert chosen is not None
    assert chosen.name == "llama-b4600-bin-win-cuda-cu12.4-x64.zip"


def test_selection_does_not_depend_on_the_order_the_assets_arrive_in() -> None:
    reversed_assets = tuple(reversed(fixture.ASSETS))
    for backend in ("cuda", "cpu", "vulkan"):
        first = select_asset(fixture.ASSETS, os_name="windows", arch="x86_64", backend=backend)
        second = select_asset(reversed_assets, os_name="windows", arch="x86_64", backend=backend)
        assert first == second


def test_an_architecture_llamafit_cannot_name_matches_nothing() -> None:
    assert select_asset(fixture.ASSETS, os_name="linux", arch="other", backend="cpu") is None


# --------------------------------------------------------------------------------------
# The CUDA runtime companion
# --------------------------------------------------------------------------------------


def test_the_windows_cuda_build_brings_the_matching_cuda_runtime() -> None:
    chosen = select_asset(fixture.ASSETS, os_name="windows", arch="x86_64", backend="cuda")
    assert chosen is not None
    extras = companion_assets(fixture.ASSETS, chosen, os_name="windows", backend="cuda")
    assert [extra.name for extra in extras] == ["cudart-llama-bin-win-cuda-12.4-x64.zip"]


def test_the_companion_matches_the_cuda_version_of_the_build_it_accompanies() -> None:
    """Three runtimes are published at once; pairing the wrong one is a broken install."""
    newer = next(a for a in fixture.ASSETS if a.name.endswith("win-cuda-13.3-x64.zip"))
    extras = companion_assets(fixture.ASSETS, newer, os_name="windows", backend="cuda")
    assert [extra.name for extra in extras] == ["cudart-llama-bin-win-cuda-13.3-x64.zip"]


def test_nothing_else_needs_a_companion() -> None:
    chosen = select_asset(fixture.ASSETS, os_name="windows", arch="x86_64", backend="vulkan")
    assert chosen is not None
    assert companion_assets(fixture.ASSETS, chosen, os_name="windows", backend="vulkan") == ()
    mac = select_asset(fixture.ASSETS, os_name="macos", arch="arm64", backend="metal")
    assert mac is not None
    assert companion_assets(fixture.ASSETS, mac, os_name="macos", backend="metal") == ()


# --------------------------------------------------------------------------------------
# Choosing the backend
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("os_name", "arch", "vendor", "expected"),
    [
        ("windows", "x86_64", "nvidia", ("cuda", "vulkan", "cpu")),
        ("windows", "x86_64", "amd", ("vulkan", "cpu")),
        ("linux", "x86_64", "amd", ("hip", "vulkan", "cpu")),
        ("linux", "x86_64", "intel", ("vulkan", "cpu")),
        ("macos", "arm64", "apple", ("metal", "cpu")),
        ("macos", "x86_64", "intel", ("cpu",)),
        ("linux", "x86_64", None, ("cpu",)),
    ],
)
def test_the_preference_order_matches_the_specified_matrix(
    os_name: OsName, arch: Arch, vendor: Vendor | None, expected: tuple[Backend, ...]
) -> None:
    assert backend_preference(os_name=os_name, arch=arch, vendor=vendor) == expected


def test_the_reference_windows_machine_gets_the_cuda_build_its_driver_can_load() -> None:
    resolved = resolve_backend(
        fixture.ASSETS,
        os_name="windows",
        arch="x86_64",
        vendor="nvidia",
        wanted=None,
        driver="610.88",
    )
    assert resolved is not None
    backend, asset = resolved
    assert backend == "cuda"
    assert asset.name == "llama-b10892-bin-win-cuda-13.3-x64.zip"


def test_an_apple_silicon_mac_gets_the_metal_build() -> None:
    resolved = resolve_backend(
        fixture.ASSETS, os_name="macos", arch="arm64", vendor="apple", wanted=None
    )
    assert resolved is not None
    assert resolved == ("metal", fixture.RELEASE.asset("llama-b10892-bin-macos-arm64.tar.gz"))


def test_a_linux_amd_box_gets_the_rocm_build_now_that_one_is_published() -> None:
    resolved = resolve_backend(
        fixture.ASSETS, os_name="linux", arch="x86_64", vendor="amd", wanted=None
    )
    assert resolved is not None
    backend, asset = resolved
    assert backend == "hip"
    assert asset.name == "llama-b10892-bin-ubuntu-rocm-10.0-x64.tar.gz"


def test_a_linux_nvidia_box_falls_to_vulkan_because_no_linux_cuda_is_published() -> None:
    resolved = resolve_backend(
        fixture.ASSETS, os_name="linux", arch="x86_64", vendor="nvidia", wanted=None
    )
    assert resolved is not None
    backend, asset = resolved
    assert backend == "vulkan"
    assert asset.name == "llama-b10892-bin-ubuntu-vulkan-x64.tar.gz"


def test_a_backend_asked_for_by_name_is_never_quietly_replaced() -> None:
    """Nothing published for it means nothing, not a build the user did not ask for."""
    assert (
        resolve_backend(
            fixture.ASSETS, os_name="linux", arch="x86_64", vendor="nvidia", wanted="cuda"
        )
        is None
    )


# --------------------------------------------------------------------------------------
# The GitHub client
# --------------------------------------------------------------------------------------


def _api_handler(
    payload: object,
    *,
    status: int = 200,
    listing: object | None = None,
    ref: dict[str, object] | None = None,
    seen: list[httpx.Request] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve the three endpoints the client uses: the listing, one release, one tag ref."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if "/git/ref/" in request.url.path:
            if ref is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=ref)
        if request.url.path.endswith("/releases"):
            return httpx.Response(
                status, json=listing if listing is not None else fixture.api_releases_list()
            )
        return httpx.Response(status, json=payload)

    return handler


def test_the_newest_build_is_found_by_listing_not_by_githubs_latest_endpoint() -> None:
    """The build releases are prereleases; ``/releases/latest`` names something else."""
    seen: list[httpx.Request] = []
    with _client(_api_handler(fixture.api_release(), seen=seen)) as http:
        release = HttpReleaseClient(client=http).latest()
    assert release.tag == "b10892"
    assert release.build == 10892
    assert not any(request.url.path.endswith("/releases/latest") for request in seen)


def test_a_semantic_version_tag_in_the_listing_is_not_mistaken_for_a_build() -> None:
    with _client(_api_handler(fixture.api_release())) as http:
        release = HttpReleaseClient(client=http).latest()
    assert release.tag != "v0.4.0"
    assert release.assets


def test_the_highest_build_wins_however_the_listing_happens_to_be_ordered() -> None:
    listing = list(reversed(fixture.api_releases_list()))
    with _client(_api_handler(fixture.api_release(), listing=listing)) as http:
        assert HttpReleaseClient(client=http).latest().tag == "b10892"


def test_a_listing_with_no_build_in_it_says_so_and_names_the_flag() -> None:
    listing = [fixture.api_releases_list()[0]]
    handler = _api_handler(fixture.api_release(), listing=listing)
    with _client(handler) as http, pytest.raises(NetworkError, match="hold no build"):
        HttpReleaseClient(client=http).latest()


def test_a_release_is_read_with_its_assets_and_published_digests() -> None:
    with _client(_api_handler(fixture.api_release())) as http:
        release = HttpReleaseClient(client=http).by_tag("b10892")
    assert len(release.assets) == len(fixture.ASSETS)
    cuda = release.asset("llama-b10892-bin-win-cuda-12.4-x64.zip")
    assert cuda is not None
    assert cuda.sha256 == "ddf287368bc09caf21e500d8c45264df6ddc56ac14ffc89aead2652fcfb12864"
    assert cuda.size == 254_082_151


def test_an_asset_without_a_digest_carries_no_checksum() -> None:
    payload = fixture.api_release(with_digests=False)
    with _client(_api_handler(payload)) as http:
        release = HttpReleaseClient(client=http).by_tag("b10892")
    assert all(asset.sha256 is None for asset in release.assets)


def test_the_commit_is_learned_from_the_tag_reference() -> None:
    ref = {"object": {"type": "commit", "sha": "a" * 40}}
    with _client(_api_handler(fixture.api_release(), ref=ref)) as http:
        release = HttpReleaseClient(client=http).by_tag("b10892")
    assert release.commit == "a" * 40


def test_a_release_is_still_usable_when_the_commit_cannot_be_learned() -> None:
    with _client(_api_handler(fixture.api_release())) as http:
        release = HttpReleaseClient(client=http).by_tag("b10892")
    assert release.commit is None
    assert release.tag == "b10892"


def test_an_annotated_tag_is_not_mistaken_for_a_commit() -> None:
    ref = {"object": {"type": "tag", "sha": "b" * 40}}
    with _client(_api_handler(fixture.api_release(), ref=ref)) as http:
        release = HttpReleaseClient(client=http).by_tag("b10892")
    assert release.commit is None


def test_a_missing_tag_says_so_and_names_the_flag() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    with _client(handler) as http, pytest.raises(NetworkError, match="no release"):
        HttpReleaseClient(client=http).by_tag("b9999")


def test_the_rate_limit_is_reported_as_itself_and_not_as_a_missing_release() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={}, headers={"x-ratelimit-remaining": "0"})

    with _client(handler) as http, pytest.raises(NetworkError, match="rate limit") as caught:
        HttpReleaseClient(client=http).latest()
    assert caught.value.hint is not None
    assert "GITHUB_TOKEN" in caught.value.hint


def test_a_transport_failure_becomes_a_network_error_with_a_hint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    with _client(handler) as http, pytest.raises(NetworkError, match="could not reach GitHub"):
        HttpReleaseClient(client=http).latest()


def test_an_answer_that_is_not_json_is_a_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>maintenance</html>")

    with _client(handler) as http, pytest.raises(NetworkError, match="not JSON"):
        HttpReleaseClient(client=http).latest()


def test_an_answer_that_is_json_but_not_a_release_is_a_network_error() -> None:
    handler = _api_handler({"nothing": "here"})
    with _client(handler) as http, pytest.raises(NetworkError, match="did not describe"):
        HttpReleaseClient(client=http).by_tag("b10892")


def test_assets_that_are_not_shaped_like_assets_are_skipped_not_fatal() -> None:
    payload = {
        "tag_name": "b1000",
        "assets": [
            "not an object",
            {"name": "no url"},
            {"name": "good.zip", "browser_download_url": "https://example.invalid/good.zip"},
        ],
    }
    with _client(_api_handler(payload)) as http:
        release = HttpReleaseClient(client=http).by_tag("b1000")
    assert [asset.name for asset in release.assets] == ["good.zip"]
    assert release.assets[0].size == 0


def test_a_token_in_the_environment_is_sent_as_a_bearer_header() -> None:
    seen: list[httpx.Request] = []
    with _client(_api_handler(fixture.api_release(), seen=seen)) as http:
        HttpReleaseClient(client=http, env={"GITHUB_TOKEN": "secret"}).latest()
    assert seen[0].headers["authorization"] == "Bearer secret"


def test_no_token_means_no_authorization_header() -> None:
    seen: list[httpx.Request] = []
    with _client(_api_handler(fixture.api_release(), seen=seen)) as http:
        HttpReleaseClient(client=http, env={}).latest()
    assert "authorization" not in seen[0].headers


def test_a_client_this_module_created_is_the_only_one_it_closes() -> None:
    borrowed = _client(_api_handler(fixture.api_release()))
    HttpReleaseClient(client=borrowed).close()
    assert not borrowed.is_closed
    borrowed.close()


def test_the_fake_client_answers_from_its_canned_releases() -> None:
    client = FakeReleaseClient(releases={"b10892": fixture.RELEASE}, latest_tag="b10892")
    assert client.latest().tag == "b10892"
    assert client.by_tag("b10892") is fixture.RELEASE
    with pytest.raises(NetworkError):
        client.by_tag("b1")
    assert client.calls == ["latest", "b10892", "b10892", "b1"]


def test_a_digest_in_an_unexpected_shape_is_read_as_no_checksum() -> None:
    payload = json.loads(json.dumps(fixture.api_release()))
    payload["assets"][0]["digest"] = "md5:0123"
    with _client(_api_handler(payload)) as http:
        release = HttpReleaseClient(client=http).by_tag("b10892")
    assert release.assets[0].sha256 is None


def test_a_release_reports_no_asset_it_does_not_have() -> None:
    assert Release(tag="b1", assets=(ReleaseAsset("a.zip", 1, "u"),)).asset("b.zip") is None


# --------------------------------------------------------------------------------------
# Against the real release page
# --------------------------------------------------------------------------------------


@pytest.mark.network
def test_the_selector_still_finds_a_build_in_the_release_llama_cpp_publishes_today() -> None:
    """The one check a recorded fixture cannot make: that the names have not moved.

    Every rule in this module reads asset names, and the llama.cpp project has renamed
    them before: ``cu12.4`` became ``cuda-12.4``, and the per-instruction-set Windows
    CPU archives became one. A rename would make the selector silently find nothing, and
    a suite of recorded fixtures would stay green through all of it. Marked ``network``,
    so it stays out of continuous integration and is run deliberately before a release.

    A release with no assets at all is skipped rather than failed. llama.cpp tags several
    builds a day and the binaries are uploaded after the tag, so there is a window of
    minutes where the newest release carries nothing; that is a fact about their upload
    and not about this selector. Failing there reported "nothing selected for
    windows/x86_64/nvidia", which sends the next reader to the one place that is working.
    """
    with httpx.Client(follow_redirects=True) as http:
        release = HttpReleaseClient(client=http).latest()
    assert release.build is not None
    if not release.assets:
        pytest.skip(
            f"release {release.tag} carries no assets yet, so there is nothing to select "
            "from. That is llama.cpp tagging a build before its binaries finish uploading. "
            "Run again in a few minutes."
        )
    for os_name, arch, vendor in [
        ("windows", "x86_64", "nvidia"),
        ("windows", "x86_64", None),
        ("macos", "arm64", "apple"),
        ("linux", "x86_64", "amd"),
        ("linux", "x86_64", None),
    ]:
        resolved = resolve_backend(
            release.assets, os_name=os_name, arch=arch, vendor=vendor, wanted=None
        )
        assert resolved is not None, f"nothing selected for {os_name}/{arch}/{vendor}"
        backend, asset = resolved
        assert asset.sha256, f"{asset.name} publishes no checksum any more"
        if os_name == "windows" and backend == "cuda":
            assert companion_assets(release.assets, asset, os_name=os_name, backend=backend), (
                "the CUDA runtime archive is no longer published beside the CUDA build"
            )
