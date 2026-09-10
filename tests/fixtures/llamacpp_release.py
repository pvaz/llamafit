"""A recorded llama.cpp release listing, and the pieces tests build installs from.

`ASSETS` is the real asset list of release `b10892`, read from the GitHub API on
2026-09-10: the names, the sizes and the published SHA-256 digests, unedited. Every rule
in `llamafit.llamacpp.releases` is a rule about those names, so a test against invented
ones would prove nothing — and the first run against the real listing found three things
no invented fixture would have: the builds are prereleases, macOS and Linux ship as
`.tar.gz`, and the OpenVINO archive carries no word but `openvino` to mark it as
anything other than a plain CPU build.

`EARLIER_ASSETS` and `OLD_STYLE_ASSETS` are two earlier namings the project has used.
They are kept so the selector is held to reading all three: the names have moved twice
already, and a selector that only reads today's would fail silently the next time.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from llamafit.llamacpp.releases import Release, ReleaseAsset

TAG = "b10892"
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}"

RECORDED: list[tuple[str, int, str]] = [
    (
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        391_443_627,
        "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
    ),
    (
        "cudart-llama-bin-win-cuda-13.3-x64.zip",
        390_970_417,
        "1462a050eb4c684921ba51dcc4cc488a036674c3e73e9945ee705b854808d03e",
    ),
    (
        "cudart-llama-bin-win-cuda-13.4-arm64.zip",
        153_318_797,
        "5a40dc7c5fa3d0a80ceeba4f16f9e8d25d87bcf1399c9233588953c43436c33c",
    ),
    (
        "llama-b10892-bin-android-arm64.tar.gz",
        75_456_525,
        "688035b9297b826eb757323a5693bb0797d3265a9569c3aa4b3c2988e3ead05e",
    ),
    (
        "llama-b10892-bin-macos-arm64.tar.gz",
        11_141_106,
        "3e7c86d1c6e638f83326ab59a2e3c01e7e2fbfc776a4112695107cc61469a1e0",
    ),
    (
        "llama-b10892-bin-macos-x64.tar.gz",
        11_193_833,
        "ebd10b6354fd70d58e6ed9dddca0f5156bdcb6972d7d0b1b743b692f6e9e1fb1",
    ),
    (
        "llama-b10892-bin-ubuntu-arm64.tar.gz",
        13_456_929,
        "a04f02226bc52b6dd12852dff1bf56f67feae6f2ce69fa1cf49c653a61717541",
    ),
    (
        "llama-b10892-bin-ubuntu-openvino-2026.3.1-x64.tar.gz",
        100_782_529,
        "ece40ba327abe51d1302a2c2b970256feb608136b69cee08271ee743a2616cd9",
    ),
    (
        "llama-b10892-bin-ubuntu-rocm-10.0-x64.tar.gz",
        218_364_636,
        "e4cdac3961063ff765df1041ec178ba45b702e0b6079fe7a49052ff9a0e7b611",
    ),
    (
        "llama-b10892-bin-ubuntu-s390x.tar.gz",
        15_349_670,
        "2531042cb7117487b4c0aa74cd8d3e99d337b3f65cf574a7f81a28f6ab56726e",
    ),
    (
        "llama-b10892-bin-ubuntu-sycl-fp16-x64.tar.gz",
        53_977_944,
        "cdad37e5a3710633469944f31da1e68f49fa6a047f31a0c8243e059a3d2c24dd",
    ),
    (
        "llama-b10892-bin-ubuntu-sycl-fp32-x64.tar.gz",
        53_751_755,
        "a2c9ce0b9dc412d2218d21a124beb083f4b765a46526173becda18d9c6258969",
    ),
    (
        "llama-b10892-bin-ubuntu-vulkan-arm64.tar.gz",
        24_216_351,
        "5084097ef843d0e03bd91eb806e297732be1390b3cc8cdf02480c53cbffe29ba",
    ),
    (
        "llama-b10892-bin-ubuntu-vulkan-x64.tar.gz",
        30_160_971,
        "87975352278b7fa5de25c2cd7454fb0abd62a42375d360e5b33f0e6b9712ac6c",
    ),
    (
        "llama-b10892-bin-ubuntu-x64.tar.gz",
        16_814_543,
        "fb262cab2adf3a94807177cd8d57c8a9d9d8d54a7be8de84d5b05ceace52c03d",
    ),
    (
        "llama-b10892-bin-win-cpu-arm64.zip",
        11_991_177,
        "46124f927182cd153958a458b2c9020a5265742a731c5c1cf58253cecbbac5da",
    ),
    (
        "llama-b10892-bin-win-cpu-x64.zip",
        18_424_730,
        "71de55b9a4ca6115e1536a37b43065040dc6c231d4c2c0cf46d071b4d32d814b",
    ),
    (
        "llama-b10892-bin-win-cuda-12.4-x64.zip",
        254_082_151,
        "ddf287368bc09caf21e500d8c45264df6ddc56ac14ffc89aead2652fcfb12864",
    ),
    (
        "llama-b10892-bin-win-cuda-13.3-x64.zip",
        149_708_335,
        "f626cf6781c91afa1cb06e12cc37ec04831af1e7021514ef2b64e93fb8a35eef",
    ),
    (
        "llama-b10892-bin-win-cuda-13.4-arm64.zip",
        142_950_754,
        "282cd0312f531e546a8b3ab5af8bfe3cded8ba5f50902e1cb0bfc3f50c9c88ae",
    ),
    (
        "llama-b10892-bin-win-opencl-adreno-arm64.zip",
        12_783_688,
        "2f2be80d0c857bf02695694facdbf93b788a44ac1fa51e178136edefbcddde27",
    ),
    (
        "llama-b10892-bin-win-openvino-2026.3.1-x64.zip",
        80_298_786,
        "be020a93cc4bc8b139f9e5fcd217a7c55ad78ce98666b88c2beb7dd8bc88d79f",
    ),
    (
        "llama-b10892-bin-win-rocm-10.0-x64.zip",
        244_152_757,
        "4dba689f707e95eb4137cf66b01313d8cc847793f23d9c4844f9674f28421c65",
    ),
    (
        "llama-b10892-bin-win-sycl-x64.zip",
        119_782_691,
        "80b763caa6a082e809f1879e1df336e178710a678a04b485087577d77ce7e8dc",
    ),
    (
        "llama-b10892-bin-win-vulkan-x64.zip",
        31_662_584,
        "37d713e9aa7a391e6e2143711358b91bc88f3494b33c6d6d34ea82c9bb1e5dff",
    ),
    (
        "llama-b10892-ui.tar.gz",
        3_085_919,
        "988eaae836fd9e7ab78db0aa7d9512a2a8252e4830be9b6ae4810d40670e3c2c",
    ),
    (
        "llama-b10892-xcframework.zip",
        87_026_579,
        "5d9f94beccfbb88984e6a2ad2b85f7e6c53282089d0c9684308795f9bfcb69c9",
    ),
]

EARLIER_NAMES = [
    "llama-b6100-bin-macos-arm64.zip",
    "llama-b6100-bin-ubuntu-vulkan-x64.zip",
    "llama-b6100-bin-ubuntu-x64.zip",
    "llama-b6100-bin-win-cpu-x64.zip",
    "llama-b6100-bin-win-cuda-12.4-x64.zip",
    "llama-b6100-bin-win-hip-radeon-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
]

OLD_STYLE_NAMES = [
    "llama-b4600-bin-win-avx2-x64.zip",
    "llama-b4600-bin-win-avx512-x64.zip",
    "llama-b4600-bin-win-noavx-x64.zip",
    "llama-b4600-bin-win-cuda-cu12.4-x64.zip",
    "llama-b4600-bin-win-vulkan-x64.zip",
    "llama-b4600-bin-macos-arm64.zip",
    "llama-b4600-bin-ubuntu-x64.zip",
]


def _digest(name: str) -> str:
    """A stand-in digest for the earlier namings, whose real ones were not recorded.

    Derived from the name rather than from ``hash()``, which is salted per process: a
    fixture whose values changed between runs would make a failure unreproducible.
    """
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


ASSETS: tuple[ReleaseAsset, ...] = tuple(
    ReleaseAsset(name=name, size=size, url=f"{BASE}/{name}", sha256=sha256)
    for name, size, sha256 in RECORDED
)

EARLIER_ASSETS: tuple[ReleaseAsset, ...] = tuple(
    ReleaseAsset(name=name, size=1_000, url=f"{BASE}/{name}", sha256=_digest(name))
    for name in EARLIER_NAMES
)

OLD_STYLE_ASSETS: tuple[ReleaseAsset, ...] = tuple(
    ReleaseAsset(name=name, size=1_000, url=f"{BASE}/{name}", sha256=_digest(name))
    for name in OLD_STYLE_NAMES
)

RELEASE = Release(
    tag=TAG,
    assets=ASSETS,
    url=f"https://github.com/ggml-org/llama.cpp/releases/tag/{TAG}",
    published_at="2026-09-09T22:14:05Z",
    commit="3f9c1a2b7d5e4c6a8b0d2f1e3c5a7b9d0e2f4a6c",
)


def api_release(tag: str = TAG, *, with_digests: bool = True) -> dict[str, object]:
    """The GitHub API's own shape for one release, as the HTTP client parses it."""
    return {
        "tag_name": tag,
        "prerelease": True,
        "html_url": f"https://github.com/ggml-org/llama.cpp/releases/tag/{tag}",
        "published_at": "2026-09-09T22:14:05Z",
        "assets": [
            {
                "name": name,
                "size": size,
                "browser_download_url": f"{BASE}/{name}",
                **({"digest": f"sha256:{sha256}"} if with_digests else {}),
            }
            for name, size, sha256 in RECORDED
        ],
    }


def api_releases_list() -> list[dict[str, object]]:
    """The listing endpoint's answer, semantic-version tag and all.

    The shape that matters: `/releases/latest` answers with the `v0.4.0` tag, because
    every build release is marked a prerelease, and that tag's only attachment is a text
    file naming the current nightly. The listing is where the builds are.
    """
    return [
        {
            "tag_name": "v0.4.0",
            "prerelease": False,
            "html_url": "https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0",
            "assets": [
                {
                    "name": "nightly-tag.txt",
                    "size": 7,
                    "browser_download_url": "https://example.invalid/nightly-tag.txt",
                }
            ],
        },
        api_release("b10890"),
        api_release(TAG),
        api_release("b10888"),
    ]


def build_zip(
    members: dict[str, bytes], *, executable: tuple[str, ...] = (), prefix: str = ""
) -> bytes:
    """A zip archive holding ``members``, optionally under a directory prefix.

    Args:
        members: File contents by path inside the archive.
        executable: Which of those paths carry the POSIX executable bit, which is what
            the extractor has to restore for ``llama-server`` to be runnable at all.
        prefix: A directory to nest everything under, so the nested ``build/bin``
            layout can be exercised as well as the flat one.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            path = f"{prefix}{name}" if prefix else name
            info = zipfile.ZipInfo(path)
            info.external_attr = ((0o755 if name in executable else 0o644) & 0xFFFF) << 16
            archive.writestr(info, content)
    return buffer.getvalue()


def windows_build_zip() -> bytes:
    """A flat archive shaped like the published Windows CUDA build."""
    return build_zip(
        {
            "llama-server.exe": b"server binary",
            "llama-cli.exe": b"cli binary",
            "ggml-cuda.dll": b"cuda backend",
            "ggml-cpu-haswell.dll": b"cpu backend",
        }
    )


def linux_build_zip() -> bytes:
    """A nested archive shaped like a build that keeps its binaries in ``build/bin``."""
    return build_zip(
        {
            "llama-server": b"server binary",
            "libggml-vulkan.so": b"vulkan backend",
        },
        executable=("llama-server",),
        prefix="build/bin/",
    )


def cudart_zip() -> bytes:
    """The companion archive holding the CUDA runtime libraries."""
    return build_zip({"cudart64_12.dll": b"cuda runtime", "cublas64_12.dll": b"cublas"})


def write(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path``, creating parents, and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
