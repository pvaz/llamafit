import httpx
import pytest

from llamafit.catalog.hf import HttpHfClient, RepoFile, match_quant_files
from llamafit.errors import NetworkError

API = {
    "siblings": [
        {"rfilename": "README.md", "size": 100},
        {
            "rfilename": "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf",
            "size": 10,
            "lfs": {"sha256": "aa"},
        },
        {
            "rfilename": "UD-Q4_K_XL/M-UD-Q4_K_XL-00002-of-00002.gguf",
            "size": 20,
            "lfs": {"sha256": "bb"},
        },
        {"rfilename": "UD-Q2_K_XL/M-UD-Q2_K_XL.gguf", "size": 5, "lfs": {"sha256": "cc"}},
        {"rfilename": "mmproj-F16.gguf", "size": 7, "lfs": {"sha256": "dd"}},
    ]
}


def client_for(handler: object) -> HttpHfClient:
    return HttpHfClient(client=httpx.Client(transport=httpx.MockTransport(handler)))  # type: ignore[arg-type]


def test_lists_files_with_sizes_and_checksums() -> None:
    files = client_for(lambda request: httpx.Response(200, json=API)).list_files("org/model")
    by_name = {f.path: f for f in files}
    assert by_name["mmproj-F16.gguf"].size == 7
    assert by_name["mmproj-F16.gguf"].sha256 == "dd"
    assert by_name["README.md"].sha256 is None


def test_a_failure_is_a_network_error_naming_the_repository() -> None:
    with pytest.raises(NetworkError, match="org/model"):
        client_for(lambda request: httpx.Response(404, json={})).list_files("org/model")


def test_builds_a_resolve_url() -> None:
    url = client_for(lambda request: httpx.Response(200, json=API)).file_url("org/m", "a/b.gguf")
    assert url == "https://huggingface.co/org/m/resolve/main/a/b.gguf"


def test_matches_a_shard_set_in_order() -> None:
    files = [
        RepoFile(path=s["rfilename"], size=s.get("size"), sha256=(s.get("lfs") or {}).get("sha256"))
        for s in API["siblings"]
    ]
    matched = match_quant_files(files, "UD-Q4_K_XL")
    assert [f.path.rsplit("/", 1)[-1] for f in matched] == [
        "M-UD-Q4_K_XL-00001-of-00002.gguf",
        "M-UD-Q4_K_XL-00002-of-00002.gguf",
    ]


def test_matches_a_single_file_quant_and_ignores_the_other_quant() -> None:
    files = [
        RepoFile(path=s["rfilename"], size=s.get("size"), sha256=(s.get("lfs") or {}).get("sha256"))
        for s in API["siblings"]
    ]
    matched = match_quant_files(files, "UD-Q2_K_XL")
    assert [f.path for f in matched] == ["UD-Q2_K_XL/M-UD-Q2_K_XL.gguf"]


def test_an_unknown_quant_matches_nothing() -> None:
    files = [RepoFile(path="a/b-Q4_K_M.gguf", size=1, sha256=None)]
    assert match_quant_files(files, "Q8_0") == []
