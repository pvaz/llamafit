import httpx
import pytest

from llamafit.catalog.hf import (
    HttpHfClient,
    RepoFile,
    assign_files_to_quants,
    match_quant_files,
)
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


def test_a_redirect_is_followed_even_on_an_injected_client() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if "api/models" in str(request.url):
            return httpx.Response(302, headers={"location": "https://huggingface.co/api/moved"})
        return httpx.Response(200, json=API)

    files = client_for(handler).list_files("org/model")
    assert len(requests) == 2
    assert {f.path for f in files} == {s["rfilename"] for s in API["siblings"]}


def test_a_transport_failure_is_a_network_error_naming_the_repository() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(NetworkError, match="org/model"):
        client_for(handler).list_files("org/model")


def test_a_non_json_body_is_a_network_error_naming_the_repository() -> None:
    with pytest.raises(NetworkError, match="org/model"):
        client_for(lambda request: httpx.Response(200, text="not json at all")).list_files(
            "org/model"
        )


def test_a_body_without_a_siblings_list_is_a_network_error_naming_the_repository() -> None:
    with pytest.raises(NetworkError, match="org/model"):
        client_for(lambda request: httpx.Response(200, json={})).list_files("org/model")


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


def test_does_not_match_a_longer_quant_name_that_starts_with_the_one_asked_for() -> None:
    files = [RepoFile(path="model-Q4_K_M-XL.gguf", size=1, sha256=None)]
    assert match_quant_files(files, "Q4_K_M") == []


def test_matches_a_plain_file_at_top_level_and_in_a_subdirectory() -> None:
    files = [
        RepoFile(path="model-Q4_K_M.gguf", size=1, sha256=None),
        RepoFile(path="Q4_K_M/model-Q4_K_M.gguf", size=2, sha256=None),
    ]
    matched = match_quant_files(files, "Q4_K_M")
    assert {f.path for f in matched} == {"model-Q4_K_M.gguf", "Q4_K_M/model-Q4_K_M.gguf"}


def test_assigns_each_file_to_the_longest_matching_quant() -> None:
    files = [
        RepoFile(path=s["rfilename"], size=s.get("size"), sha256=(s.get("lfs") or {}).get("sha256"))
        for s in API["siblings"]
    ] + [RepoFile(path="model-Q4_K_XL.gguf", size=1, sha256=None)]

    groups = assign_files_to_quants(files, ["Q4_K_XL", "UD-Q4_K_XL", "UD-Q2_K_XL"])

    assert [f.path for f in groups["Q4_K_XL"]] == ["model-Q4_K_XL.gguf"]
    assert [f.path for f in groups["UD-Q4_K_XL"]] == [
        "UD-Q4_K_XL/M-UD-Q4_K_XL-00001-of-00002.gguf",
        "UD-Q4_K_XL/M-UD-Q4_K_XL-00002-of-00002.gguf",
    ]
    assert [f.path for f in groups["UD-Q2_K_XL"]] == ["UD-Q2_K_XL/M-UD-Q2_K_XL.gguf"]

    assigned = [f.path for bucket in groups.values() for f in bucket]
    assert len(assigned) == len(set(assigned))


def test_two_files_sharing_a_shard_index_are_ordered_not_compared() -> None:
    files = [
        RepoFile(path="b/M-Q4_K_M-00001-of-00002.gguf", size=2, sha256=None),
        RepoFile(path="a/M-Q4_K_M-00002-of-00002.gguf", size=3, sha256=None),
        RepoFile(path="a/M-Q4_K_M-00001-of-00002.gguf", size=1, sha256=None),
    ]

    matched = match_quant_files(files, "Q4_K_M")

    assert [file.path for file in matched] == [
        "a/M-Q4_K_M-00001-of-00002.gguf",
        "b/M-Q4_K_M-00001-of-00002.gguf",
        "a/M-Q4_K_M-00002-of-00002.gguf",
    ]


def test_a_short_shard_set_is_left_out_rather_than_passed_on() -> None:
    files = [
        RepoFile(path="M-Q4_K_M-00001-of-00004.gguf", size=1, sha256=None),
        RepoFile(path="M-Q4_K_M-00002-of-00004.gguf", size=1, sha256=None),
    ]

    assert match_quant_files(files, "Q4_K_M") == []


def test_a_complete_set_is_preferred_to_a_larger_short_one() -> None:
    files = [
        RepoFile(path="M-Q4_K_M-00001-of-00004.gguf", size=1, sha256=None),
        RepoFile(path="M-Q4_K_M-00001-of-00002.gguf", size=1, sha256=None),
        RepoFile(path="M-Q4_K_M-00002-of-00002.gguf", size=1, sha256=None),
    ]

    assert [file.path for file in match_quant_files(files, "Q4_K_M")] == [
        "M-Q4_K_M-00001-of-00002.gguf",
        "M-Q4_K_M-00002-of-00002.gguf",
    ]


def test_a_short_shard_set_is_left_out_of_every_quant_it_belongs_to() -> None:
    files = [
        RepoFile(path="M-Q4_K_M-00001-of-00003.gguf", size=1, sha256=None),
        RepoFile(path="M-Q8_0.gguf", size=1, sha256=None),
    ]

    assigned = assign_files_to_quants(files, ["Q4_K_M", "Q8_0"])

    assert assigned["Q4_K_M"] == []
    assert [file.path for file in assigned["Q8_0"]] == ["M-Q8_0.gguf"]
