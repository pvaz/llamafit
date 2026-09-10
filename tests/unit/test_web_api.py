"""The JSON API, driven through FastAPI's own test client rather than a running server.

Nothing here starts a socket and nothing here scans a machine: the dashboard's scan is
injected, so every assertion holds on any machine in the matrix.

The test that matters most is the last one. Section 13.3 promises that ``--json`` and the
API never disagree, and a promise like that is only worth what it is checked against, so
one test runs the command and calls the endpoint and compares the two documents.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from llamafit.catalog.loader import Problem, load_catalog
from llamafit.cli.app import app as cli_app
from llamafit.errors import ProbeError
from llamafit.models.catalog import Catalog
from llamafit.models.plan import Candidate, QualityBreakdown, SpeedEstimate
from llamafit.services.recommend import BoardRow
from llamafit.web.api import (
    BOARD_PARAMETERS,
    SORT_KEYS,
    Dashboard,
    _sorted_rows,
    create_app,
    endpoints,
)
from tests.unit.test_cli import fake_report

runner = CliRunner()


@pytest.fixture
def dashboard() -> Dashboard:
    """A dashboard whose scan is the reference machine and whose catalog is the real one."""
    return Dashboard(scan=fake_report, catalog_loader=load_catalog)


@pytest.fixture
def client(dashboard: Dashboard) -> Iterator[TestClient]:
    """A client that addresses the server the way a browser on this machine would."""
    with TestClient(create_app(dashboard), base_url="http://127.0.0.1:8765") as test_client:
        yield test_client


def test_health_says_the_version(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"]


def test_the_ui_payload_carries_the_words_and_the_punctuation(client: TestClient) -> None:
    body = client.get("/api/v1/ui").json()
    assert body["language"] == "en"
    assert body["direction"] == "ltr"
    assert body["strings"]["panel.board"] == "Board"
    assert body["labels"]["verdict"]["too-tight"] == "pages"
    assert body["columns"]["gen"] == "Tok/s"
    assert body["format"] == {"group": ",", "decimal": ".", "unknown_size": "unknown"}


def test_the_page_is_told_once_that_no_speed_was_measured(client: TestClient) -> None:
    """Section 10.3: today every figure is the formula's, and the page has to say so."""
    notice = client.get("/api/v1/ui").json()["estimate_notice"]
    assert "Nothing has been benchmarked on this machine yet" in notice
    assert "no figure here is a measurement" in notice


def test_system_returns_the_scan_and_scan_takes_a_new_one(client: TestClient) -> None:
    first = client.get("/api/v1/system").json()
    assert first["host"]["cpu"]["model"] == "Intel i9-14900KF"
    again = client.post("/api/v1/scan").json()
    assert again == first


def test_the_scan_is_taken_once_and_then_reused(dashboard: Dashboard) -> None:
    calls: list[int] = []

    def counted() -> object:
        calls.append(1)
        return fake_report()

    state = Dashboard(scan=counted, catalog_loader=load_catalog)  # type: ignore[arg-type]
    with TestClient(create_app(state), base_url="http://127.0.0.1") as client:
        client.get("/api/v1/system")
        client.get("/api/v1/system")
        assert len(calls) == 1
        client.post("/api/v1/scan")
        assert len(calls) == 2


def test_doctor_reports_findings(client: TestClient) -> None:
    body = client.get("/api/v1/doctor").json()
    assert body["findings"]
    assert {"level", "title", "detail"} <= set(body["findings"][0])


def test_the_board_ranks_and_keeps_the_reason_for_every_candidate(client: TestClient) -> None:
    body = client.get("/api/v1/models/top", params={"use_case": "coding", "limit": 3}).json()
    assert body["needs"]["use_case"] == "coding"
    assert len(body["rows"]) <= 3
    assert all(row["rank"] for row in body["rows"])
    assert all(row["candidate"]["excluded_because"] for row in body["excluded"])


def test_models_shows_every_quantisation(client: TestClient) -> None:
    top = client.get("/api/v1/models/top", params={"limit": 50}).json()
    every = client.get("/api/v1/models", params={"limit": 50}).json()
    counted = len(every["rows"]) + len(every["excluded"])
    assert counted >= len(top["rows"]) + len(top["excluded"])


def test_a_misspelled_parameter_is_refused_by_name(client: TestClient) -> None:
    """A dropped parameter would answer a question nobody asked, confidently."""
    response = client.get("/api/v1/models/top", params={"use-case": "coding"})
    assert response.status_code == 400
    error = response.json()["error"]
    assert "use-case" in error["message"]
    assert "use_case" in error["hint"]


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"use_case": "knitting"}, "use-case"),
        ({"require": ["telepathy"]}, "require"),
        ({"prefer": "loud"}, "prefer"),
        ({"sort": "vibes"}, "sort"),
        ({"max_download": "many"}, "max_download"),
    ],
)
def test_a_value_the_services_do_not_know_is_refused(
    client: TestClient, params: dict[str, object], expected: str
) -> None:
    response = client.get("/api/v1/models/top", params=params)
    assert response.status_code == 400
    assert expected in response.json()["error"]["message"]


@pytest.mark.parametrize("key", SORT_KEYS)
def test_every_sort_key_orders_the_rows_without_renumbering_them(
    client: TestClient, key: str
) -> None:
    body = client.get("/api/v1/models/top", params={"sort": key, "limit": 8}).json()
    ranks = [row["rank"] for row in body["rows"]]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))
    if key == "speed":
        speeds = [
            row["candidate"]["speed"]["gen_tps"]
            for row in body["rows"]
            if row["candidate"]["speed"]
        ]
        assert speeds == sorted(speeds, reverse=True)


def test_a_profile_scores_against_another_machine(client: TestClient) -> None:
    body = client.get(
        "/api/v1/models/top", params={"profile": "reference-rtx4060-128gb", "limit": 2}
    ).json()
    assert body["rows"] or body["excluded"]


def test_a_profile_that_is_a_path_is_refused(client: TestClient) -> None:
    """Over HTTP a profile is a name; ``--profile`` takes a path because a shell may."""
    response = client.get("/api/v1/models/top", params={"profile": "../../etc/passwd"})
    assert response.status_code == 400
    assert "not a path" in response.json()["error"]["message"]


def test_an_unknown_profile_is_not_found(client: TestClient) -> None:
    response = client.get("/api/v1/models/top", params={"profile": "no-such-machine"})
    assert response.status_code == 404
    assert "no-such-machine" in response.json()["error"]["message"]


def test_an_override_that_cannot_apply_is_refused(client: TestClient) -> None:
    response = client.get("/api/v1/models/top", params={"memory": "0"})
    assert response.status_code == 400


def test_one_model_carries_its_quantisations_and_facts(client: TestClient) -> None:
    body = client.get("/api/v1/models/qwen3-coder-next").json()
    assert body["model"]["id"] == "qwen3-coder-next"
    assert body["quants"]
    assert "bytes" in body["quants"][0], "the alias --json prints must be the API's too"


def test_an_unknown_model_is_a_404_with_the_command_lines_own_hint(client: TestClient) -> None:
    response = client.get("/api/v1/models/qwen3-codr")
    assert response.status_code == 404
    error = response.json()["error"]
    assert "qwen3-codr" in error["message"]
    assert "qwen3-coder" in error["hint"]


def test_plan_returns_a_command_line(client: TestClient) -> None:
    body = client.post("/api/v1/plan", json={"model": "qwen3-coder-next"}).json()
    assert body["command"][0].endswith("llama-server")
    assert body["placement"]["budget"]["lines"]
    assert body["flags"]


def test_plan_accepts_a_hand_set_micro_batch_and_context(client: TestClient) -> None:
    body = client.post(
        "/api/v1/plan",
        json={"model": "qwen3-coder-next", "context": 8192, "ub": 512, "vision": False},
    ).json()
    assert body["placement"]["micro_batch"] == 512


def test_plan_refuses_a_quantisation_the_model_does_not_publish(client: TestClient) -> None:
    response = client.post("/api/v1/plan", json={"model": "qwen3-coder-next", "quant": "Q0_0"})
    assert response.status_code == 400
    assert "Q0_0" in response.json()["error"]["message"]


def test_plan_refuses_a_field_it_does_not_know(client: TestClient) -> None:
    response = client.post("/api/v1/plan", json={"model": "qwen3-coder-next", "colour": "red"})
    assert response.status_code == 422


def test_profiles_are_the_shape_hardware_list_prints(client: TestClient) -> None:
    body = client.get("/api/v1/profiles").json()
    assert body
    assert set(body[0]) == {"name", "bundled", "path", "profile"}


def test_the_catalog_schema_is_served_for_other_peoples_tooling(client: TestClient) -> None:
    body = client.get("/api/v1/catalog/schema").json()
    assert body["title"] == "CatalogModel"
    assert "$defs" in body


def test_a_catalog_problem_is_said_once_in_the_readers_language(dashboard: Dashboard) -> None:
    def broken() -> tuple[Catalog, list[Problem]]:
        catalog, _problems = load_catalog()
        return catalog, [Problem(file="custom.yaml", model_id=None, location="file", message="x")]

    state = Dashboard(scan=fake_report, catalog_loader=broken)
    with TestClient(create_app(state), base_url="http://127.0.0.1") as client:
        body = client.get("/api/v1/catalog/problems").json()
    assert body["count"] == 1
    assert body["message"] == (
        "1 catalog problem found; run `llamafit catalog validate` for details."
    )
    assert body["problems"][0]["file"] == "custom.yaml"


def test_the_page_is_served_from_the_package(client: TestClient) -> None:
    index = client.get("/")
    assert index.status_code == 200
    assert "text/html" in index.headers["content-type"]
    assert "<title>LlamaFit</title>" in index.text
    for asset in ("/app.js", "/styles.css"):
        assert client.get(asset).status_code == 200


def test_a_request_addressed_to_another_hostname_is_refused(dashboard: Dashboard) -> None:
    """DNS rebinding: a site resolving its own name to 127.0.0.1 must not read this API."""
    with TestClient(create_app(dashboard), base_url="http://evil.example") as client:
        assert client.get("/api/v1/system").status_code == 400


def test_a_deliberate_bind_can_still_be_addressed_by_its_own_name(dashboard: Dashboard) -> None:
    app = create_app(dashboard, extra_hosts=["192.168.1.10"])
    with TestClient(app, base_url="http://192.168.1.10:8765") as client:
        assert client.get("/health").status_code == 200


def test_no_cross_origin_headers_are_offered(client: TestClient) -> None:
    response = client.get("/api/v1/system", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in {k.lower() for k in response.headers}


def test_every_parameter_the_board_accepts_is_one_the_dependency_reads() -> None:
    """The refusal list and the signature are one list, or the refusal is a lie."""
    from llamafit.web.api import board_query

    signature = set(board_query.__annotations__) - {"request", "state", "return"}
    named = {name.rstrip("_") for name in signature}
    assert named == BOARD_PARAMETERS


def test_the_api_and_the_command_line_return_the_same_board(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Section 13.3's promise, checked rather than asserted in a docstring."""
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: fake_report())
    result = runner.invoke(cli_app, ["--json", "recommend", "--use-case", "coding", "--limit", "5"])
    assert result.exit_code == 0, result.output
    from_command = json.loads(result.stdout)
    from_api = client.get("/api/v1/models/top", params={"use_case": "coding", "limit": 5}).json()
    assert from_api == from_command


def test_the_api_and_the_command_line_return_the_same_plan(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("llamafit.cli.common.scan", lambda **_kwargs: fake_report())
    result = runner.invoke(cli_app, ["--json", "plan", "qwen3-coder-next"])
    assert result.exit_code == 0, result.output
    from_command = json.loads(result.stdout)
    from_api = client.post("/api/v1/plan", json={"model": "qwen3-coder-next"}).json()
    assert from_api == from_command


def test_a_machine_problem_is_a_503_and_a_request_problem_a_400(dashboard: Dashboard) -> None:
    """A script has to be able to tell "fix your machine" from "fix your request"."""

    def broken() -> object:
        raise ProbeError("nvidia-smi did not run", hint="install the driver")

    state = Dashboard(scan=broken, catalog_loader=load_catalog)  # type: ignore[arg-type]
    with TestClient(create_app(state), base_url="http://127.0.0.1") as client:
        response = client.get("/api/v1/system")
    assert response.status_code == 503
    assert response.json()["error"]["hint"] == "install the driver"


def test_an_override_failure_that_is_not_about_the_profile_stays_a_400(
    client: TestClient,
) -> None:
    """``memory`` on a machine with a card is fine; the refusal below is about the value."""
    response = client.get(
        "/api/v1/models/top",
        params={"profile": "reference-rtx4060-128gb", "cpu_cores": 0},
    )
    assert response.status_code == 422


def test_the_api_surface_is_the_one_the_documentation_publishes(dashboard: Dashboard) -> None:
    """docs/web.md is the contract; a path that quietly went missing fails here."""
    served = set(endpoints(create_app(dashboard)))
    for path in (
        "/health",
        "/api/v1/ui",
        "/api/v1/system",
        "/api/v1/scan",
        "/api/v1/doctor",
        "/api/v1/models",
        "/api/v1/models/top",
        "/api/v1/models/{model_id}",
        "/api/v1/plan",
        "/api/v1/profiles",
        "/api/v1/catalog/schema",
        "/api/v1/catalog/problems",
    ):
        assert path in served, f"docs/web.md publishes {path} and nothing serves it"


def _row(model_id: str, *, size: int | None) -> BoardRow:
    """A candidate the planner could not place: no speed, no quality, no placement."""
    return BoardRow(
        rank=None,
        model_id=model_id,
        name=model_id,
        quant="Q4",
        download_bytes=size,
        candidate=Candidate(model_id=model_id, quant="Q4", excluded_because="no room"),
    )


def _scored_row(model_id: str) -> BoardRow:
    """A candidate with the two figures a board can be sorted by."""
    return BoardRow(
        rank=1,
        model_id=model_id,
        name=model_id,
        quant="Q4",
        download_bytes=4,
        candidate=Candidate(
            model_id=model_id,
            quant="Q4",
            speed=SpeedEstimate(gen_tps=12.0, pp_tps=400.0, confidence="estimated"),
            quality=QualityBreakdown(baseline=70, quant_penalty=0, alignment_bonus=0, quality=70),
        ),
    )


@pytest.mark.parametrize("key", ["speed", "quality", "size"])
def test_a_row_with_nothing_to_sort_on_goes_last(key: str) -> None:
    """An unplaceable candidate must not lead a board sorted by a figure it does not have."""
    ordered = _sorted_rows([_row("a", size=None), _scored_row("b")], key)
    assert [row.model_id for row in ordered] == ["b", "a"]


def test_sorting_never_drops_a_row_it_cannot_order() -> None:
    """The reason a candidate is missing is the thing an ordering must not hide."""
    rows = [_row("a", size=None), _scored_row("b")]
    assert {row.model_id for row in _sorted_rows(rows, "context")} == {"a", "b"}
