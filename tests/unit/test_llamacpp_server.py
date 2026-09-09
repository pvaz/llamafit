from llamafit.hardware.runner import FakeRunner
from llamafit.llamacpp import detect_llamacpp
from llamafit.llamacpp.server import (
    HEALTH_TIMEOUT_S,
    FakeHttp,
    candidate_ports,
    discover_servers,
    discover_with_probes,
)

HEALTH = {"status": "ok"}
MODELS = {"data": [{"id": "qwen3-coder-next", "object": "model"}]}
PROPS = {"default_generation_settings": {"n_ctx": 262144}, "build_info": "b10867-f3f1a8f27"}


def test_candidate_ports_env_first() -> None:
    assert candidate_ports({"LLAMA_SERVER_PORT": "9000"}) == [9000, 8080, 8081, 8098]
    assert candidate_ports({}) == [8080, 8081, 8098]


def test_discover_finds_one_server() -> None:
    http = FakeHttp(
        {
            "http://127.0.0.1:8080/health": HEALTH,
            "http://127.0.0.1:8080/v1/models": MODELS,
            "http://127.0.0.1:8080/props": PROPS,
        }
    )
    servers = discover_servers(http, [8080, 8081])
    assert len(servers) == 1
    server = servers[0]
    assert server.url == "http://127.0.0.1:8080"
    assert server.model == "qwen3-coder-next"
    assert server.n_ctx == 262144
    assert server.build == "b10867-f3f1a8f27"


def test_discover_ignores_non_llama_services() -> None:
    http = FakeHttp({"http://127.0.0.1:8080/health": {"status": "degraded"}})
    assert discover_servers(http, [8080]) == []


def test_discover_tolerates_malformed_models_and_props() -> None:
    http = FakeHttp(
        {
            "http://127.0.0.1:8080/health": HEALTH,
            "http://127.0.0.1:8080/v1/models": {"data": ["not-a-dict"]},
            "http://127.0.0.1:8080/props": {
                "default_generation_settings": ["not-a-dict"],
                "build_info": 42,
            },
        }
    )
    servers = discover_servers(http, [8080])
    assert len(servers) == 1
    assert servers[0].model is None and servers[0].n_ctx is None and servers[0].build is None


def test_discover_reads_string_and_zero_context_sizes() -> None:
    http = FakeHttp(
        {
            "http://127.0.0.1:8080/health": HEALTH,
            "http://127.0.0.1:8080/props": {"default_generation_settings": {"n_ctx": "4096"}},
            "http://127.0.0.1:8081/health": HEALTH,
            "http://127.0.0.1:8081/props": {"default_generation_settings": {"n_ctx": 0}},
        }
    )
    servers = discover_servers(http, [8080, 8081])
    assert [s.n_ctx for s in servers] == [4096, 0]


def test_candidate_ports_ignores_non_numeric_and_duplicates() -> None:
    assert candidate_ports({"LLAMA_SERVER_PORT": "abc"}) == [8080, 8081, 8098]
    assert candidate_ports({"LLAMA_SERVER_PORT": "8081"}) == [8081, 8080, 8098]


def test_health_checks_use_the_health_timeout() -> None:
    assert HEALTH_TIMEOUT_S == 1.0, "long enough for a busy server, and refusals return at once"
    http = FakeHttp({"http://127.0.0.1:8080/health": HEALTH})
    discover_servers(http, [8080])
    assert http.calls[0] == ("http://127.0.0.1:8080/health", HEALTH_TIMEOUT_S)


def test_discovered_servers_follow_the_order_of_the_ports() -> None:
    responses = {}
    for port in (8080, 8081, 8098):
        responses[f"http://127.0.0.1:{port}/health"] = HEALTH
        responses[f"http://127.0.0.1:{port}/v1/models"] = {"data": [{"id": f"model-{port}"}]}
    servers, probes = discover_with_probes(FakeHttp(responses), [8098, 8080, 8081])
    assert [s.model for s in servers] == ["model-8098", "model-8080", "model-8081"]
    assert [p.name for p in probes] == ["server:8098", "server:8080", "server:8081"]


def test_detect_llamacpp_composes() -> None:
    http = FakeHttp(
        {
            "http://127.0.0.1:8080/health": HEALTH,
            "http://127.0.0.1:8080/v1/models": MODELS,
            "http://127.0.0.1:8080/props": PROPS,
        }
    )
    result = detect_llamacpp(
        FakeRunner({}), os_name="linux", env={"PATH": ""}, http=http, well_known=[]
    )
    assert not result.installed
    assert [s.model for s in result.running_servers] == ["qwen3-coder-next"]
    assert any(p.name == "server:8080" and p.ok for p in result.probes)
