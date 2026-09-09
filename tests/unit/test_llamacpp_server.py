from llamafit.hardware.runner import FakeRunner
from llamafit.llamacpp import detect_llamacpp
from llamafit.llamacpp.server import FakeHttp, candidate_ports, discover_servers

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
