"""Read-only container id → name/image lookup with a 30 s cache (spec §5)."""
import docker.errors

from dgxmm.docker_ctl import ContainerDirectory, DockerController
from dgxmm.engines.base import managed_labels
from tests.fakes import FakeDockerClient


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class CountingClient(FakeDockerClient):
    def __init__(self):
        super().__init__()
        self.list_calls = 0
        self.fail_list = False
        real = self.containers.list

        def list_(*a, **kw):
            self.list_calls += 1
            if self.fail_list:
                raise docker.errors.APIError("daemon down")
            return real(*a, **kw)
        self.containers.list = list_


def test_lookup_resolves_name_image_and_deployment_label():
    sdk = CountingClient()
    lite = sdk.add_unmanaged("litellm-proxy", image="ghcr.io/berriai/litellm:main-latest")
    eng = sdk.add_unmanaged("dgxmm-qwen", image="vllm/vllm-openai:v0.11", labels=managed_labels(7, "vllm"))
    d = ContainerDirectory(sdk, clock=Clock())
    assert d.lookup(lite.id) == {"id": lite.id[:12], "name": "litellm-proxy", "image": "ghcr.io/berriai/litellm:main-latest",
                                 "deployment_id": None}
    assert d.lookup(eng.id)["deployment_id"] == 7
    assert sdk.list_calls == 1  # one docker ps for many lookups


def test_cache_ttl_30s_and_rate_limited_refresh_on_miss():
    sdk, clock = CountingClient(), Clock()
    d = ContainerDirectory(sdk, clock=clock)
    assert d.lookup("f" * 64) is None and sdk.list_calls == 1
    new = sdk.add_unmanaged("kokoro-tts")
    assert d.lookup(new.id) is None and sdk.list_calls == 1  # misses don't hammer the daemon
    clock.t += 6
    assert d.lookup(new.id)["name"] == "kokoro-tts" and sdk.list_calls == 2  # miss after 5 s refreshes
    sdk.container(new.id).name = "renamed"
    clock.t += 29
    assert d.lookup(new.id)["name"] == "kokoro-tts" and sdk.list_calls == 2  # hit within 30 s: cached
    clock.t += 2
    assert d.lookup(new.id)["name"] == "renamed" and sdk.list_calls == 3


def test_docker_errors_keep_last_known_map():
    sdk, clock = CountingClient(), Clock()
    c = sdk.add_unmanaged("livekit-server")
    d = ContainerDirectory(sdk, clock=clock)
    assert d.lookup(c.id)["name"] == "livekit-server"
    sdk.fail_list = True
    clock.t += 31
    assert d.lookup(c.id)["name"] == "livekit-server"


def test_lookup_is_read_only():
    sdk = CountingClient()
    c = sdk.add_unmanaged("litellm-proxy")
    ctl = DockerController(sdk)
    assert ctl.lookup(c.id)["name"] == "litellm-proxy"
    assert ctl.lookup(None) is None
    assert not sdk.run_calls and not sdk.container(c.id).stopped and c.id in sdk._containers


def test_sparse_listing_shape_is_understood():
    class Sparse:
        id = "d" * 64
        name = None  # docker-py: sparse objects have no Name attr
        attrs = {"Id": "d" * 64, "Names": ["/vllm-qwen"], "Image": "vllm/vllm-openai:latest",
                 "Labels": {"dgxmm.managed": "true", "dgxmm.deployment_id": "3"}}

    class Client:
        class containers:
            @staticmethod
            def list(**kw):
                assert kw.get("sparse") is True
                return [Sparse()]
    d = ContainerDirectory(Client(), clock=Clock())
    assert d.lookup("d" * 64) == {"id": "d" * 12, "name": "vllm-qwen", "image": "vllm/vllm-openai:latest", "deployment_id": 3}
