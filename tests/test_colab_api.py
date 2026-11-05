from __future__ import annotations

import json

import pytest

from hermes_colab.colab_api import XSSI_PREFIX, ColabError, ColabTunnel, TooManyRuntimes


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, timeout):
        self.calls.append((method, url, dict(headers)))
        status, body = self.responses.pop(0)
        if isinstance(body, (dict, list)):
            body = XSSI_PREFIX + json.dumps(body)
        return status, {}, body.encode()


ASSIGNMENT = {"endpoint": "gpu-t4-s-abc", "accelerator": "T4", "variant": 1,
              "runtimeProxyInfo": {"url": "https://8080-gpu-t4-s-abc.prod.colab.dev/", "token": "proxy",
                                   "tokenExpiresInSeconds": 3600}}


def test_assign_two_step_with_xsrf_token():
    transport = FakeTransport([(200, {"acc": "T4", "nbh": "x", "token": "xsrf", "variant": "GPU"}),
                               (200, ASSIGNMENT)])
    runtime = ColabTunnel(transport).assign("t4")
    assert runtime.endpoint == "gpu-t4-s-abc"
    assert runtime.url == "https://8080-gpu-t4-s-abc.prod.colab.dev"
    (m1, u1, h1), (m2, u2, h2) = transport.calls
    assert (m1, m2) == ("GET", "POST")
    assert u1 == u2 and "accelerator=T4" in u1 and "variant=GPU" in u1 and u1.endswith("authuser=0")
    assert u1.startswith("https://colab.research.google.com/tun/m/assign?")
    assert h2["X-Goog-Colab-Token"] == "xsrf" and "X-Goog-Colab-Token" not in h1
    assert "Authorization" not in h1  # the bearer is attached by Hermes core, never here


def test_assign_reuses_existing_assignment():
    transport = FakeTransport([(200, ASSIGNMENT)])
    assert ColabTunnel(transport).assign("T4").proxy_token == "proxy"
    assert len(transport.calls) == 1


def test_cpu_and_tpu_variants():
    transport = FakeTransport([(200, ASSIGNMENT), (200, ASSIGNMENT)])
    tunnel = ColabTunnel(transport)
    tunnel.assign("NONE")
    tunnel.assign("V6E1", high_mem=True)
    assert "variant" not in transport.calls[0][1]
    assert "variant=TPU" in transport.calls[1][1] and "shape" not in transport.calls[1][1]


def test_unknown_accelerator_rejected():
    with pytest.raises(ValueError):
        ColabTunnel(FakeTransport([])).assign("B200")


def test_too_many_runtimes_maps_412():
    with pytest.raises(TooManyRuntimes):
        ColabTunnel(FakeTransport([(412, {"error": {"message": "too many"}})])).list_runtimes()


def test_error_message_is_extracted():
    with pytest.raises(ColabError) as info:
        ColabTunnel(FakeTransport([(500, {"error": {"message": "backend down"}})])).list_runtimes()
    assert "backend down" in str(info.value) and info.value.status == 500


def test_keep_alive_404_means_gone():
    tunnel = ColabTunnel(FakeTransport([(200, ""), (404, "")]))
    assert tunnel.keep_alive("ep") is True
    assert tunnel.keep_alive("ep") is False


def test_keep_alive_sends_tunnel_header():
    transport = FakeTransport([(200, "")])
    ColabTunnel(transport).keep_alive("gpu/ep")
    method, url, headers = transport.calls[0]
    assert url.startswith("https://colab.research.google.com/tun/m/gpu%2Fep/keep-alive/")
    assert headers["X-Colab-Tunnel"] == "Google"


def test_release_two_step():
    transport = FakeTransport([(200, {"token": "t"}), (200, "")])
    ColabTunnel(transport).release("ep")
    assert [c[0] for c in transport.calls] == ["GET", "POST"]
    assert transport.calls[1][2]["X-Goog-Colab-Token"] == "t"


def test_list_runtimes():
    tunnel = ColabTunnel(FakeTransport([(200, {"assignments": [ASSIGNMENT]})]))
    assert [r.endpoint for r in tunnel.list_runtimes()] == ["gpu-t4-s-abc"]
