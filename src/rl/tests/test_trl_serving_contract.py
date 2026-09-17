import io
import json

import pytest

from rl.frameworks.trl import serving_contract


def _responses(monkeypatch, schema):
    requests = []
    payloads = {"/openapi.json": schema, "/health/": {"status": "ok"},
                "/get_world_size/": {"world_size": 1}}

    def open_url(url, timeout):
        assert timeout == 5
        requests.append(url)
        return io.StringIO(json.dumps(payloads[url.removeprefix("http://127.0.0.1:8205")]))

    monkeypatch.setattr(serving_contract, "urlopen", open_url)
    return requests


def test_real_trl_routes_are_checked_read_only(monkeypatch):
    schema = {"paths": {p: {m: {}} for p, m in serving_contract.REQUIRED_ROUTES.items()}}
    requests = _responses(monkeypatch, schema)
    audit = serving_contract.probe_trl_server("127.0.0.1", 8205)
    assert audit["world_size"] == 1
    assert audit["weight_sync_verified"] is False
    assert len(audit["openapi_sha256"]) == 64
    assert len(requests) == 3


def test_healthy_openai_server_is_rejected(monkeypatch):
    requests = _responses(monkeypatch, {"paths": {"/health": {"get": {}}, "/v1/completions": {"post": {}}}})
    with pytest.raises(RuntimeError, match="missing TRL weight-sync routes"):
        serving_contract.probe_trl_server("127.0.0.1", 8205)
    assert len(requests) == 1


def test_connection_failure_does_not_silently_pass(monkeypatch):
    def disconnected(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr(serving_contract, "urlopen", disconnected)
    with pytest.raises(RuntimeError, match="connection refused"):
        serving_contract.probe_trl_server("127.0.0.1", 8205)
