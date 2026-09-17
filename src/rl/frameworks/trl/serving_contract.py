"""Read-only capability gate for the TRL online weight-sync server.

Passing this gate proves HTTP compatibility, not successful NCCL synchronization.
The first rollout/update remains a separate runtime gate.
"""
from __future__ import annotations

import hashlib
import json
from urllib.request import urlopen


REQUIRED_ROUTES = {
    "/health/": "get",
    "/get_world_size/": "get",
    "/generate/": "post",
    "/init_communicator/": "post",
    "/update_named_param/": "post",
    "/reset_prefix_cache/": "post",
    "/close_communicator/": "post",
}


def probe_trl_server(host: str, port: int, *, timeout: float = 5.0) -> dict:
    """Reject evaluation-only OpenAI servers without allocating GPU memory."""
    base_url = f"http://{host}:{port}"

    def read_json(path):
        with urlopen(base_url + path, timeout=timeout) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise ValueError(f"{path} did not return a JSON object")
        return result

    try:
        schema = read_json("/openapi.json")
        paths = schema.get("paths", {})
        missing = [path for path, method in REQUIRED_ROUTES.items()
                   if not isinstance(paths.get(path), dict) or method not in paths[path]]
        if missing:
            raise ValueError(f"missing TRL weight-sync routes: {missing}")
        if read_json("/health/").get("status") != "ok":
            raise ValueError("TRL server is not healthy")
        world_size = read_json("/get_world_size/").get("world_size")
        if type(world_size) is not int or world_size < 1:
            raise ValueError("invalid TRL serving world_size")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            f"online RL requires a ready `trl vllm-serve` at {base_url}: {exc}"
        ) from exc
    return {
        "schema_version": "trl-serving-capability-v1",
        "base_url": base_url,
        "world_size": world_size,
        "openapi_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "required_routes": REQUIRED_ROUTES,
        "weight_sync_verified": False,
    }
