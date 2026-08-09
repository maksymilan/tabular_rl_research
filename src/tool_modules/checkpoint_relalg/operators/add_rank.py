"""Window ranking operator facade."""
from __future__ import annotations
from typing import Any, Mapping
NAME = "add_rank"
def execute(executor: Any, arguments: Mapping[str, Any], artifact_handle: str | None = None) -> Any:
    return executor.execute(NAME, arguments, artifact_handle)
add_rank = execute
__all__ = ["NAME", "execute", "add_rank"]
