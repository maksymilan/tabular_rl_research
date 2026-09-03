"""Positionally aligned set/bag operator facade."""
from __future__ import annotations
from typing import Any, Mapping
NAME = "set_operation"
def execute(executor: Any, arguments: Mapping[str, Any], artifact_handle: str | None = None) -> Any:
    return executor.execute(NAME, arguments, artifact_handle)
set_operation = execute
__all__ = ["NAME", "execute", "set_operation"]
