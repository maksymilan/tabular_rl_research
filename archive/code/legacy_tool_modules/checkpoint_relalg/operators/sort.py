"""Ordering-establishment operator facade."""
from __future__ import annotations
from typing import Any, Mapping
NAME = "sort"
def execute(executor: Any, arguments: Mapping[str, Any], artifact_handle: str | None = None) -> Any:
    return executor.execute(NAME, arguments, artifact_handle)
sort = execute
__all__ = ["NAME", "execute", "sort"]
