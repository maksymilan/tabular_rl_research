"""Selection operator facade."""
from __future__ import annotations
from typing import Any, Mapping
NAME = "filter_rows"
def execute(executor: Any, arguments: Mapping[str, Any], artifact_handle: str | None = None) -> Any:
    return executor.execute(NAME, arguments, artifact_handle)
filter_rows = execute
__all__ = ["NAME", "execute", "filter_rows"]
