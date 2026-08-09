"""Structured errors for the checkpointed relational-agent protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckpointRelalgError(Exception):
    """One model-visible, state-preserving protocol or execution failure."""

    error_type: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.error_type,
            "code": self.code,
            "message": self.message,
            "details": dict(self.details),
        }


def structured_error(
    exc: BaseException,
    *,
    default_type: str = "execution_error",
    default_code: str = "execution_failed",
) -> CheckpointRelalgError:
    """Normalize package-local and SQLite failures without exposing private inputs."""

    if isinstance(exc, CheckpointRelalgError):
        return exc
    error_type = getattr(exc, "error_type", None) or getattr(exc, "type", None)
    code = getattr(exc, "code", None)
    details = getattr(exc, "details", None)
    return CheckpointRelalgError(
        str(error_type or default_type),
        str(code or default_code),
        str(exc),
        dict(details) if isinstance(details, dict) else {},
    )
