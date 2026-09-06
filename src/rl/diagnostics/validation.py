"""Small fail-closed validation primitives for diagnostic scenarios."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when an audit invariant is not satisfied."""

    if not condition:
        raise ValueError(message)


def regular_file(path: Path, label: str = "file") -> None:
    """Require a non-symlink regular file and report the logical label."""

    require(
        path.is_file() and not path.is_symlink(),
        f"{label} must be a regular non-symlink file: {path}",
    )


def regular_dir(path: Path, label: str = "directory") -> None:
    """Require a non-symlink directory."""

    require(
        path.is_dir() and not path.is_symlink(),
        f"{label} must be a regular non-symlink directory: {path}",
    )


def finite_number(value: Any) -> bool:
    """Return whether ``value`` is an actual finite int/float scalar."""

    return type(value) in {int, float} and math.isfinite(float(value))


def require_fields(
    record: Mapping[str, Any], expected: Mapping[str, Any], label: str = "record"
) -> None:
    """Require exact values for a set of manifest or artifact fields."""

    for field, expected_value in expected.items():
        require(
            record.get(field) == expected_value,
            f"{label} {field} mismatch",
        )


def task_id(
    record: Mapping[str, Any],
    *,
    fields: tuple[str, ...] = ("task_id", "example_id", "instance_id"),
) -> str:
    """Extract a stable task identifier from common artifact schemas."""

    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"record has no usable task identifier; fields={fields}")


__all__ = [
    "finite_number",
    "regular_dir",
    "regular_file",
    "require",
    "require_fields",
    "task_id",
]
