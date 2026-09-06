"""Reusable, strict file I/O for diagnostics and audit scenarios.

Scenario modules should keep their domain logic here and use these helpers at
the artifact boundary.  The helpers deliberately use the same JSON settings
everywhere (UTF-8, non-ASCII preserved, sorted keys) and make writes atomic.
This gives reports and manifests stable bytes without forcing every scenario to
reimplement parsing and temporary-file cleanup.

The module is a diagnostics-facing API.  The lower-level functions in
``rl.shared.io`` remain available for other RL code; importing this module is
the preferred choice for new diagnostic scenarios.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


JsonObject = dict[str, Any]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _source_name(source: Path | str | None) -> str:
    return str(source) if source is not None else "<bytes>"


def parse_json_bytes(
    data: bytes,
    *,
    source: Path | str | None = None,
    require_object: bool = False,
) -> Any:
    """Parse UTF-8 JSON and optionally require a top-level object."""

    name = _source_name(source)
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid JSON") from exc
    if require_object and not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def read_json(path: Path, *, require_object: bool = False) -> Any:
    """Read a UTF-8 JSON file, with an artifact path in validation errors."""

    return parse_json_bytes(path.read_bytes(), source=path, require_object=require_object)


def parse_jsonl_bytes(
    data: bytes,
    *,
    source: Path | str | None = None,
    require_object: bool = True,
    allow_blank: bool = True,
) -> list[Any]:
    """Parse JSONL bytes with line-aware errors.

    Empty lines are ignored.  Diagnostic row files conventionally contain
    objects, so ``require_object`` defaults to true while remaining explicit
    for callers that intentionally store scalar rows.
    """

    name = _source_name(source)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name}: invalid UTF-8") from exc
    rows: list[Any] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            if not allow_blank:
                raise ValueError(f"{name}:{line_number}: blank JSONL row")
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name}:{line_number}: invalid JSON") from exc
        if require_object and not isinstance(value, dict):
            raise ValueError(f"{name}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def read_jsonl(
    path: Path, *, require_object: bool = True, allow_blank: bool = True
) -> list[Any]:
    """Read a JSONL artifact, ignoring blank lines and validating each row."""

    return parse_jsonl_bytes(
        path.read_bytes(), source=path, require_object=require_object, allow_blank=allow_blank
    )


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Encode JSON with the repository's deterministic defaults."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def canonical_json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return canonical_json(value, indent=indent).encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Any]) -> bytes:
    """Encode rows as deterministic JSONL bytes, including a final newline."""

    return b"".join(canonical_json(row).encode("utf-8") + b"\n" for row in rows)


def _temporary_path(path: Path) -> tuple[int, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    return descriptor, Path(name)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically replace ``path`` with ``data`` after flushing the file."""

    descriptor, temporary = _temporary_path(path)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def write_json(
    path: Path,
    value: Any,
    *,
    indent: int | None = 2,
    trailing_newline: bool = True,
    atomic: bool = True,
) -> None:
    """Write a JSON artifact using deterministic formatting."""

    text = canonical_json(value, indent=indent)
    if trailing_newline:
        text += "\n"
    if atomic:
        atomic_write_text(path, text)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def write_jsonl(
    path: Path,
    rows: Iterable[Any],
    *,
    trailing_newline: bool = True,
    atomic: bool = True,
) -> None:
    """Write deterministic JSONL rows, optionally as an atomic replacement."""

    data = canonical_jsonl_bytes(rows)
    if not trailing_newline and data.endswith(b"\n"):
        data = data[:-1]
    if atomic:
        atomic_write_bytes(path, data)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str, *, encoding: str = "utf-8") -> str:
    return sha256_bytes(text.encode(encoding))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_jsonl(rows: Iterable[Any]) -> str:
    return sha256_bytes(canonical_jsonl_bytes(rows))


def verify_sha256(path: Path, expected: str) -> bool:
    """Return whether a file has the declared lowercase SHA-256 digest."""

    if not _SHA256_RE.fullmatch(expected):
        raise ValueError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    return sha256_file(path) == expected


def require_sha256(path: Path, expected: str, *, label: str | None = None) -> str:
    """Validate a file digest and return it, using a useful audit error."""

    actual = sha256_file(path)
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError(f"{label or path} declared SHA-256 is malformed")
    if actual != expected:
        raise ValueError(f"{label or path} SHA-256 mismatch: {actual} != {expected}")
    return actual


def as_json_object(value: Mapping[str, Any], *, context: str = "value") -> JsonObject:
    """Make a shallow JSON-object copy for callers building reports."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return dict(value)


__all__ = [
    "JsonObject",
    "as_json_object",
    "atomic_write_bytes",
    "atomic_write_text",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "parse_json_bytes",
    "parse_jsonl_bytes",
    "read_json",
    "read_jsonl",
    "require_sha256",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_jsonl",
    "sha256_text",
    "verify_sha256",
    "write_json",
    "write_jsonl",
]
