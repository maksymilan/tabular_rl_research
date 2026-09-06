"""Common report and manifest assembly for diagnostic scenarios.

Most audit scripts produce the same small pieces of provenance: a schema
version, input/output artifact paths, SHA-256 digests, and optional row counts.
``ManifestBuilder`` makes that pattern explicit while leaving each scenario in
charge of its domain-specific checks and result fields.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .io import sha256_file, write_json


def _stored_path(path: Path, root: Path | None) -> str:
    resolved = path.resolve()
    if root is None:
        return str(resolved)
    return str(resolved.relative_to(root.resolve()))


@dataclass(frozen=True)
class ArtifactRecord:
    """A stable manifest binding for one file artifact."""

    path: str
    sha256: str
    records: int | None = None
    size_bytes: int | None = None

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"path": self.path, "sha256": self.sha256}
        if self.records is not None:
            value["records"] = self.records
        if self.size_bytes is not None:
            value["size_bytes"] = self.size_bytes
        return value


def artifact_record(
    path: Path,
    *,
    records: int | None = None,
    root: Path | None = None,
    include_size: bool = False,
) -> dict[str, Any]:
    """Return the canonical path/digest record used by new manifests."""

    if records is not None and (isinstance(records, bool) or records < 0):
        raise ValueError("records must be a non-negative integer")
    path = Path(path)
    record = ArtifactRecord(
        path=_stored_path(path, root),
        sha256=sha256_file(path),
        records=int(records) if records is not None else None,
        size_bytes=path.stat().st_size if include_size else None,
    )
    return record.as_dict()


def _artifact_value(value: Path | str | Mapping[str, Any], *, root: Path | None) -> dict[str, Any]:
    if isinstance(value, (Path, str)):
        return artifact_record(Path(value), root=root)
    if not isinstance(value, Mapping):
        raise TypeError("artifact must be a path or mapping")
    record = dict(value)
    path_value = record.get("path")
    if path_value is not None and "sha256" not in record:
        record.update(artifact_record(Path(str(path_value)), root=root))
    return record


class ManifestBuilder:
    """Build a provenance manifest without coupling it to a scenario."""

    def __init__(self, schema_version: str, **metadata: Any) -> None:
        if not schema_version:
            raise ValueError("schema_version must not be empty")
        self._manifest: dict[str, Any] = {
            "schema_version": schema_version,
            **deepcopy(metadata),
            "inputs": {},
            "outputs": {},
        }

    def set(self, key: str, value: Any) -> "ManifestBuilder":
        if key in {"schema_version", "inputs", "outputs"}:
            raise ValueError(f"reserved manifest key: {key}")
        self._manifest[key] = deepcopy(value)
        return self

    def add_artifact(
        self,
        bucket: str,
        name: str,
        path: Path,
        *,
        records: int | None = None,
        root: Path | None = None,
        include_size: bool = False,
        **metadata: Any,
    ) -> "ManifestBuilder":
        if bucket not in {"inputs", "outputs"}:
            raise ValueError("bucket must be 'inputs' or 'outputs'")
        record = artifact_record(
            path, records=records, root=root, include_size=include_size
        )
        record.update(deepcopy(metadata))
        self._manifest[bucket][name] = record
        return self

    def add_input(self, name: str, path: Path, **kwargs: Any) -> "ManifestBuilder":
        return self.add_artifact("inputs", name, path, **kwargs)

    def add_output(self, name: str, path: Path, **kwargs: Any) -> "ManifestBuilder":
        return self.add_artifact("outputs", name, path, **kwargs)

    def build(self) -> dict[str, Any]:
        return deepcopy(self._manifest)

    def write(self, path: Path, *, indent: int | None = 2) -> dict[str, Any]:
        manifest = self.build()
        write_json(path, manifest, indent=indent)
        return manifest


def build_manifest(
    schema_version: str,
    *,
    inputs: Mapping[str, Path | str | Mapping[str, Any]] | None = None,
    outputs: Mapping[str, Path | str | Mapping[str, Any]] | None = None,
    root: Path | None = None,
    **metadata: Any,
) -> dict[str, Any]:
    """Convenience wrapper for one-shot manifest construction."""

    builder = ManifestBuilder(schema_version, **metadata)
    for bucket, values in (("inputs", inputs), ("outputs", outputs)):
        for name, value in (values or {}).items():
            builder._manifest[bucket][name] = _artifact_value(value, root=root)
    return builder.build()


def write_report(path: Path, report: Mapping[str, Any], *, indent: int | None = 2) -> dict[str, Any]:
    """Persist a report and return the exact object written."""

    value = deepcopy(dict(report))
    write_json(path, value, indent=indent)
    return value


__all__ = [
    "ArtifactRecord",
    "ManifestBuilder",
    "artifact_record",
    "build_manifest",
    "write_report",
]
