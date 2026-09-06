from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rl.diagnostics.io import (
    atomic_write_text,
    canonical_json,
    parse_jsonl_bytes,
    read_json,
    read_jsonl,
    sha256_json,
    verify_sha256,
    write_json,
    write_jsonl,
)
from rl.diagnostics.reporting import ManifestBuilder, artifact_record, build_manifest, write_report


def test_json_and_jsonl_helpers_are_deterministic_and_line_aware(tmp_path: Path) -> None:
    json_path = tmp_path / "nested" / "report.json"
    write_json(json_path, {"z": 1, "中文": "保留"})
    assert read_json(json_path, require_object=True) == {"z": 1, "中文": "保留"}
    assert json_path.read_text(encoding="utf-8") == '{\n  "z": 1,\n  "中文": "保留"\n}\n'

    jsonl_path = tmp_path / "rows.jsonl"
    write_jsonl(jsonl_path, [{"b": 2, "a": 1}, {"id": 2}])
    assert read_jsonl(jsonl_path) == [{"a": 1, "b": 2}, {"id": 2}]
    with pytest.raises(ValueError, match=r"<bytes>:2: expected a JSON object"):
        parse_jsonl_bytes(b'{"id": 1}\n[2]\n')


def test_atomic_write_preserves_previous_artifact_when_encoding_fails(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    atomic_write_text(path, "old\n")
    with pytest.raises(TypeError):
        write_json(path, {"bad": object()})
    assert path.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob(".report.json.*.tmp"))


def test_hash_helpers_and_manifest_builder_bind_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "out.json"
    write_jsonl(input_path, [{"id": 1}, {"id": 2}])
    write_report(output_path, {"ok": True})

    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert verify_sha256(input_path, digest)
    assert sha256_json({"b": 2, "a": 1}) == hashlib.sha256(
        canonical_json({"a": 1, "b": 2}).encode()
    ).hexdigest()
    input_record = artifact_record(input_path, records=2, root=tmp_path, include_size=True)
    assert input_record == {
        "path": "input.jsonl",
        "sha256": digest,
        "records": 2,
        "size_bytes": input_path.stat().st_size,
    }

    builder = ManifestBuilder("diagnostic-test-v1", purpose="unit-test")
    builder.add_input("rows", input_path, records=2, root=tmp_path)
    builder.add_output("report", output_path, root=tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = builder.write(manifest_path)
    assert read_json(manifest_path, require_object=True) == manifest
    assert manifest["inputs"]["rows"]["sha256"] == digest
    assert manifest["outputs"]["report"]["path"] == "out.json"

    one_shot = build_manifest(
        "diagnostic-test-v1",
        inputs={"rows": input_path},
        outputs={"report": output_path},
        root=tmp_path,
    )
    assert one_shot["inputs"]["rows"]["path"] == "input.jsonl"


def test_digest_validation_rejects_malformed_declarations(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_bytes(b"x")
    with pytest.raises(ValueError, match="64 lowercase"):
        verify_sha256(path, "A")
