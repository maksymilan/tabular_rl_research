#!/usr/bin/env python3
"""Create one read-only provenance manifest for the Qwen3-8B atomic-v26 SFT1 run.

The training artifacts live on NewGNN while evaluation results are written in the local
checkout.  This collector can hash the training side through SSH without copying or changing any
source artifact.  Evaluation result directories are intentionally local: the collector validates
and hashes ``all.jsonl``, ``manifest.json``, the Qwen3 model gate, and the exact-version26 runtime
gate in place.

Two modes are available:

* ``draft`` records stable artifacts that already exist and reports missing/incomplete work.
* ``final`` is fail-closed.  Any missing, unstable, incomplete, malformed, or contract-drifting
  artifact produces a non-zero exit status and a manifest whose status is ``failed``.

The only write is an atomic replacement of the explicitly supplied ``--output`` path.  Existing
outputs are refused unless ``--overwrite`` is supplied.
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


SCHEMA_VERSION = "qwen3-8b-atomic-v26-sft1-artifact-archive-v1"
EXPECTED_SHARDS = {
    "model-00001-of-00005.safetensors": (
        "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f"
    ),
    "model-00002-of-00005.safetensors": (
        "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282"
    ),
    "model-00003-of-00005.safetensors": (
        "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836"
    ),
    "model-00004-of-00005.safetensors": (
        "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a"
    ),
    "model-00005-of-00005.safetensors": (
        "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff"
    ),
}


@dataclass(frozen=True)
class ExperimentContract:
    model_repo: str = "Qwen/Qwen3-8B"
    model_revision: str = "b968826d9c46dd6066d109eabc6255188de91218"
    shard_sha256: dict[str, str] = field(default_factory=lambda: dict(EXPECTED_SHARDS))
    launch_manifest_sha256: str | None = (
        "70a5be796e02ae80ac7cf697a8b6bb1c88cf6744942404210e13df0c31645f2c"
    )
    dataset_sha256: str | None = (
        "c19742ec55d99980075e51a52e79285e4322f89a1d1991dea592748c279e9a14"
    )
    dataset_manifest_sha256: str | None = (
        "58e9eea8ad6ffd056bfe0bd42dd83a3f2592094392dac137b354fc7ee441ed24"
    )
    preparation_manifest_sha256: str | None = (
        "8f6890d951ae8bc66ffc5ec49fbeabf97912d28960c7b266bc65ffb80ee6bca7"
    )
    expected_training_records: int = 4471
    expected_global_step: int = 560
    expected_eval_tasks: int = 1534
    evaluation_source_sha256: str = (
        "8bf5a8bfe93ab49788656e2cc789bf80e729e0ec5f7f40159be01a1ab7b923e0"
    )
    evaluation_derived_sha256: str = (
        "636e096babe2db9dde096b655ed01a0e1e6aae1ea5cbf4e952770e02841721f5"
    )
    version26_source_commit: str = "4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
    version26_content_tree_sha256: str = (
        "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
    )
    student_prompt_sha256: str = (
        "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
    )
    tokenizer_config_sha256: str = (
        "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
    )
    chat_template_sha256: str = (
        "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
    )
    training_python_version: str | None = "3.11.15"
    evaluation_python_version: str | None = "3.11.15"
    training_packages: dict[str, str] = field(
        default_factory=lambda: {
            "torch": "2.6.0",
            "transformers": "5.6.0",
            "peft": "0.18.1",
            "bitsandbytes": "0.46.1",
            "datasets": "4.0.0",
            "llamafactory": "0.9.5",
            "liger-kernel": "0.8.0",
            "safetensors": "0.8.0",
            "accelerate": "1.11.0",
        }
    )
    evaluation_packages: dict[str, str] = field(
        default_factory=lambda: {
            "vllm": "0.19.1",
            "torch": "2.10.0+cu126",
            "transformers": "5.12.0",
            "safetensors": "0.8.0",
        }
    )


DEFAULT_CONTRACT = ExperimentContract()
PACKAGE_NAMES = (
    "torch",
    "transformers",
    "peft",
    "bitsandbytes",
    "datasets",
    "llamafactory",
    "liger-kernel",
    "safetensors",
    "accelerate",
    "vllm",
)


@dataclass
class IssueBook:
    incomplete: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def add_incomplete(self, code: str, message: str, *, path: str = "") -> None:
        item = {"code": code, "message": message}
        if path:
            item["path"] = path
        self.incomplete.append(item)

    def add_error(self, code: str, message: str, *, path: str = "") -> None:
        item = {"code": code, "message": message}
        if path:
            item["path"] = path
        self.errors.append(item)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _snapshot_local(path: str, *, include_content: bool, count_lines: bool) -> dict[str, Any]:
    source = Path(path)
    record: dict[str, Any] = {"path": str(source), "location": "local"}
    try:
        before = source.stat()
    except FileNotFoundError:
        return {**record, "state": "missing"}
    except OSError as exc:
        return {**record, "state": "error", "error": str(exc)}
    if not source.is_file():
        return {**record, "state": "error", "error": "not a regular file"}

    digest = hashlib.sha256()
    content = bytearray() if include_content else None
    nonempty_lines = 0
    try:
        with source.open("rb") as handle:
            if count_lines:
                for line in handle:
                    digest.update(line)
                    if line.strip():
                        nonempty_lines += 1
                    if content is not None:
                        content.extend(line)
            else:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
                    if content is not None:
                        content.extend(chunk)
        after = source.stat()
    except OSError as exc:
        return {**record, "state": "error", "error": str(exc)}
    stable = (
        before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    record.update(
        {
            "state": "present" if stable else "unstable",
            "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest(),
        }
    )
    if count_lines:
        record["nonempty_lines"] = nonempty_lines
    if content is not None:
        record["content_base64"] = base64.b64encode(bytes(content)).decode("ascii")
    return record


REMOTE_HELPER = r'''import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import sys

request = json.load(sys.stdin)
operation = request.get("operation")
if operation == "snapshot":
    path = request["path"]
    include_content = bool(request.get("include_content"))
    count_lines = bool(request.get("count_lines"))
    record = {"path": path, "location": request.get("location", "remote")}
    try:
        before = os.stat(path)
    except FileNotFoundError:
        print(json.dumps({**record, "state": "missing"}))
        raise SystemExit(0)
    except OSError as exc:
        print(json.dumps({**record, "state": "error", "error": str(exc)}))
        raise SystemExit(0)
    if not os.path.isfile(path):
        print(json.dumps({**record, "state": "error", "error": "not a regular file"}))
        raise SystemExit(0)
    digest = hashlib.sha256()
    content = bytearray() if include_content else None
    nonempty_lines = 0
    try:
        with open(path, "rb") as handle:
            if count_lines:
                for line in handle:
                    digest.update(line)
                    if line.strip():
                        nonempty_lines += 1
                    if content is not None:
                        content.extend(line)
            else:
                while True:
                    chunk = handle.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    if content is not None:
                        content.extend(chunk)
        after = os.stat(path)
    except OSError as exc:
        print(json.dumps({**record, "state": "error", "error": str(exc)}))
        raise SystemExit(0)
    stable = (
        before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    record.update({
        "state": "present" if stable else "unstable",
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": digest.hexdigest(),
    })
    if count_lines:
        record["nonempty_lines"] = nonempty_lines
    if content is not None:
        record["content_base64"] = base64.b64encode(bytes(content)).decode("ascii")
    print(json.dumps(record))
elif operation == "environment":
    versions = {}
    for name in request.get("packages", []):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    payload = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }
    payload["probe_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    print(json.dumps(payload))
else:
    raise SystemExit("unknown helper operation")
'''


class Backend:
    def __init__(self, host: str | None, helper_python: str = "python3") -> None:
        self.host = host
        self.helper_python = helper_python
        self.location = f"ssh:{host}" if host else "local"
        if host and not re.fullmatch(r"[A-Za-z0-9_.@-]+", host):
            raise ValueError(f"unsafe SSH host: {host!r}")

    def _run_helper(self, python: str, request: dict[str, Any]) -> dict[str, Any]:
        bootstrap = (
            "import base64;exec(base64.b64decode("
            + repr(base64.b64encode(REMOTE_HELPER.encode("utf-8")).decode("ascii"))
            + "))"
        )
        if self.host:
            command = (
                "PYTHONDONTWRITEBYTECODE=1 "
                + shlex.quote(python)
                + " -c "
                + shlex.quote(bootstrap)
            )
            argv = ["ssh", self.host, command]
        else:
            argv = [python, "-c", bootstrap]
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            argv,
            input=json.dumps({**request, "location": self.location}),
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"helper failed at {self.location} with status {completed.returncode}: {detail}"
            )
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"helper at {self.location} returned invalid JSON: {completed.stdout[:500]!r}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"helper at {self.location} returned a non-object")
        return value

    def snapshot(
        self, path: str, *, include_content: bool = False, count_lines: bool = False
    ) -> dict[str, Any]:
        if not self.host:
            return _snapshot_local(path, include_content=include_content, count_lines=count_lines)
        return self._run_helper(
            self.helper_python,
            {
                "operation": "snapshot",
                "path": path,
                "include_content": include_content,
                "count_lines": count_lines,
            },
        )

    def environment(self, python: str) -> dict[str, Any]:
        return self._run_helper(
            python,
            {"operation": "environment", "packages": list(PACKAGE_NAMES)},
        )


def _public_record(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in snapshot.items() if key != "content_base64"}


def _record_snapshot_issue(snapshot: dict[str, Any], issues: IssueBook, label: str) -> bool:
    state = snapshot.get("state")
    path = str(snapshot.get("path", ""))
    if state == "present":
        return True
    if state == "missing":
        issues.add_incomplete("missing_artifact", f"missing {label}", path=path)
    elif state == "unstable":
        issues.add_incomplete(
            "artifact_changed_while_reading", f"{label} changed while being hashed", path=path
        )
    else:
        issues.add_error(
            "artifact_read_error",
            f"could not read {label}: {snapshot.get('error', 'unknown error')}",
            path=path,
        )
    return False


def _snapshot(
    backend: Backend,
    path: str,
    issues: IssueBook,
    label: str,
    *,
    include_content: bool = False,
    count_lines: bool = False,
) -> dict[str, Any]:
    try:
        value = backend.snapshot(
            path, include_content=include_content, count_lines=count_lines
        )
    except (OSError, RuntimeError, ValueError) as exc:
        value = {
            "path": path,
            "location": backend.location,
            "state": "error",
            "error": str(exc),
        }
    _record_snapshot_issue(value, issues, label)
    return value


def _json_snapshot(
    backend: Backend,
    path: str,
    issues: IssueBook,
    label: str,
    *,
    allow_incomplete_json: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    snapshot = _snapshot(backend, path, issues, label, include_content=True)
    if snapshot.get("state") != "present":
        return snapshot, None
    try:
        raw = base64.b64decode(snapshot["content_base64"], validate=True)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level value is not an object")
    except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        if allow_incomplete_json:
            issues.add_incomplete(
                "incomplete_json_artifact",
                f"{label} is not yet a complete JSON object: {exc}",
                path=str(path),
            )
        else:
            issues.add_error(
                "invalid_json_artifact", f"invalid {label}: {exc}", path=str(path)
            )
        return snapshot, None
    return snapshot, value


def _optional_json_snapshot(
    backend: Backend,
    path: str,
    issues: IssueBook,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Read an optional JSON artifact without treating simple absence as incompleteness.

    ``remote_eval_asset_gate.json`` was introduced after the first local-result format.  Two old
    result directories that both lack it remain supported.  Once a gate exists, however, an
    unstable, unreadable, or incomplete file is still reported and final mode remains fail-closed.
    """
    try:
        snapshot = backend.snapshot(path, include_content=True)
    except (OSError, RuntimeError, ValueError) as exc:
        snapshot = {
            "path": path,
            "location": backend.location,
            "state": "error",
            "error": str(exc),
        }
    if snapshot.get("state") == "missing":
        return snapshot, None
    if not _record_snapshot_issue(snapshot, issues, label):
        return snapshot, None
    try:
        raw = base64.b64decode(snapshot["content_base64"], validate=True)
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("top-level value is not an object")
    except (KeyError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.add_incomplete(
            "incomplete_json_artifact",
            f"{label} is not yet a complete JSON object: {exc}",
            path=path,
        )
        return snapshot, None
    return snapshot, value


def _expect_hash(
    snapshot: dict[str, Any], expected: str | None, issues: IssueBook, label: str
) -> None:
    if expected is None or snapshot.get("state") != "present":
        return
    if snapshot.get("sha256") != expected:
        issues.add_error(
            "sha256_mismatch",
            f"{label} sha256 is {snapshot.get('sha256')}, expected {expected}",
            path=str(snapshot.get("path", "")),
        )


def _expect_equal(
    actual: Any,
    expected: Any,
    issues: IssueBook,
    code: str,
    field_name: str,
    *,
    path: str = "",
) -> None:
    if actual != expected:
        issues.add_error(
            code,
            f"{field_name} is {actual!r}, expected {expected!r}",
            path=path,
        )


def _get_path(override: str | None, source: dict[str, Any] | None, field_name: str) -> str:
    if override:
        return override
    if source and isinstance(source.get(field_name), str):
        return source[field_name]
    return ""


def _join_remote(root: str, name: str) -> str:
    return str(PurePosixPath(root) / name)


def _validate_environment(
    environment: dict[str, Any],
    expected_python: str | None,
    expected_packages: dict[str, str],
    issues: IssueBook,
    label: str,
) -> None:
    if expected_python is not None:
        _expect_equal(
            environment.get("python_version"),
            expected_python,
            issues,
            "environment_version_mismatch",
            f"{label} Python version",
        )
    packages = environment.get("packages")
    if not isinstance(packages, dict):
        issues.add_error("environment_probe_invalid", f"{label} package map is missing")
        return
    for name, expected in expected_packages.items():
        _expect_equal(
            packages.get(name),
            expected,
            issues,
            "environment_version_mismatch",
            f"{label} package {name}",
        )


def _probe_environment(
    backend: Backend,
    python: str,
    expected_python: str | None,
    expected_packages: dict[str, str],
    issues: IssueBook,
    label: str,
) -> dict[str, Any]:
    try:
        environment = backend.environment(python)
    except (OSError, RuntimeError, ValueError) as exc:
        issues.add_error("environment_probe_failed", f"{label} probe failed: {exc}")
        return {"state": "error", "error": str(exc), "python": python}
    environment["state"] = "present"
    environment["location"] = backend.location
    _validate_environment(environment, expected_python, expected_packages, issues, label)
    return environment


def _validate_launch(
    launch: dict[str, Any], contract: ExperimentContract, issues: IssueBook, path: str
) -> None:
    expected = {
        "run_kind": "full",
        "experiment": "qwen3-8b-historical-atomic-version26-sft1-qlora",
        "model_repo": contract.model_repo,
        "model_revision": contract.model_revision,
        "records": contract.expected_training_records,
        "world_size": 2,
        "effective_global_batch": 16,
        "expected_optimizer_steps": contract.expected_global_step,
    }
    for key, value in expected.items():
        _expect_equal(
            launch.get(key), value, issues, "launch_contract_mismatch", f"launch.{key}", path=path
        )
    declared = launch.get("model_shards_sha256")
    _expect_equal(
        declared,
        contract.shard_sha256,
        issues,
        "launch_contract_mismatch",
        "launch.model_shards_sha256",
        path=path,
    )
    launch_versions = {
        "torch": "2.6.0+cu124",
        "transformers": "5.6.0",
        "peft": "0.18.1",
        "bitsandbytes": "0.46.1",
        "datasets": "4.0.0",
        "llamafactory": "0.9.5",
    }
    for key, value in launch_versions.items():
        _expect_equal(
            launch.get(key),
            value,
            issues,
            "launch_environment_mismatch",
            f"launch.{key}",
            path=path,
        )


def _validate_checkpoint(
    checkpoint: dict[str, Any], contract: ExperimentContract, issues: IssueBook
) -> None:
    state = checkpoint.get("trainer_state_json")
    if isinstance(state, dict):
        _expect_equal(
            state.get("global_step"),
            contract.expected_global_step,
            issues,
            "checkpoint_global_step_mismatch",
            "trainer_state.global_step",
        )
    adapter = checkpoint.get("adapter_config_json")
    if isinstance(adapter, dict):
        expected = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
        }
        for key, value in expected.items():
            _expect_equal(
                adapter.get(key),
                value,
                issues,
                "adapter_config_mismatch",
                f"adapter_config.{key}",
            )
        dropout = adapter.get("lora_dropout")
        if not isinstance(dropout, (int, float)) or float(dropout) != 0.05:
            issues.add_error(
                "adapter_config_mismatch",
                f"adapter_config.lora_dropout is {dropout!r}, expected 0.05",
            )


def _collect_training(
    args: argparse.Namespace,
    contract: ExperimentContract,
    issues: IssueBook,
) -> dict[str, Any]:
    backend = Backend(args.training_host, args.remote_helper_python)
    result: dict[str, Any] = {"location": backend.location}

    launch_snapshot, launch = _json_snapshot(
        backend, args.training_launch_manifest, issues, "training launch manifest"
    )
    result["launch_manifest"] = _public_record(launch_snapshot)
    _expect_hash(
        launch_snapshot,
        contract.launch_manifest_sha256,
        issues,
        "training launch manifest",
    )
    if launch is not None:
        _validate_launch(launch, contract, issues, args.training_launch_manifest)
        result["launch_contract"] = launch

    status_snapshot, status = _json_snapshot(
        backend,
        args.training_status_manifest,
        issues,
        "training status manifest",
        allow_incomplete_json=args.mode == "draft",
    )
    result["status_manifest"] = _public_record(status_snapshot)
    if status is not None:
        result["status"] = status
        _expect_equal(
            status.get("success"), True, issues, "training_failed", "status.success"
        )
        _expect_equal(
            status.get("exit_status"), 0, issues, "training_failed", "status.exit_status"
        )
        _expect_equal(
            status.get("expected_global_step"),
            contract.expected_global_step,
            issues,
            "training_status_mismatch",
            "status.expected_global_step",
        )
        _expect_equal(
            status.get("run_kind"),
            "full",
            issues,
            "training_status_mismatch",
            "status.run_kind",
        )
        _expect_equal(
            status.get("log"),
            args.training_log,
            issues,
            "training_status_mismatch",
            "status.log",
        )
        if launch is not None:
            _expect_equal(
                status.get("output_dir"),
                launch.get("output_dir"),
                issues,
                "training_status_mismatch",
                "status.output_dir",
            )
        adapter_model = status.get("adapter_model")
        if not isinstance(adapter_model, str) or not adapter_model:
            issues.add_error(
                "training_status_mismatch",
                "status.adapter_model must identify the saved final adapter weights",
            )

    log_snapshot = _snapshot(
        backend, args.training_log, issues, "training log", include_content=True
    )
    result["log"] = _public_record(log_snapshot)
    if log_snapshot.get("state") == "present" and status is not None:
        raw_log = base64.b64decode(log_snapshot["content_base64"])
        marker = f"final global_step gate passed: {contract.expected_global_step}".encode()
        if marker not in raw_log:
            issues.add_error(
                "training_log_completion_marker_missing",
                f"training log lacks {marker.decode()!r}",
                path=args.training_log,
            )

    base_root = _get_path(args.base_model_root, launch, "model")
    if not base_root:
        issues.add_incomplete("base_model_path_unknown", "cannot derive base model path")
        result["base_model"] = {"state": "unknown"}
    elif args.mode == "draft" and not getattr(args, "rehash_base_shards_in_draft", False):
        # Reading all five shards means streaming roughly the whole 8B checkpoint.  An interim
        # archive must not add that I/O load to a live training/evaluation run.  The immutable
        # launch declaration is retained here, but only final mode (or the explicit opt-in) may
        # claim that the files themselves were rehashed.
        result["base_model"] = {
            "root": base_root,
            "shards": {
                name: {
                    "path": _join_remote(base_root, name),
                    "location": backend.location,
                    "state": "deferred",
                    "declared_sha256": (
                        (launch.get("model_shards_sha256") or {}).get(name)
                        if isinstance(launch, dict)
                        else None
                    ),
                    "expected_sha256": expected_hash,
                }
                for name, expected_hash in sorted(contract.shard_sha256.items())
            },
        }
        issues.add_incomplete(
            "base_shard_rehash_deferred",
            "draft mode deferred the five large base-shard reads; final mode rehashes them",
            path=base_root,
        )
    else:
        base_artifacts: dict[str, Any] = {"root": base_root, "shards": {}}
        for name, expected_hash in sorted(contract.shard_sha256.items()):
            snapshot = _snapshot(
                backend,
                _join_remote(base_root, name),
                issues,
                f"base model shard {name}",
            )
            _expect_hash(snapshot, expected_hash, issues, f"base model shard {name}")
            base_artifacts["shards"][name] = _public_record(snapshot)
        result["base_model"] = base_artifacts

    output_root = _get_path(None, launch, "output_dir")
    checkpoint_root = args.checkpoint_dir or (
        _join_remote(output_root, f"checkpoint-{contract.expected_global_step}")
        if output_root
        else ""
    )
    checkpoint: dict[str, Any] = {"path": checkpoint_root, "files": {}}
    if not checkpoint_root:
        issues.add_incomplete("checkpoint_path_unknown", "cannot derive checkpoint-560 path")
    else:
        json_values: dict[str, dict[str, Any] | None] = {}
        for name in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "trainer_state.json",
            "training_args.bin",
        ):
            path = _join_remote(checkpoint_root, name)
            if name.endswith(".json"):
                snapshot, value = _json_snapshot(
                    backend,
                    path,
                    issues,
                    f"checkpoint {name}",
                    allow_incomplete_json=args.mode == "draft",
                )
                json_values[name] = value
            else:
                snapshot = _snapshot(backend, path, issues, f"checkpoint {name}")
            checkpoint["files"][name] = _public_record(snapshot)
        checkpoint["trainer_state_json"] = json_values.get("trainer_state.json")
        checkpoint["adapter_config_json"] = json_values.get("adapter_config.json")
        _validate_checkpoint(checkpoint, contract, issues)
    result["checkpoint_560"] = checkpoint

    dataset_path = _get_path(args.training_dataset, launch, "dataset")
    dataset_manifest_path = _get_path(args.training_dataset_manifest, launch, "dataset_manifest")
    preparation_path = _get_path(args.preparation_manifest, launch, "preparation_manifest")
    dataset_info_path = _get_path(args.dataset_info, launch, "dataset_info")
    data_artifacts: dict[str, Any] = {}
    dataset_snapshot = _snapshot(
        backend,
        dataset_path,
        issues,
        "training dataset",
        count_lines=True,
    ) if dataset_path else {"path": "", "location": backend.location, "state": "missing"}
    if not dataset_path:
        _record_snapshot_issue(dataset_snapshot, issues, "training dataset")
    _expect_hash(dataset_snapshot, contract.dataset_sha256, issues, "training dataset")
    if dataset_snapshot.get("state") == "present":
        _expect_equal(
            dataset_snapshot.get("nonempty_lines"),
            contract.expected_training_records,
            issues,
            "training_record_count_mismatch",
            "training dataset nonempty line count",
            path=dataset_path,
        )
    data_artifacts["dataset"] = _public_record(dataset_snapshot)

    dataset_manifest_snapshot, dataset_manifest = _json_snapshot(
        backend, dataset_manifest_path, issues, "training dataset manifest"
    ) if dataset_manifest_path else (
        {"path": "", "location": backend.location, "state": "missing"}, None
    )
    if not dataset_manifest_path:
        _record_snapshot_issue(dataset_manifest_snapshot, issues, "training dataset manifest")
    _expect_hash(
        dataset_manifest_snapshot,
        contract.dataset_manifest_sha256,
        issues,
        "training dataset manifest",
    )
    data_artifacts["dataset_manifest"] = _public_record(dataset_manifest_snapshot)
    if dataset_manifest is not None:
        data_artifacts["dataset_manifest_contract"] = dataset_manifest
        _expect_equal(
            dataset_manifest.get("output_sha256"),
            contract.dataset_sha256,
            issues,
            "dataset_manifest_binding_mismatch",
            "dataset manifest output_sha256",
            path=dataset_manifest_path,
        )

    prep_snapshot, prep = _json_snapshot(
        backend, preparation_path, issues, "preparation manifest"
    ) if preparation_path else (
        {"path": "", "location": backend.location, "state": "missing"}, None
    )
    if not preparation_path:
        _record_snapshot_issue(prep_snapshot, issues, "preparation manifest")
    _expect_hash(
        prep_snapshot,
        contract.preparation_manifest_sha256,
        issues,
        "preparation manifest",
    )
    data_artifacts["preparation_manifest"] = _public_record(prep_snapshot)
    if prep is not None:
        data_artifacts["preparation_contract"] = prep
        files = prep.get("files")
        if dataset_path and isinstance(files, dict):
            _expect_equal(
                files.get(dataset_path),
                contract.dataset_sha256,
                issues,
                "preparation_manifest_binding_mismatch",
                "preparation manifest dataset hash",
                path=preparation_path,
            )

    if dataset_info_path:
        info_snapshot = _snapshot(backend, dataset_info_path, issues, "dataset_info.json")
        data_artifacts["dataset_info"] = _public_record(info_snapshot)
        if launch is not None:
            _expect_hash(
                info_snapshot,
                launch.get("dataset_info_sha256"),
                issues,
                "dataset_info.json",
            )
    result["data"] = data_artifacts

    training_python = args.training_python or (
        launch.get("python") if isinstance(launch, dict) else ""
    )
    if not isinstance(training_python, str) or not training_python:
        issues.add_incomplete("training_python_unknown", "cannot derive training Python")
        result["environment"] = {"state": "unknown"}
    else:
        result["environment"] = _probe_environment(
            backend,
            training_python,
            contract.training_python_version,
            contract.training_packages,
            issues,
            "training environment",
        )
    return result


EVAL_MANIFEST_LOCK = {
    "runner": "tool_rollout_passk",
    "tool_scheme": "atomic",
    "assistant_carrier": "think-json-v1",
    "protocol_version": "version26",
    "n_samples": 1,
    "stop_on_success": False,
    "pass_k": [1],
    "sample_detail": "full",
    "temperature": 0.0,
    "top_p": 1.0,
    "max_steps": 30,
    "few_shot": 0,
    "context_mode": "rolling-legal-history",
    "history_turns": 4,
    "rolling_prompt_variant": "full",
    "rolling_observation_style": "resident",
    "denotation_comparison": "bird-set",
    "dataset_purpose": "evaluation",
}


def _validate_eval_manifest(
    manifest: dict[str, Any], contract: ExperimentContract, issues: IssueBook, label: str, path: str
) -> None:
    for key, value in EVAL_MANIFEST_LOCK.items():
        _expect_equal(
            manifest.get(key),
            value,
            issues,
            "evaluation_manifest_mismatch",
            f"{label} manifest.{key}",
            path=path,
        )
    _expect_equal(
        manifest.get("requested_size"),
        contract.expected_eval_tasks,
        issues,
        "evaluation_manifest_mismatch",
        f"{label} manifest.requested_size",
        path=path,
    )
    enable_thinking = str(manifest.get("enable_thinking")).lower()
    if enable_thinking not in {"1", "true"}:
        issues.add_error(
            "evaluation_manifest_mismatch",
            f"{label} manifest.enable_thinking is not true",
            path=path,
        )
    max_tokens = manifest.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 2048:
        issues.add_error(
            "evaluation_manifest_mismatch",
            f"{label} manifest.max_tokens is {max_tokens!r}, expected >=2048",
            path=path,
        )
    system_prompt = manifest.get("system_prompt")
    if not isinstance(system_prompt, str) or _sha256_bytes(system_prompt.encode()) != contract.student_prompt_sha256:
        issues.add_error(
            "evaluation_prompt_mismatch",
            f"{label} system prompt does not match the frozen version26 prompt hash",
            path=path,
        )


def _scan_all_jsonl(
    path: Path, contract: ExperimentContract, issues: IssueBook, label: str
) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "location": "local"}
    try:
        before = path.stat()
    except FileNotFoundError:
        record["state"] = "missing"
        _record_snapshot_issue(record, issues, f"{label} all.jsonl")
        return record
    except OSError as exc:
        record.update({"state": "error", "error": str(exc)})
        _record_snapshot_issue(record, issues, f"{label} all.jsonl")
        return record
    digest = hashlib.sha256()
    indices: set[int] = set()
    correct_count = 0
    legal_count = 0
    malformed: str | None = None
    try:
        with path.open("rb") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                digest.update(raw_line)
                if malformed is not None:
                    continue
                if not raw_line.strip():
                    malformed = f"blank line {line_number}"
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    malformed = f"invalid JSON at line {line_number}: {exc}"
                    continue
                index = row.get("example_index") if isinstance(row, dict) else None
                if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                    malformed = f"invalid example_index at line {line_number}: {index!r}"
                    continue
                if index in indices:
                    malformed = f"duplicate example_index {index} at line {line_number}"
                    continue
                indices.add(index)
                if row.get("tool_scheme") != "atomic" or row.get("protocol_version") != "version26":
                    malformed = f"protocol drift at line {line_number}"
                    continue
                samples = row.get("samples")
                if not isinstance(samples, list) or len(samples) != 1 or not isinstance(samples[0], dict):
                    malformed = f"expected exactly one sample at line {line_number}"
                    continue
                sample = samples[0]
                if type(sample.get("correct")) is not bool or type(sample.get("legal")) is not bool:
                    malformed = f"non-boolean sample outcome at line {line_number}"
                    continue
                correct_count += int(sample["correct"])
                legal_count += int(sample["legal"])
        after = path.stat()
    except OSError as exc:
        record.update({"state": "error", "error": str(exc)})
        _record_snapshot_issue(record, issues, f"{label} all.jsonl")
        return record
    stable = (
        before.st_ino == after.st_ino
        and before.st_size == after.st_size
        and before.st_mtime_ns == after.st_mtime_ns
    )
    record.update(
        {
            "state": "present" if stable else "unstable",
            "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
            "sha256": digest.hexdigest(),
            "records": len(indices),
            "unique_example_indices": len(indices),
            "correct": correct_count,
            "legal": legal_count,
        }
    )
    _record_snapshot_issue(record, issues, f"{label} all.jsonl")
    if malformed:
        issues.add_error("invalid_evaluation_jsonl", f"{label}: {malformed}", path=str(path))
        record["validation_error"] = malformed
        return record
    expected_indices = set(range(contract.expected_eval_tasks))
    if indices != expected_indices:
        if len(indices) < contract.expected_eval_tasks and indices.issubset(expected_indices):
            issues.add_incomplete(
                "evaluation_incomplete",
                f"{label} has {len(indices)}/{contract.expected_eval_tasks} task records",
                path=str(path),
            )
        else:
            issues.add_error(
                "evaluation_index_set_mismatch",
                f"{label} example-index set is not 0..{contract.expected_eval_tasks - 1}",
                path=str(path),
            )
    return record


def _validate_runtime_gate(
    gate: dict[str, Any], contract: ExperimentContract, issues: IssueBook, label: str, path: str
) -> None:
    expected = {
        "status": "ok",
        "source_commit": contract.version26_source_commit,
        "content_tree_sha256": contract.version26_content_tree_sha256,
        "reasoning_parser": None,
    }
    for key, value in expected.items():
        _expect_equal(
            gate.get(key),
            value,
            issues,
            "runtime_gate_mismatch",
            f"{label} runtime gate.{key}",
            path=path,
        )
    protocol = gate.get("protocol")
    if not isinstance(protocol, dict):
        issues.add_error("runtime_gate_mismatch", f"{label} runtime gate protocol is missing")
        return
    protocol_expected = {
        "protocol_version": "version26",
        "tool_scheme": "atomic",
        "assistant_carrier": "think-json-v1",
        "rolling_system_prompt_sha256": contract.student_prompt_sha256,
    }
    for key, value in protocol_expected.items():
        _expect_equal(
            protocol.get(key),
            value,
            issues,
            "runtime_gate_mismatch",
            f"{label} runtime gate.protocol.{key}",
            path=path,
        )


def _validate_model_gate(
    gate: dict[str, Any],
    contract: ExperimentContract,
    issues: IssueBook,
    label: str,
    path: str,
    *,
    adapter: bool,
    checkpoint_path: str,
) -> None:
    expected = {
        "status": "ok",
        "assumed_huggingface_revision": contract.model_revision,
        "model_type": "qwen3",
        "architecture": "Qwen3ForCausalLM",
        "tokenizer_config_sha256": contract.tokenizer_config_sha256,
        "chat_template_sha256": contract.chat_template_sha256,
        "official_chat_template": True,
        "enable_thinking": True,
        "reasoning_parser": None,
    }
    for key, value in expected.items():
        _expect_equal(
            gate.get(key),
            value,
            issues,
            "model_gate_mismatch",
            f"{label} model gate.{key}",
            path=path,
        )
    adapter_value = gate.get("adapter")
    if not adapter:
        _expect_equal(
            adapter_value,
            None,
            issues,
            "model_gate_mismatch",
            f"{label} model gate.adapter",
            path=path,
        )
    elif not isinstance(adapter_value, dict):
        issues.add_error("model_gate_mismatch", f"{label} adapter model gate has no adapter")
    else:
        _expect_equal(
            adapter_value.get("peft_type"),
            "LORA",
            issues,
            "model_gate_mismatch",
            f"{label} adapter peft_type",
            path=path,
        )
        _expect_equal(
            adapter_value.get("rank"),
            16,
            issues,
            "model_gate_mismatch",
            f"{label} adapter rank",
            path=path,
        )
        if checkpoint_path:
            _expect_equal(
                str(PurePosixPath(str(adapter_value.get("path", "")))),
                str(PurePosixPath(checkpoint_path)),
                issues,
                "model_gate_mismatch",
                f"{label} adapter checkpoint path",
                path=path,
            )


def _validate_remote_eval_asset_gate(
    gate: dict[str, Any],
    contract: ExperimentContract,
    issues: IssueBook,
    label: str,
    path: str,
) -> None:
    expected_top_level = {
        "status": "ok",
        "schema_version": "qwen3-atomic-v26-all-newgnn-assets-v1",
        "mode": label,
    }
    for key, value in expected_top_level.items():
        _expect_equal(
            gate.get(key),
            value,
            issues,
            "remote_eval_asset_gate_mismatch",
            f"{label} remote asset gate.{key}",
            path=path,
        )
    evaluation_input = gate.get("evaluation_input")
    if not isinstance(evaluation_input, dict):
        issues.add_error(
            "remote_eval_asset_gate_mismatch",
            f"{label} remote asset gate has no evaluation_input object",
            path=path,
        )
        return
    expected_input = {
        "source_sha256": contract.evaluation_source_sha256,
        "derived_sha256": contract.evaluation_derived_sha256,
        "records": contract.expected_eval_tasks,
        "only_db_path_changed": True,
    }
    for key, value in expected_input.items():
        _expect_equal(
            evaluation_input.get(key),
            value,
            issues,
            "remote_eval_asset_gate_mismatch",
            f"{label} remote asset gate.evaluation_input.{key}",
            path=path,
        )
    databases = evaluation_input.get("databases")
    if not isinstance(databases, dict) or not databases:
        issues.add_error(
            "remote_eval_asset_gate_mismatch",
            f"{label} remote asset gate has no verified database map",
            path=path,
        )
        return
    database_records = 0
    for db_id, entry in databases.items():
        if not isinstance(db_id, str) or not db_id or not isinstance(entry, dict):
            issues.add_error(
                "remote_eval_asset_gate_mismatch",
                f"{label} remote asset gate has a malformed database entry",
                path=path,
            )
            return
        records = entry.get("records")
        digest = entry.get("sha256")
        if not isinstance(records, int) or isinstance(records, bool) or records < 0:
            issues.add_error(
                "remote_eval_asset_gate_mismatch",
                f"{label} remote asset gate database {db_id} has invalid records={records!r}",
                path=path,
            )
            return
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            issues.add_error(
                "remote_eval_asset_gate_mismatch",
                f"{label} remote asset gate database {db_id} has an invalid sha256",
                path=path,
            )
            return
        database_records += records
    _expect_equal(
        database_records,
        contract.expected_eval_tasks,
        issues,
        "remote_eval_asset_gate_mismatch",
        f"{label} remote asset gate summed database records",
        path=path,
    )


def _collect_eval_run(
    label: str,
    result_dir: Path,
    contract: ExperimentContract,
    issues: IssueBook,
    *,
    adapter: bool,
    checkpoint_path: str,
) -> dict[str, Any]:
    backend = Backend(None)
    result: dict[str, Any] = {"result_dir": str(result_dir), "location": "local"}
    result["all_jsonl"] = _scan_all_jsonl(
        result_dir / "all.jsonl", contract, issues, label
    )
    manifest_snapshot, manifest = _json_snapshot(
        backend,
        str(result_dir / "manifest.json"),
        issues,
        f"{label} evaluation manifest",
        allow_incomplete_json=True,
    )
    result["manifest"] = _public_record(manifest_snapshot)
    if manifest is not None:
        result["manifest_contract"] = manifest
        _validate_eval_manifest(
            manifest, contract, issues, label, str(result_dir / "manifest.json")
        )
    model_snapshot, model_gate = _json_snapshot(
        backend,
        str(result_dir / "qwen3_model_gate.json"),
        issues,
        f"{label} model gate",
        allow_incomplete_json=True,
    )
    result["model_gate"] = _public_record(model_snapshot)
    if model_gate is not None:
        result["model_gate_contract"] = model_gate
        _validate_model_gate(
            model_gate,
            contract,
            issues,
            label,
            str(result_dir / "qwen3_model_gate.json"),
            adapter=adapter,
            checkpoint_path=checkpoint_path,
        )
    runtime_snapshot, runtime_gate = _json_snapshot(
        backend,
        str(result_dir / "version26_runtime_gate.json"),
        issues,
        f"{label} version26 runtime gate",
        allow_incomplete_json=True,
    )
    result["runtime_gate"] = _public_record(runtime_snapshot)
    if runtime_gate is not None:
        result["runtime_gate_contract"] = runtime_gate
        _validate_runtime_gate(
            runtime_gate,
            contract,
            issues,
            label,
            str(result_dir / "version26_runtime_gate.json"),
        )
    asset_gate_path = str(result_dir / "remote_eval_asset_gate.json")
    asset_snapshot, asset_gate = _optional_json_snapshot(
        backend,
        asset_gate_path,
        issues,
        f"{label} remote evaluation asset gate",
    )
    result["remote_eval_asset_gate"] = _public_record(asset_snapshot)
    result["remote_eval_asset_gate_present"] = asset_snapshot.get("state") != "missing"
    if asset_gate is not None:
        result["remote_eval_asset_gate_contract"] = asset_gate
        _validate_remote_eval_asset_gate(
            asset_gate,
            contract,
            issues,
            label,
            asset_gate_path,
        )
    return result


EVAL_COMPARE_FIELDS = (
    "runner",
    "tool_scheme",
    "tool_scheme_registry_version",
    "assistant_carrier",
    "protocol_version",
    "protocol_hash",
    "requested_size",
    "n_samples",
    "stop_on_success",
    "pass_k",
    "sample_detail",
    "temperature",
    "top_p",
    "max_tokens",
    "max_steps",
    "few_shot",
    "enable_thinking",
    "system_prompt_variant",
    "system_prompt",
    "context_mode",
    "history_turns",
    "rolling_prompt_variant",
    "rolling_observation_style",
    "denotation_comparison",
    "dataset_purpose",
)


def _compare_evaluations(
    base: dict[str, Any], adapter: dict[str, Any], issues: IssueBook
) -> dict[str, Any]:
    base_manifest = base.get("manifest_contract")
    adapter_manifest = adapter.get("manifest_contract")
    base_asset = base.get("remote_eval_asset_gate_contract")
    adapter_asset = adapter.get("remote_eval_asset_gate_contract")
    base_asset_present = bool(base.get("remote_eval_asset_gate_present"))
    adapter_asset_present = bool(adapter.get("remote_eval_asset_gate_present"))
    compared: dict[str, Any] = {
        "fields": list(EVAL_COMPARE_FIELDS),
        "equal": None,
        "manifest_fields_equal": None,
        "dataset_identity_equal": None,
        "dataset_identity_strategy": None,
    }
    if isinstance(base_manifest, dict) and isinstance(adapter_manifest, dict):
        differences = {
            field: {"base": base_manifest.get(field), "adapter": adapter_manifest.get(field)}
            for field in EVAL_COMPARE_FIELDS
            if base_manifest.get(field) != adapter_manifest.get(field)
        }
        compared["manifest_fields_equal"] = not differences
        compared["differences"] = differences
        if differences:
            issues.add_error(
                "evaluation_contracts_differ",
                f"base and adapter evaluation manifests differ: {sorted(differences)}",
            )
    if isinstance(base_asset, dict) and isinstance(adapter_asset, dict):
        base_input = base_asset.get("evaluation_input") or {}
        adapter_input = adapter_asset.get("evaluation_input") or {}
        base_derived = base_input.get("derived_sha256")
        adapter_derived = adapter_input.get("derived_sha256")
        compared.update(
            {
                "dataset_identity_strategy": "remote_eval_asset_gate_derived_sha256",
                "base_manifest_dataset": (
                    base_manifest.get("dataset") if isinstance(base_manifest, dict) else None
                ),
                "adapter_manifest_dataset": (
                    adapter_manifest.get("dataset")
                    if isinstance(adapter_manifest, dict)
                    else None
                ),
                "base_derived_sha256": base_derived,
                "adapter_derived_sha256": adapter_derived,
                "derived_sha256_equal": base_derived == adapter_derived,
                "dataset_identity_equal": base_derived == adapter_derived,
            }
        )
        if base_derived != adapter_derived:
            issues.add_error(
                "evaluation_dataset_hashes_differ",
                "base and adapter remote asset gates have different derived evaluation hashes",
            )
    elif not base_asset_present and not adapter_asset_present:
        # Backward compatibility for the original local launcher.  Those result manifests both
        # point at the same local BIRD input, so absolute-path equality remains meaningful there.
        base_dataset = base_manifest.get("dataset") if isinstance(base_manifest, dict) else None
        adapter_dataset = (
            adapter_manifest.get("dataset") if isinstance(adapter_manifest, dict) else None
        )
        compared.update(
            {
                "dataset_identity_strategy": "legacy_manifest_dataset_path",
                "base_manifest_dataset": base_dataset,
                "adapter_manifest_dataset": adapter_dataset,
                "manifest_dataset_equal": (
                    base_dataset == adapter_dataset
                    if base_dataset is not None and adapter_dataset is not None
                    else None
                ),
                "dataset_identity_equal": (
                    base_dataset == adapter_dataset
                    if base_dataset is not None and adapter_dataset is not None
                    else None
                ),
            }
        )
        if base_dataset is not None and adapter_dataset is not None and base_dataset != adapter_dataset:
            issues.add_error(
                "evaluation_dataset_paths_differ",
                "legacy base and adapter manifests point at different evaluation datasets",
            )
    else:
        compared["dataset_identity_strategy"] = "incomplete_remote_eval_asset_gate_pair"
        issues.add_incomplete(
            "remote_eval_asset_gate_pair_incomplete",
            "remote_eval_asset_gate.json is present/valid for only one evaluation arm",
        )
    base_runtime = (base.get("runtime_gate") or {}).get("sha256")
    adapter_runtime = (adapter.get("runtime_gate") or {}).get("sha256")
    compared["runtime_gate_sha256_equal"] = (
        base_runtime == adapter_runtime if base_runtime and adapter_runtime else None
    )
    if base_runtime and adapter_runtime and base_runtime != adapter_runtime:
        issues.add_error(
            "runtime_gate_hashes_differ",
            "base and adapter version26 runtime-gate files are not byte-identical",
        )
    equality_parts = (
        compared.get("manifest_fields_equal"),
        compared.get("dataset_identity_equal"),
        compared.get("runtime_gate_sha256_equal"),
    )
    compared["equal"] = (
        all(equality_parts) if all(value is not None for value in equality_parts) else None
    )
    return compared


def collect_archive(
    args: argparse.Namespace, contract: ExperimentContract = DEFAULT_CONTRACT
) -> tuple[dict[str, Any], int]:
    issues = IssueBook()
    training = _collect_training(args, contract, issues)
    checkpoint_path = str((training.get("checkpoint_560") or {}).get("path", ""))
    base_eval = _collect_eval_run(
        "base",
        Path(args.base_result_dir),
        contract,
        issues,
        adapter=False,
        checkpoint_path=checkpoint_path,
    )
    adapter_eval = _collect_eval_run(
        "adapter",
        Path(args.adapter_result_dir),
        contract,
        issues,
        adapter=True,
        checkpoint_path=checkpoint_path,
    )
    comparison = _compare_evaluations(base_eval, adapter_eval, issues)

    evaluation_backend = Backend(args.evaluation_host, args.remote_helper_python)
    evaluation_environment = _probe_environment(
        evaluation_backend,
        args.evaluation_python,
        contract.evaluation_python_version,
        contract.evaluation_packages,
        issues,
        "evaluation environment",
    )

    complete = not issues.incomplete and not issues.errors
    if args.mode == "final":
        status = "ok" if complete else "failed"
        exit_status = 0 if complete else 2
    else:
        status = "ok" if complete else ("invalid" if issues.errors else "incomplete")
        exit_status = 2 if issues.errors else 0

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "status": status,
        "complete": complete,
        "final_ready": complete,
        "read_only_source_audit": True,
        "experiment": "qwen3-8b-historical-atomic-version26-sft1-qlora",
        "protocol_boundary": (
            "frozen historical atomic version26 think-json-v1; not current version54 "
            "native-tool-bundle and not an exact TRUST-SQL training reproduction"
        ),
        "contract": dataclasses.asdict(contract),
        "artifacts": {
            "training": training,
            "evaluations": {"base": base_eval, "adapter": adapter_eval},
            "evaluation_environment": evaluation_environment,
        },
        "cross_run_checks": comparison,
        "issues": {
            "incomplete": issues.incomplete,
            "errors": issues.errors,
        },
        "issue_counts": {
            "incomplete": len(issues.incomplete),
            "errors": len(issues.errors),
        },
    }
    manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
    return manifest, exit_status


def _write_manifest(path: Path, manifest: dict[str, Any], *, overwrite: bool) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("draft", "final"), default="final")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--training-host", default="NewGNN")
    parser.add_argument("--evaluation-host", default="NewGNN")
    parser.add_argument("--remote-helper-python", default="python3")
    parser.add_argument(
        "--rehash-base-shards-in-draft",
        action="store_true",
        help="opt in to streaming all five large base shards in draft mode; final mode always rehashes",
    )
    parser.add_argument(
        "--training-launch-manifest",
        default=(
            "/home/dengyan/tabular_rl_outputs/logs/"
            "qwen3_8b_atomic_v26_sft1_full_20260806_235100.launch_manifest.json"
        ),
    )
    parser.add_argument(
        "--training-status-manifest",
        default=(
            "/home/dengyan/tabular_rl_outputs/logs/"
            "qwen3_8b_atomic_v26_sft1_full_20260806_235100.status.json"
        ),
    )
    parser.add_argument(
        "--training-log",
        default=(
            "/home/dengyan/tabular_rl_outputs/logs/"
            "qwen3_8b_atomic_v26_sft1_full_20260806_235100.log"
        ),
    )
    parser.add_argument("--base-model-root")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--training-dataset")
    parser.add_argument("--training-dataset-manifest")
    parser.add_argument("--preparation-manifest")
    parser.add_argument("--dataset-info")
    parser.add_argument("--training-python")
    parser.add_argument(
        "--evaluation-python",
        default="/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python",
    )
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument(
        "--base-result-dir",
        default=str(
            project_root
            / "data/results/qwen3_8b_base_atomic_v26_bird_dev1534_greedy1_bird_set"
        ),
    )
    parser.add_argument(
        "--adapter-result-dir",
        default=str(
            project_root
            / "data/results/qwen3_8b_sft1_qlora_atomic_v26_bird_dev1534_greedy1_bird_set"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest, exit_status = collect_archive(args)
        _write_manifest(args.output, manifest, overwrite=args.overwrite)
    except (FileExistsError, OSError, RuntimeError, ValueError) as exc:
        print(f"artifact archive failed before output: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "mode": manifest["mode"],
                "status": manifest["status"],
                "complete": manifest["complete"],
                "incomplete": manifest["issue_counts"]["incomplete"],
                "errors": manifest["issue_counts"]["errors"],
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            },
            sort_keys=True,
        )
    )
    return exit_status


if __name__ == "__main__":
    raise SystemExit(main())
