#!/usr/bin/env python3
"""Fail closed unless an exported evaluator is the pinned version26 runtime.

The verifier deliberately imports protocol modules in an isolated child interpreter.  It never
places the current checkout's ``src`` directories on ``sys.path``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_LOCK = HERE / "runtime_lock.json"
FORBIDDEN_RUNTIME_MARKERS = (
    "reasoning_parser",
    "reasoning_content",
    "--reasoning-parser",
    "--enable-reasoning",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_tree(root: Path, exported_paths: list[str]) -> tuple[int, str]:
    generated_directories = {
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
    generated_names = {".DS_Store"}
    generated_suffixes = {".pyc", ".pyo"}
    files: list[Path] = []
    for relative in exported_paths:
        directory = root / relative
        if not directory.is_dir():
            raise ValueError(f"missing exported directory: {directory}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not any(part in generated_directories for part in path.parts)
            and path.name not in generated_names
            and path.suffix not in generated_suffixes
        )

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


def isolated_protocol_audit(runtime_root: Path) -> dict:
    audit_program = r'''
import hashlib
import inspect
import json
import pathlib
import sys

runtime = pathlib.Path(sys.argv[1]).resolve()
for relative in ("src/eval", "src/harness", "src/sft"):
    sys.path.insert(0, str(runtime / relative))

import action_carrier
import protocol
import tool_schemes

system = protocol.get_system_prompt()
rolling = protocol.rolling_system_prompt(system, compact=False)
module_paths = {
    "action_carrier": pathlib.Path(inspect.getfile(action_carrier)).resolve(),
    "protocol": pathlib.Path(inspect.getfile(protocol)).resolve(),
    "tool_schemes": pathlib.Path(inspect.getfile(tool_schemes)).resolve(),
}
for name, module_path in module_paths.items():
    if runtime not in module_path.parents:
        raise SystemExit(f"{name} escaped isolated runtime: {module_path}")

print(json.dumps({
    "protocol_version": protocol.PROTOCOL_VERSION,
    "tool_scheme": tool_schemes.ATOMIC_TOOL_SCHEME,
    "tool_scheme_registry_version": tool_schemes.TOOL_SCHEME_REGISTRY_VERSION,
    "assistant_carrier": tool_schemes.ATOMIC_ASSISTANT_CARRIER,
    "rolling_system_prompt_sha256": hashlib.sha256(rolling.encode("utf-8")).hexdigest(),
    "rolling_protocol_hash": protocol.protocol_hash(rolling),
    "tool_names": sorted(protocol.TOOLS),
    "module_paths": {key: str(value) for key, value in module_paths.items()},
}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", audit_program, str(runtime_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"isolated protocol import failed: {detail}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"isolated protocol audit returned invalid JSON: {completed.stdout!r}") from exc


def verify(runtime_root: Path, lock_path: Path) -> dict:
    runtime_root = runtime_root.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_commit = lock["source_commit"]

    copied_lock = runtime_root / "runtime_lock.json"
    if not copied_lock.is_file():
        raise ValueError(f"runtime lock is missing: {copied_lock}")
    if json.loads(copied_lock.read_text(encoding="utf-8")) != lock:
        raise ValueError("runtime lock differs from the repository lock")

    file_count, tree_hash = content_tree(runtime_root, lock["exported_paths"])
    if tree_hash != lock["content_tree_sha256"]:
        raise ValueError(
            f"runtime content tree mismatch: expected {lock['content_tree_sha256']}, got {tree_hash}"
        )

    key_hashes = {}
    for relative, expected_hash in lock["key_file_sha256"].items():
        path = runtime_root / relative
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"key file hash mismatch for {relative}: expected {expected_hash}, got {actual_hash}"
            )
        key_hashes[relative] = actual_hash

    rollout_client = (runtime_root / "src/eval/rollout_passk.py").read_text(encoding="utf-8")
    forbidden_hits = [marker for marker in FORBIDDEN_RUNTIME_MARKERS if marker in rollout_client]
    if forbidden_hits:
        raise ValueError(f"reasoning-parser boundary violated: {forbidden_hits}")
    required_client_fragments = (
        'payload["chat_template_kwargs"] = {"enable_thinking": think == "1"}',
        'return data["choices"][0]["message"]["content"]',
    )
    missing_fragments = [item for item in required_client_fragments if item not in rollout_client]
    if missing_fragments:
        raise ValueError(f"pinned chat client boundary is missing: {missing_fragments}")

    protocol_audit = isolated_protocol_audit(runtime_root)
    expected_protocol = lock["protocol_contract"]
    for field in (
        "protocol_version",
        "tool_scheme",
        "tool_scheme_registry_version",
        "assistant_carrier",
        "rolling_system_prompt_sha256",
        "rolling_protocol_hash",
    ):
        if protocol_audit[field] != expected_protocol[field]:
            raise ValueError(
                f"protocol boundary mismatch for {field}: "
                f"expected {expected_protocol[field]!r}, got {protocol_audit[field]!r}"
            )

    return {
        "status": "ok",
        "isolation": "git-archive-exact-commit",
        "runtime_root": str(runtime_root),
        "source_commit": expected_commit,
        "exported_file_count": file_count,
        "content_tree_sha256": tree_hash,
        "key_file_sha256": key_hashes,
        "protocol": protocol_audit,
        "evaluation_contract": lock["evaluation_contract"],
        "reasoning_parser": None,
        "current_checkout_protocol_modules_imported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--json", action="store_true", help="emit the full machine-readable gate")
    args = parser.parse_args()

    try:
        report = verify(args.runtime_root, args.lock)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"version26 runtime gate failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        protocol = report["protocol"]
        print(
            "version26 runtime gate: OK "
            f"commit={report['source_commit']} "
            f"prompt_sha256={protocol['rolling_system_prompt_sha256']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
