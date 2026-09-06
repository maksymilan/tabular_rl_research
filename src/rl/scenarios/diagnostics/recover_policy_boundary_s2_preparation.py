#!/usr/bin/env python3
"""Recover the two-file publication of the frozen S2 cohort preparer."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence
from unittest import mock


def _load_preparer(path: Path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"preparer must be a regular non-symlink file: {path}")
    spec = importlib.util.spec_from_file_location(
        "_frozen_policy_boundary_s2_preparer_for_recovery", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load preparer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)) or not callable(
        getattr(module, "write_atomic", None)
    ):
        raise ValueError("preparer does not expose the frozen main/write_atomic contract")
    return module


def _capture_expected(
    *, preparer_path: Path, argv: Sequence[str], targets: set[Path]
) -> dict[Path, bytes]:
    preparer = _load_preparer(preparer_path)
    captured: dict[Path, bytes] = {}

    def capture(path: Path, data: bytes) -> None:
        resolved = path.resolve()
        if resolved not in targets or resolved in captured:
            raise RuntimeError(f"preparer emitted an unexpected/duplicate output: {path}")
        if not isinstance(data, bytes):
            raise RuntimeError(f"preparer output is not bytes: {path}")
        captured[resolved] = data

    original_exists = preparer.Path.exists

    def hidden_target_exists(path: Path) -> bool:
        if path.resolve() in targets:
            return False
        return original_exists(path)

    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        mock.patch.object(preparer, "write_atomic", capture),
        mock.patch.object(preparer.Path, "exists", hidden_target_exists),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        result = preparer.main(argv)
    if result != 0 or set(captured) != targets:
        raise ValueError(
            "frozen S2 preparer failed during recovery recomputation: "
            f"code={result} stdout={stdout.getvalue()!r} stderr={stderr.getvalue()!r}"
        )
    return captured


def recover_preparation(
    *,
    preparer_path: Path,
    preparer_argv: Sequence[str],
    output_path: Path,
    manifest_path: Path,
    quarantine_dir: Path,
) -> dict[str, Any]:
    if output_path.parent.resolve() != manifest_path.parent.resolve():
        raise ValueError("S2 output and manifest must share one dedicated directory")
    root = output_path.parent
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"S2 input directory must be a regular directory: {root}")
    if quarantine_dir.exists() and (
        not quarantine_dir.is_dir() or quarantine_dir.is_symlink()
    ):
        raise ValueError(f"quarantine-dir must be a regular directory: {quarantine_dir}")
    targets = (output_path, manifest_path)
    target_set = {path.resolve() for path in targets}
    temporaries = {path: path.with_suffix(path.suffix + ".tmp") for path in targets}
    allowed = {path.name for path in targets} | {
        path.name for path in temporaries.values()
    }
    entries = list(root.iterdir())
    unknown = [path for path in entries if path.name not in allowed]
    if unknown:
        raise ValueError(f"unknown S2 preparation output blocks recovery: {unknown[0]}")
    for path in entries:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"S2 preparation entry is not a regular file: {path}")

    expected = _capture_expected(
        preparer_path=preparer_path,
        argv=preparer_argv,
        targets=target_set,
    )
    # Compare every existing final/temporary before publishing either output.
    for target in targets:
        payload = expected[target.resolve()]
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"existing S2 preparation output differs: {target}")
        temporary = temporaries[target]
        if temporary.exists() and temporary.read_bytes() != payload:
            raise ValueError(f"existing S2 preparation temporary differs: {temporary}")

    published: list[str] = []
    quarantined: list[dict[str, str]] = []
    for target in targets:  # Tasks first, manifest last: same frozen publication order.
        payload = expected[target.resolve()]
        temporary = temporaries[target]
        if not target.exists():
            if not temporary.exists():
                temporary.write_bytes(payload)
            os.replace(temporary, target)
            published.append(str(target.resolve()))
        elif temporary.exists():
            quarantine_dir.mkdir(parents=True, exist_ok=True)
            destination = quarantine_dir / temporary.name
            suffix = 1
            while destination.exists():
                destination = quarantine_dir / f"{temporary.name}.{suffix}"
                suffix += 1
            os.replace(temporary, destination)
            quarantined.append(
                {"from": str(temporary.resolve()), "to": str(destination.resolve())}
            )
    for target in targets:
        if (
            not target.is_file()
            or target.is_symlink()
            or target.read_bytes() != expected[target.resolve()]
        ):
            raise RuntimeError(f"S2 preparation recovery did not publish exact output: {target}")
    return {
        "status": "complete",
        "output": str(output_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "published_missing": published,
        "quarantined_redundant_temporaries": quarantined,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preparer", type=Path, required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    parser.add_argument("--s1-audit", type=Path, required=True)
    parser.add_argument("--expected-s1-audit-sha256", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eligible", type=Path, required=True)
    parser.add_argument("--s1-tasks", type=Path, required=True)
    parser.add_argument("--s1-cohort-manifest", type=Path, required=True)
    parser.add_argument("--baseline-eval300", type=Path, required=True)
    parser.add_argument("--sft1-index", type=Path, required=True)
    parser.add_argument("--old-mixed60", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-db-root", type=Path, required=True)
    args = parser.parse_args(argv)
    preparer_argv = []
    for option, value in (
        ("--s1-audit", args.s1_audit),
        ("--expected-s1-audit-sha256", args.expected_s1_audit_sha256),
        ("--reference", args.reference),
        ("--eligible", args.eligible),
        ("--s1-tasks", args.s1_tasks),
        ("--s1-cohort-manifest", args.s1_cohort_manifest),
        ("--baseline-eval300", args.baseline_eval300),
        ("--sft1-index", args.sft1_index),
        ("--old-mixed60", args.old_mixed60),
        ("--output", args.output),
        ("--manifest", args.manifest),
        ("--remote-db-root", args.remote_db_root),
    ):
        preparer_argv.extend((option, str(value)))
    try:
        result = recover_preparation(
            preparer_path=args.preparer,
            preparer_argv=preparer_argv,
            output_path=args.output,
            manifest_path=args.manifest,
            quarantine_dir=args.quarantine_dir,
        )
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"S2 preparation recovery blocked: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
