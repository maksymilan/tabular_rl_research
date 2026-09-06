#!/usr/bin/env python3
"""Recover publication of a deterministic policy-boundary selection.

The frozen selector computes all cohort semantics. This helper executes that
exact selector with its write function captured in memory, then publishes only
missing canonical files after proving every existing final/temporary byte is
identical. It adds crash recovery without reimplementing selection policy.
"""
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


OUTPUT_NAMES = (
    "boundary332.jsonl",
    "train300.jsonl",
    "validation32.jsonl",
    "boundary_cohort_manifest.json",
)


def _load_selector(path: Path):
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"selector must be a regular non-symlink file: {path}")
    spec = importlib.util.spec_from_file_location(
        "_frozen_policy_boundary_selector_for_recovery", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"could not load selector: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "main", None)) or not callable(
        getattr(module, "_write", None)
    ):
        raise ValueError("selector does not expose the frozen main/_write contract")
    return module


def _capture_expected(
    *,
    selector_path: Path,
    screen_audits: Sequence[Path],
    tasks: Sequence[Path],
    output_dir: Path,
    seed: str,
) -> dict[Path, bytes]:
    selector = _load_selector(selector_path)
    targets = {(output_dir / name).resolve() for name in OUTPUT_NAMES}
    captured: dict[Path, bytes] = {}

    def capture(path: Path, data: bytes) -> None:
        resolved = path.resolve()
        if resolved not in targets or resolved in captured:
            raise RuntimeError(f"selector emitted an unexpected/duplicate output: {path}")
        if not isinstance(data, bytes):
            raise RuntimeError(f"selector output is not bytes: {path}")
        captured[resolved] = data

    original_exists = selector.Path.exists

    def hidden_target_exists(path: Path) -> bool:
        if path.resolve() in targets:
            return False
        return original_exists(path)

    argv: list[str] = []
    for audit in screen_audits:
        argv.extend(("--screen-audit", str(audit)))
    for task_path in tasks:
        argv.extend(("--tasks", str(task_path)))
    argv.extend(("--output-dir", str(output_dir), "--seed", seed))
    stdout, stderr = io.StringIO(), io.StringIO()
    with (
        mock.patch.object(selector, "_write", capture),
        mock.patch.object(selector.Path, "exists", hidden_target_exists),
        contextlib.redirect_stdout(stdout),
        contextlib.redirect_stderr(stderr),
    ):
        result = selector.main(argv)
    if result != 0 or set(captured) != targets:
        raise ValueError(
            "frozen selector failed during recovery recomputation: "
            f"code={result} stdout={stdout.getvalue()!r} stderr={stderr.getvalue()!r}"
        )
    return captured


def recover_selection(
    *,
    selector_path: Path,
    screen_audits: Sequence[Path],
    tasks: Sequence[Path],
    output_dir: Path,
    quarantine_dir: Path,
    seed: str,
) -> dict[str, Any]:
    if len(screen_audits) != len(tasks) or len(tasks) not in {1, 2}:
        raise ValueError("pass one tasks file per screen audit, for S1 or S1+S2")
    for path in (*screen_audits, *tasks):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"selector input must be a regular non-symlink file: {path}")
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError(f"output-dir must be a regular directory: {output_dir}")
    if quarantine_dir.exists() and (
        not quarantine_dir.is_dir() or quarantine_dir.is_symlink()
    ):
        raise ValueError(f"quarantine-dir must be a regular directory: {quarantine_dir}")

    finals = {name: output_dir / name for name in OUTPUT_NAMES}
    temporaries = {name: output_dir / f"{name}.next" for name in OUTPUT_NAMES}
    allowed = set(finals) | {path.name for path in temporaries.values()}
    verification = output_dir / "boundary_selection_verification.json"
    verification_temporary = output_dir / "boundary_selection_verification.json.next"
    allowed.add(verification_temporary.name)
    entries = list(output_dir.iterdir())
    unknown = [
        path
        for path in entries
        if path.name not in allowed and path.name != verification.name
    ]
    if unknown:
        raise ValueError(f"unknown boundary selection output blocks recovery: {unknown[0]}")
    for path in entries:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"selection output entry is not a regular file: {path}")
    existing_finals = sum(path.exists() for path in finals.values())
    if verification.exists() and existing_finals != len(OUTPUT_NAMES):
        raise ValueError("verification exists beside a partial boundary selection")

    expected = _capture_expected(
        selector_path=selector_path,
        screen_audits=screen_audits,
        tasks=tasks,
        output_dir=output_dir,
        seed=seed,
    )
    # Complete the full byte comparison before publishing anything.
    for name, path in finals.items():
        if path.exists() and path.read_bytes() != expected[path.resolve()]:
            raise ValueError(f"existing boundary selection output differs: {path}")
        temporary = temporaries[name]
        if temporary.exists() and temporary.read_bytes() != expected[path.resolve()]:
            raise ValueError(f"existing boundary selection temporary differs: {temporary}")
    if verification_temporary.exists() and existing_finals != len(OUTPUT_NAMES):
        raise ValueError("verification temporary exists beside a partial boundary selection")

    published: list[str] = []
    quarantined: list[dict[str, str]] = []
    for name in OUTPUT_NAMES:
        path, temporary = finals[name], temporaries[name]
        payload = expected[path.resolve()]
        if not path.exists():
            if not temporary.exists():
                temporary.write_bytes(payload)
            os.replace(temporary, path)
            published.append(str(path.resolve()))
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

    if verification_temporary.exists():
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        destination = quarantine_dir / verification_temporary.name
        suffix = 1
        while destination.exists():
            destination = quarantine_dir / f"{verification_temporary.name}.{suffix}"
            suffix += 1
        os.replace(verification_temporary, destination)
        quarantined.append(
            {
                "from": str(verification_temporary.resolve()),
                "to": str(destination.resolve()),
            }
        )

    for path in finals.values():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != expected[path.resolve()]:
            raise RuntimeError(f"selection recovery did not publish exact output: {path}")
    return {
        "status": "complete",
        "outputs": {name: str(path.resolve()) for name, path in finals.items()},
        "published_missing": published,
        "quarantined_redundant_temporaries": quarantined,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--screen-audit", type=Path, action="append", required=True)
    parser.add_argument("--tasks", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quarantine-dir", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    args = parser.parse_args(argv)
    try:
        result = recover_selection(
            selector_path=args.selector,
            screen_audits=args.screen_audit,
            tasks=args.tasks,
            output_dir=args.output_dir,
            quarantine_dir=args.quarantine_dir,
            seed=args.seed,
        )
    except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"boundary selection recovery blocked: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
