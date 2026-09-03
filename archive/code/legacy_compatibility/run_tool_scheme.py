#!/usr/bin/env python3
"""Stable evaluation launcher for one explicitly selected table-tool scheme.

The selected scheme is exclusive: a model sees exactly one registered top-level action space,
never a merged combination. Existing runner CLIs remain available for artifact reproduction; this
launcher is the preferred entry point for paired future runs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from tool_modules.registry import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    ATOMIC_TOOL_SCHEME,
    CHECKPOINT_RELALG_TOOL_SCHEME,
    DIRECT_SQL_SEARCH_TOOL_SCHEME,
    ITERATIVE_SQL_TOOL_SCHEME,
    NATIVE_TOOL_BUNDLE_SCHEME,
    RELATIONAL_PROGRAM_TOOL_SCHEME,
    TOOL_SCHEME_NAMES,
)
from tool_modules.direct_sql_search.protocol import (  # noqa: E402
    DIRECT_SQL_SEARCH_INTERFACE,
)
from tool_modules.iterative_sql.protocol import ITERATIVE_SQL_INTERFACE  # noqa: E402


def runner_argv(tool_scheme: str, forwarded: list[str]) -> list[str]:
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if tool_scheme == ATOMIC_TOOL_SCHEME:
        script = HERE / "rollout.py"
        return [sys.executable, str(script), *forwarded]
    if tool_scheme == NATIVE_TOOL_BUNDLE_SCHEME:
        script = ROOT / "src" / "sft" / "generate_teacher_rollouts.py"
        return [
            sys.executable,
            str(script),
            *forwarded,
            "--atomic-protocol-version",
            "version54",
            "--deepseek-carrier",
            "native-tool-bundle",
            "--diagnostic-only",
        ]
    if tool_scheme == CHECKPOINT_RELALG_TOOL_SCHEME:
        script = ROOT / "src" / "tool_modules" / "checkpoint_relalg" / "runner.py"
        return [sys.executable, str(script), *forwarded]
    if tool_scheme == ACTION_BLOCK_TOOL_SCHEME:
        script = ROOT / "src" / "tool_modules" / "action_block" / "evaluator.py"
        return [sys.executable, str(script), *forwarded]
    if tool_scheme == RELATIONAL_PROGRAM_TOOL_SCHEME:
        script = ROOT / "src" / "tool_modules" / "relational_program" / "evaluator.py"
        return [sys.executable, str(script), *forwarded]
    if tool_scheme == DIRECT_SQL_SEARCH_TOOL_SCHEME:
        script = ROOT / "src" / "tool_modules" / "sql_common" / "runner.py"
        return [
            sys.executable,
            str(script),
            *forwarded,
            "--interface",
            DIRECT_SQL_SEARCH_INTERFACE,
        ]
    if tool_scheme == ITERATIVE_SQL_TOOL_SCHEME:
        script = ROOT / "src" / "tool_modules" / "sql_common" / "runner.py"
        return [
            sys.executable,
            str(script),
            *forwarded,
            "--interface",
            ITERATIVE_SQL_INTERFACE,
        ]
    raise ValueError(f"unsupported tool scheme: {tool_scheme}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one explicitly selected tool scheme. Place runner-specific arguments "
            "after --."
        )
    )
    parser.add_argument("--tool-scheme", choices=TOOL_SCHEME_NAMES, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved child command without executing it",
    )
    args, forwarded = parser.parse_known_args()
    command = runner_argv(args.tool_scheme, forwarded)
    if args.dry_run:
        print(json.dumps({
            "tool_scheme": args.tool_scheme,
            "command": command,
        }, ensure_ascii=False))
        return 0
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
