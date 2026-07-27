#!/usr/bin/env python3
"""Stable causal-rollout launcher for either public table-tool scheme."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src" / "eval"), str(HERE)]

from tool_schemes import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    ATOMIC_TOOL_SCHEME,
    TOOL_SCHEME_NAMES,
)


def generator_argv(tool_scheme: str, forwarded: list[str]) -> list[str]:
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if tool_scheme == ATOMIC_TOOL_SCHEME:
        return [
            sys.executable,
            str(HERE / "generate_teacher_rollouts.py"),
            *forwarded,
        ]
    if tool_scheme == ACTION_BLOCK_TOOL_SCHEME:
        return [
            sys.executable,
            str(ROOT / "src" / "eval" / "evaluate_batch_plan.py"),
            *forwarded,
        ]
    raise ValueError(f"unsupported tool scheme: {tool_scheme}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-scheme", choices=TOOL_SCHEME_NAMES, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args()
    command = generator_argv(args.tool_scheme, forwarded)
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
