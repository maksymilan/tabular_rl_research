#!/usr/bin/env python3
"""Plan one of the four supported project workflows.

Use ``--json`` for launcher integration.  Planning is side-effect free; the
selected canonical entrypoint remains responsible for preflight and execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import WorkflowConfig, WorkflowKind, build_plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=[kind.value for kind in WorkflowKind])
    parser.add_argument("--questions", type=int)
    parser.add_argument("--prompts-per-update", type=int)
    parser.add_argument("--dataset", default="bird")
    parser.add_argument("--dataset-path")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--partition", choices=("round_robin", "contiguous"), default="round_robin")
    parser.add_argument("--gpu", type=int, action="append", dest="gpu_ids")
    parser.add_argument("--mechanism", default="saam-asymmetric-error")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--checkpoint", default="checkpoint-6380")
    parser.add_argument("--output-dir")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    kind = WorkflowKind(args.workflow)
    default_prompts = 14 if kind is WorkflowKind.RL_VERIFY else 30
    config = WorkflowConfig(
        kind=kind,
        question_count=args.questions,
        prompts_per_update=args.prompts_per_update or default_prompts if kind in (WorkflowKind.RL_VERIFY, WorkflowKind.RL_FULL) else None,
        mechanism=args.mechanism,
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        split=args.split,
        partition=args.partition,
        gpu_ids=tuple(args.gpu_ids or ()) if kind is WorkflowKind.EVALUATE else (),
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        optimizer_updates=args.updates,
    )
    plan = build_plan(config, repo_root=args.repo_root)
    print(json.dumps(plan, indent=2, ensure_ascii=False) if args.json else "\n".join(plan["command"]))


if __name__ == "__main__":
    main()
