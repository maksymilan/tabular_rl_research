#!/usr/bin/env python3
"""Run OmniSQL's pinned BIRD scorer and save structured audit artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-train-eval-dir", required=True, type=pathlib.Path)
    parser.add_argument("--pred", required=True, type=pathlib.Path)
    parser.add_argument("--gold", required=True, type=pathlib.Path)
    parser.add_argument("--db-path", required=True, type=pathlib.Path)
    parser.add_argument(
        "--mode", required=True, choices=("greedy_search", "major_voting")
    )
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.upstream_train_eval_dir))
    evaluate_bird = importlib.import_module("evaluate_bird")
    accuracy, selected_sqls = evaluate_bird.run_eval(
        str(args.gold),
        str(args.pred),
        str(args.db_path),
        args.mode,
        True,
    )

    predictions = json.loads(args.pred.read_text())
    candidate_sqls = [
        sql for prediction in predictions for sql in prediction["pred_sqls"]
    ]
    records = list(evaluate_bird.evaluation_results)
    summary = {
        "mode": args.mode,
        "total": len(records),
        "correct": sum(record["correctness"] for record in records),
        "accuracy": accuracy,
        "selected_sql_count": len(selected_sqls),
        "selected_empty_sql_count": sum(not sql for sql in selected_sqls),
        "candidate_count": len(candidate_sqls),
        "candidate_empty_sql_count": sum(not sql for sql in candidate_sqls),
    }
    if args.mode == "major_voting":
        candidate_execution_results = list(evaluate_bird.execution_results)
        summary["candidate_execution_count"] = len(candidate_execution_results)
        summary["candidate_valid_count"] = sum(
            result["valid"] for result in candidate_execution_results
        )
        summary["candidate_invalid_count"] = sum(
            not result["valid"] for result in candidate_execution_results
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "greedy" if args.mode == "greedy_search" else "majority_n8"
    (args.output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (args.output_dir / f"{stem}_records.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
