#!/usr/bin/env python3
"""Run the conservative distance/error classifier on an SMC audit pair file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.insert(0, str(ROOT / "src"))

from rl.diagnostics.distance_credit import (  # noqa: E402
    classify_pair,
    load_smc_pairs,
    summarize_classifications,
)
from rl.diagnostics.io import write_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.audit.is_file() or not args.rollouts.is_file():
        raise SystemExit("audit and rollouts must be existing regular files")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")

    records = []
    for audit, positive, negative in load_smc_pairs(str(args.audit), str(args.rollouts)):
        classification = classify_pair(positive, negative)
        records.append(
            {
                "example_index": audit["example_index"],
                "policy_global_step": audit["policy_global_step"],
                "positive_trajectory_id": positive["trajectory_id"],
                "negative_trajectory_id": negative["trajectory_id"],
                "classification": classification.to_dict(),
            }
        )
    write_jsonl(args.output, records)
    print(json.dumps(summarize_classifications(records), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
