"""CLI for the CPU-only fixed-suffix action deletion audit."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from rl.diagnostics.action_counterfactual import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--db-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=30)
    parser.add_argument("--trajectory-seconds", type=float, default=60.0)
    args = parser.parse_args()
    summary = run_audit(
        rollouts=args.rollouts, source_manifest=args.source_manifest,
        partitions_path=args.partitions, runtime_root=args.runtime_root,
        db_roots=args.db_root, output=args.output, groups=args.groups,
        trajectory_seconds=args.trajectory_seconds,
    )
    print(summary)


if __name__ == "__main__":
    main()
