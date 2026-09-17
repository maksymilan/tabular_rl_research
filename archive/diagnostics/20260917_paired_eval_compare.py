"""Paired comparison of two merged BIRD-dev1534 checkpoint evals (same example_index set).

Reads only `results/merged/all.jsonl` from each eval root; every row carries the per-question
`correct` flag produced by the harness denotation scorer. No gold SQL is read or printed.
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def read_correct(path: Path) -> dict[int, bool]:
    rows: dict[int, bool] = {}
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            index = record.get("example_index")
            if index is None:
                continue
            if index in rows:
                raise SystemExit(f"{path}: duplicate example_index {index}")
            rows[int(index)] = bool(record.get("correct"))
    return rows


def exact_mcnemar(gain: int, loss: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = gain + loss
    if n == 0:
        return 1.0
    k = min(gain, loss)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True, help="baseline eval root")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate eval root")
    parser.add_argument("--base-label", default="base")
    parser.add_argument("--candidate-label", default="candidate")
    args = parser.parse_args()

    base = read_correct(args.base / "results/merged/all.jsonl")
    candidate = read_correct(args.candidate / "results/merged/all.jsonl")
    shared = sorted(set(base) & set(candidate))
    if len(shared) != len(base) or len(shared) != len(candidate):
        print(
            f"WARNING unmatched example sets: base={len(base)} candidate={len(candidate)} "
            f"shared={len(shared)}"
        )
    both = sum(base[i] and candidate[i] for i in shared)
    base_only = sum(base[i] and not candidate[i] for i in shared)
    candidate_only = sum(candidate[i] and not base[i] for i in shared)
    neither = len(shared) - both - base_only - candidate_only
    base_total = both + base_only
    candidate_total = candidate_only + both
    net = candidate_only - base_only
    p_value = exact_mcnemar(candidate_only, base_only)

    print(f"questions compared: {len(shared)}")
    print(f"{args.base_label:>28s}: {base_total}/{len(shared)} = {base_total / len(shared):.4f}")
    print(
        f"{args.candidate_label:>28s}: {candidate_total}/{len(shared)} = "
        f"{candidate_total / len(shared):.4f}"
    )
    print(f"both correct: {both}   neither: {neither}")
    print(f"candidate-only (gain): {candidate_only}   base-only (loss): {base_only}")
    print(f"net: {net:+d} questions ({net / len(shared) * 100:+.3f} pp), exact McNemar p={p_value:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
