#!/usr/bin/env python3
import collections
import itertools
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
GEN_INFRA = {"generation_length", "context_overflow", "generation_oom"}


def types(row):
    return {str(e.get("error_type") or "unknown") for e in (row.get("error_events") or [])}


def model_error(row):
    ts = types(row)
    return bool(ts - GEN_INFRA) or (int(row.get("errors") or 0) > 0 and not ts)


def excluded(row):
    return row.get("failure_type") in GEN_INFRA or row.get("optimization_exclusion") == "max_token_completion"


def reward(row):
    if excluded(row):
        return None
    if row.get("correct") and not model_error(row):
        return 1.5
    if row.get("correct"):
        return 1.0
    if row.get("legal") and not model_error(row):
        return -0.5
    return -1.0


groups = [list(g) for _, g in itertools.groupby(rows, key=lambda x: x["example_index"])]
sets = collections.Counter(tuple(sorted(set(reward(x) for x in g if reward(x) is not None))) for g in groups)
print(json.dumps({
    "label_counts": dict(collections.Counter(
        "excluded_generation" if excluded(x) else (
            "correct_clean" if x.get("correct") and not model_error(x) else
            "correct_error" if x.get("correct") else
            "wrong_legal_clean" if x.get("legal") and not model_error(x) else
            "wrong_error_or_illegal"
        ) for x in rows
    )),
    "eligible_rows": sum(reward(x) is not None for x in rows),
    "group_reward_sets": {str(k): v for k, v in sorted(sets.items())},
    "homogeneous_groups": sum(v for k, v in sets.items() if len(k) == 1),
    "mixed_groups": sum(v for k, v in sets.items() if len(k) > 1),
}, ensure_ascii=False, indent=2))
