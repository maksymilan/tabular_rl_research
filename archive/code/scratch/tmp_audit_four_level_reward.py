#!/usr/bin/env python3
import collections
import itertools
import json
import sys

path = sys.argv[1]
rows = [json.loads(line) for line in open(path, encoding="utf-8")]


def has_model_error(row):
    return bool(row.get("error_events")) or int(row.get("errors") or 0) > 0


def label(row):
    if row.get("optimization_exclusion") is not None:
        return "infra_excluded"
    if bool(row.get("correct")) and not has_model_error(row):
        return "clean_correct"
    if bool(row.get("correct")) and has_model_error(row):
        return "correct_recovered_error"
    if bool(row.get("legal")) and not has_model_error(row):
        return "wrong_legal_clean"
    return "wrong_with_error_or_illegal"


def proposed(row):
    kind = label(row)
    return {
        "clean_correct": 1.5,
        "correct_recovered_error": 1.0,
        "wrong_legal_clean": -0.5,
        "wrong_with_error_or_illegal": -1.0,
        "infra_excluded": None,
    }[kind]


counts = collections.Counter(label(row) for row in rows)
groups = [list(group) for _, group in itertools.groupby(rows, key=lambda row: row["example_index"])]
eligible_groups = []
for group in groups:
    eligible = [row for row in group if proposed(row) is not None]
    eligible_groups.append(eligible)

reward_sets = collections.Counter(
    tuple(sorted(set(float(proposed(row)) for row in group)))
    for group in eligible_groups
)
binary_sets = collections.Counter(
    tuple(sorted(set(float(row["result_reward"]["value"]) for row in group)))
    for group in eligible_groups
)

print(json.dumps({
    "rows": len(rows),
    "groups": len(groups),
    "label_counts": dict(counts),
    "proposed_reward_counts": dict(collections.Counter(
        str(proposed(row)) if proposed(row) is not None else "excluded"
        for row in rows
    )),
    "binary_group_reward_sets": {str(k): v for k, v in sorted(binary_sets.items())},
    "proposed_group_reward_sets": {str(k): v for k, v in sorted(reward_sets.items())},
    "binary_homogeneous_eligible_groups": sum(v for k, v in binary_sets.items() if len(k) == 1),
    "proposed_homogeneous_eligible_groups": sum(v for k, v in reward_sets.items() if len(k) == 1),
    "binary_mixed_eligible_groups": sum(v for k, v in binary_sets.items() if len(k) > 1),
    "proposed_mixed_eligible_groups": sum(v for k, v in reward_sets.items() if len(k) > 1),
}, ensure_ascii=False, indent=2))
