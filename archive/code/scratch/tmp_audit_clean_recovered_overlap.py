#!/usr/bin/env python3
import collections
import itertools
import json
import sys

sys.path.insert(0, "/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_saam_asymmetric_gate60_20260829/src/rl/diagnostics")
from audit_saam_lineage_replay import LineageReplay

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
GEN = {"generation_length", "context_overflow", "generation_oom"}


def model_error(row):
    types = {str(e.get("error_type") or "unknown") for e in (row.get("error_events") or [])}
    return bool(types - GEN) or (int(row.get("errors") or 0) > 0 and not types)


def raw_groups():
    return [list(g) for _, g in itertools.groupby(rows, key=lambda x: x["example_index"])]


def legal_keys(row):
    events, _ = LineageReplay(row).replay()
    turns = row.get("turns") or []
    output = []
    if len(events) != len(turns):
        raise ValueError("event/turn length mismatch")
    for event, turn in zip(events, turns):
        if turn.get("error_event") is not None:
            continue
        if not event.get("matched") or not event.get("action_matchable"):
            continue
        key = (str(event.get("state_digest")), str(event.get("action_digest")))
        output.append(key)
    return output


def summarize(group_selector):
    selected = [g for g in raw_groups() if group_selector(g)]
    groups_with_overlap = 0
    unique_shared = 0
    clean_shared_occ = 0
    recovered_shared_occ = 0
    clean_total = 0
    recovered_total = 0
    error_actions = 0
    for group in selected:
        clean = [r for r in group if bool(r.get("correct")) and not model_error(r) and r.get("optimization_exclusion") is None]
        recovered = [r for r in group if bool(r.get("correct")) and model_error(r)]
        positive = [r for r in group if bool(r.get("correct"))]
        negative = [r for r in group if not bool(r.get("correct")) and r.get("optimization_exclusion") is None]
        if group_selector.__name__ == "all_correct_error_group":
            left, right = clean, recovered
        else:
            left, right = positive, negative
        left_keys = [key for row in left for key in legal_keys(row)]
        right_keys = [key for row in right for key in legal_keys(row)]
        left_set, right_set = set(left_keys), set(right_keys)
        shared = left_set & right_set
        if shared:
            groups_with_overlap += 1
        unique_shared += len(shared)
        clean_shared_occ += sum(key in shared for key in left_keys)
        recovered_shared_occ += sum(key in shared for key in right_keys)
        clean_total += len(left_keys)
        recovered_total += len(right_keys)
        error_actions += sum(len(r.get("error_events") or []) for r in recovered)
    return {
        "groups": len(selected),
        "groups_with_shared_legal_state_action": groups_with_overlap,
        "unique_shared_state_action_keys": unique_shared,
        "left_shared_occurrences": clean_shared_occ,
        "right_shared_occurrences": recovered_shared_occ,
        "left_legal_occurrences": clean_total,
        "right_legal_occurrences": recovered_total,
        "left_shared_fraction": clean_shared_occ / clean_total if clean_total else 0.0,
        "right_shared_fraction": recovered_shared_occ / recovered_total if recovered_total else 0.0,
        "right_error_events": error_actions,
    }


def all_correct_error_group(g):
    return all(bool(r.get("correct")) for r in g) and any(model_error(r) for r in g)


def terminal_mixed_group(g):
    eligible = [r for r in g if r.get("optimization_exclusion") is None]
    return any(bool(r.get("correct")) for r in eligible) and any(not bool(r.get("correct")) for r in eligible)


print(json.dumps({
    "all_correct_with_process_error": summarize(all_correct_error_group),
    "terminal_correct_wrong_mixed": summarize(terminal_mixed_group),
}, ensure_ascii=False, indent=2))
