#!/usr/bin/env python3
"""Failure attribution for tool-use evaluation runs (stdlib only).

A self-contained companion to ``server.py``: it turns a directory of rollout records
(``all.jsonl`` / ``failure.jsonl``) into a structured *why-did-it-fail* view so the dashboard
can review a run instead of only reporting a single accuracy number.

Two layers:

* ``attribute_record`` — pure per-case classifier. It buckets a wrong answer by comparing the
  predicted vs gold result SHAPE, and records whether the model actively probed the data.
  The discriminative buckets for metadata-only (catalog/handle) blindness are
  ``arity_mismatch`` (projected the wrong columns) and ``empty_pred`` (over-filtered to zero
  rows and answered anyway) — exactly the upstream-decision errors a final read cannot fix.

* ``attribution_summary`` — mtime-cached aggregate over ``all.jsonl``: the bucket distribution
  plus the perception behaviour of wrong answers. The key contrast it surfaces is
  ``read_subtable`` (did it read at all) vs ``decision_point_read`` (did it read at a step that
  could still change a downstream tool choice, rather than as the scripted pre-answer ritual).

Run standalone:

    python attribution.py data/results/qwen2.5_7b_sft_v2ctx/tool_zero_shot_dev1034
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# The four wrong_answer sub-buckets, ordered most→least diagnostic of blindness.
WRONG_ANSWER_BUCKETS = (
    "empty_pred",            # 0 predicted rows where gold has rows — over-filtered, answered anyway
    "arity_mismatch",        # wrong number of columns — wrong projection / selected the wrong fields
    "cardinality_mismatch",  # right columns, wrong number of rows — bad limit / group / distinct
    "value_mismatch",        # right shape, wrong cell values — wrong column, filter value, or aggregate
)

BUCKET_LABELS = {
    "correct": "正确",
    "empty_pred": "空结果(过滤过头)",
    "arity_mismatch": "投影错列(列数不符)",
    "cardinality_mismatch": "行数不符",
    "value_mismatch": "取值错(列/值/聚合)",
    "execution_error": "执行错误",
    "protocol_error": "协议/格式错误",
    "api_error": "API 错误",
    "max_steps": "超出最大步数",
    "unknown": "未知",
}


def _sample_shape(sample) -> tuple[int, int]:
    """(row_count, column_count) of a result sample, tolerating scalar (flat) rows."""
    if not sample:
        return (0, 0)
    rows = [row if isinstance(row, (list, tuple)) else [row] for row in sample]
    return (len(rows), max((len(row) for row in rows), default=0))


def _tool_sequence(record: dict) -> list[str]:
    return [
        turn["parsed"].get("tool")
        for turn in record.get("turns", [])
        if isinstance(turn.get("parsed"), dict)
    ]


def record_step_count(record: dict) -> int:
    """Trajectory length (number of tool-call steps), across both record formats:

    * eval rollout record  → ``len(turns)``
    * sharegpt training record (LLaMA-Factory) → number of ``gpt`` (assistant) turns

    This matches the manifest's ``trajectory_length_hist`` bucketing, so a histogram bar of
    length N and a ``min_steps=max_steps=N`` filter select the same trajectories.
    """
    turns = record.get("turns")
    if isinstance(turns, list):
        return len(turns)
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        return sum(1 for c in conversations if isinstance(c, dict) and c.get("from") == "gpt")
    return 0


def attribute_record(record: dict) -> dict:
    """Classify a single eval case. Pure; safe on success and non-wrong_answer failures."""
    tools = _tool_sequence(record)
    # A decision-point read is a read_subtable that is NOT the penultimate step (the scripted
    # pre-answer read sits at index len-2, right before answer_from_context at len-1).
    decision_read = any(
        tool == "read_subtable" and index < len(tools) - 2 for index, tool in enumerate(tools)
    )
    info = {
        "failure_type": record.get("failure_type"),
        "n_steps": len(record.get("turns", [])),
        "tool_sequence": tools,
        "used_read_subtable": "read_subtable" in tools,
        "used_inspect_column": "inspect_column" in tools,
        "decision_point_read": decision_read,
    }
    if record.get("correct") is True:
        info["bucket"] = "correct"
        return info
    if record.get("failure_type") != "wrong_answer":
        # execution_error / protocol_error / api_error / max_steps — keep as-is.
        info["bucket"] = record.get("failure_type") or "unknown"
        return info

    pr, pc = _sample_shape(record.get("pred_sample"))
    gr, gc = _sample_shape(record.get("gold_sample"))
    info["pred_shape"] = [pr, pc]
    info["gold_shape"] = [gr, gc]
    if pr == 0 and gr > 0:
        info["bucket"] = "empty_pred"
    elif pc != gc:
        info["bucket"] = "arity_mismatch"
    elif pr != gr:
        info["bucket"] = "cardinality_mismatch"
    else:
        info["bucket"] = "value_mismatch"
    return info


_summary_cache: dict[str, tuple[float, dict]] = {}


def attribution_summary(directory: Path) -> dict:
    """Aggregate attribution over ``directory/all.jsonl`` (mtime-cached)."""
    all_path = Path(directory) / "all.jsonl"
    if not all_path.exists():
        return {"available": False}
    cache_key = str(all_path)
    mtime = all_path.stat().st_mtime
    cached = _summary_cache.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    buckets: Counter = Counter()
    wrong_total = read_any = read_decision = inspected = 0
    with all_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            info = attribute_record(json.loads(line))
            buckets[info["bucket"]] += 1
            if info["bucket"] in WRONG_ANSWER_BUCKETS:
                wrong_total += 1
                read_any += int(info["used_read_subtable"])
                read_decision += int(info["decision_point_read"])
                inspected += int(info["used_inspect_column"])

    total = sum(buckets.values())
    summary = {
        "available": True,
        "total": total,
        "buckets": dict(buckets),
        "labels": BUCKET_LABELS,
        "wrong_answer": {
            "total": wrong_total,
            "sub_buckets": {b: buckets.get(b, 0) for b in WRONG_ANSWER_BUCKETS},
            "read_subtable": read_any,
            "decision_point_read": read_decision,
            "inspect_column": inspected,
        },
    }
    _summary_cache[cache_key] = (mtime, summary)
    return summary


def _main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    summary = attribution_summary(Path(argv[0]))
    if not summary.get("available"):
        print(f"no all.jsonl under {argv[0]}")
        return 1
    print(f"total {summary['total']}  buckets:")
    for bucket, count in sorted(summary["buckets"].items(), key=lambda kv: -kv[1]):
        print(f"  {count:5d}  {bucket}  ({BUCKET_LABELS.get(bucket, bucket)})")
    wa = summary["wrong_answer"]
    print(f"\nwrong_answer {wa['total']}:")
    for bucket, count in wa["sub_buckets"].items():
        print(f"  {count:5d}  {bucket}")
    print(
        f"  perception → read_subtable {wa['read_subtable']}, "
        f"decision_point_read {wa['decision_point_read']}, inspect_column {wa['inspect_column']}"
    )
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
