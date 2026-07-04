#!/usr/bin/env python3
"""Select a high-quality CURRENT-PROTOCOL skeleton subset for the perception/recovery data pipeline.

The output is still a skeleton trajectory file, not final SFT-ready clean data. For clean SFT, run
`enrich_traj.py` on the selected ids and train only on the resulting `quality_status == "ready"`
v3-enriched file.

Why: the full set is overkill for the new (more expensive, LLM-in-the-loop) enrichment. A length-
stratified, DB-diverse subset keeps the difficulty spread of the original while cutting API cost and
training time.

Outputs (under data/trajectories/):
  subset_<N>.jsonl        selected current-protocol skeleton trajectories (v3, no memory)
  subset_<N>.ids.json     {"subset": [...ids], "smoke": [...10 ids]} for downstream stages
The 10 smoke ids span the length range EVENLY (not all short), for the human-reviewed smoke test.

Usage:
  .venv/bin/python src/sft/select_subset.py [--n 180] [--per-db-cap 2] [--smoke 10]
  .venv/bin/python src/sft/select_subset.py --strategy tool-balanced --n 2000 \
    --source data/trajectories/spider_train_nway.jsonl --out-prefix subset_tool_balanced_2000
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_SOURCE = os.path.join(ROOT, "data", "trajectories", "spider_train_v3.jsonl")
MEMORY_TOOLS = {"add_to_memory", "refine_memory"}
OPERATION_TOOLS = {
    "condition_filter",
    "project",
    "join_tables",
    "group_aggregate",
    "aggregate",
    "extreme_value_select",
    "set_op",
    "derive_column",
    "window",
}
RARE_FEATURES = {
    "nway_join_3plus",
    "value_ref_filter",
    "in_table_filter",
    "like_filter",
    "set_op_union",
    "set_op_intersect",
    "set_op_except",
    "multi_aggregate",
    "whole_table_multi_agg",
    "grouped_multi_agg",
}


def has_memory_residue(traj: dict) -> bool:
    text = json.dumps(traj, ensure_ascii=False)
    if any(marker in text for marker in ("memory_id", "supporting_memory_ids", "mem_")):
        return True
    for step in traj.get("steps", []):
        if (step.get("tool_call") or {}).get("tool") in MEMORY_TOOLS:
            return True
    return False


def load_skeletons(path: str, *, allow_legacy_source: bool = False) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                traj = json.loads(line)
                schema_version = str(traj.get("schema_version", ""))
                if not allow_legacy_source and not schema_version.startswith("v3"):
                    raise ValueError(
                        f"{path}:{line_no}: expected current v3 schema, got {schema_version!r}. "
                        "Regenerate with `src/harness/gen_trajectories.py train --tag=_v3` or pass "
                        "--allow-legacy-source only for migration/debugging."
                    )
                if not allow_legacy_source and has_memory_residue(traj):
                    raise ValueError(
                        f"{path}:{line_no}: memory residue found in skeleton source; do not use this "
                        "file for current no-memory SFT data."
                    )
                out.append(traj)
    return out


def iter_conditions(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_conditions(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_conditions(child)


def trajectory_tool_calls(traj: dict) -> list[tuple[str, dict]]:
    calls = []
    for step in traj.get("steps", []):
        call = step.get("tool_call") or {}
        tool = call.get("tool")
        if tool:
            calls.append((tool, call.get("arguments") or {}))
    return calls


def trajectory_tags(traj: dict) -> set[str]:
    """Tags used for tool-balanced sampling.

    Tool tags are prefixed with `tool:`; mode tags are prefixed with `feature:`.  Sampling balances
    operation tools and rare modes, not terminal/perception calls that are either mandatory or
    inserted by a later enrichment stage.
    """
    tags: set[str] = set()
    for tool, args in trajectory_tool_calls(traj):
        if tool in OPERATION_TOOLS:
            tags.add(f"tool:{tool}")
        if tool == "join_tables":
            n_tables = len(args.get("tables") or [])
            tags.add("feature:nway_join_3plus" if n_tables >= 3 else "feature:join_2way")
        elif tool == "condition_filter":
            for cond in iter_conditions(args.get("conditions")):
                op = str(cond.get("op", "")).lower()
                if "value_ref" in cond:
                    tags.add("feature:value_ref_filter")
                if "in_table" in cond:
                    tags.add("feature:in_table_filter")
                if op in {"like", "not like"}:
                    tags.add("feature:like_filter")
        elif tool == "set_op":
            op = str(args.get("op", "")).lower() or "unknown"
            tags.add(f"feature:set_op_{op}")
        elif tool == "group_aggregate":
            group_by = args.get("group_by") or []
            aggs = args.get("aggregations") or []
            if not group_by:
                tags.add("feature:whole_table_group_aggregate")
            if len(aggs) >= 2:
                tags.add("feature:multi_aggregate")
                tags.add("feature:grouped_multi_agg" if group_by else "feature:whole_table_multi_agg")
        elif tool == "aggregate":
            tags.add("feature:single_scalar_aggregate")
        elif tool == "extreme_value_select":
            tags.add("feature:extreme_value_select")
    return tags


def tool_stats(trajs: list[dict]) -> tuple[collections.Counter, collections.Counter, collections.Counter]:
    call_counts: collections.Counter = collections.Counter()
    traj_counts: collections.Counter = collections.Counter()
    tag_counts: collections.Counter = collections.Counter()
    for traj in trajs:
        local_tools = set()
        for tool, _args in trajectory_tool_calls(traj):
            call_counts[tool] += 1
            local_tools.add(tool)
        traj_counts.update(local_tools)
        tag_counts.update(trajectory_tags(traj))
    return call_counts, traj_counts, tag_counts


def db_id(traj: dict) -> str:
    return str((traj.get("source") or {}).get("db_id") or traj.get("db_id") or "")


def load_excluded_ids(paths: list[str]) -> set[str]:
    excluded = set()
    for path in paths:
        if not path:
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
            stripped = text.lstrip()
            if not stripped:
                continue
            if stripped[0] in "[{":
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, list):
                    excluded.update(str(item) for item in payload)
                    continue
                if isinstance(payload, dict):
                    if "trajectory_id" in payload:
                        excluded.add(payload["trajectory_id"])
                    for value in payload.values():
                        if isinstance(value, list):
                            excluded.update(str(item) for item in value)
                    continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if "trajectory_id" in item:
                    excluded.add(item["trajectory_id"])
    return excluded


def length_targets(by_len: dict[int, list], n: int) -> dict[int, int]:
    """Proportional per-length quota that sums to ~n, with >=1 for every present length so the
    distribution's tail (rare long trajectories) is preserved, then trimmed/topped to hit n."""
    total = sum(len(v) for v in by_len.values())
    target = {L: max(1, round(n * len(v) / total)) for L, v in by_len.items()}
    # Reconcile the rounded sum to exactly n: add to / remove from the largest buckets first.
    order = sorted(by_len, key=lambda L: -len(by_len[L]))
    while sum(target.values()) > n:
        for L in reversed(order):  # trim smallest-population buckets but never below 1
            if sum(target.values()) <= n:
                break
            if target[L] > 1:
                target[L] -= 1
    while sum(target.values()) < n:
        for L in order:
            if sum(target.values()) >= n:
                break
            if target[L] < len(by_len[L]):
                target[L] += 1
    return target


def pick_db_diverse(cands: list[dict], k: int, used_db: collections.Counter, cap: int) -> list[dict]:
    """Pick k trajectories preferring databases used the fewest times so far (<= cap each)."""
    pool = sorted(cands, key=lambda t: (used_db[db_id(t)], t["trajectory_id"]))
    chosen = []
    for t in pool:
        if len(chosen) >= k:
            break
        if used_db[db_id(t)] < cap:
            chosen.append(t)
            used_db[db_id(t)] += 1
    # If the cap blocked us from reaching k, relax and fill from the remainder.
    if len(chosen) < k:
        for t in pool:
            if len(chosen) >= k:
                break
            if t not in chosen:
                chosen.append(t)
                used_db[db_id(t)] += 1
    return chosen


def build_tool_targets(
    skeletons: list[dict],
    n: int,
    *,
    min_tool_frac: float,
    min_feature_frac: float,
    rare_feature_frac: float,
) -> dict[str, int]:
    """Build trajectory-level minimum quotas for operation tools and rare modes.

    The target is capped by availability in the source pool.  This makes small pools such as
    `set_op_union` saturate instead of asking the selector for impossible coverage.
    """
    _call_counts, traj_counts, tag_counts = tool_stats(skeletons)
    targets: dict[str, int] = {}
    tool_min = math.ceil(n * min_tool_frac)
    feature_min = math.ceil(n * min_feature_frac)
    rare_min = math.ceil(n * rare_feature_frac)
    for tool in sorted(OPERATION_TOOLS):
        tag = f"tool:{tool}"
        available = traj_counts.get(tool, 0)
        if available:
            targets[tag] = min(available, tool_min)
    for tag, available in tag_counts.items():
        if not tag.startswith("feature:"):
            continue
        name = tag.removeprefix("feature:")
        if name in RARE_FEATURES:
            targets[tag] = min(available, max(rare_min, min(available, feature_min)))
        elif name in {"join_2way", "single_scalar_aggregate", "extreme_value_select"}:
            targets[tag] = min(available, feature_min)
    return {tag: quota for tag, quota in targets.items() if quota > 0}


def pick_tool_balanced(
    skeletons: list[dict],
    n: int,
    *,
    per_db_cap: int,
    seed: int,
    min_tool_frac: float,
    min_feature_frac: float,
    rare_feature_frac: float,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    by_len: dict[int, list] = collections.defaultdict(list)
    for traj in skeletons:
        by_len[len(traj["steps"])].append(traj)
    len_targets = length_targets(by_len, n)
    tag_targets = build_tool_targets(
        skeletons, n,
        min_tool_frac=min_tool_frac,
        min_feature_frac=min_feature_frac,
        rare_feature_frac=rare_feature_frac,
    )
    tag_weights = {}
    for tag, target in tag_targets.items():
        name = tag.split(":", 1)[1]
        if tag.startswith("tool:"):
            tag_weights[tag] = 1.0
        elif name in RARE_FEATURES:
            tag_weights[tag] = 2.0
        else:
            tag_weights[tag] = 0.8

    shuffled = list(skeletons)
    rng.shuffle(shuffled)
    tag_by_id = {t["trajectory_id"]: trajectory_tags(t) for t in shuffled}
    source_tag_counts: collections.Counter = collections.Counter()
    for tags in tag_by_id.values():
        source_tag_counts.update(tags)
    used_db: collections.Counter = collections.Counter()
    used_len: collections.Counter = collections.Counter()
    used_tags: collections.Counter = collections.Counter()
    selected: list[dict] = []
    selected_ids: set[str] = set()

    def need_score(traj: dict) -> float:
        tags = tag_by_id[traj["trajectory_id"]]
        score = 0.0
        for tag in tags:
            target = tag_targets.get(tag, 0)
            if target and used_tags[tag] < target:
                score += tag_weights.get(tag, 1.0) * (target - used_tags[tag]) / target
            elif target:
                score -= 0.04 * (used_tags[tag] - target + 1) / max(1, target)
        L = len(traj["steps"])
        if used_len[L] < len_targets.get(L, 0):
            score += 0.35 * (len_targets[L] - used_len[L]) / max(1, len_targets[L])
        else:
            score -= 0.02 * (used_len[L] - len_targets.get(L, 0) + 1)
        db = db_id(traj)
        if db and used_db[db] >= per_db_cap:
            score -= 0.25 * (used_db[db] - per_db_cap + 1)
        return score

    def add(traj: dict) -> None:
        selected.append(traj)
        selected_ids.add(traj["trajectory_id"])
        used_db[db_id(traj)] += 1
        used_len[len(traj["steps"])] += 1
        used_tags.update(tag_by_id[traj["trajectory_id"]])

    def eligible_remaining() -> list[dict]:
        remaining = [t for t in shuffled if t["trajectory_id"] not in selected_ids]
        under_cap = [t for t in remaining if used_db[db_id(t)] < per_db_cap]
        return under_cap or remaining

    def tag_priority(tag: str) -> tuple[int, int, str]:
        name = tag.split(":", 1)[1]
        if tag.startswith("feature:") and name in RARE_FEATURES:
            return (0, source_tag_counts[tag], tag)
        if tag == "tool:set_op":
            return (1, source_tag_counts[tag], tag)
        if tag.startswith("tool:"):
            return (2, source_tag_counts[tag], tag)
        return (3, source_tag_counts[tag], tag)

    # Phase 0: first reserve scarce tool modes. Without this, common multi-tag examples can crowd
    # out union / grouped-multi-agg / like cases even when their quotas are explicit.
    for target_tag in sorted(tag_targets, key=tag_priority):
        while len(selected) < n and used_tags[target_tag] < tag_targets[target_tag]:
            candidates = [
                t for t in eligible_remaining()
                if target_tag in tag_by_id[t["trajectory_id"]]
            ]
            if not candidates:
                break
            best = max(
                candidates,
                key=lambda t: (need_score(t), -used_db[db_id(t)], -len(t["steps"]), t["trajectory_id"]),
            )
            add(best)

    # Phase 1: satisfy remaining operation/mode quotas as much as possible. The heap is rebuilt
    # each round because every selected trajectory changes multiple tag deficits at once.
    while len(selected) < n:
        remaining = eligible_remaining()
        if not remaining:
            break
        missing = {tag: target for tag, target in tag_targets.items() if used_tags[tag] < target}
        if not missing:
            break
        best = max(
            remaining,
            key=lambda t: (need_score(t), -used_db[db_id(t)], -len(t["steps"]), t["trajectory_id"]),
        )
        if need_score(best) <= 0:
            break
        add(best)

    # Phase 2: fill to n while preserving length/DB diversity and avoiding already-saturated tags.
    while len(selected) < n:
        remaining = eligible_remaining()
        if not remaining:
            break
        best = max(
            remaining,
            key=lambda t: (need_score(t), -used_db[db_id(t)], t["trajectory_id"]),
        )
        add(best)

    metadata = {
        "strategy": "tool-balanced",
        "seed": seed,
        "length_targets": {str(k): v for k, v in sorted(len_targets.items())},
        "tag_targets": {k: v for k, v in sorted(tag_targets.items())},
        "tag_coverage": {k: used_tags[k] for k in sorted(tag_targets)},
    }
    return selected, metadata


def print_tool_report(label: str, trajs: list[dict]) -> None:
    call_counts, traj_counts, tag_counts = tool_stats(trajs)
    total_calls = sum(call_counts.values())
    total = len(trajs)
    print(f"\n{label} tool distribution:")
    print(f"{'tool':<22} {'calls':>7} {'call%':>7} {'traj':>7} {'traj%':>7}")
    for tool, count in call_counts.most_common():
        print(f"{tool:<22} {count:>7} {100*count/max(1,total_calls):>6.2f}% "
              f"{traj_counts[tool]:>7} {100*traj_counts[tool]/max(1,total):>6.2f}%")
    interesting = {
        tag: count for tag, count in tag_counts.items()
        if tag.startswith("feature:") and (
            tag.removeprefix("feature:") in RARE_FEATURES
            or tag.removeprefix("feature:") in {"join_2way", "single_scalar_aggregate", "extreme_value_select"}
        )
    }
    if interesting:
        print(f"\n{label} feature coverage:")
        print(f"{'feature':<30} {'traj':>7} {'traj%':>7}")
        for tag, count in sorted(interesting.items(), key=lambda kv: (-kv[1], kv[0])):
            name = tag.removeprefix("feature:")
            print(f"{name:<30} {count:>7} {100*count/max(1,total):>6.2f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180, help="subset size (<200)")
    ap.add_argument("--per-db-cap", type=int, default=2)
    ap.add_argument("--smoke", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strategy", choices=("length", "tool-balanced"), default="length",
                    help="length keeps the historical length-stratified sampler; tool-balanced "
                         "adds minimum quotas for operation tools and rare tool modes")
    ap.add_argument("--min-tool-frac", type=float, default=0.25,
                    help="tool-balanced: target at least this fraction of selected trajectories "
                         "for each available operation tool, capped by source availability")
    ap.add_argument("--min-feature-frac", type=float, default=0.15,
                    help="tool-balanced: target at least this fraction for common feature tags")
    ap.add_argument("--rare-feature-frac", type=float, default=0.20,
                    help="tool-balanced: target at least this fraction for rare feature tags, "
                         "capped by source availability")
    ap.add_argument("--out-prefix", default=None,
                    help="output prefix under data/trajectories; default subset_<N>")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="current-protocol skeleton JSONL source; default spider_train_v3.jsonl")
    ap.add_argument("--allow-legacy-source", action="store_true",
                    help="allow non-v3/memory-bearing sources for one-off migration/debugging only")
    ap.add_argument("--exclude-ids", action="append", default=[],
                    help="JSON/JSONL file containing trajectory ids to exclude; may be repeated")
    args = ap.parse_args()
    random.seed(args.seed)

    excluded = load_excluded_ids(args.exclude_ids)
    skeletons = [
        t for t in load_skeletons(args.source, allow_legacy_source=args.allow_legacy_source)
        if t["trajectory_id"] not in excluded
    ]
    by_len: dict[int, list] = collections.defaultdict(list)
    for t in skeletons:
        by_len[len(t["steps"])].append(t)

    used_db: collections.Counter = collections.Counter()
    selection_meta: dict = {
        "strategy": args.strategy,
        "seed": args.seed,
        "per_db_cap": args.per_db_cap,
    }
    if args.strategy == "length":
        target = length_targets(by_len, args.n)
        subset = []
        for L in sorted(by_len):
            cands = list(by_len[L])
            random.shuffle(cands)
            subset.extend(pick_db_diverse(cands, target[L], used_db, args.per_db_cap))
        selection_meta["length_targets"] = {str(k): v for k, v in sorted(target.items())}
    else:
        subset, selection_meta = pick_tool_balanced(
            skeletons, args.n,
            per_db_cap=args.per_db_cap,
            seed=args.seed,
            min_tool_frac=args.min_tool_frac,
            min_feature_frac=args.min_feature_frac,
            rare_feature_frac=args.rare_feature_frac,
        )
        used_db = collections.Counter(db_id(t) for t in subset)
        selection_meta.update({
            "per_db_cap": args.per_db_cap,
            "min_tool_frac": args.min_tool_frac,
            "min_feature_frac": args.min_feature_frac,
            "rare_feature_frac": args.rare_feature_frac,
        })

    # Smoke: EVENLY spread lengths (never all-short). Take distinct subset lengths, pick `smoke` of
    # them evenly across the range, one trajectory each (prefer a fresh DB).
    sub_by_len: dict[int, list] = collections.defaultdict(list)
    for t in subset:
        sub_by_len[len(t["steps"])].append(t)
    lengths = sorted(sub_by_len)
    if len(lengths) <= args.smoke:
        smoke_lengths = lengths
    else:
        idx = [round(i * (len(lengths) - 1) / (args.smoke - 1)) for i in range(args.smoke)]
        smoke_lengths = sorted({lengths[i] for i in idx})
    smoke = [sub_by_len[L][0] for L in smoke_lengths][: args.smoke]

    # Write artifacts.
    out_prefix = args.out_prefix or f"subset_{args.n}"
    out_jsonl = os.path.join(ROOT, "data", "trajectories", f"{out_prefix}.jsonl")
    out_ids = os.path.join(ROOT, "data", "trajectories", f"{out_prefix}.ids.json")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in subset:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    json.dump(
        {
            "subset": [t["trajectory_id"] for t in subset],
            "smoke": [t["trajectory_id"] for t in smoke],
            "selection": selection_meta,
        },
        open(out_ids, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
    )

    # Report: full vs subset length distribution (proportional preservation) + DB coverage.
    full_total, sub_total = len(skeletons), len(subset)
    print(f"source {os.path.relpath(args.source, ROOT)}")
    print(f"selected {sub_total} / {full_total}  (DB coverage {len(used_db)} dbs, "
          f"max {max(used_db.values())}/db)")
    if excluded:
        print(f"excluded {len(excluded)} trajectory ids")
    print(f"{'len':>4} {'full':>6} {'full%':>7} {'sub':>5} {'sub%':>7}")
    for L in sorted(by_len):
        fn, sn = len(by_len[L]), len(sub_by_len.get(L, []))
        print(f"{L:>4} {fn:>6} {100*fn/full_total:>6.1f}% {sn:>5} {100*sn/max(1,sub_total):>6.1f}%")
    print_tool_report("source", skeletons)
    print_tool_report("subset", subset)
    print(f"\nsmoke {len(smoke)} (lengths {[len(t['steps']) for t in smoke]}):")
    for t in smoke:
        print(f"  {t['trajectory_id']:<22} len={len(t['steps']):<3} db={db_id(t)}")
    print(f"\n-> {out_jsonl}\n-> {out_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
