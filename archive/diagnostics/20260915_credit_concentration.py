"""CPU-only coefficient concentration audit against an isolated training runtime.

This is an offline diagnostic, not a launcher or a proposed credit mechanism.
Source rollouts are snapshotted by byte length; gold and prompt fields are discarded.
Reported masses omit PPO ratios/clipping and are not gradient/loss measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--credit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    from rl.frameworks.trl.mechanism import RLMechanism
    from rl.frameworks.trl.state_action_ambiguity import _local_error_kind
    from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn

    torch.set_num_threads(1)
    path = args.run_root / "train/rollouts.jsonl"
    source_size = path.stat().st_size
    with path.open("rb") as handle:
        snapshot = handle.read(source_size)
    complete = snapshot[: snapshot.rfind(b"\n") + 1]
    rows = []
    for line in complete.splitlines():
        raw = json.loads(line)
        row = {k: raw[k] for k in (
            "trajectory_id", "example_index", "correct", "legal", "result_reward",
            "failure_type", "errors", "error_events", "process_update", "policy_global_step",
        ) if k in raw}
        row["turns"] = [{k: t[k] for k in (
            "turn_index", "parsed", "tool_output", "error_event", "execution_error_type",
            "execution_error", "model_output",
        ) if k in t} for t in raw["turns"]]
        rows.append(row)
    del snapshot, complete
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/dengyan/models/Qwen3-4B-TrustSQL-baseline", local_files_only=True,
    )
    updates_rows = defaultdict(list)
    for row in rows:
        updates_rows[int(row["policy_global_step"])].append(row)
    report = {
        "schema": "credit-concentration-diagnostic-v1", "run_root": str(args.run_root),
        "source_size": source_size, "complete_rows": len(rows), "credit": args.credit,
        "source_sha256": hashlib.sha256(path.read_bytes()[:source_size]).hexdigest(),
        "gold_fields_used": False, "gpu_used": False,
        "limitations": "Authored response text is re-tokenized; sampled token IDs/logprobs not saved. Mass is the sum of absolute normalized loss coefficients before PPO ratios and clipping, not measured loss or gradient. Raw rows excluded from optimization do not contribute.",
        "implementation_sha256": {}, "updates": [],
    }
    for name in ("mechanism.py", "transition_batch.py", "state_action_ambiguity.py", "transition_grpo.py"):
        p = args.run_root / "project/src/rl/frameworks/trl" / name
        report["implementation_sha256"][name] = hashlib.sha256(p.read_bytes()).hexdigest()
    all_details = []
    for step, batch in sorted(updates_rows.items()):
        groups = defaultdict(list)
        for row in batch:
            groups[row["example_index"]].append(row)
        if any(len(g) != 8 for g in groups.values()):
            raise ValueError("incomplete K8 group in snapshot")
        if len(groups) != 30:
            raise ValueError("incomplete 30-prompt update; retry after complete rollout write")
        episodes = []
        info = {}
        group_stats = []
        for example, group in sorted(groups.items()):
            c = sum(bool(r["correct"]) for r in group)
            eligible = [r for r in group if r["process_update"]]
            ec = sum(bool(r["correct"]) for r in eligible)
            n = len(eligible)
            group_stats.append({"example_index": example, "correct": c, "eligible": n, "eligible_correct": ec})
            for row in group:
                tid = row["trajectory_id"]
                if tid in info:
                    raise ValueError("trajectory ID collision inside update")
                tokens = []
                for turn in (row["turns"] if row["process_update"] else []):
                    ids = tuple(tokenizer.encode(turn.get("model_output") or "", add_special_tokens=False))
                    tokens.append(PolicyTurn((1,), ids, (0.0,) * len(ids)))
                kinds = [_local_error_kind(t) for t in row["turns"]]
                sample = SimpleNamespace(
                    reward=1.0 if row["correct"] else 0.0, correct=bool(row["correct"]),
                    process_update=bool(row["process_update"]), step_rewards=None,
                    turns=[(list(t.prompt_ids), list(t.response_ids)) for t in tokens], audit_record=row,
                )
                episodes.append(PolicyEpisode(sample, tokens))
                info[tid] = {
                    "step": step, "trajectory_id": tid, "example_index": example,
                    "correct": row["correct"], "eligible": row["process_update"],
                    "group_c": c, "eligible_group_c": ec, "eligible_group_n": n,
                    "raw_extreme_group": c in (1, 7),
                    "raw_minority": (c == 1 and row["correct"]) or (c == 7 and not row["correct"]),
                    "effective_minority": n >= 3 and ((ec == 1 and row["correct"]) or (ec == n-1 and not row["correct"])),
                    "error_turns": sum(k is not None for k in kinds) if tokens else 0,
                    "deterministic_errors": sum(k is not None and k != "infrastructure_timeout" for k in kinds) if tokens else 0,
                    "recorded_error_events": len(row.get("error_events", [])),
                    "response_tokens": sum(len(t.response_ids) for t in tokens),
                    "abs_mass": 0.0, "neg_mass": 0.0, "error_abs_mass": 0.0,
                    "error_neg_delta_vs_trajectory": 0.0, "error_sign_flips": 0,
                    "error_replacements": 0, "turns": len(tokens),
                }
        mechanism = RLMechanism(policy_reduction="trajectory_token_mean", credit_assignment=args.credit)
        original = mechanism.build_updates(episodes, train_turns="all")
        credited, audit = mechanism.apply_credit(episodes, original)
        N = len(original)
        J = len({u.trajectory_id for u in original})
        coeffs = mechanism.effective_advantages(credited, transition_count=N, trajectory_count=J)
        ep_by_id = {e.sample.audit_record["trajectory_id"]: e for e in episodes}
        for old, new, coef in zip(original, credited, coeffs, strict=True):
            d = info[new.trajectory_id]
            weight = new.trajectory_token_weight
            assert abs(coef / N - new.advantage * weight / J) < 1e-9
            d["base_advantage"] = old.advantage
            d["abs_mass"] += abs(new.advantage) * weight
            d["neg_mass"] += max(-new.advantage, 0) * weight
            kind = _local_error_kind(ep_by_id[new.trajectory_id].sample.audit_record["turns"][new.turn_index])
            if kind is not None:
                d["error_abs_mass"] += abs(new.advantage) * weight
                d["error_neg_delta_vs_trajectory"] += (max(-new.advantage, 0) - max(-old.advantage, 0)) * weight
                d["error_sign_flips"] += int(old.advantage >= 0 and new.advantage < 0)
                d["error_replacements"] += int(new.advantage != old.advantage)
        trainable = [d for d in info.values() if d["eligible"]]
        total_abs = sum(d["abs_mass"] for d in trainable)
        total_neg = sum(d["neg_mass"] for d in trainable)
        for d in trainable:
            d["update_abs_share"] = d["abs_mass"] / total_abs
            d["update_neg_share"] = d["neg_mass"] / total_neg if total_neg else 0
            d["objective_abs_mass"] = d["abs_mass"] / J
            d["objective_neg_mass"] = d["neg_mass"] / J
            assert d["abs_mass"] <= max(abs(d.get("base_advantage", 0)), 1) + 1e-6
        categories = {
            "raw_extreme_group_trajectories": lambda d: d["raw_extreme_group"],
            "raw_extreme_minority": lambda d: d["raw_minority"],
            "effective_extreme_minority": lambda d: d["effective_minority"],
            "any_error": lambda d: d["error_turns"] > 0,
            "multi_error": lambda d: d["error_turns"] >= 2,
            "multi_deterministic_error": lambda d: d["deterministic_errors"] >= 2,
            "no_error": lambda d: d["error_turns"] == 0,
            "homogeneous_with_local_update": lambda d: d["eligible_group_c"] in (0, d["eligible_group_n"]) and d["abs_mass"] > 0,
        }
        summaries = {}
        for name, predicate in categories.items():
            selected = [d for d in trainable if predicate(d)]
            summaries[name] = {
                "count": len(selected), "abs_share": sum(d["abs_mass"] for d in selected) / total_abs,
                "neg_share": sum(d["neg_mass"] for d in selected) / total_neg if total_neg else 0,
                "max_abs_share": max((d["update_abs_share"] for d in selected), default=0),
                "threshold_counts": {str(t): sum(d["update_abs_share"] >= t for d in selected) for t in (.01, .02, .05, .1, .5)},
            }
        ordered = sorted(trainable, key=lambda d: d["abs_mass"], reverse=True)
        report["updates"].append({
            "step": step, "rows": len(batch), "groups": len(groups), "eligible": J,
            "transitions": N, "raw_group_correct_counts": dict(sorted(Counter(g["correct"] for g in group_stats).items())),
            "effective_group_n_c": dict(Counter(f"{g['eligible']}:{g['eligible_correct']}" for g in group_stats)),
            "total_abs_mass": total_abs, "total_neg_mass": total_neg,
            "error_action_abs_share": sum(d["error_abs_mass"] for d in trainable) / total_abs,
            "error_negative_delta_vs_trajectory": sum(d["error_neg_delta_vs_trajectory"] for d in trainable),
            "error_replacements": sum(d["error_replacements"] for d in trainable),
            "error_sign_flips": sum(d["error_sign_flips"] for d in trainable),
            "top_shares": {str(k): sum(d["update_abs_share"] for d in ordered[:k]) for k in (1, 5, 10)},
            "categories": summaries, "top10": ordered[:10],
            "top_multi_error": sorted([d for d in trainable if d["error_turns"] >= 2], key=lambda d: d["abs_mass"], reverse=True)[:10],
            "groups_detail": group_stats, "credit_audit": audit.to_dict(),
        })
        all_details.extend(info.values())
    report["trajectories"] = all_details
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(args.output), "rows": len(rows), "updates": [
        {k:v for k,v in u.items() if k not in ("top10", "top_multi_error", "groups_detail", "credit_audit")} for u in report["updates"]
    ]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
