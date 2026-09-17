"""CPU-only retrospective cap audit using each run's frozen credit implementation.

No rollout, GPU, optimizer, gold SQL, or training mutation. Re-tokenization is an
estimate of saved sampled lengths; EOS sensitivity is reported separately.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fallback-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = args.run_root / "train/implementation_source_snapshot/src"
    sys.path[:0] = [str(snapshot), str(args.fallback_src)]
    # Frozen snapshots can be namespace trees beneath a regular rl package.
    # Load the locked files explicitly instead of allowing package shadowing.
    import rl.frameworks.trl
    for leaf in ("transition_batch", "state_action_ambiguity"):
        name = "rl.frameworks.trl." + leaf
        source = snapshot / "rl/frameworks/trl" / (leaf + ".py")
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    import torch
    from transformers import AutoTokenizer
    from rl.diagnostics.io import read_json, read_jsonl
    from rl.frameworks.trl.transition_batch import (
        PolicyEpisode, PolicyTurn, build_transition_updates,
        policy_reduction_advantages,
    )
    from rl.frameworks.trl.state_action_ambiguity import (
        _local_error_kind, apply_asymmetric_error_credit,
    )

    torch.set_num_threads(1)
    manifest = read_json(args.run_root / "train/run_manifest.json")
    path = args.run_root / "train/rollouts.jsonl"
    raw_rows = read_jsonl(path)
    rows = []
    for raw in raw_rows:
        row = {k: raw.get(k) for k in (
            "trajectory_id", "example_index", "db_id", "question", "correct",
            "legal", "result_reward", "failure_type", "errors", "error_events",
            "process_update", "policy_global_step",
        )}
        row["turns"] = [{k: t[k] for k in (
            "turn_index", "parsed", "tool_output", "error_event",
            "execution_error_type", "execution_error", "model_output",
        ) if k in t} for t in raw["turns"]]
        rows.append(row)
    del raw_rows
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_path"], local_files_only=True)
    by_update = defaultdict(list)
    for row in rows:
        by_update[row["policy_global_step"]].append(row)
    stats = {str(c): Counter() for c in (1.0, 1.5, 2.0)}
    raw_stats = {str(c): Counter() for c in (1.0, 1.5, 2.0)}
    tool_stats = {str(c): Counter() for c in (1.0, 1.5, 2.0)}
    details, update_stats = [], []
    for step, batch in sorted(by_update.items()):
        groups = defaultdict(list)
        for row in batch:
            groups[row["example_index"]].append(row)
        assert len(batch) == 240 and len(groups) == 30
        assert all(len(g) == 8 for g in groups.values())
        episodes, info = [], {}
        for row in batch:
            tid = row["trajectory_id"]
            assert tid not in info
            group = groups[row["example_index"]]
            turns = []
            for t in (row["turns"] if row["process_update"] else []):
                ids = tuple(tokenizer.encode(t["model_output"], add_special_tokens=False))
                assert ids
                turns.append(PolicyTurn((1,), ids, (0.0,) * len(ids)))
            sample = SimpleNamespace(
                reward=float(row["result_reward"]["value"]), correct=bool(row["correct"]),
                process_update=bool(row["process_update"]), step_rewards=None,
                turns=[(list(t.prompt_ids), list(t.response_ids)) for t in turns], audit_record=row,
            )
            episodes.append(PolicyEpisode(sample, turns))
            kinds = [_local_error_kind(t) for t in row["turns"]]
            info[tid] = {
                **{k:v for k,v in row.items() if k != "turns"},
                "update": step + 1, "eligible_group_n": sum(bool(x["process_update"]) for x in group),
                "eligible_group_correct": sum(bool(x["process_update"] and x["correct"]) for x in group),
                "group_correct": sum(bool(x["correct"]) for x in group),
                "deterministic_error_turns": [i for i,k in enumerate(kinds) if k and k != "infrastructure_timeout"],
                "timeout_turns": [i for i,k in enumerate(kinds) if k == "infrastructure_timeout"],
                "turn_count": len(turns), "response_tokens": sum(len(t.response_ids) for t in turns),
                "turns": [], "caps": {},
            }
        original = build_transition_updates(episodes, reward_mode="result-only", train_turns="all")
        credited, audit = apply_asymmetric_error_credit(episodes, original, error_penalty=manifest["error_penalty"])
        n, j = len(original), len({u.trajectory_id for u in original})
        coeffs = policy_reduction_advantages(credited, reduction="trajectory_token_mean",
            normalization_transition_count=n, normalization_trajectory_count=j)
        row_map = {r["trajectory_id"]:r for r in batch}
        for old, new, b in zip(original, credited, coeffs, strict=True):
            d = info[new.trajectory_id]
            t = row_map[new.trajectory_id]["turns"][new.turn_index]
            kind = _local_error_kind(t)
            length = len(new.response_ids)
            total = d["response_tokens"]
            b_plus_eos = new.advantage * ((length+1)/(total+d["turn_count"])) * n/j
            # A +/-2-token stress check around text reconstruction, not a certified bound.
            lo = abs(new.advantage)*max(1,length-2)/(total+2*d["turn_count"])*n/j
            hi = abs(new.advantage)*(length+2)/max(1,total-2*d["turn_count"])*n/j
            d["base_advantage"] = old.advantage
            d["turns"].append({"i":new.turn_index, "raw_a":old.advantage,
                "credit_a":new.advantage, "coefficient":b, "coefficient_plus_eos":b_plus_eos,
                "abs_sensitivity_lo":lo, "abs_sensitivity_hi":hi, "tokens":length,
                "token_share":new.trajectory_token_weight, "error_kind":kind,
                "action":{k:v for k,v in (t.get("parsed") or {}).items() if k != "think"},
                "reasoning":(t.get("parsed") or {}).get("think"),
                "tool_output":t.get("tool_output"),
            })
        per_update = {"update":step+1,"transitions":n,"trajectories":j,"mean_turns":n/j,
            "post_credit_nonzero":sum(u.advantage!=0 for u in credited), "caps":{}}
        for c in (1.0,1.5,2.0):
            k=str(c); us=Counter()
            for d in info.values():
                if not d["process_update"]:
                    continue
                rr=raw_stats[k]
                a=d.get("base_advantage",0.)
                if abs(a)>c:
                    rr["trajectories"]+=1;rr["positive" if a>0 else "negative"]+=1
                hit=[t for t in d["turns"] if abs(t["coefficient"])>c]
                pos=[t for t in hit if t["coefficient"]>0]
                neg=[t for t in hit if t["coefficient"]<0]
                eos=[t for t in d["turns"] if abs(t["coefficient_plus_eos"])>c]
                d["caps"][k]={"hit_turns":[t["i"] for t in hit],"positive_turns":[t["i"] for t in pos],
                    "negative_turns":[t["i"] for t in neg],
                    "removed_positive":sum(t["coefficient"]-c for t in pos)/n,
                    "removed_negative":sum(-t["coefficient"]-c for t in neg)/n,
                    "retained_abs_fraction":sum(min(c,abs(t["coefficient"])) for t in d["turns"])/sum(abs(t["coefficient"]) for t in d["turns"]) if any(t["coefficient"] for t in d["turns"]) else 1.,
                }
                us["eligible"]+=1;us["trajectories"]+=bool(hit)
                us["positive_trajectories"]+=bool(pos);us["negative_trajectories"]+=bool(neg)
                us["both_sign_trajectories"]+=bool(pos and neg)
                us["correct_trajectories"]+=bool(hit and d["correct"])
                us["wrong_trajectories"]+=bool(hit and not d["correct"])
                us["correct_clean_trajectories"]+=bool(hit and d["correct"] and not d["deterministic_error_turns"] and not d["timeout_turns"])
                us["raw_a_le_cap_trajectories"]+=bool(hit and abs(a)<=c)
                us["transitions"]+=len(hit);us["positive_transitions"]+=len(pos);us["negative_transitions"]+=len(neg)
                us["plus_eos_trajectories"]+=bool(eos);us["plus_eos_transitions"]+=len(eos)
                us["plus_eos_membership_changed"]+=set(t["i"] for t in hit)!=set(t["i"] for t in eos)
                us["sensitivity_uncertain_transitions"]+=sum(t["abs_sensitivity_lo"]<=c<t["abs_sensitivity_hi"] for t in d["turns"])
                for t in d["turns"]:
                    sign="positive" if t["coefficient"]>0 else "negative"
                    us[sign+"_before_mass"]+=abs(t["coefficient"])/n
                    us[sign+"_removed_mass"]+=max(0,abs(t["coefficient"])-c)/n
                    if abs(t["coefficient"])>c:
                        tool_stats[k][str(t["action"].get("tool"))]+=1
                        us["negative_error_transitions"]+=bool(t["coefficient"]<0 and t["error_kind"])
                        us["negative_nonerror_transitions"]+=bool(t["coefficient"]<0 and not t["error_kind"])
                        us["correct_negative_error_transitions"]+=bool(t["coefficient"]<0 and t["error_kind"] and d["correct"])
            stats[k].update(us);per_update["caps"][k]=dict(us)
        update_stats.append(per_update);details.extend(info.values())
    args.output.mkdir(parents=True,exist_ok=True)
    result={"run_root":str(args.run_root),"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "identity":{k:manifest.get(k) for k in ("seed","selection_sha256","examples_json_sha256","initial_adapter_sha256","protocol_hash","student_prompt_sha256","credit_assignment","policy_reduction","advantage_magnitude_cap")},
        "rows":len(rows),"stats":{k:dict(v) for k,v in stats.items()},"raw_advantage_stats":{k:dict(v) for k,v in raw_stats.items()},
        "tools":{k:dict(v) for k,v in tool_stats.items()},"updates":update_stats,
        "limitations":"Response tokens reconstructed from saved authored text, not exact sampled IDs. Plus-EOS and +/-2 sensitivity supplied. Coefficient mass excludes PPO ratios, span masks, gradient cancellation and Adam. No gold SQL, new generation, fresh replay or GPU used.",
        "loaded_modules":{m:{"path":sys.modules[m].__file__,"sha256":hashlib.sha256(Path(sys.modules[m].__file__).read_bytes()).hexdigest()} for m in ("rl.frameworks.trl.transition_batch","rl.frameworks.trl.state_action_ambiguity")}}
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'trajectories.jsonl').open('w') as f:
        for d in details:f.write(json.dumps(d,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('updates','tools')},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
