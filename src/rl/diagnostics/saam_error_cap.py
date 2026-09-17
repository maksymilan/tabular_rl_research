"""Fixed raw-rollout credit comparison; no generation, optimizer or gold path."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from types import SimpleNamespace

from rl.frameworks.trl.mechanism import RLMechanism
from rl.frameworks.trl.state_action_ambiguity import _local_error_kind
from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn


def audit_error_cap(rows, tokenizer, *, expected_group_size=8, soft_alphas=()):
    groups = defaultdict(list)
    for row in rows:
        groups[(int(row["policy_global_step"]), int(row["example_index"]))].append(row)
    variants = {name: Counter() for name in ("A_binary_saam", "B_first_error_capped", "C_three_level_saam", "D_later_error_half")}
    soft_variants = {f"soft_alpha_{float(alpha):g}": Counter() for alpha in soft_alphas}
    outcomes = Counter()
    for key, group in sorted(groups.items()):
        if len(group) != expected_group_size:
            raise ValueError(f"incomplete group {key}: {len(group)}")
        episodes = []
        for row in group:
            # Explicit allowlist: never pass gold SQL/results into credit analysis.
            audit = {k: row[k] for k in ("trajectory_id", "example_index", "correct", "turns", "result_reward", "failure_type", "errors", "error_events") if k in row}
            audit["turns"] = [{k: turn[k] for k in ("turn_index", "parsed", "tool_output", "error_event", "execution_error_type", "execution_error") if k in turn} for turn in row["turns"]]
            turns = []
            for turn in (row["turns"] if row.get("process_update", False) else []):
                ids = tuple(tokenizer.encode(turn.get("model_output") or "", add_special_tokens=False))
                if not ids:
                    raise ValueError(f"empty authored response: {row['trajectory_id']}")
                turns.append(PolicyTurn((1,), ids, tuple(0.0 for _ in ids)))
            sample = SimpleNamespace(
                reward=1.0 if row["correct"] else 0.0,
                correct=bool(row["correct"]), process_update=bool(row.get("process_update", False)),
                turns=[(list(t.prompt_ids), list(t.response_ids)) for t in turns],
                step_rewards=None, audit_record=audit,
            )
            episodes.append(PolicyEpisode(sample, turns))
            kinds = [_local_error_kind(turn) for turn in audit["turns"]]
            outcomes["episodes"] += 1
            outcomes["eligible"] += int(sample.process_update)
            outcomes["correct"] += int(sample.correct)
            outcomes["timeouts"] += kinds.count("infrastructure_timeout")
            outcomes["eligible_deterministic_errors"] += sum(k is not None and k != "infrastructure_timeout" for k in kinds) * int(sample.process_update)
        for name, total in variants.items():
            selected = episodes
            if name == "C_three_level_saam":
                selected = []
                for ep in episodes:
                    has_error = bool(ep.sample.audit_record.get("errors") or ep.sample.audit_record.get("error_events")) or any(_local_error_kind(t) for t in ep.sample.audit_record["turns"])
                    reward = (0.75 if has_error else 1.25) if ep.sample.correct else -1.0
                    sample = SimpleNamespace(**{**vars(ep.sample), "reward": reward})
                    selected.append(replace(ep, sample=sample))
            mechanism = RLMechanism(
                policy_reduction="trajectory_token_mean",
                credit_assignment=(
                    "saam-first-error-capped" if name.startswith("B_")
                    else "saam-later-error-half" if name.startswith("D_")
                    else "saam-asymmetric-error"
                ),
            )
            original = mechanism.build_updates(selected, train_turns="all")
            updates, audit = mechanism.apply_credit(selected, original)
            coefficients = mechanism.effective_advantages(updates, transition_count=len(original), trajectory_count=len({u.trajectory_id for u in original}))
            total["transitions"] += len(updates)
            for update, coefficient in zip(updates, coefficients, strict=True):
                total["zero_transitions"] += int(update.advantage == 0)
                if update.trajectory_correct and coefficient < 0:
                    total["correct_negative_transitions"] += 1
                    total["correct_negative_mass"] += abs(coefficient)
                if not update.trajectory_correct and coefficient > 0:
                    total["wrong_positive_transitions"] += 1
                    total["wrong_positive_mass"] += coefficient
            for field in ("deterministic_error_transitions", "correct_error_positive_flips", "capped_error_transitions", "capped_correct_error_transitions", "shared_success_wrong_suppressed", "timeout_penalized_transitions"):
                total[field] += getattr(audit, field)
        if soft_variants:
            # Start from the ordinary asymmetric SAAM result, then attenuate
            # only later deterministic errors. This is an offline diagnostic;
            # it does not register a trainer credit assignment.
            mechanism = RLMechanism(policy_reduction="trajectory_token_mean", credit_assignment="saam-asymmetric-error")
            original = mechanism.build_updates(episodes, train_turns="all")
            full_updates, _ = mechanism.apply_credit(episodes, original)
            full_coefficients = mechanism.effective_advantages(
                full_updates,
                transition_count=len(original),
                trajectory_count=len({u.trajectory_id for u in original}),
            )
            error_keys = {}
            first_error_keys = set()
            by_trajectory = defaultdict(list)
            for episode in episodes:
                trajectory_id = str(episode.sample.audit_record["trajectory_id"])
                for turn in episode.sample.audit_record.get("turns", []):
                    kind = _local_error_kind(turn)
                    if kind is not None:
                        key = (trajectory_id, int(turn["turn_index"]))
                        error_keys[key] = kind
                        if kind != "infrastructure_timeout":
                            by_trajectory[trajectory_id].append(key)
            for keys in by_trajectory.values():
                if keys:
                    first_error_keys.add(min(keys, key=lambda key: key[1]))
            for alpha, (variant_name, total) in zip(soft_alphas, soft_variants.items(), strict=True):
                adjusted = []
                update_keys = {(update.trajectory_id, update.turn_index) for update in original}
                for update, full_update in zip(original, full_updates, strict=True):
                    key = (update.trajectory_id, update.turn_index)
                    kind = error_keys.get(key)
                    raw = full_update.advantage
                    if kind == "infrastructure_timeout":
                        raw = -max(abs(float(update.advantage)), 1.0)
                    elif kind in {"no_progress_repeat", "argument_validation_error", "protocol_error", "carrier_error", "invalid_response", "execution_error"} and key not in first_error_keys:
                        raw = -float(alpha) * max(abs(float(update.advantage)), 1.0)
                    adjusted.append(replace(full_update, advantage=raw))
                coefficients = mechanism.effective_advantages(
                    adjusted,
                    transition_count=len(original),
                    trajectory_count=len({u.trajectory_id for u in original}),
                )
                total["transitions"] += len(adjusted)
                for update, coefficient in zip(adjusted, coefficients, strict=True):
                    total["zero_transitions"] += int(update.advantage == 0)
                    if update.trajectory_correct and coefficient < 0:
                        total["correct_negative_transitions"] += 1
                        total["correct_negative_mass"] += abs(coefficient)
                    if not update.trajectory_correct and coefficient > 0:
                        total["wrong_positive_transitions"] += 1
                        total["wrong_positive_mass"] += coefficient
                later_error_count = sum(
                    kind in {"no_progress_repeat", "argument_validation_error", "protocol_error", "carrier_error", "invalid_response", "execution_error"}
                    and key in update_keys
                    and key not in first_error_keys
                    for key, kind in error_keys.items()
                )
                correct_later_error_count = sum(
                    key in update_keys and key not in first_error_keys and kind in {"no_progress_repeat", "argument_validation_error", "protocol_error", "carrier_error", "invalid_response", "execution_error"}
                    and any(u.trajectory_id == key[0] and u.trajectory_correct for u in original)
                    for key, kind in error_keys.items()
                )
                total["later_error_transitions"] += later_error_count
                total["correct_later_error_transitions"] += correct_later_error_count
                total["wrong_later_error_transitions"] += later_error_count - correct_later_error_count
                total["timeout_penalized_transitions"] += sum(
                    key in update_keys and kind == "infrastructure_timeout"
                    for key, kind in error_keys.items()
                )
    a, b = variants["A_binary_saam"], variants["B_first_error_capped"]
    passed = b["correct_negative_mass"] < a["correct_negative_mass"] and b["wrong_positive_mass"] <= a["wrong_positive_mass"] and b["capped_correct_error_transitions"] > 0
    for total in soft_variants.values():
        for field in (
            "wrong_positive_transitions",
            "wrong_positive_mass",
            "correct_later_error_transitions",
            "wrong_later_error_transitions",
            "timeout_penalized_transitions",
        ):
            total[field] += 0.0
    variants.update({k: dict(v) for k, v in soft_variants.items()})
    return {
        "schema_version": "saam-first-error-cap-static-v1", "gold_sql_read": False,
        "groups": len(groups), "outcomes": dict(outcomes),
        "variants": {k: dict(v) for k, v in variants.items()},
        "signal_gate_passed": passed,
        "limitations": "Response tokens re-tokenized from authored text; no fresh Harness execution, exact sampled-token gradient, or accuracy claim. Less negative credit is a mechanism effect, not proof that removed error penalties were harmful.",
    }
