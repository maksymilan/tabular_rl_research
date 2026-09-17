"""Offline audits that decide the next RL change, using a frozen run's own credit code.

Three questions, all CPU-only, no gold SQL, no generation, no GPU:

  A. Group normalization: how much advantage mass moves between difficulty bins
     if the population-std divisor is replaced by a constant divisor?
  B. Credit placement: where does the negative coefficient mass actually land -
     terminal submission, last relational action, earlier exploration, plan?
  C. Lineage-state coverage: for a wrong trajectory's turns, how often does the
     same lineage state also appear in a correct trajectory of the same group,
     and does the action match (SAAM mask) or differ (unexploited contrast)?
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fallback-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--sample-per-class", type=int, default=25)
    args = parser.parse_args()

    snapshot = args.run_root / "train/implementation_source_snapshot/src"
    sys.path[:0] = [str(snapshot), str(args.fallback_src)]
    # Parent packages come from the complete fallback tree; every credit-critical
    # module is then replaced by the run's own frozen file.
    import rl.frameworks.trl  # noqa: F401
    import rl.scenarios.diagnostics  # noqa: F401

    locked = (
        ("rl.frameworks.trl.transition_batch",
         snapshot / "rl/frameworks/trl/transition_batch.py"),
        ("rl.scenarios.diagnostics.audit_saam_lineage_replay",
         snapshot / "rl/scenarios/diagnostics/audit_saam_lineage_replay.py"),
        ("rl.frameworks.trl.state_action_ambiguity",
         snapshot / "rl/frameworks/trl/state_action_ambiguity.py"),
    )
    for name, source in locked:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    import torch
    from transformers import AutoTokenizer

    from rl.diagnostics.io import read_json, read_jsonl
    from rl.frameworks.trl.transition_batch import (
        PolicyEpisode,
        PolicyTurn,
        build_transition_updates,
        policy_reduction_advantages,
        standardized_group_advantages,
    )
    from rl.frameworks.trl.state_action_ambiguity import (
        _local_error_kind,
        apply_asymmetric_error_credit,
    )
    from rl.scenarios.diagnostics.audit_saam_lineage_replay import LineageReplay

    torch.set_num_threads(1)
    manifest = read_json(args.run_root / "train/run_manifest.json")
    raw_rows = read_jsonl(args.run_root / "train/rollouts.jsonl")
    rows = []
    for raw in raw_rows:
        row = {k: raw.get(k) for k in (
            "trajectory_id", "example_index", "db_id", "question", "correct", "legal",
            "result_reward", "failure_type", "errors", "error_events",
            "process_update", "policy_global_step",
        )}
        row["turns"] = [{k: t[k] for k in (
            "turn_index", "parsed", "tool_output", "error_event",
            "execution_error_type", "execution_error", "model_output",
        ) if k in t} for t in raw["turns"]]
        rows.append(row)
    del raw_rows

    tokenizer = AutoTokenizer.from_pretrained(
        manifest["model_path"], local_files_only=True
    )

    # ---- audit A accumulators -------------------------------------------------
    norm = {
        key: {str(bin_): Counter() for bin_ in range(9)}
        for key in ("std", "const1", "const05")
    }

    # ---- audit B accumulators -------------------------------------------------
    placement = {name: Counter() for name in (
        "wrong_error", "wrong_clean", "correct_error", "correct_clean",
    )}
    role_tokens = Counter()
    role_mass = Counter()

    # ---- audit C accumulators -------------------------------------------------
    coverage = Counter()
    per_update_coverage = []

    # ---- audit E accumulators -------------------------------------------------
    divergence = Counter()
    divergence_chain = Counter()
    contrast_supply = Counter()
    credit_source = Counter()

    # ---- audit F: where do masked / contrast turns sit? -----------------------
    depth_table = defaultdict(Counter)
    depth_denominator = Counter()
    tool_by_category = defaultdict(Counter)
    position_by_category = defaultdict(list)
    depth_by_category = defaultdict(Counter)
    mass_by_category = Counter()
    samples = {"contrast_strict": [], "contrast_diff_action": []}
    sample_seen_examples = defaultdict(Counter)
    digest_summary = {}
    contrast_quality = Counter()

    def short(value, limit=90):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        return text if len(text) <= limit else text[: limit] + "…"

    def action_summary(parsed):
        if not isinstance(parsed, dict):
            return None
        return {
            "tool": parsed.get("tool"),
            "arguments": short(parsed.get("arguments"), 220),
        }

    def output_summary(output):
        if not isinstance(output, dict):
            return None
        summary = {}
        if isinstance(output.get("rows"), list):
            summary["rows"] = len(output["rows"])
            summary["row0"] = short(output["rows"][0], 120) if output["rows"] else None
        if isinstance(output.get("table"), str):
            summary["table"] = output["table"]
        if isinstance(output.get("tables"), list):
            summary["tables"] = short(
                [item.get("table_name") if isinstance(item, dict) else item
                 for item in output["tables"]], 160
            )
        if isinstance(output.get("columns"), list):
            summary["columns"] = short(
                [item.get("name") if isinstance(item, dict) else item
                 for item in output["columns"]], 160
            )
        if output.get("error") is not None:
            summary["error"] = short(output.get("error"), 160)
        if not summary:
            summary["raw"] = short(output, 160)
        return summary

    def collect_strings(value):
        if isinstance(value, str):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from collect_strings(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from collect_strings(item)

    def answer_ancestry(row):
        """Depths of the derivation chain that produced the cited answer table."""
        produced, refs = {}, {}
        for depth, turn in enumerate(row["turns"]):
            output = turn.get("tool_output")
            if isinstance(output, dict):
                for key in ("table", "handle"):
                    value = output.get(key)
                    if isinstance(value, str):
                        produced[value] = depth
                tables = output.get("tables")
                if isinstance(tables, list):
                    for item in tables:
                        if isinstance(item, dict) and isinstance(item.get("table"), str):
                            produced.setdefault(item["table"], depth)
            parsed = turn.get("parsed") or {}
            refs[depth] = set(collect_strings(parsed.get("arguments")))
        stack, seen = [], set()
        for depth, turn in enumerate(row["turns"]):
            parsed = turn.get("parsed") or {}
            if parsed.get("tool") == "answer_from_context":
                for value in collect_strings(parsed.get("arguments")):
                    if value in produced:
                        stack.append(produced[value])
        while stack:
            depth = stack.pop()
            if depth in seen:
                continue
            seen.add(depth)
            for value in refs.get(depth, ()):
                if value in produced and produced[value] not in seen:
                    stack.append(produced[value])
        return seen

    by_update = defaultdict(list)
    for row in rows:
        by_update[row["policy_global_step"]].append(row)

    for step, batch in sorted(by_update.items()):
        groups = defaultdict(list)
        for row in batch:
            groups[row["example_index"]].append(row)

        # ---------------- audit A: advantage mass by difficulty bin -----------
        update_norm = {key: {str(bin_): Counter() for bin_ in range(9)} for key in norm}
        for example_index, group in groups.items():
            eligible = [bool(row["process_update"]) for row in group]
            rewards = [float(row["result_reward"]["value"]) for row in group]
            correct_flags = [bool(row["correct"]) for row in group]
            eligible_correct = sum(ok and keep for ok, keep in zip(correct_flags, eligible, strict=True))
            bin_ = str(eligible_correct)
            std_adv = standardized_group_advantages(rewards, eligible)
            kept = [rewards[i] for i, keep in enumerate(eligible) if keep]
            if kept and len(set(kept)) > 1:
                mean = sum(kept) / len(kept)
            else:
                mean = None
            for scheme, divisor in (("std", None), ("const1", 1.0), ("const05", 0.5)):
                for index, keep in enumerate(eligible):
                    if not keep:
                        continue
                    if scheme == "std":
                        value = std_adv[index]
                    elif mean is None:
                        value = 0.0
                    else:
                        value = (rewards[index] - mean) / divisor
                    cell = update_norm[scheme][bin_]
                    cell["transitions_trajectories"] += 1
                    cell["abs_mass"] += abs(value)
                    cell["positive_mass"] += max(0.0, value)
                    cell["negative_mass"] += max(0.0, -value)
                    if correct_flags[index]:
                        cell["correct_abs_mass"] += abs(value)
                    else:
                        cell["wrong_abs_mass"] += abs(value)
        for scheme in norm:
            for bin_, cell in update_norm[scheme].items():
                norm[scheme][bin_].update(cell)
                norm[scheme][bin_]["updates"] += 1

        # ---------------- rebuild transitions for audits B and C --------------
        episodes, audit_rows = [], {}
        for row in batch:
            turns = []
            for turn in (row["turns"] if row["process_update"] else []):
                ids = tuple(tokenizer.encode(turn["model_output"], add_special_tokens=False))
                turns.append(PolicyTurn((1,), ids, (0.0,) * len(ids)))
            sample = SimpleNamespace(
                reward=float(row["result_reward"]["value"]),
                correct=bool(row["correct"]),
                process_update=bool(row["process_update"]),
                step_rewards=None,
                turns=[(list(t.prompt_ids), list(t.response_ids)) for t in turns],
                audit_record=row,
            )
            episodes.append(PolicyEpisode(sample, turns))
            kinds = [_local_error_kind(t) for t in row["turns"]]
            audit_rows[row["trajectory_id"]] = {
                "correct": bool(row["correct"]),
                "error_turns": [i for i, k in enumerate(kinds) if k],
                "turn_count": len(turns),
                "response_tokens": sum(len(t.response_ids) for t in turns),
            }

        original = build_transition_updates(episodes, reward_mode="result-only", train_turns="all")
        credited, _ = apply_asymmetric_error_credit(
            episodes, original, error_penalty=manifest["error_penalty"]
        )
        update_count = len(original)
        trajectory_count = len({u.trajectory_id for u in original})
        coeffs = policy_reduction_advantages(
            credited,
            reduction="trajectory_token_mean",
            normalization_transition_count=update_count,
            normalization_trajectory_count=trajectory_count,
        )
        row_map = {row["trajectory_id"]: row for row in batch}
        ancestry = {
            row["trajectory_id"]: answer_ancestry(row) for row in batch
        }
        tool_sequence = {}
        for row in batch:
            tools = [str((t.get("parsed") or {}).get("tool")) for t in row["turns"]]
            tool_sequence[row["trajectory_id"]] = tools

        # ---- audit B: where the coefficient mass lands -----------------------
        coeff_map = {}
        for old, new, coefficient in zip(original, credited, coeffs, strict=True):
            coeff_map[(new.trajectory_id, new.turn_index)] = coefficient
            row = row_map[new.trajectory_id]
            info = audit_rows[new.trajectory_id]
            if not info["error_turns"]:
                klass = "correct_clean" if info["correct"] else "wrong_clean"
            else:
                klass = "correct_error" if info["correct"] else "wrong_error"
            tools = tool_sequence[new.trajectory_id]
            tool = tools[new.turn_index]
            later_actions = [
                index for index, name in enumerate(tools)
                if name not in ("plan", "answer_from_context") and index > new.turn_index
            ]
            if tool == "answer_from_context":
                role = "terminal"
            elif tool == "plan":
                role = "plan"
            elif later_actions:
                role = "earlier_action"
            else:
                role = "last_relational_action"
            turn = row["turns"][new.turn_index]
            parsed = turn.get("parsed") or {}
            think = parsed.get("think") or ""
            action_text = json.dumps(
                {k: v for k, v in parsed.items() if k != "think"},
                ensure_ascii=False, sort_keys=True,
            )
            think_tokens = len(tokenizer.encode(think, add_special_tokens=False)) if think else 0
            action_tokens = len(tokenizer.encode(action_text, add_special_tokens=False))
            cell = placement[klass]
            cell["transitions"] += 1
            cell["response_tokens"] += len(new.response_ids or ())
            on_chain = new.turn_index in ancestry[new.trajectory_id]
            cell["transitions_on_answer_chain" if on_chain else "transitions_off_answer_chain"] += 1
            cell[f"mass_{role}"] += abs(coefficient)
            if coefficient < 0:
                cell["negative_transitions"] += 1
                cell["negative_mass"] += abs(coefficient)
                cell[f"negative_mass_{role}"] += abs(coefficient)
                cell["negative_mass_on_answer_chain" if on_chain else "negative_mass_off_answer_chain"] += abs(coefficient)
                cell["negative_think_tokens"] += think_tokens
                cell["negative_action_tokens"] += action_tokens
            elif coefficient > 0:
                cell["positive_transitions"] += 1
                cell["positive_mass"] += abs(coefficient)
                cell[f"positive_mass_{role}"] += abs(coefficient)
                cell["positive_mass_on_answer_chain" if on_chain else "positive_mass_off_answer_chain"] += abs(coefficient)
            else:
                cell["zeroed_transitions"] += 1
            role_tokens[role] += think_tokens + action_tokens
            role_mass[role] += abs(coefficient)

        # ---- audit C: lineage state coverage ---------------------------------
        update_cov = Counter()
        per_traj_states = {}
        per_traj_error_turns = {}
        for row in batch:
            if not row["process_update"]:
                continue
            events, _ = LineageReplay(row).replay()
            per_traj_states[row["trajectory_id"]] = [
                (event.get("state_digest"), event.get("action_digest"))
                for event in events
            ]
            per_traj_error_turns[row["trajectory_id"]] = {
                i for i, k in enumerate(
                    _local_error_kind(t) for t in row["turns"]
                ) if k
            }
        for example_index, group in groups.items():
            eligible = [row for row in group if row["process_update"]]
            correct = [row for row in eligible if row["correct"]]
            wrong = [row for row in eligible if not row["correct"]]
            if not correct or not wrong:
                continue
            correct_states = defaultdict(set)
            correct_pairs = defaultdict(set)
            for row in correct:
                for depth, (state, action) in enumerate(per_traj_states[row["trajectory_id"]]):
                    correct_states[state].add(depth)
                    correct_pairs[(state, action)].add(depth)
            for row in wrong:
                states = per_traj_states[row["trajectory_id"]]
                for depth, (state, action) in enumerate(states):
                    update_cov["wrong_turns"] += 1
                    if state in correct_states:
                        update_cov["shared_state_turns"] += 1
                        if (state, action) in correct_pairs:
                            update_cov["shared_state_action_turns"] += 1
                        else:
                            update_cov["shared_state_other_action_turns"] += 1
                    elif action is not None:
                        update_cov["unshared_state_turns"] += 1
        coverage.update(update_cov)
        per_update_coverage.append({"update": step + 1, **update_cov})

        # ---- audit E1: where does a wrong trajectory first diverge? ----------
        for example_index, group in groups.items():
            eligible = [row for row in group if row["process_update"]]
            correct = [row for row in eligible if row["correct"]]
            wrong = [row for row in eligible if not row["correct"]]
            if not correct or not wrong:
                continue
            for wrong_row in wrong:
                wrong_states = per_traj_states[wrong_row["trajectory_id"]]
                wrong_chain = ancestry[wrong_row["trajectory_id"]]
                for correct_row in correct:
                    correct_states = per_traj_states[correct_row["trajectory_id"]]
                    limit = min(len(wrong_states), len(correct_states))
                    first = None
                    for depth in range(limit):
                        if wrong_states[depth] != correct_states[depth]:
                            first = depth
                            break
                    if first is None:
                        divergence["no_divergence_within_shared_depth"] += 1
                        continue
                    divergence["pairs"] += 1
                    divergence[f"depth_{min(first, 5)}"] += 1
                    if first in wrong_chain:
                        divergence_chain["first_divergence_on_wrong_answer_chain"] += 1
                    else:
                        divergence_chain["first_divergence_off_wrong_answer_chain"] += 1
                    if first in ancestry[correct_row["trajectory_id"]]:
                        divergence_chain["first_divergence_on_correct_answer_chain"] += 1
                    if first in per_traj_error_turns[wrong_row["trajectory_id"]]:
                        divergence_chain["first_divergence_is_wrong_side_harness_error"] += 1

        # ---- audit E2: strict supply for a state-contrast penalty ------------
        for example_index, group in groups.items():
            eligible = [row for row in group if row["process_update"]]
            correct = [row for row in eligible if row["correct"]]
            wrong = [row for row in eligible if not row["correct"]]
            if not correct or not wrong:
                continue
            state_to_actions = defaultdict(lambda: defaultdict(set))
            for row in correct:
                for depth, (state, action) in enumerate(per_traj_states[row["trajectory_id"]]):
                    if action is not None:
                        state_to_actions[state][action].add(row["trajectory_id"])
                        if action not in digest_summary and depth < len(row["turns"]):
                            parsed_now = row["turns"][depth].get("parsed") or {}
                            digest_summary[action] = {
                                "tool": parsed_now.get("tool"),
                                "arguments": parsed_now.get("arguments"),
                            }
            for row in wrong:
                turns = row["turns"]
                span = max(1, len(turns) - 1)
                for depth, (state, action) in enumerate(per_traj_states[row["trajectory_id"]]):
                    depth_denominator[min(depth, 8)] += 1
                    tool = str((turns[depth].get("parsed") or {}).get("tool")) if depth < len(turns) else "?"
                    if state is None or action is None:
                        continue
                    if any(candidate == action for candidate in state_to_actions.get(state, {})):
                        category = "saam_same_action"
                    elif state in state_to_actions:
                        differing = [
                            trajs for candidate, trajs in state_to_actions[state].items()
                            if candidate != action
                        ]
                        category = (
                            "contrast_strict"
                            if any(len(trajs) >= 2 for trajs in differing)
                            else "contrast_diff_action"
                        )
                    else:
                        category = "no_match"
                    depth_table[category][min(depth, 8)] += 1
                    depth_by_category[category][min(depth, 8)] += 1
                    tool_by_category[category][tool] += 1
                    position_by_category[category].append(depth / span)
                    mass_by_category[category] += abs(
                        coeff_map.get((row["trajectory_id"], depth), 0.0)
                    )
                    if category.startswith("contrast"):
                        turn_now = turns[depth] if depth < len(turns) else {}
                        output_now = turn_now.get("tool_output")
                        produced_now = (
                            output_now.get("table")
                            if isinstance(output_now, dict)
                            else None
                        )
                        used_later = any(
                            isinstance(produced_now, str)
                            and produced_now
                            in json.dumps(
                                (later.get("parsed") or {}).get("arguments"),
                                ensure_ascii=False,
                            )
                            for later in turns[depth + 1:]
                        )
                        wrong_tool = (turn_now.get("parsed") or {}).get("tool")
                        wrong_args = (turn_now.get("parsed") or {}).get("arguments") or {}
                        drop_keys = {"return_columns", "columns", "limit"}
                        stripped_wrong = {
                            k: v for k, v in wrong_args.items() if k not in drop_keys
                        }
                        describe_variation = wrong_tool == "describe_table"
                        projection_only = True
                        wrong_tables = set(wrong_args.get("tables") or [])
                        for candidate, trajs in state_to_actions[state].items():
                            if candidate == action:
                                continue
                            detail = digest_summary.get(candidate) or {}
                            if detail.get("tool") != "describe_table":
                                describe_variation = False
                            if detail.get("tool") != wrong_tool:
                                projection_only = False
                                continue
                            candidate_args = detail.get("arguments") or {}
                            stripped_candidate = {
                                k: v for k, v in candidate_args.items() if k not in drop_keys
                            }
                            if stripped_candidate != stripped_wrong:
                                projection_only = False
                            try:
                                sorted_args = json.loads(
                                    json.dumps(candidate_args, sort_keys=True)
                                )
                            except Exception:
                                sorted_args = {}
                            candidate_tables = set(sorted_args.get("tables") or [])
                            if not (
                                wrong_tables >= candidate_tables
                                or candidate_tables >= wrong_tables
                            ):
                                describe_variation = False
                        if describe_variation:
                            contrast_quality["describe_set_variation"] += 1
                        elif projection_only:
                            contrast_quality["projection_or_limit_only"] += 1
                        else:
                            contrast_quality["other_action_change"] += 1
                            contrast_quality[
                                "other_action_change_used_later"
                                if used_later
                                else "other_action_change_unused"
                            ] += 1
                            if depth in ancestry[row["trajectory_id"]]:
                                contrast_quality["other_action_change_on_answer_chain"] += 1
                        if used_later:
                            contrast_quality["any_used_later"] += 1
                    if (
                        args.sample_output is not None
                        and category in samples
                        and len(samples[category]) < args.sample_per_class
                        and sample_seen_examples[category][example_index] < 2
                    ):
                        sample_seen_examples[category][example_index] += 1
                        turn = turns[depth] if depth < len(turns) else {}
                        output = turn.get("tool_output")
                        produced = (
                            output.get("table") if isinstance(output, dict) else None
                        )
                        later_refs = any(
                            isinstance(produced, str)
                            and produced in json.dumps(
                                (later.get("parsed") or {}).get("arguments"),
                                ensure_ascii=False,
                            )
                            for later in turns[depth + 1:]
                        )
                        alternatives_detail = []
                        for candidate, trajs in list(state_to_actions[state].items())[:4]:
                            if candidate == action:
                                continue
                            alternatives_detail.append(
                                {
                                    "action_digest": candidate[:16],
                                    "correct_trajectories": len(trajs),
                                    "action": {
                                        "tool": (digest_summary.get(candidate) or {}).get("tool"),
                                        "arguments": short(
                                            (digest_summary.get(candidate) or {}).get("arguments"),
                                            220,
                                        ),
                                    },
                                }
                            )
                        correct_example = None
                        if alternatives_detail:
                            for candidate_row in correct:
                                for cand_depth, (cand_state, cand_action) in enumerate(
                                    per_traj_states[candidate_row["trajectory_id"]]
                                ):
                                    if (
                                        cand_state == state
                                        and cand_action is not None
                                        and cand_action != action
                                        and cand_depth < len(candidate_row["turns"])
                                    ):
                                        candidate_turn = candidate_row["turns"][cand_depth]
                                        correct_example = {
                                            "trajectory_id": candidate_row["trajectory_id"],
                                            "depth": cand_depth,
                                            "action": action_summary(candidate_turn.get("parsed")),
                                            "output": output_summary(candidate_turn.get("tool_output")),
                                        }
                                        break
                                if correct_example:
                                    break
                        samples[category].append(
                            {
                                "category": category,
                                "example_index": example_index,
                                "db_id": row.get("db_id"),
                                "question": (row.get("question") or "")[:180],
                                "update": step + 1,
                                "trajectory_id": row["trajectory_id"],
                                "turn_index": depth,
                                "turn_count": len(turns),
                                "prior_actions": [
                                    action_summary((t.get("parsed") or {}))
                                    for t in turns[:depth]
                                ],
                                "prior_outputs": [
                                    output_summary(t.get("tool_output")) for t in turns[:depth]
                                ],
                                "wrong_action": action_summary(turn.get("parsed")),
                                "wrong_output": output_summary(output),
                                "wrong_error": turn.get("execution_error_type")
                                or turn.get("execution_error"),
                                "wrong_flags": {
                                    "on_answer_chain": depth in ancestry[row["trajectory_id"]],
                                    "output_used_later": later_refs,
                                    "final_legal": bool(row.get("legal")),
                                    "final_failure": row.get("failure_type"),
                                    "harness_errors_total": row.get("errors"),
                                },
                                "correct_alternatives": alternatives_detail,
                                "correct_example": correct_example,
                                "coefficient": coeff_map.get((row["trajectory_id"], depth), 0.0),
                            }
                        )
                    alternatives = state_to_actions.get(state)
                    if not alternatives:
                        continue
                    contrast_supply["state_matched_turns"] += 1
                    differing = {
                        candidate: trajs
                        for candidate, trajs in alternatives.items()
                        if candidate != action
                    }
                    if not differing:
                        continue
                    contrast_supply["state_matched_different_action_turns"] += 1
                    mass = abs(coeff_map.get((row["trajectory_id"], depth), 0.0))
                    contrast_supply["different_action_negative_mass"] += mass
                    if any(len(trajs) >= 2 for trajs in differing.values()):
                        contrast_supply["with_repeated_correct_alternative"] += 1
                        contrast_supply["repeated_alternative_negative_mass"] += mass
                    if depth in ancestry[row["trajectory_id"]]:
                        contrast_supply["on_wrong_answer_chain"] += 1
            contrast_supply["all_wrong_turn_negative_mass"] = contrast_supply.get(
                "all_wrong_turn_negative_mass", 0.0
            )
        for row in batch:
            if not row["process_update"] or row["correct"]:
                continue
            for depth in range(len(row["turns"])):
                coefficient = coeff_map.get((row["trajectory_id"], depth), 0.0)
                if coefficient < 0:
                    contrast_supply["all_wrong_turn_negative_mass"] += abs(coefficient)

        # ---- audit E3: negative mass by credit source ------------------------
        for old, new, coefficient in zip(original, credited, coeffs, strict=True):
            if coefficient >= 0:
                continue
            on_chain = new.turn_index in ancestry[new.trajectory_id]
            modified = abs(new.advantage - old.advantage) > 1e-9
            source = "local_error_override" if modified else "trajectory_credit"
            credit_source[f"{source}_{'on_chain' if on_chain else 'off_chain'}"] += abs(coefficient)
        print(json.dumps({"update": step + 1, "coverage": dict(update_cov)}), flush=True)

    total_mass = {scheme: sum(c["abs_mass"] for c in norm[scheme].values()) for scheme in norm}
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "run_root": str(args.run_root),
        "rollouts_sha256": hashlib.sha256((args.run_root / "train/rollouts.jsonl").read_bytes()).hexdigest(),
        "identity": {k: manifest.get(k) for k in (
            "credit_assignment", "policy_reduction", "error_penalty",
            "advantage_magnitude_cap", "result_advantage_profile", "span_balance_alpha",
        )},
        "audit_a_normalization": {
            scheme: {
                "total_abs_mass": total_mass[scheme],
                "by_bin": {
                    bin_: {
                        **dict(cell),
                        "mass_share": cell["abs_mass"] / total_mass[scheme] if total_mass[scheme] else 0.0,
                        "per_positive_sample": (
                            cell["positive_mass"] / cell["transitions_trajectories"]
                        ),
                    }
                    for bin_, cell in sorted(norm[scheme].items()) if cell
                },
            }
            for scheme in norm
        },
        "audit_b_credit_placement": {
            klass: dict(cell) for klass, cell in placement.items() if cell
        },
        "audit_b_role_tokens_vs_mass": {
            role: {"tokens": role_tokens[role], "abs_mass": role_mass[role]}
            for role in role_mass
        },
        "audit_c_lineage_coverage": {
            **dict(coverage),
            "shared_state_rate": coverage["shared_state_turns"] / coverage["wrong_turns"] if coverage["wrong_turns"] else 0.0,
            "shared_state_same_action_rate": coverage["shared_state_action_turns"] / coverage["wrong_turns"] if coverage["wrong_turns"] else 0.0,
            "shared_state_other_action_rate": coverage["shared_state_other_action_turns"] / coverage["wrong_turns"] if coverage["wrong_turns"] else 0.0,
        },
        "audit_c_per_update": per_update_coverage,
        "audit_e1_first_divergence": {
            **dict(divergence),
            **dict(divergence_chain),
            "on_wrong_chain_rate": (
                divergence_chain["first_divergence_on_wrong_answer_chain"]
                / max(1, divergence["pairs"])
            ),
        },
        "audit_e2_contrast_supply": dict(contrast_supply),
        "audit_e3_negative_mass_by_source": dict(credit_source),
        "audit_f_depth_and_tools": {
            "wrong_turns_by_depth": dict(depth_denominator),
            "category_by_depth": {
                category: dict(counts) for category, counts in depth_by_category.items()
            },
            "category_tools": {
                category: dict(counts.most_common(12))
                for category, counts in tool_by_category.items()
            },
            "category_position_fraction": {
                category: {
                    "n": len(values),
                    "median": sorted(values)[len(values) // 2] if values else None,
                    "mean": (sum(values) / len(values)) if values else None,
                }
                for category, values in position_by_category.items()
            },
            "category_negative_mass": dict(mass_by_category),
            "contrast_quality": dict(contrast_quality),
        },
        "limitations": (
            "Response tokens reconstructed from saved authored text. Coefficient mass is the "
            "trajectory_token_mean reduction value before span masks, PPO ratios, Adam and "
            "gradient cancellation. No gold SQL, generation, replay or GPU used."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.sample_output is not None:
        with args.sample_output.open("w", encoding="utf-8") as handle:
            for category, rows in samples.items():
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k.startswith("audit")}, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
