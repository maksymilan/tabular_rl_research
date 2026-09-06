#!/usr/bin/env python3
"""Audit environment exploration and whether grounded reward graphs are more than chains."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "harness"))

from provenance import build_grounding_references, build_references  # noqa: E402


PERCEPTION_TOOLS = {"describe_table", "inspect_column", "read_subtable"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def referenced_steps(refs: list[dict[str, Any]]) -> set[str]:
    return {str(ref["step"]) for ref in refs if isinstance(ref.get("step"), str)}


def rebuild_references(steps: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Rebuild runtime provenance because rollout artifacts do not persist every reference edge."""
    history: dict[str, dict[str, Any]] = {}
    handle_to_step: dict[str, str] = {}
    rebuilt: dict[str, list[dict[str, Any]]] = {}
    for step in steps:
        step_id = str(step.get("step_id"))
        call = step.get("tool_call") or {}
        tool = str(call.get("tool"))
        arguments = call.get("arguments") or {}

        def resolve_step(ref):
            if ref in handle_to_step:
                return handle_to_step[ref]
            if ref in history:
                return ref
            return None

        refs = build_references(tool, arguments, resolve_step)
        refs.extend(build_grounding_references(tool, arguments, history))
        rebuilt[step_id] = refs
        output = step.get("tool_output") or {}
        record = {"tool": tool, "arguments": arguments, "output": output, "references": refs}
        if tool == "read_subtable":
            requested_columns = arguments.get("columns")
            if requested_columns:
                record["observed_columns"] = list(requested_columns)
        history[step_id] = record
        handle = output.get("table")
        if isinstance(handle, str):
            handle_to_step[handle] = step_id
    return rebuilt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, action="append", required=True)
    parser.add_argument("--scored", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.trajectories) != len(args.scored):
        parser.error("--trajectories and --scored counts must match")

    episodes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    sources = []
    for trajectory_path, scored_path in zip(args.trajectories, args.scored):
        trajectory_path = trajectory_path.resolve()
        scored_path = scored_path.resolve()
        trajectories = {row["trajectory_id"]: row for row in read_jsonl(trajectory_path)}
        scores = {row["trajectory_id"]: row for row in read_jsonl(scored_path)}
        if set(trajectories) != set(scores):
            raise ValueError(f"trajectory/scored ids differ for {trajectory_path}")
        for trajectory_id, trajectory in trajectories.items():
            score = scores[trajectory_id]
            diagnostics = score.get("diagnostics") or {}
            if score.get("correct") and diagnostics.get("deterministic_grounding_complete"):
                episodes.append((trajectory, score))
        sources.append({
            "trajectories": str(trajectory_path),
            "trajectories_sha256": sha256(trajectory_path),
            "scored": str(scored_path),
            "scored_sha256": sha256(scored_path),
        })

    counters = collections.Counter()
    tool_calls = collections.Counter()
    positive_by_tool = collections.Counter()
    zero_by_tool = collections.Counter()
    negative_by_tool = collections.Counter()
    step_counts: list[int] = []
    positive_counts: list[int] = []
    max_positive_shares: list[float] = []
    described_table_counts: list[int] = []
    tool_diversity: list[int] = []
    topology_examples: list[dict[str, Any]] = []

    for trajectory, score in episodes:
        steps = trajectory.get("steps") or []
        scored_steps = score.get("steps") or []
        step_counts.append(len(steps))
        tools = [str((step.get("tool_call") or {}).get("tool")) for step in steps]
        tool_diversity.append(len(set(tools)))
        for tool in tools:
            tool_calls[tool] += 1

        described_tables = {
            table
            for step in steps
            if (step.get("tool_call") or {}).get("tool") == "describe_table"
            for table in ((step.get("tool_call") or {}).get("arguments") or {}).get("tables", [])
        }
        described_table_counts.append(len(described_tables))
        if described_tables:
            counters["episodes_with_describe_table"] += 1
        if len(described_tables) > 1:
            counters["episodes_describing_multiple_tables"] += 1
        if "inspect_column" in tools:
            counters["episodes_with_inspect_column"] += 1
        if "read_subtable" in tools:
            counters["episodes_with_read_subtable"] += 1
        if "join_tables" in tools:
            counters["episodes_with_join"] += 1
        if "plan" in tools:
            counters["episodes_with_plan"] += 1

        generation = trajectory.get("rollout_generation") or {}
        if generation.get("error_events"):
            counters["episodes_with_error_feedback"] += 1
        if any(step.get("feedback_recovery") for step in steps):
            counters["episodes_with_feedback_recovery"] += 1
        if any((item.get("features") or {}).get("empty_result") for item in scored_steps):
            counters["episodes_with_empty_result"] += 1
        if any((item.get("features") or {}).get("action_changed_after_empty") for item in scored_steps):
            counters["episodes_adjusting_after_empty"] += 1

        back_slice = set(score.get("back_slice_step_ids") or [])
        off_slice = [
            step for step in steps
            if step.get("step_id") not in back_slice
            and (step.get("tool_call") or {}).get("tool") != "answer_from_context"
        ]
        if off_slice:
            counters["episodes_with_off_slice_actions"] += 1
        off_slice_perception = [
            step for step in off_slice
            if (step.get("tool_call") or {}).get("tool") in PERCEPTION_TOOLS
        ]
        if off_slice_perception:
            counters["episodes_with_off_slice_perception"] += 1

        positive = 0
        for item in scored_steps:
            tool = item.get("tool") or (item.get("features") or {}).get("error_type") or "unknown"
            reward = float(item.get("reward") or 0.0)
            if reward > 0:
                positive += 1
                positive_by_tool[tool] += 1
            elif reward < 0:
                negative_by_tool[tool] += 1
            else:
                zero_by_tool[tool] += 1
        positive_counts.append(positive)
        shares = [float(item.get("c_positive") or 0.0) for item in scored_steps]
        max_positive_shares.append(max(shares, default=0.0))
        if positive > 1:
            counters["episodes_rewarding_multiple_steps"] += 1

        references = rebuild_references(steps)
        for step in steps:
            tool = (step.get("tool_call") or {}).get("tool")
            if tool not in {"join_tables", "set_op"}:
                continue
            refs = references.get(str(step.get("step_id"))) or []
            step_parents = len({ref["step"] for ref in refs if ref.get("type") == "data" and ref.get("step")})
            source_parents = len({str(ref["source"]) for ref in refs if ref.get("type") == "data" and ref.get("source")})
            counters[f"{tool}_calls"] += 1
            counters[f"{tool}_calls_with_{step_parents}_step_data_parents"] += 1
            counters[f"{tool}_calls_with_{source_parents}_source_data_parents"] += 1
        terminal_id = str(steps[-1].get("step_id")) if steps else "terminal"
        references[terminal_id] = list(references.get(terminal_id) or []) + list(
            (score.get("diagnostics") or {}).get("final_references") or []
        )
        graph_nodes = back_slice | {terminal_id}
        edges: set[tuple[str, str, str]] = set()
        for child, refs in references.items():
            if child not in graph_nodes:
                continue
            for ref in refs:
                parent = ref.get("step")
                if isinstance(parent, str) and parent in graph_nodes and parent != child:
                    edges.add((parent, child, str(ref.get("type") or "data")))
        indegree = collections.Counter(child for _, child, _ in edges)
        outdegree = collections.Counter(parent for parent, _, _ in edges)
        merge_nodes = [node for node, degree in indegree.items() if degree > 1]
        fork_nodes = [node for node, degree in outdegree.items() if degree > 1]
        incoming_types = {
            node: collections.Counter(edge_type for _, child, edge_type in edges if child == node)
            for node in merge_nodes
        }
        if any(types["data"] >= 2 for types in incoming_types.values()):
            counters["episodes_with_multi_data_parent_merge"] += 1
        if any(types["data"] and types["value"] for types in incoming_types.values()):
            counters["episodes_with_data_value_merge"] += 1
        if any(types["data"] and types["grounding"] for types in incoming_types.values()):
            counters["episodes_with_data_grounding_merge"] += 1
        if merge_nodes:
            counters["episodes_with_reward_graph_merge"] += 1
        if fork_nodes:
            counters["episodes_with_reward_graph_fork"] += 1
        if merge_nodes or fork_nodes:
            counters["episodes_with_non_chain_reward_graph"] += 1
            if len(topology_examples) < 10:
                topology_examples.append({
                    "trajectory_id": trajectory.get("trajectory_id"),
                    "difficulty": trajectory.get("difficulty"),
                    "merge_nodes": merge_nodes,
                    "fork_nodes": fork_nodes,
                    "merge_incoming_types": {
                        node: dict(sorted(incoming_types[node].items())) for node in merge_nodes
                    },
                    "edges": sorted(edges),
                })
        if len((score.get("diagnostics") or {}).get("grounding_handles") or []) > 1:
            counters["episodes_with_multi_handle_final_grounding"] += 1

    count = len(episodes)
    report = {
        "sources": sources,
        "eligible_episodes": count,
        "exploration": {
            key: {"episodes": counters[key], "rate": counters[key] / count if count else 0.0}
            for key in (
                "episodes_with_describe_table", "episodes_describing_multiple_tables",
                "episodes_with_inspect_column", "episodes_with_read_subtable", "episodes_with_join",
                "episodes_with_plan", "episodes_with_error_feedback", "episodes_with_feedback_recovery",
                "episodes_with_empty_result", "episodes_adjusting_after_empty",
                "episodes_with_off_slice_actions", "episodes_with_off_slice_perception",
            )
        },
        "trajectory_shape": {
            "steps_p50": statistics.median(step_counts) if step_counts else 0,
            "steps_p90": percentile([float(value) for value in step_counts], 0.9),
            "described_tables_p50": statistics.median(described_table_counts) if described_table_counts else 0,
            "tool_diversity_p50": statistics.median(tool_diversity) if tool_diversity else 0,
            "tool_calls": dict(sorted(tool_calls.items())),
        },
        "reward_distribution": {
            "episodes_rewarding_multiple_steps": counters["episodes_rewarding_multiple_steps"],
            "rate_rewarding_multiple_steps": counters["episodes_rewarding_multiple_steps"] / count if count else 0.0,
            "positive_steps_p50": statistics.median(positive_counts) if positive_counts else 0,
            "positive_steps_p90": percentile([float(value) for value in positive_counts], 0.9),
            "max_positive_share_p50": statistics.median(max_positive_shares) if max_positive_shares else 0.0,
            "max_positive_share_p90": percentile(max_positive_shares, 0.9),
            "positive_step_count_by_tool": dict(sorted(positive_by_tool.items())),
            "zero_step_count_by_tool": dict(sorted(zero_by_tool.items())),
            "negative_step_count_by_tool": dict(sorted(negative_by_tool.items())),
        },
        "reward_graph_topology": {
            key: {"episodes": counters[key], "rate": counters[key] / count if count else 0.0}
            for key in (
                "episodes_with_reward_graph_merge", "episodes_with_reward_graph_fork",
                "episodes_with_non_chain_reward_graph", "episodes_with_multi_data_parent_merge",
                "episodes_with_data_value_merge", "episodes_with_data_grounding_merge",
                "episodes_with_multi_handle_final_grounding",
            )
        },
        "operator_reference_edges": {
            key: value for key, value in sorted(counters.items())
            if key.startswith("join_tables_") or key.startswith("set_op_")
        },
        "non_chain_examples": topology_examples,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
