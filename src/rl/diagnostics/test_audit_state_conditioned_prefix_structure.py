from __future__ import annotations

import json

from src.rl.diagnostics.audit_state_conditioned_prefix_structure import (
    audit_rows,
    load_group_dirs,
)


def _assistant(tool: str, arguments: dict) -> dict:
    action = json.dumps(
        {"tool": tool, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {"role": "assistant", "content": f"<think>x</think>{action}"}


def _row(
    sample_index: int,
    *,
    correct: bool,
    first_tables: list[str],
    second_tool: str,
    second_prompt: list[int],
) -> dict:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "question"},
        _assistant("describe_table", {"tables": first_tables}),
        {"role": "user", "content": '{"status":"success","step_id":"step_1"}'},
        _assistant(second_tool, {"table": "t"}),
        {"role": "user", "content": '{"status":"success","step_id":"step_2"}'},
    ]
    return {
        "environment": {
            "dataset_split": "train",
            "task_id": "bird_train_00001",
        },
        "sample": {
            "correct": correct,
            "process_update": True,
            "audit_record": {
                "sample_index": sample_index,
                "protocol_version": "version26",
                "protocol_hash": "hash",
                "final_messages": messages,
            },
        },
        "policy_turns": [
            {"prompt_ids": [1, 2, 3]},
            {"prompt_ids": second_prompt},
        ],
    }


def test_full_arguments_distinguish_actions_and_exact_state_anchor() -> None:
    rows = [
        _row(
            0,
            correct=True,
            first_tables=["A"],
            second_tool="group_aggregate",
            second_prompt=[4, 5],
        ),
        _row(
            1,
            correct=False,
            first_tables=["A"],
            second_tool="project",
            second_prompt=[4, 5],
        ),
        _row(
            2,
            correct=False,
            first_tables=["B"],
            second_tool="project",
            second_prompt=[8, 9],
        ),
    ]
    result = audit_rows(rows)
    first = result["first_action"]
    assert first["unanimous_first_tool_tasks"] == 1
    assert first["unanimous_schema_normalized_first_action_tasks"] == 0
    assert first["same_tool_but_different_arguments_tasks"] == 1

    exact = result["state_anchors"]["exact_environment_prefix"]
    assert exact["counts"]["noninitial_informative"] == 1
    assert exact["task_counts"]["noninitial_informative"] == 1

    strict = result["state_anchors"]["strict_policy_prompt"]
    assert strict["counts"]["noninitial_informative"] == 1


def test_describe_table_order_is_normalized_but_authored_identity_is_retained() -> None:
    rows = [
        _row(
            0,
            correct=True,
            first_tables=["A", "B"],
            second_tool="project",
            second_prompt=[4],
        ),
        _row(
            1,
            correct=False,
            first_tables=["B", "A"],
            second_tool="project",
            second_prompt=[5],
        ),
    ]
    result = audit_rows(rows)
    first = result["first_action"]
    assert first["unanimous_schema_normalized_first_action_tasks"] == 1
    assert first["unanimous_authored_first_action_tasks"] == 0
    prefix = result["common_prefix"]
    assert prefix["tasks_with_schema_normalized_action_lcp_at_least_1"] == 1


def test_group_loader_selects_only_clean_mixed_k8(tmp_path) -> None:
    group_dir = tmp_path / "groups"
    group_dir.mkdir()
    mixed = [
        _row(
            index,
            correct=index < 4,
            first_tables=["A"],
            second_tool="project",
            second_prompt=[4, index],
        )
        for index in range(8)
    ]
    homogeneous = [
        _row(
            index,
            correct=True,
            first_tables=["A"],
            second_tool="project",
            second_prompt=[5, index],
        )
        for index in range(8)
    ]
    for row in homogeneous:
        row["environment"]["task_id"] = "bird_train_00002"
    (group_dir / "bird_train_00001.json").write_text(json.dumps(mixed))
    (group_dir / "bird_train_00002.json").write_text(json.dumps(homogeneous))

    rows, summary = load_group_dirs(
        [group_dir],
        only_clean_mixed_groups=True,
    )
    assert len(rows) == 8
    assert summary["selected_groups"] == 1
    assert summary["rejected_homogeneous"] == 1


def test_recovered_malformed_carrier_excludes_whole_episode() -> None:
    valid = _row(
        0,
        correct=True,
        first_tables=["A"],
        second_tool="project",
        second_prompt=[4],
    )
    malformed = _row(
        1,
        correct=False,
        first_tables=["A"],
        second_tool="project",
        second_prompt=[4],
    )
    malformed["sample"]["audit_record"]["final_messages"][2]["content"] = "not an action"

    result = audit_rows([valid, malformed])
    assert result["observed"]["rows"] == 2
    assert result["observed"]["parsed_episodes"] == 1
    assert result["observed"]["eligible_episodes"] == 1
    assert result["observed"]["issues"] == 1


def test_leave_one_out_repeated_actions_can_improve_over_state_mean() -> None:
    rows = []
    for index in range(6):
        rows.append(
            _row(
                index,
                correct=index < 3,
                first_tables=["A"],
                second_tool=("project" if index < 3 else "inspect_column"),
                second_prompt=[4, 5],
            )
        )
    result = audit_rows(rows)
    validation = result["predictive_validation"]
    assert validation["eligible_states"] == 1
    assert validation["prediction_cases"] == 6
    assert validation["losses"]["action_minus_state_brier"] < 0
    assert validation["losses"]["action_minus_state_log_loss"] < 0


def test_flat_training_rollout_adapter_uses_exact_model_input_state() -> None:
    canonical = _row(
        3,
        correct=True,
        first_tables=["A"],
        second_tool="project",
        second_prompt=[4],
    )
    messages = canonical["sample"]["audit_record"]["final_messages"]
    flat = {
        "example_index": 1,
        "trajectory_id": "rl_1_sample_3",
        "correct": True,
        "protocol_version": "version26",
        "protocol_hash": "hash",
        "result_reward": {"executable_terminal": True},
        "final_messages": messages,
        "turns": [
            {"model_input": [{"role": "user", "content": "first"}]},
            {"model_input": [{"role": "user", "content": "second"}]},
        ],
        "gold_sql": "must not be read",
    }
    result = audit_rows([flat])
    assert result["observed"]["eligible_episodes"] == 1
    assert result["observed"]["eligible_decision_events"] == 2
    assert result["identity"]["dataset_split_counts"] == {"train": 1}
