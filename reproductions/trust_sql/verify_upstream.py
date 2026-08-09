#!/usr/bin/env python3
"""Verify the pinned TRUST-SQL source snapshot and its executable reward core."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any


PINNED_COMMIT = "89df0661ad6b8e29ed8e61f7c950fbc2c1678b08"
EXPECTED_HASHES = {
    "data_for_sql/filtered_questions.jsonl": "af68f0d0d1d89fd9d4f40e5f5f028131fa0ca55a71675f2598cbbba26c9ab414",
    "examples/nl2sql/generate_sql_token.py": "da6f84de7d7a2cf80ea0dd3c81151a22ea192f6b383a3b00aad5f93dffded699",
    "examples/nl2sql/sql_reward_with_schema.py": "939ac3ac7cbca96ba1160e9b54451585841c781bdce451d5927ec0055a3f6daf",
    "trustsql_eval/prompt_template.txt": "31e7031dcd0c0313b698ec0e742fc1666284179115e10b41838d5a9386cc41e1",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("trustsql_reward_snapshot", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_markers(text: str, markers: list[str], source: str) -> None:
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"{source} is missing pinned semantic markers: {missing}")


async def reward_smoke(reward: ModuleType) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="trustsql_reward_") as temporary:
        root = Path(temporary)
        db_id = "tiny"
        db_dir = root / db_id
        db_dir.mkdir()
        db_path = db_dir / f"{db_id}.sqlite"
        connection = sqlite3.connect(db_path)
        try:
            connection.executescript(
                "CREATE TABLE items(id INTEGER PRIMARY KEY, value TEXT);"
                "INSERT INTO items(value) VALUES ('alpha'),('beta');"
            )
            connection.commit()
        finally:
            connection.close()

        label = {
            "data_source": db_id,
            "ground_truth": {
                "target": ["SELECT value FROM items ORDER BY id"],
                "schema": {
                    "tables": ["items"],
                    "columns": {"items": ["id", "value"]},
                    "joins": [],
                },
            },
        }
        correct = await reward.compute_score_sql_async(
            solution_str="<answer>SELECT value FROM items ORDER BY id</answer>",
            ground_truth=label,
            db_root_path=str(root),
            timeout=2.0,
        )
        executable_wrong = await reward.compute_score_sql_async(
            solution_str="<answer>SELECT id FROM items ORDER BY id</answer>",
            ground_truth=label,
            db_root_path=str(root),
            timeout=2.0,
        )
        invalid = await reward.compute_score_sql_async(
            solution_str="<answer>SELECT missing FROM items</answer>",
            ground_truth=label,
            db_root_path=str(root),
            timeout=2.0,
        )

    if (correct, executable_wrong, invalid) != (1.0, 0.2, 0.0):
        raise AssertionError(
            "released execution reward changed: "
            f"{(correct, executable_wrong, invalid)}"
        )

    proposed = {
        "tables": ["ITEMS"],
        "columns": {"ITEMS": ["VALUE", "ID"]},
        "joins": ["ignored.by.reward"],
    }
    schema_scores = reward.compute_schema_linking_score(
        proposed,
        label["ground_truth"]["schema"],
        mode="truematch",
    )
    if schema_scores["total_score"] != 1.0:
        raise AssertionError(f"released exact schema reward changed: {schema_scores}")

    turns = {
        "explore_schema": (
            "<think>inspect</think><action>explore_schema</action>"
            '<tool_call>{"name":"execute_sql_query","arguments":'
            '{"db_id":"tiny","sql":"PRAGMA table_info(items)"}}</tool_call>'
        ),
        "propose_schema": (
            "<think>commit</think><action>propose_schema</action>"
            '<schema>{"tables":["items"],"columns":{"items":["id","value"]}}</schema>'
        ),
        "generate_sql": (
            "<think>verify</think><action>generate_sql</action>"
            '<tool_call>{"name":"execute_sql_query","arguments":'
            '{"db_id":"tiny","sql":"SELECT value FROM items ORDER BY id"}}</tool_call>'
        ),
        "confirm_answer": (
            "<think>done</think><action>confirm_answer</action>"
            "<answer>SELECT value FROM items ORDER BY id</answer>"
        ),
    }
    format_scores = {
        phase: reward.check_single_turn_format(text)
        for phase, text in turns.items()
    }
    if not all(result["is_valid"] for result in format_scores.values()):
        raise AssertionError(f"released format contract rejected valid turns: {format_scores}")

    return {
        "execution_reward": {
            "correct": correct,
            "executable_wrong": executable_wrong,
            "invalid": invalid,
        },
        "schema_reward": schema_scores,
        "valid_action_formats": sorted(turns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    upstream = args.upstream.resolve()
    head = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != PINNED_COMMIT:
        raise AssertionError(f"upstream HEAD {head} != pinned {PINNED_COMMIT}")

    observed_hashes = {
        relative: sha256_file(upstream / relative)
        for relative in EXPECTED_HASHES
    }
    if observed_hashes != EXPECTED_HASHES:
        raise AssertionError(
            f"pinned upstream file hashes changed: {observed_hashes}"
        )

    rollout_path = upstream / "examples/nl2sql/generate_sql_token.py"
    rollout_text = rollout_path.read_text(encoding="utf-8")
    require_markers(
        rollout_text,
        [
            '"max_turns": 10',
            'action == "explore_schema"',
            'action == "schema"',
            "token_rewards[response_length - 1] = sql_reward_total",
            "if sql_execution_score == 1.0:",
            "token_rewards[schema_end_position - 1] = schema_score * schema_weight",
        ],
        str(rollout_path),
    )
    loss_path = upstream / "slime/backends/megatron_utils/loss.py"
    loss_text = loss_path.read_text(encoding="utf-8")
    require_markers(
        loss_text,
        [
            "def get_grpo_returns_schema_answer_separate(",
            "schema_adv = _compute_group_advantages",
            "answer_adv = _compute_group_advantages",
            'getattr(args, "use_weighted_schema_policy_loss", False)',
            "schema_loss_weight * pg_loss_schema + answer_loss_weight * pg_loss_answer",
        ],
        str(loss_path),
    )
    launcher_path = upstream / "submit_training.sh"
    launcher_text = launcher_path.read_text(encoding="utf-8")
    require_markers(
        launcher_text,
        [
            "N_SAMPLES_PER_PROMPT=8",
            "ROLLOUT_BATCH_SIZE=32",
            "ROLLOUT_TEMPERATURE=0.8",
            "SCHEMA_WEIGHT=0.25",
            'SCHEMA_SCORING_MODE="truematch"',
        ],
        str(launcher_path),
    )

    reward_module = load_module(
        upstream / "examples/nl2sql/sql_reward_with_schema.py"
    )
    smoke = asyncio.run(reward_smoke(reward_module))
    reward_module.cleanup()
    missing_runtime_helpers = [
        relative
        for relative in ("utils/export_env_slime.sh", "utils/low_gpu_utilization.py")
        if not (upstream / relative).exists()
    ]
    audit = {
        "status": "passed_with_reproduction_gaps",
        "upstream": "https://github.com/JaneEyre0530/TrustSQL",
        "commit": head,
        "file_sha256": observed_hashes,
        "reward_and_protocol_smoke": smoke,
        "dual_track_static_contract": {
            "group_size": 8,
            "separate_schema_and_full_advantages": True,
            "schema_loss_weight": 0.25,
            "sparse_coupled_schema_reward": True,
        },
        "release_vs_paper_differences": [
            "paper Table 8 specifies three RL epochs; submit_training.sh sets NUM_EPISODES=2",
            "paper Table 8 specifies 8B LR=8e-7; submit_training.sh sets LR_ACTOR=1e-6",
            "paper says a 30-row tool result cap; rollout displays up to 100 while reward compares at most 30",
            "released RL prompt contains schema examples absent from the Appendix C prompt",
        ],
        "missing_release_artifacts": [
            "the 9.2k SFT trajectories / 70,970 expanded SFT turns",
            "the SFT warm-up checkpoints",
            "the final TRUST-SQL-4B and TRUST-SQL-8B checkpoints",
            "a locked paper-run dependency image or tag",
            *missing_runtime_helpers,
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
