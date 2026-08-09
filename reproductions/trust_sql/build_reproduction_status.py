#!/usr/bin/env python3
"""Combine TRUST-SQL source, host, and data audits into a compact status manifest."""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-audit", type=Path, required=True)
    parser.add_argument("--host-preflight", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    upstream = read_json(args.upstream_audit)
    host = read_json(args.host_preflight)
    data = read_json(args.data_audit)
    status = {
        "status": "partial_contract_reproduction_complete_paper_training_blocked",
        "updated_on": date.today().isoformat(),
        "paper": {
            "title": "TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown Schemas",
            "arxiv": "2603.16448v2",
            "paper_pdf_sha256": "f9a85b93a88cd8c48780c558f24da9c1090a3a8b1de4889df894c9d289008473",
            "bird_dev_targets_percent": {
                "qwen3_4b_greedy": 64.9,
                "qwen3_4b_majority": 67.2,
                "qwen3_8b_greedy": 65.8,
                "qwen3_8b_majority": 67.7,
            },
        },
        "source": {
            "repository": upstream["upstream"],
            "commit": upstream["commit"],
            "file_sha256": upstream["file_sha256"],
        },
        "verified_contract": {
            "actions": upstream["reward_and_protocol_smoke"]["valid_action_formats"],
            "execution_reward": upstream["reward_and_protocol_smoke"]["execution_reward"],
            "schema_reward_exact_match": upstream["reward_and_protocol_smoke"]["schema_reward"]["total_score"],
            "dual_track": upstream["dual_track_static_contract"],
        },
        "released_rl_data": {
            "source_sha256": data["source_sha256"],
            "counts": data["counts"],
            "database_coverage": data["database_coverage"],
            "actor_prompt_gold_sql_exact_match_leaks": data[
                "actor_prompt_gold_sql_exact_match_leaks"
            ],
            "schema_catalog_issue_count": data["schema_catalog_issue_count"],
            "schema_catalog_issue_summary": data.get("schema_catalog_issue_summary"),
            "gold_execution_checked": data["gold_execution_checked"],
            "gold_execution_failure_count": data["gold_execution_failure_count"],
            "gold_execution_timeout_count": data.get("gold_execution_timeout_count"),
            "gold_execution_non_timeout_failure_count": data.get(
                "gold_execution_non_timeout_failure_count"
            ),
            "gold_execution_failure_summary": data.get("gold_execution_failure_summary"),
            "pass_rate_filter_audit": data["pass_rate_filter_audit"],
        },
        "table_rl": {
            "host": host["host"],
            "gpus": host["gpus"],
            "packages": host["packages"],
            "paper_scale_ready": host["paper_scale_ready"],
            "allowed_scope": host["allowed_scope"],
        },
        "paper_training": {
            "started": False,
            "qwen3_4b": {
                "sft": "16xA100, batch 256, 2 epochs, lr 1e-5",
                "rl": "8xA100, batch 32 prompts, K=8, 3 epochs, lr 1e-6, lambda=0.25, max_turns=10",
            },
            "qwen3_8b": {
                "sft": "16xA100, batch 256, 2 epochs, lr 1.5e-6",
                "rl": "32xA100, batch 32 prompts, K=8, 3 epochs, lr 8e-7, lambda=0.25, max_turns=10",
            },
            "blocking_reasons": [
                "the paper SFT trajectories and warm-up checkpoints are not released",
                "the final 4B/8B checkpoints are not released",
                "table_rl has 2xRTX3090 24GiB rather than the paper's 8-32 A100 GPUs",
                "the pinned release lacks a complete locked Slime/SGLang/Megatron runtime",
                "the available BIRD/Spider release produces stable schema/SQL errors plus load-sensitive SQL timeouts",
            ],
        },
        "release_differences": upstream["release_vs_paper_differences"],
        "missing_release_artifacts": upstream["missing_release_artifacts"],
        "interpretation": (
            "Protocol, reward, pinned-source, and released-data audits are reproducible. "
            "Headline model accuracy and end-to-end paper training are not reproducible on this host "
            "from the currently public artifacts."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
