#!/usr/bin/env python3
"""Build and audit an offline Harness-conditioned SMDP/IQL dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from rl.frameworks.trl.iql import IQLConfig  # noqa: E402
from rl.frameworks.trl import iql as iql_module  # noqa: E402
from rl.frameworks.trl import mdp_critic as mdp_critic_module  # noqa: E402
from rl.runtime import terminal_reward as terminal_reward_module  # noqa: E402
from rl.frameworks.trl.mdp_critic import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    audit_transitions,
    build_transitions,
    load_jsonl,
    write_jsonl,
)


EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-checkpoint", default="checkpoint-6380")
    parser.add_argument("--expected-protocol-version", default="version26")
    parser.add_argument("--expected-protocol-hash", default="4da19387399bd3a5")
    parser.add_argument(
        "--student-prompt-sha256",
        default=EXPECTED_STUDENT_PROMPT_SHA256,
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile-tau", type=float, default=0.7)
    parser.add_argument("--inverse-temperature", type=float, default=3.0)
    parser.add_argument("--max-advantage-weight", type=float, default=20.0)
    args = parser.parse_args()

    if not args.rollouts.is_file():
        raise SystemExit(f"rollout file does not exist: {args.rollouts}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "transitions": args.output_dir / "transitions.jsonl",
        "issues": args.output_dir / "issues.jsonl",
        "audit": args.output_dir / "audit.json",
        "manifest": args.output_dir / "manifest.json",
    }
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing artifacts: {existing}")

    config = IQLConfig(
        gamma=args.gamma,
        expectile_tau=args.expectile_tau,
        inverse_temperature=args.inverse_temperature,
        max_advantage_weight=args.max_advantage_weight,
    )
    config.validate()
    rows = load_jsonl(args.rollouts)
    if not rows:
        raise SystemExit("rollout file is empty")
    protocol_pairs = {
        (row.get("protocol_version"), row.get("protocol_hash"))
        for row in rows
    }
    expected_pair = (args.expected_protocol_version, args.expected_protocol_hash)
    if protocol_pairs != {expected_pair}:
        raise SystemExit(
            "rollouts do not have one expected protocol identity: "
            f"{sorted(protocol_pairs)!r}"
        )

    source_sha256 = sha256_file(args.rollouts)
    transitions, issues = build_transitions(rows)
    audit = audit_transitions(transitions, issues)
    audit["source"] = {
        "rollouts": str(args.rollouts),
        "rollout_sha256": source_sha256,
        "rows": len(rows),
        "valid_trajectory_rows": len({transition.episode_key for transition in transitions}),
        "excluded_trajectory_rows": len(issues),
    }
    manifest: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "implementation": "harness-conditioned-smdp-iql-diagnostic",
        "implementation_git_commit": git_commit(),
        "implementation_files_sha256": {
            str(Path(module.__file__).resolve()): sha256_file(Path(module.__file__).resolve())
            for module in (iql_module, mdp_critic_module, terminal_reward_module)
        },
        "base_checkpoint": args.base_checkpoint,
        "protocol_version": args.expected_protocol_version,
        "protocol_hash": args.expected_protocol_hash,
        "student_prompt_sha256": args.student_prompt_sha256,
        "reward": {
            "source": "Atomic Harness result_reward",
            "profile": "four-level",
            "nonterminal_reward": 0.0,
            "semantic_step_duration": 1,
        },
        "iql": {
            "gamma": config.gamma,
            "expectile_tau": config.expectile_tau,
            "inverse_temperature": config.inverse_temperature,
            "max_advantage_weight": config.max_advantage_weight,
        },
        "forbidden_reference_fields": [
            "gold_sql",
            "gold_sample",
            "reference_answer",
            "reference_sql",
        ],
        "source_sha256": source_sha256,
        "transition_schema_version": "atomic-v26-harness-smdp-transition-v1",
        "actor_update": False,
        "admission_status": "diagnostic_only",
    }
    write_jsonl(
        outputs["transitions"],
        (transition.to_dict() for transition in transitions),
    )
    write_jsonl(outputs["issues"], issues)
    outputs["audit"].write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"manifest": str(outputs["manifest"]), **audit["observed"], **audit["gate"]},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
