#!/usr/bin/env python3
"""Evaluate the relational-program scheme in the real causal table harness.

DeepSeek sees only the current model-visible prefix. Hidden gold SQL remains inside the evaluator
for terminal ``bird-set`` scoring and is never placed in a provider request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from denotation import add_denotation_comparison_argument  # noqa: E402
from evaluate_batch_plan import (  # noqa: E402
    DEFAULT_MAX_ERRORS_PER_TYPE,
    DEFAULT_MAX_TOKENS,
    read_completed,
    run_episode,
    summarize_records,
)
from generate_teacher_rollouts import (  # noqa: E402
    append_jsonl,
    load_examples,
    trajectory_id,
    write_manifest,
)
from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    is_deepseek_split_model,
    provider_request_options,
)
from provider_client import load_api_config  # noqa: E402
from protocol import (  # noqa: E402
    PROTOCOL_VERSION as UNDERLYING_ATOMIC_PROTOCOL_VERSION,
    tool_schema_hash,
)
from relational_program_protocol import (  # noqa: E402
    ASSISTANT_CARRIERS,
    INLINE_THINK_CARRIER,
    MAX_RELATIONAL_PROGRAM_CALLS,
    PROVIDER_NATIVE_CARRIER,
    RELATIONAL_PROGRAM_PROTOCOL_VERSION,
    TYPED_REFERENCE_SCHEMA,
    build_relational_program_messages,
    build_relational_program_system_prompt,
    parse_relational_program_action,
    parse_relational_program_assistant,
    prepare_relational_work_action,
    relational_program_protocol_hash,
    render_relational_program_observation,
    validate_relational_program_call,
)
from rollout import TERMINAL_ANSWER_CONTRACT  # noqa: E402
from tool_schemes import (  # noqa: E402
    RELATIONAL_PROGRAM_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
)


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_WORK_TURNS = 30
DEFAULT_HISTORY_TURNS = 4
TRAINING_ADMISSION = "diagnostic_only_pending_protocol_scale_gate"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a causal relational-program diagnostic. Perception stays interactive; "
            "the harness derives and executes each program DAG."
        )
    )
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--examples-file", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--assistant-carrier",
        choices=ASSISTANT_CARRIERS,
        default=PROVIDER_NATIVE_CARRIER,
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-work-turns",
        type=int,
        default=DEFAULT_MAX_WORK_TURNS,
    )
    parser.add_argument(
        "--max-program-calls",
        type=int,
        default=MAX_RELATIONAL_PROGRAM_CALLS,
    )
    parser.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY_TURNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument(
        "--max-errors-per-type",
        type=int,
        default=DEFAULT_MAX_ERRORS_PER_TYPE,
    )
    parser.add_argument("--table-output-rows", type=int, default=0)
    parser.add_argument(
        "--trajectory-id",
        action="append",
        default=[],
        help="run only this exact frozen trajectory id; repeat as needed",
    )
    add_denotation_comparison_argument(parser)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if (
        args.assistant_carrier == PROVIDER_NATIVE_CARRIER
        and not is_deepseek_split_model(args.model)
    ):
        parser.error(
            "provider-native carrier requires deepseek-v4-flash or deepseek-v4-pro"
        )
    if (
        args.assistant_carrier == INLINE_THINK_CARRIER
        and is_deepseek_split_model(args.model)
    ):
        parser.error(
            "DeepSeek split-response models require the provider-native carrier"
        )
    if args.denotation_comparison != "bird-set":
        parser.error("new BIRD evaluations must use --denotation-comparison bird-set")
    if args.max_work_turns < 2:
        parser.error("--max-work-turns must be at least 2")
    if not 1 <= args.max_program_calls <= MAX_RELATIONAL_PROGRAM_CALLS:
        parser.error(
            f"--max-program-calls must be in 1..{MAX_RELATIONAL_PROGRAM_CALLS}"
        )
    if args.history_turns != DEFAULT_HISTORY_TURNS:
        parser.error(
            f"{RELATIONAL_PROGRAM_PROTOCOL_VERSION} requires --history-turns 4"
        )

    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    if out_path.exists() and not args.resume:
        parser.error(f"{out_path} already exists; use a new path or --resume")

    requested_ids = list(dict.fromkeys(args.trajectory_id))
    if requested_ids:
        load_args = deepcopy(args)
        load_args.start = 0
        load_args.limit = None
        examples = load_examples(load_args)
        available = {
            trajectory_id(args.split, index, ex)
            for index, ex in examples
        }
        missing = sorted(set(requested_ids) - available)
        if missing:
            parser.error(
                "requested trajectory ids are absent: " + ", ".join(missing)
            )
        examples = [
            (index, ex)
            for index, ex in examples
            if trajectory_id(args.split, index, ex) in set(requested_ids)
        ]
    else:
        examples = load_examples(args)

    completed = read_completed(out_path) if args.resume else set()
    work = [
        (index, ex)
        for index, ex in examples
        if trajectory_id(args.split, index, ex) not in completed
    ]
    system_prompt = build_relational_program_system_prompt(
        args.max_program_calls,
        assistant_carrier=args.assistant_carrier,
    )
    protocol_hash = relational_program_protocol_hash(
        system_prompt,
        args.max_program_calls,
    )
    started = time.time()

    def process(item: tuple[int, dict]) -> dict:
        index, ex = item
        try:
            return run_episode(
                example_index=index,
                ex=ex,
                split=args.split,
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                system_prompt=system_prompt,
                protocol_hash=protocol_hash,
                max_action_blocks=args.max_work_turns,
                max_batch_calls=args.max_program_calls,
                max_tokens=args.max_tokens,
                api_retries=args.api_retries,
                api_timeout=args.api_timeout,
                max_errors_per_type=args.max_errors_per_type,
                table_output_rows=args.table_output_rows,
                history_turns=args.history_turns,
                denotation_comparison=args.denotation_comparison,
                protocol_version=RELATIONAL_PROGRAM_PROTOCOL_VERSION,
                assistant_carrier=args.assistant_carrier,
                tool_scheme=RELATIONAL_PROGRAM_TOOL_SCHEME,
                build_messages=build_relational_program_messages,
                parse_provider_action=parse_relational_program_action,
                parse_inline_action=parse_relational_program_assistant,
                prepare_work_action=lambda tool, arguments: (
                    prepare_relational_work_action(
                        tool,
                        arguments,
                        max_program_calls=args.max_program_calls,
                    )
                ),
                render_work_observation=render_relational_program_observation,
                validate_call=validate_relational_program_call,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "tool_scheme": RELATIONAL_PROGRAM_TOOL_SCHEME,
                "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
                "assistant_carrier": args.assistant_carrier,
                "example_index": index,
                "trajectory_id": trajectory_id(args.split, index, ex),
                "db_id": ex.get("db_id"),
                "question": ex.get("question"),
                "correct": False,
                "legal": False,
                "model_turns": 0,
                "work_actions": 0,
                "atomic_actions": 0,
                "submitted_calls": 0,
                "blocked_nodes": 0,
                "errors": 0,
                "failure_type": "runner_error",
                "fail": f"{type(exc).__name__}: {exc}",
                "turns": [],
                "atomic_events": [],
                "error_events": [],
                "usage": {},
                "denotation_comparison": args.denotation_comparison,
                "terminal_answer_contract": TERMINAL_ANSWER_CONTRACT,
                "protocol_version": RELATIONAL_PROGRAM_PROTOCOL_VERSION,
                "typed_reference_schema": TYPED_REFERENCE_SCHEMA,
                "underlying_atomic_protocol_version": (
                    UNDERLYING_ATOMIC_PROTOCOL_VERSION
                ),
                "underlying_atomic_tool_schema_sha256": tool_schema_hash(),
                "protocol_hash": protocol_hash,
                "sft_export_eligible": False,
                "training_admission": TRAINING_ADMISSION,
            }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process, item) for item in work]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            record["training_admission"] = TRAINING_ADMISSION
            append_jsonl(out_path, record)
            status = "OK " if record.get("correct") else "ERR"
            print(
                f"[{done}/{len(work)}] {status} turns={record.get('model_turns')} "
                f"work={record.get('work_actions')} atomic={record.get('atomic_actions')} "
                f"blocked={record.get('blocked_nodes')} errors={record.get('errors')} "
                f"type={record.get('failure_type')} {record.get('trajectory_id')}",
                flush=True,
            )

    summary = summarize_records(out_path)
    manifest = {
        "generator": "src/eval/evaluate_relational_program.py",
        "method": (
            "interactive_perception_plus_typed_references_plus_"
            "harness_derived_relational_program_dag"
        ),
        "model_role": "external_teacher_diagnostic",
        "model": args.model,
        "tool_scheme": RELATIONAL_PROGRAM_TOOL_SCHEME,
        "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
        "assistant_carrier": args.assistant_carrier,
        "protocol_version": RELATIONAL_PROGRAM_PROTOCOL_VERSION,
        "typed_reference_schema": TYPED_REFERENCE_SCHEMA,
        "underlying_atomic_protocol_version": (
            UNDERLYING_ATOMIC_PROTOCOL_VERSION
        ),
        "underlying_atomic_tool_schema_sha256": tool_schema_hash(),
        "protocol_hash": protocol_hash,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "system_prompt_characters": len(system_prompt),
        "system_prompt": system_prompt,
        "split": args.split,
        "examples_file": args.examples_file,
        "source_count": len(examples),
        "requested_trajectory_ids": requested_ids,
        "attempted_this_run": len(work),
        "resume": args.resume,
        "output": str(out_path),
        "max_work_turns": args.max_work_turns,
        "max_program_calls": args.max_program_calls,
        "max_errors_per_type": args.max_errors_per_type,
        "history_turns": args.history_turns,
        "max_tokens": args.max_tokens,
        "workers": max(1, args.workers),
        "table_output_rows": args.table_output_rows,
        "temperature": 0,
        "thinking": (
            "enabled"
            if args.assistant_carrier == PROVIDER_NATIVE_CARRIER
            else "inline"
        ),
        "reasoning_effort": (
            "high"
            if args.assistant_carrier == PROVIDER_NATIVE_CARRIER
            else None
        ),
        "deepseek_carrier": (
            DEEPSEEK_CARRIER_JSON_OUTPUT
            if args.assistant_carrier == PROVIDER_NATIVE_CARRIER
            else None
        ),
        "provider_request_options": provider_request_options(
            args.model,
            carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
        ),
        "denotation_comparison": args.denotation_comparison,
        "terminal_answer_contract": TERMINAL_ANSWER_CONTRACT,
        "gold_sql_visible_to_model": False,
        "sft_export_eligible": False,
        "training_admission": TRAINING_ADMISSION,
        "summary": summary,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
