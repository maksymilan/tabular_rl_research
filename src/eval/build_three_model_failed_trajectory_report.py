#!/usr/bin/env python3
"""Build a complete three-checkpoint qualitative audit of failed tool trajectories.

The input artifacts are matched BIRD pass@K runs.  Every (example_index, sample_index)
position for which at least one checkpoint failed is rendered side by side.  The report preserves
model-authored reasoning, parsed calls, full recorded tool outputs, and errors.  A deterministic
replay adds a hierarchy of relaxed *output-contract* diagnostics without treating model prose as
factual authority.

This audit deliberately distinguishes:

* strict BIRD correctness;
* a correct executed table hidden by carrier/terminal-citation failure;
* column-order or extra-helper-column output-shape failures;
* incomplete but projectable result tables;
* genuinely different denotations; and
* trajectories that never produced enough executable state to judge.

The three sample positions are independently sampled.  Positional alignment is useful for reading
the artifacts but is not a shared-randomness causal pairing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))
sys.path.insert(0, str(ROOT / "src" / "rl"))

from denotation import bird_rows_equal  # noqa: E402
from executor import Harness  # noqa: E402
from process_credit import remap_replay_handles  # noqa: E402
from rollout import (  # noqa: E402
    _evidence_table,
    execute_tool,
    new_ctx,
    overview,
    task_db_path,
    task_gold_sql,
)


MODEL_ORDER = ("sft2", "result_only", "process")
MODEL_LABELS = {
    "sft2": "SFT2",
    "result_only": "Result-only RL",
    "process": "Process-RL",
}
FORMAT_AGNOSTIC_CORRECT = {
    "format_only_terminal_recovered",
    "correct_table_wrong_or_missing_terminal",
    "column_order_only",
    "extra_columns_projection_only",
}
CORE_LOGIC_CORRECT_OUTPUT_INCOMPLETE = {
    "projectable_table_wrong_or_missing_terminal",
}
PARTIAL_LOGIC = {
    "overbroad_result",
    "underbroad_result",
    "partial_row_overlap",
    "numeric_representation_close",
}


def json_dump(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=indent, default=str)


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def jsonl_offsets_by_example(path: Path) -> dict[int, int]:
    """Index a large JSONL without retaining the records in memory."""
    offsets = {}
    with path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            example_index = int(record["example_index"])
            if example_index in offsets:
                raise ValueError(
                    f"duplicate example_index {example_index} in {path}"
                )
            offsets[example_index] = offset
    return offsets


def read_jsonl_record_at(source: Any, offset: int) -> dict[str, Any]:
    source.seek(offset)
    line = source.readline()
    if not line:
        raise ValueError(f"missing JSONL record at byte offset {offset}")
    return json.loads(line)


def load_task_map(path: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for row in jsonl_rows(path):
        index = int(row.get("example_index", row.get("index")))
        if index in result:
            raise ValueError(f"duplicate task example_index {index}")
        result[index] = row
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows(rows: Iterable[Iterable[Any]]) -> set[tuple[Any, ...]]:
    return {tuple(row) for row in rows}


def cell_equal(left: Any, right: Any, tolerance: float = 1e-4) -> bool:
    if left == right:
        return True
    try:
        left_number = float(left)
        right_number = float(right)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()
    if not math.isfinite(left_number) or not math.isfinite(right_number):
        return left_number == right_number
    return math.isclose(left_number, right_number, rel_tol=tolerance, abs_tol=tolerance)


def rows_numeric_close(predicted: list[list[Any]], gold: list[list[Any]]) -> bool:
    if not predicted and not gold:
        return True
    if len({len(row) for row in predicted + gold}) > 1:
        return False
    remaining = list(gold)
    for predicted_row in predicted:
        match = next(
            (
                index
                for index, gold_row in enumerate(remaining)
                if len(predicted_row) == len(gold_row)
                and all(cell_equal(a, b) for a, b in zip(predicted_row, gold_row))
            ),
            None,
        )
        if match is None:
            return False
        remaining.pop(match)
    return not remaining


def column_signatures(rows: list[list[Any]], width: int) -> list[set[str]]:
    return [
        {json_dump(row[column]) for row in rows if len(row) > column}
        for column in range(width)
    ]


def find_projection(
    predicted: list[list[Any]],
    gold: list[list[Any]],
    *,
    max_combinations: int = 100_000,
) -> list[int] | None:
    if not predicted or not gold:
        return None
    predicted_widths = {len(row) for row in predicted}
    gold_widths = {len(row) for row in gold}
    if len(predicted_widths) != 1 or len(gold_widths) != 1:
        return None
    predicted_width = next(iter(predicted_widths))
    gold_width = next(iter(gold_widths))
    if predicted_width < gold_width or gold_width <= 0:
        return None

    predicted_signatures = column_signatures(predicted, predicted_width)
    gold_signatures = column_signatures(gold, gold_width)
    candidates = []
    for gold_column in range(gold_width):
        exact = [
            predicted_column
            for predicted_column in range(predicted_width)
            if predicted_signatures[predicted_column] == gold_signatures[gold_column]
        ]
        candidates.append(exact or list(range(predicted_width)))

    tried = 0
    for columns in itertools.product(*candidates):
        if len(set(columns)) != len(columns):
            continue
        tried += 1
        if tried > max_combinations:
            return None
        projected = [[row[column] for column in columns] for row in predicted]
        if bird_rows_equal(projected, gold):
            return list(columns)
    return None


def row_relation(predicted: list[list[Any]], gold: list[list[Any]]) -> dict[str, Any]:
    predicted_set = canonical_rows(predicted)
    gold_set = canonical_rows(gold)
    if not predicted_set and not gold_set:
        return {"relation": "equal_empty", "overlap": 1.0}
    if predicted_set and predicted_set < gold_set:
        return {
            "relation": "underbroad",
            "overlap": len(predicted_set & gold_set) / len(predicted_set | gold_set),
        }
    if gold_set and gold_set < predicted_set:
        return {
            "relation": "overbroad",
            "overlap": len(predicted_set & gold_set) / len(predicted_set | gold_set),
        }
    union = predicted_set | gold_set
    overlap = len(predicted_set & gold_set) / len(union) if union else 0.0
    return {"relation": "different", "overlap": overlap}


def tolerant_json_calls(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    calls = []
    for position, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("tool"), str)
            and isinstance(value.get("arguments"), dict)
        ):
            calls.append(value)
    unique = []
    seen = set()
    for call in calls:
        key = json_dump(call)
        if key not in seen:
            seen.add(key)
            unique.append(call)
    return unique


def reasoning_text(sample: dict[str, Any]) -> str:
    parts = []
    for turn in sample.get("turns") or []:
        parsed = turn.get("parsed") or {}
        think = parsed.get("think")
        if isinstance(think, str) and think.strip():
            parts.append(think.strip())
            continue
        output = str(turn.get("model_output") or "")
        match = re.search(r"<think>(.*?)(?:</think>|$)", output, flags=re.DOTALL)
        if match:
            parts.append(match.group(1).strip())
        elif output:
            parts.append(output.strip())
    return "\n".join(parts)


def text_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", text)
        if len(token) > 1
    }


def jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def tool_sequence(sample: dict[str, Any]) -> list[str]:
    result = []
    for turn in sample.get("turns") or []:
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        if tool:
            result.append(str(tool))
        elif turn.get("execution_error_type"):
            result.append(f"<{turn['execution_error_type']}>")
    return result


def action_sequence(sample: dict[str, Any]) -> list[str]:
    result = []
    for turn in sample.get("turns") or []:
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if tool and isinstance(arguments, dict):
            result.append(json_dump({"tool": tool, "arguments": arguments}))
        elif turn.get("execution_error_type"):
            result.append(f"<{turn['execution_error_type']}>")
    return result


def logic_rank(audit: dict[str, Any]) -> int:
    if audit["category"] == "strict_correct":
        return 4
    if audit["category"] in FORMAT_AGNOSTIC_CORRECT:
        return 3
    if audit["category"] in CORE_LOGIC_CORRECT_OUTPUT_INCOMPLETE:
        return 2
    if audit["category"] in PARTIAL_LOGIC:
        return 1
    return 0


def logic_status(category: str) -> str:
    if category == "strict_correct":
        return "strict_correct"
    if category in FORMAT_AGNOSTIC_CORRECT:
        return "logic_correct_ignoring_strict_output"
    if category in CORE_LOGIC_CORRECT_OUTPUT_INCOMPLETE:
        return "core_logic_correct_output_incomplete"
    if category in PARTIAL_LOGIC:
        return "partially_correct"
    if category in {"replay_error", "replay_timeout"}:
        return "indeterminate"
    return "logic_incorrect_or_unresolved"


@dataclass
class ReplayState:
    predicted: list[list[Any]] | None
    gold: list[list[Any]]
    cited_table: str | None
    created_tables: dict[str, list[list[Any]]]
    created_columns: dict[str, list[str]]
    tolerant_terminal: dict[str, Any] | None
    replay_error: str | None


def replay_sample(
    sample: dict[str, Any],
    task: dict[str, Any],
    *,
    timeout_seconds: float,
) -> ReplayState:
    harness = Harness(task_db_path(task))
    deadline = time.monotonic() + timeout_seconds

    def interrupt_expensive_replay() -> int:
        return int(time.monotonic() >= deadline)

    harness.conn.set_progress_handler(interrupt_expensive_replay, 10_000)
    context = new_ctx(overview(harness))
    created: set[str] = set()
    created_order: list[str] = []
    handle_map: dict[str, str] = {}
    terminal_arguments = None
    cited_table = None
    tolerant_terminal = None
    replay_error = None
    try:
        for position, turn in enumerate(sample.get("turns") or []):
            parsed = turn.get("parsed") or {}
            if turn.get("execution_error_type"):
                continue
            tool = parsed.get("tool")
            authored_arguments = parsed.get("arguments")
            if not tool or not isinstance(authored_arguments, dict):
                continue
            arguments = remap_replay_handles(authored_arguments, handle_map)
            if tool == "answer_from_context":
                terminal_arguments = arguments
                cited_table = _evidence_table(arguments.get("evidence"))
                break
            step_id = f"step_{int(turn.get('turn_index', position)) + 1}"
            output, table_name = execute_tool(
                harness,
                tool,
                arguments,
                context,
                step_id,
            )
            if table_name:
                created.add(table_name)
                created_order.append(table_name)
                recorded_table = (turn.get("tool_output") or {}).get("table")
                if isinstance(recorded_table, str):
                    handle_map[recorded_table] = table_name

        if terminal_arguments is None:
            last_turn = (sample.get("turns") or [None])[-1]
            if isinstance(last_turn, dict) and last_turn.get("execution_error_type") == "protocol_error":
                for call in reversed(tolerant_json_calls(str(last_turn.get("model_output") or ""))):
                    if call["tool"] != "answer_from_context":
                        continue
                    arguments = remap_replay_handles(call["arguments"], handle_map)
                    evidence = _evidence_table(arguments.get("evidence"))
                    if evidence and (evidence in created or evidence in harness.views):
                        tolerant_terminal = {
                            "call": call,
                            "replay_arguments": arguments,
                            "evidence_table": evidence,
                        }
                        terminal_arguments = arguments
                        cited_table = evidence
                        break

        gold_sql = task_gold_sql(task)
        if not gold_sql:
            raise ValueError("task has no gold SQL")
        gold = [list(row) for row in harness.gold(gold_sql)]

        predicted = None
        if cited_table and (cited_table in created or cited_table in harness.views):
            predicted = [list(row) for row in harness.rows(cited_table)]

        created_tables = {}
        created_columns = {}
        for table in created_order:
            try:
                created_tables[table] = [list(row) for row in harness.rows(table)]
                # Do not issue PRAGMA table_info against deeply nested SQLite views here.
                # SQLite may expand the full view tree while preparing that metadata query;
                # projection indices are sufficient for this audit.
                created_columns[table] = []
            except Exception:
                continue
        return ReplayState(
            predicted=predicted,
            gold=gold,
            cited_table=cited_table,
            created_tables=created_tables,
            created_columns=created_columns,
            tolerant_terminal=tolerant_terminal,
            replay_error=None,
        )
    except Exception as exc:  # noqa: BLE001 - audit records deterministic replay failure.
        replay_error = f"{type(exc).__name__}: {exc}"
        harness.conn.set_progress_handler(None, 0)
        try:
            gold = [list(row) for row in harness.gold(task_gold_sql(task))]
        except Exception:
            gold = []
        return ReplayState(
            predicted=None,
            gold=gold,
            cited_table=cited_table,
            created_tables={},
            created_columns={},
            tolerant_terminal=tolerant_terminal,
            replay_error=replay_error,
        )
    finally:
        harness.conn.set_progress_handler(None, 0)
        harness.conn.close()


def summarize_rows(rows: list[list[Any]] | None, limit: int = 10) -> dict[str, Any]:
    if rows is None:
        return {"available": False, "row_count": None, "width": None, "sample": []}
    widths = sorted({len(row) for row in rows})
    return {
        "available": True,
        "row_count": len(rows),
        "width": widths[0] if len(widths) == 1 else widths,
        "sample": rows[:limit],
        "sha256": hashlib.sha256(json_dump(sorted(map(json_dump, rows))).encode()).hexdigest(),
    }


def audit_failed_sample(
    sample: dict[str, Any],
    task: dict[str, Any],
    *,
    replay_timeout_seconds: float,
) -> dict[str, Any]:
    if sample.get("correct"):
        return {
            "category": "strict_correct",
            "logic_status": "strict_correct",
            "confidence": "deterministic",
            "strict_correct": True,
            "failure_type": None,
            "replay": None,
        }

    state = replay_sample(
        sample,
        task,
        timeout_seconds=replay_timeout_seconds,
    )
    base = {
        "strict_correct": False,
        "failure_type": sample.get("failure_type"),
        "recorded_legal": bool(sample.get("legal")),
        "recorded_errors": sample.get("errors"),
        "cited_table": state.cited_table,
        "tolerant_terminal": state.tolerant_terminal,
        "replay_error": state.replay_error,
        "predicted": summarize_rows(state.predicted),
        "gold": summarize_rows(state.gold),
    }
    if state.replay_error:
        category = (
            "replay_timeout"
            if "interrupted" in state.replay_error.casefold()
            else "replay_error"
        )
        return {
            **base,
            "category": category,
            "logic_status": logic_status(category),
            "confidence": "indeterminate",
        }

    if state.predicted is not None and bird_rows_equal(state.predicted, state.gold):
        category = (
            "format_only_terminal_recovered"
            if state.tolerant_terminal
            else "correct_table_wrong_or_missing_terminal"
        )
        return {
            **base,
            "category": category,
            "logic_status": logic_status(category),
            "confidence": "deterministic",
        }

    exact_created = [
        table
        for table, rows in state.created_tables.items()
        if bird_rows_equal(rows, state.gold)
    ]
    if exact_created:
        category = "correct_table_wrong_or_missing_terminal"
        return {
            **base,
            "category": category,
            "logic_status": logic_status(category),
            "confidence": "deterministic",
            "matching_created_tables": exact_created,
        }

    if state.predicted is not None:
        projection = find_projection(state.predicted, state.gold)
        if projection is not None:
            predicted_width = (
                len(state.predicted[0]) if state.predicted else 0
            )
            gold_width = len(state.gold[0]) if state.gold else 0
            category = (
                "column_order_only"
                if predicted_width == gold_width
                else "extra_columns_projection_only"
            )
            return {
                **base,
                "category": category,
                "logic_status": logic_status(category),
                "confidence": "deterministic",
                "matching_projection_columns": projection,
            }
        if rows_numeric_close(state.predicted, state.gold):
            category = "numeric_representation_close"
            return {
                **base,
                "category": category,
                "logic_status": logic_status(category),
                "confidence": "deterministic_with_1e-4_tolerance",
            }

    projectable_created = []
    for table, rows in state.created_tables.items():
        projection = find_projection(rows, state.gold)
        if projection is not None:
            projectable_created.append(
                {
                    "table": table,
                    "columns": state.created_columns.get(table),
                    "projection_indices": projection,
                }
            )
    if projectable_created:
        category = "projectable_table_wrong_or_missing_terminal"
        return {
            **base,
            "category": category,
            "logic_status": logic_status(category),
            "confidence": "deterministic_core_relation",
            "projectable_created_tables": projectable_created,
        }

    if state.predicted is None:
        category = "no_executable_terminal"
        return {
            **base,
            "category": category,
            "logic_status": logic_status(category),
            "confidence": "unresolved",
            "last_tolerant_calls": tolerant_json_calls(
                str(((sample.get("turns") or [{}])[-1]).get("model_output") or "")
            )[-3:],
        }

    relation = row_relation(state.predicted, state.gold)
    if relation["relation"] == "overbroad":
        category = "overbroad_result"
    elif relation["relation"] == "underbroad":
        category = "underbroad_result"
    elif relation["overlap"] > 0:
        category = "partial_row_overlap"
    else:
        category = "semantic_denotation_wrong"
    return {
        **base,
        "category": category,
        "logic_status": logic_status(category),
        "confidence": "deterministic_denotation",
        "row_relation": relation,
    }


def comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_reasoning = reasoning_text(left["sample"])
    right_reasoning = reasoning_text(right["sample"])
    left_tools = tool_sequence(left["sample"])
    right_tools = tool_sequence(right["sample"])
    left_actions = action_sequence(left["sample"])
    right_actions = action_sequence(right["sample"])
    delta = logic_rank(right["audit"]) - logic_rank(left["audit"])
    return {
        "reasoning_token_jaccard": jaccard(
            text_tokens(left_reasoning), text_tokens(right_reasoning)
        ),
        "tool_sequence_equal": left_tools == right_tools,
        "action_sequence_equal": left_actions == right_actions,
        "left_tools": left_tools,
        "right_tools": right_tools,
        "logic_rank_delta": delta,
        "logic_transition": f"{left['audit']['category']} -> {right['audit']['category']}",
    }


def compact_trajectory(
    model: str,
    record: dict[str, Any],
    sample: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "example_index": record.get("example_index"),
        "sample_index": sample.get("sample_index"),
        "trajectory_id": sample.get("trajectory_id") or record.get("trajectory_id"),
        "correct": bool(sample.get("correct")),
        "legal": bool(sample.get("legal")),
        "steps": sample.get("steps"),
        "errors": sample.get("errors"),
        "failure_type": sample.get("failure_type"),
        "audit": audit,
        "tool_sequence": tool_sequence(sample),
        "reasoning_sha256": hashlib.sha256(
            reasoning_text(sample).encode("utf-8")
        ).hexdigest(),
    }


def markdown_code(value: Any, language: str = "json") -> str:
    text = value if isinstance(value, str) else json_dump(value, indent=2)
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}\n"


def render_turn(turn: dict[str, Any]) -> str:
    parsed = turn.get("parsed") or {}
    pieces = [f"#### Turn {int(turn.get('turn_index', 0)) + 1}\n"]
    think = parsed.get("think")
    if not isinstance(think, str):
        output = str(turn.get("model_output") or "")
        match = re.search(r"<think>(.*?)(?:</think>|$)", output, flags=re.DOTALL)
        think = match.group(1) if match else ""
    pieces.append("**推理内容（verbatim）**\n\n")
    pieces.append(markdown_code(think or "<无可解析 think>", "text"))
    pieces.append("**原始模型输出（verbatim）**\n\n")
    pieces.append(markdown_code(str(turn.get("model_output") or ""), "text"))
    pieces.append("**解析动作**\n\n")
    pieces.append(
        markdown_code(
            {
                "tool": parsed.get("tool"),
                "arguments": parsed.get("arguments"),
                "feedback_recovery": turn.get("feedback_recovery"),
                "recovered_from_error_type": turn.get("recovered_from_error_type"),
            }
        )
    )
    if turn.get("execution_error_type"):
        pieces.append("**执行错误**\n\n")
        pieces.append(
            markdown_code(
                {
                    "execution_error_type": turn.get("execution_error_type"),
                    "execution_error": turn.get("execution_error"),
                    "error_event": turn.get("error_event"),
                }
            )
        )
    else:
        pieces.append("**工具输出（完整记录）**\n\n")
        pieces.append(markdown_code(turn.get("tool_output")))
    generation = turn.get("generation_stats")
    if generation:
        pieces.append("**生成统计**\n\n")
        pieces.append(markdown_code(generation))
    return "\n".join(pieces)


def render_model_trajectory(
    model: str,
    sample: dict[str, Any],
    audit: dict[str, Any],
) -> str:
    pieces = [
        f"### {MODEL_LABELS[model]}\n",
        (
            f"- strict correct: `{bool(sample.get('correct'))}`\n"
            f"- legal: `{bool(sample.get('legal'))}`\n"
            f"- failure type: `{sample.get('failure_type')}`\n"
            f"- steps/errors: `{sample.get('steps')}` / `{sample.get('errors')}`\n"
            f"- format-agnostic audit: `{audit['category']}`\n"
            f"- logic status: `{audit['logic_status']}`\n"
            f"- confidence: `{audit.get('confidence')}`\n"
            f"- tool sequence: `{tool_sequence(sample)}`\n"
        ),
        "\n**确定性 replay / 宽松输出契约诊断**\n\n",
        markdown_code(audit),
    ]
    for turn in sample.get("turns") or []:
        pieces.append(render_turn(turn))
    if not sample.get("turns"):
        pieces.append("\n<无 semantic turn>\n")
    return "\n".join(pieces)


def render_question_report(
    record_by_model: dict[str, dict[str, Any]],
    task: dict[str, Any],
    paired_rows: list[dict[str, Any]],
) -> str:
    example_index = int(next(iter(record_by_model.values()))["example_index"])
    record = record_by_model["sft2"]
    pieces = [
        f"# BIRD example {example_index} 三模型轨迹对比\n",
        "> 审计说明：sample_index 只做位置对齐；三组轨迹是独立随机采样，不是共享随机数的一一反事实。\n",
        f"- db_id: `{record.get('db_id')}`\n",
        f"- difficulty: `{(task.get('metadata') or {}).get('difficulty')}`\n",
        f"- question: {record.get('question')}\n",
        f"- external knowledge: {task.get('external_knowledge') or '<无>'}\n",
        "- scorer: `bird-set`；行顺序和重复行已忽略，列位置仍严格。\n",
        "\n<details><summary>审计侧 gold SQL（模型生成时不可见）</summary>\n\n",
        markdown_code(task_gold_sql(task), "sql"),
        "</details>\n",
    ]
    for paired in paired_rows:
        sample_index = paired["sample_index"]
        pieces.append(f"\n## Sample position {sample_index}\n")
        pieces.append(
            "| 模型 | Strict | Legal | Failure | Logic audit | Steps | Errors |\n"
            "|---|---:|---:|---|---|---:|---:|\n"
        )
        for model in MODEL_ORDER:
            item = paired["models"][model]
            sample = item["sample"]
            audit = item["audit"]
            pieces.append(
                f"| {MODEL_LABELS[model]} | {bool(sample.get('correct'))} | "
                f"{bool(sample.get('legal'))} | `{sample.get('failure_type')}` | "
                f"`{audit['category']}` | {sample.get('steps')} | {sample.get('errors')} |\n"
            )
        pieces.append("\n**跨模型变化（仅描述独立样本，不作因果解释）**\n\n")
        pieces.append(markdown_code(paired["comparisons"]))
        for model in MODEL_ORDER:
            pieces.append(
                render_model_trajectory(
                    model,
                    paired["models"][model]["sample"],
                    paired["models"][model]["audit"],
                )
            )
    return "\n".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft2-dir", type=Path, required=True)
    parser.add_argument("--result-only-dir", type=Path, required=True)
    parser.add_argument("--process-dir", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()

    source_dirs = {
        "sft2": args.sft2_dir.resolve(),
        "result_only": args.result_only_dir.resolve(),
        "process": args.process_dir.resolve(),
    }
    tasks = load_task_map(args.tasks_json.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = args.output_dir / "by_question"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_paths = {
        model: directory / "all.jsonl"
        for model, directory in source_dirs.items()
    }
    source_offsets = {
        model: jsonl_offsets_by_example(path)
        for model, path in all_paths.items()
    }
    example_indices = set(source_offsets["sft2"])
    for model in MODEL_ORDER[1:]:
        if set(source_offsets[model]) != example_indices:
            missing = sorted(example_indices - set(source_offsets[model]))
            extra = sorted(set(source_offsets[model]) - example_indices)
            raise ValueError(
                f"{model} example selection differs: missing={missing}, extra={extra}"
            )
    streams = {
        model: all_paths[model].open("rb")
        for model in MODEL_ORDER
    }
    failure_counts = Counter()
    category_counts = {model: Counter() for model in MODEL_ORDER}
    transition_counts = {
        "result_only_vs_sft2": Counter(),
        "process_vs_sft2": Counter(),
    }
    failed_trajectory_count = 0
    paired_position_count = 0
    question_report_count = 0
    model_exact_tool_sequences = Counter()
    reasoning_similarity_sums = Counter()
    reasoning_similarity_counts = Counter()
    report_index = []
    paired_index_path = args.output_dir / "paired_trajectory_index.jsonl"
    failure_index_path = args.output_dir / "failed_trajectory_index.jsonl"

    with paired_index_path.open("w", encoding="utf-8") as paired_target, (
        failure_index_path.open("w", encoding="utf-8")
    ) as failure_target:
        try:
            for example_index in sorted(example_indices):
                records = {
                    model: read_jsonl_record_at(
                        streams[model], source_offsets[model][example_index]
                    )
                    for model in MODEL_ORDER
                }
                task = tasks.get(example_index)
                if task is None:
                    raise ValueError(f"task {example_index} missing from tasks JSONL")
                samples = {
                    model: {
                        int(sample.get("sample_index", position)): sample
                        for position, sample in enumerate(record.get("samples") or [])
                    }
                    for model, record in records.items()
                }
                sample_positions = set(samples["sft2"])
                if any(set(samples[model]) != sample_positions for model in MODEL_ORDER):
                    raise ValueError(f"sample positions differ for example {example_index}")

                question_paired_rows = []
                for sample_index in sorted(sample_positions):
                    if all(samples[model][sample_index].get("correct") for model in MODEL_ORDER):
                        continue
                    paired_position_count += 1
                    models = {}
                    for model in MODEL_ORDER:
                        sample = samples[model][sample_index]
                        audit = audit_failed_sample(
                            sample,
                            task,
                            replay_timeout_seconds=args.replay_timeout_seconds,
                        )
                        models[model] = {"sample": sample, "audit": audit}
                        if not sample.get("correct"):
                            failed_trajectory_count += 1
                            failure_counts[str(sample.get("failure_type"))] += 1
                            category_counts[model][audit["category"]] += 1
                            failure_target.write(
                                json_dump(
                                    compact_trajectory(
                                        model, records[model], sample, audit
                                    )
                                )
                                + "\n"
                            )

                    comparisons = {
                        "result_only_vs_sft2": comparison(
                            models["sft2"], models["result_only"]
                        ),
                        "process_vs_sft2": comparison(
                            models["sft2"], models["process"]
                        ),
                        "result_only_vs_process": comparison(
                            models["process"], models["result_only"]
                        ),
                    }
                    for name in ("result_only_vs_sft2", "process_vs_sft2"):
                        transition_counts[name][comparisons[name]["logic_transition"]] += 1
                        model_exact_tool_sequences[name] += int(
                            comparisons[name]["tool_sequence_equal"]
                        )
                        similarity = comparisons[name]["reasoning_token_jaccard"]
                        if similarity is not None:
                            reasoning_similarity_sums[name] += similarity
                            reasoning_similarity_counts[name] += 1
                    compact = {
                        "example_index": example_index,
                        "sample_index": sample_index,
                        "db_id": records["sft2"].get("db_id"),
                        "question": records["sft2"].get("question"),
                        "models": {
                            model: compact_trajectory(
                                model,
                                records[model],
                                models[model]["sample"],
                                models[model]["audit"],
                            )
                            for model in MODEL_ORDER
                        },
                        "comparisons": comparisons,
                    }
                    paired_target.write(json_dump(compact) + "\n")
                    question_paired_rows.append(
                        {
                            "sample_index": sample_index,
                            "models": models,
                            "comparisons": comparisons,
                        }
                    )

                if question_paired_rows:
                    filename = f"example_{example_index:05d}.md"
                    (report_dir / filename).write_text(
                        render_question_report(records, task, question_paired_rows),
                        encoding="utf-8",
                    )
                    question_report_count += 1
                    report_index.append(
                        {
                            "example_index": example_index,
                            "db_id": records["sft2"].get("db_id"),
                            "question": records["sft2"].get("question"),
                            "failed_sample_positions": [
                                row["sample_index"] for row in question_paired_rows
                            ],
                            "report": f"by_question/{filename}",
                        }
                    )
        finally:
            for stream in streams.values():
                stream.close()

    if failed_trajectory_count != sum(failure_counts.values()):
        raise AssertionError("failed trajectory accounting mismatch")

    logical_improvements = {}
    paired_rows = list(jsonl_rows(paired_index_path))
    for target in ("result_only", "process"):
        strict_gains = 0
        format_agnostic_gains = 0
        wrong_to_logic_correct_still_strict_failed = 0
        regressions = 0
        for row in paired_rows:
            source = row["models"]["sft2"]
            dest = row["models"][target]
            source_rank = logic_rank(source["audit"])
            dest_rank = logic_rank(dest["audit"])
            if not source["correct"] and dest["correct"]:
                strict_gains += 1
            if source_rank < 3 <= dest_rank:
                format_agnostic_gains += 1
            if (
                source_rank == 0
                and dest["audit"]["category"] in FORMAT_AGNOSTIC_CORRECT
                and not dest["correct"]
            ):
                wrong_to_logic_correct_still_strict_failed += 1
            if source_rank > dest_rank:
                regressions += 1
        logical_improvements[target] = {
            "sft_failed_to_target_strict_correct": strict_gains,
            "sft_below_logic_correct_to_target_logic_correct_or_strict": format_agnostic_gains,
            "sft_wrong_to_target_logic_correct_but_still_strict_failed": (
                wrong_to_logic_correct_still_strict_failed
            ),
            "logic_rank_regressions": regressions,
        }

    summary = {
        "schema_version": "three-model-failed-trajectory-audit-v1",
        "warning": (
            "sample_index alignment is positional only; the three runs used independent "
            "stochastic samples and are not shared-randomness causal pairs"
        ),
        "inputs": {
            model: {
                "all_jsonl": str((directory / "all.jsonl").resolve()),
                "all_jsonl_sha256": file_sha256(directory / "all.jsonl"),
                "manifest": str((directory / "manifest.json").resolve()),
            }
            for model, directory in source_dirs.items()
        },
        "tasks_json": str(args.tasks_json.resolve()),
        "replay_timeout_seconds_per_failed_trajectory": args.replay_timeout_seconds,
        "failed_trajectory_count": failed_trajectory_count,
        "paired_position_count_with_any_failure": paired_position_count,
        "question_report_count": question_report_count,
        "failure_types": dict(sorted(failure_counts.items())),
        "category_counts_by_model_failed_only": {
            model: dict(sorted(counts.items()))
            for model, counts in category_counts.items()
        },
        "format_agnostic_correct_categories": sorted(FORMAT_AGNOSTIC_CORRECT),
        "core_logic_correct_output_incomplete_categories": sorted(
            CORE_LOGIC_CORRECT_OUTPUT_INCOMPLETE
        ),
        "logical_improvements_positional_not_causal": logical_improvements,
        "exact_tool_sequence_count_on_paired_positions": dict(model_exact_tool_sequences),
        "mean_reasoning_token_jaccard_on_paired_positions": {
            name: (
                reasoning_similarity_sums[name] / reasoning_similarity_counts[name]
                if reasoning_similarity_counts[name]
                else None
            )
            for name in reasoning_similarity_sums
        },
        "logic_transitions": {
            name: dict(sorted(counts.items()))
            for name, counts in transition_counts.items()
        },
        "artifacts": {
            "paired_index": str(paired_index_path.resolve()),
            "failure_index": str(failure_index_path.resolve()),
            "question_reports": str(report_dir.resolve()),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json_dump(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report_index.json").write_text(
        json_dump(report_index, indent=2) + "\n",
        encoding="utf-8",
    )

    readme_lines = [
        "# Three-model failed-trajectory audit",
        "",
        summary["warning"],
        "",
        f"- Failed trajectories: {failed_trajectory_count}",
        f"- Positional groups with any failure: {paired_position_count}",
        f"- Question reports: {question_report_count}",
        "- Full repeated model inputs remain in `raw/*/all.jsonl`.",
        "- Per-question reports retain verbatim model outputs, parsed actions, full recorded tool",
        "  outputs, errors, and deterministic relaxed-output replay.",
        "",
        "## Reports",
        "",
    ]
    for row in report_index:
        readme_lines.append(
            f"- [{row['example_index']} — {row['question']}]({row['report']})"
        )
    (args.output_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n",
        encoding="utf-8",
    )
    print(json_dump(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
