#!/usr/bin/env python3
"""Build a text-only comparison of three matched model trajectory runs.

This report intentionally performs no database replay and does not render tool
observations.  It preserves only the information needed for qualitative policy
comparison:

* the model-authored reasoning text;
* the authored/parsed tool name and arguments;
* strict evaluation outcome and recorded execution error; and
* cross-model text/action-sequence comparisons.

The input JSONL files may contain the same examples in different line orders.
They are aligned by ``example_index`` using byte offsets so the large records do
not need to be retained in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MODEL_ORDER = ("sft2", "result_only", "process")
MODEL_LABELS = {
    "sft2": "SFT2",
    "result_only": "Result-only RL",
    "process": "Process-RL",
}
EXAMPLE_INDEX_RE = re.compile(rb'"example_index"\s*:\s*(\d+)')


def json_dump(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        default=str,
    )


def markdown_code(value: Any, language: str = "json") -> str:
    text = value if isinstance(value, str) else json_dump(value, indent=2)
    fence = "```"
    while fence in text:
        fence += "`"
    return f"{fence}{language}\n{text}\n{fence}\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_offsets_by_example(path: Path) -> dict[int, int]:
    offsets: dict[int, int] = {}
    with path.open("rb") as source:
        while True:
            offset = source.tell()
            line = source.readline()
            if not line:
                break
            if not line.strip():
                continue
            match = EXAMPLE_INDEX_RE.search(line)
            if not match:
                raise ValueError(f"missing example_index at byte offset {offset}: {path}")
            example_index = int(match.group(1))
            if example_index in offsets:
                raise ValueError(f"duplicate example_index {example_index}: {path}")
            offsets[example_index] = offset
    return offsets


def read_jsonl_record_at(source: Any, offset: int) -> dict[str, Any]:
    source.seek(offset)
    line = source.readline()
    if not line:
        raise ValueError(f"missing JSONL record at byte offset {offset}")
    return json.loads(line)


def tolerant_json_calls(text: str) -> list[dict[str, Any]]:
    """Recover visibly authored calls without executing or validating them."""
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


def turn_reasoning(turn: dict[str, Any]) -> str:
    parsed = turn.get("parsed") or {}
    think = parsed.get("think")
    if isinstance(think, str) and think.strip():
        return think.strip()
    output = str(turn.get("model_output") or "")
    match = re.search(r"<think>(.*?)(?:</think>|$)", output, flags=re.DOTALL)
    return match.group(1).strip() if match else output.strip()


def reasoning_text(sample: dict[str, Any]) -> str:
    return "\n".join(
        text for turn in sample.get("turns") or [] if (text := turn_reasoning(turn))
    )


def parsed_action(turn: dict[str, Any]) -> dict[str, Any] | None:
    parsed = turn.get("parsed") or {}
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if isinstance(tool, str) and isinstance(arguments, dict):
        return {"tool": tool, "arguments": arguments}
    return None


def visible_actions(turn: dict[str, Any]) -> list[dict[str, Any]]:
    action = parsed_action(turn)
    if action is not None:
        return [action]
    return tolerant_json_calls(str(turn.get("model_output") or ""))


def action_sequence(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return list(
        itertools.chain.from_iterable(
            visible_actions(turn) for turn in sample.get("turns") or []
        )
    )


def tool_sequence(sample: dict[str, Any]) -> list[str]:
    return [
        str(action["tool"])
        for action in action_sequence(sample)
    ]


def text_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[\w\u4e00-\u9fff]+", text)
        if len(token) > 1
    }


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def format_diagnostic(sample: dict[str, Any]) -> dict[str, Any]:
    if sample.get("correct"):
        return {
            "category": "strict_correct",
            "format_only_candidate": False,
        }
    turns = sample.get("turns") or []
    recovered = []
    for turn in turns:
        if parsed_action(turn) is not None:
            continue
        calls = tolerant_json_calls(str(turn.get("model_output") or ""))
        if calls:
            recovered.append(
                {
                    "turn_index": turn.get("turn_index"),
                    "execution_error_type": turn.get("execution_error_type"),
                    "visible_calls": calls,
                }
            )
    failure_type = str(sample.get("failure_type") or "unknown")
    candidate = failure_type == "protocol_error" and bool(recovered)
    return {
        "category": (
            "protocol_failure_with_visible_json_call"
            if candidate
            else failure_type
        ),
        "format_only_candidate": candidate,
        "visible_calls_not_accepted_by_strict_parser": recovered,
        "warning": (
            "This is only a carrier/format candidate. Without execution results it does "
            "not prove that the call or final answer is logically correct."
            if candidate
            else None
        ),
    }


def comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_reasoning = reasoning_text(left)
    right_reasoning = reasoning_text(right)
    left_actions = action_sequence(left)
    right_actions = action_sequence(right)
    return {
        "reasoning_exactly_equal": left_reasoning == right_reasoning,
        "reasoning_token_jaccard": jaccard(
            text_tokens(left_reasoning),
            text_tokens(right_reasoning),
        ),
        "tool_sequence_equal": tool_sequence(left) == tool_sequence(right),
        "action_sequence_equal": left_actions == right_actions,
        "left_tool_sequence": tool_sequence(left),
        "right_tool_sequence": tool_sequence(right),
        "strict_outcome": (
            f"{'correct' if left.get('correct') else 'failed'} -> "
            f"{'correct' if right.get('correct') else 'failed'}"
        ),
    }


def render_turn(turn: dict[str, Any]) -> str:
    action = parsed_action(turn)
    recovered = [] if action is not None else visible_actions(turn)
    pieces = [
        f"#### Turn {int(turn.get('turn_index', 0)) + 1}\n",
        "**推理文本（verbatim）**\n\n",
        markdown_code(turn_reasoning(turn) or "<无推理文本>", "text"),
        "**严格解析到的工具调用**\n\n",
        markdown_code(action or {"parsed_action": None}),
    ]
    if recovered:
        pieces.extend(
            [
                "**严格解析失败、但原始文本中可见的 JSON 调用（仅格式候选）**\n\n",
                markdown_code(recovered),
            ]
        )
    if turn.get("execution_error_type") or turn.get("execution_error"):
        pieces.extend(
            [
                "**记录的错误（不含数据库/工具返回）**\n\n",
                markdown_code(
                    {
                        "execution_error_type": turn.get("execution_error_type"),
                        "execution_error": turn.get("execution_error"),
                    }
                ),
            ]
        )
    if action is None:
        pieces.extend(
            [
                "**原始模型输出（仅在严格解析失败时保留）**\n\n",
                markdown_code(str(turn.get("model_output") or ""), "text"),
            ]
        )
    return "\n".join(pieces)


def render_model(model: str, sample: dict[str, Any]) -> str:
    diagnostic = format_diagnostic(sample)
    pieces = [
        f"### {MODEL_LABELS[model]}\n",
        (
            f"- strict correct: `{bool(sample.get('correct'))}`\n"
            f"- legal termination: `{bool(sample.get('legal'))}`\n"
            f"- failure type: `{sample.get('failure_type')}`\n"
            f"- steps/errors: `{sample.get('steps')}` / `{sample.get('errors')}`\n"
            f"- tool sequence: `{tool_sequence(sample)}`\n"
            f"- format diagnostic: `{diagnostic['category']}`\n"
        ),
    ]
    if diagnostic["format_only_candidate"]:
        pieces.append(
            "\n> 仅标记为格式候选：原始文本含 JSON 调用，但没有执行结果时不能断言"
            "其逻辑或最终答案正确。\n"
        )
    for turn in sample.get("turns") or []:
        pieces.append(render_turn(turn))
    if not sample.get("turns"):
        pieces.append("\n<无模型 turn>\n")
    return "\n".join(pieces)


def compact_sample(model: str, sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "sample_index": sample.get("sample_index"),
        "trajectory_id": sample.get("trajectory_id"),
        "strict_correct": bool(sample.get("correct")),
        "legal": bool(sample.get("legal")),
        "failure_type": sample.get("failure_type"),
        "steps": sample.get("steps"),
        "errors": sample.get("errors"),
        "tool_sequence": tool_sequence(sample),
        "format_diagnostic": format_diagnostic(sample),
        "reasoning_sha256": hashlib.sha256(
            reasoning_text(sample).encode("utf-8")
        ).hexdigest(),
    }


def render_question(
    records: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
) -> str:
    record = records["sft2"]
    pieces = [
        f"# BIRD example {record['example_index']} 三模型推理/工具轨迹\n",
        "> 不执行数据库 replay，不展示任何工具或数据库返回。sample_index 仅按位置"
        "对齐；三组是独立随机采样，不能解释为共享随机数的反事实变化。\n",
        f"- db_id: `{record.get('db_id')}`\n",
        f"- question: {record.get('question')}\n",
        f"- external knowledge: {record.get('external_knowledge') or '<无>'}\n",
    ]
    for group in groups:
        sample_index = group["sample_index"]
        pieces.extend(
            [
                f"\n## Sample position {sample_index}\n",
                "| 模型 | Strict | Legal | Failure | Steps | Errors | Tool sequence |\n",
                "|---|---:|---:|---|---:|---:|---|\n",
            ]
        )
        for model in MODEL_ORDER:
            sample = group["samples"][model]
            pieces.append(
                f"| {MODEL_LABELS[model]} | {bool(sample.get('correct'))} | "
                f"{bool(sample.get('legal'))} | `{sample.get('failure_type')}` | "
                f"{sample.get('steps')} | {sample.get('errors')} | "
                f"`{tool_sequence(sample)}` |\n"
            )
        pieces.extend(
            [
                "\n**跨模型文本/动作变化（独立样本，仅描述性）**\n\n",
                markdown_code(group["comparisons"]),
            ]
        )
        for model in MODEL_ORDER:
            pieces.append(render_model(model, group["samples"][model]))
    return "\n".join(pieces)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft2-dir", type=Path, required=True)
    parser.add_argument("--result-only-dir", type=Path, required=True)
    parser.add_argument("--process-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_dirs = {
        "sft2": args.sft2_dir.resolve(),
        "result_only": args.result_only_dir.resolve(),
        "process": args.process_dir.resolve(),
    }
    paths = {model: directory / "all.jsonl" for model, directory in source_dirs.items()}
    offsets = {
        model: jsonl_offsets_by_example(path)
        for model, path in paths.items()
    }
    examples = set(offsets["sft2"])
    for model in MODEL_ORDER[1:]:
        if set(offsets[model]) != examples:
            raise ValueError(f"{model} has a different example_index set")

    output_dir = args.output_dir.resolve()
    report_dir = output_dir / "by_question"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    streams = {model: paths[model].open("rb") for model in MODEL_ORDER}

    failure_counts = Counter()
    failures_by_model = Counter()
    paired_positions = 0
    exact_reasoning = Counter()
    exact_tools = Counter()
    similarity_sums = Counter()
    format_candidates = []
    strict_gains = Counter()
    report_index = []
    failed_index_path = output_dir / "failed_trajectory_index.jsonl"
    paired_index_path = output_dir / "paired_position_index.jsonl"

    with failed_index_path.open("w", encoding="utf-8") as failed_target, (
        paired_index_path.open("w", encoding="utf-8")
    ) as paired_target:
        try:
            for example_index in sorted(examples):
                records = {
                    model: read_jsonl_record_at(
                        streams[model],
                        offsets[model][example_index],
                    )
                    for model in MODEL_ORDER
                }
                samples = {
                    model: {
                        int(sample.get("sample_index", position)): sample
                        for position, sample in enumerate(records[model].get("samples") or [])
                    }
                    for model in MODEL_ORDER
                }
                positions = set(samples["sft2"])
                if any(set(samples[model]) != positions for model in MODEL_ORDER):
                    raise ValueError(f"sample positions differ for example {example_index}")

                question_groups = []
                for sample_index in sorted(positions):
                    group_samples = {
                        model: samples[model][sample_index]
                        for model in MODEL_ORDER
                    }
                    if all(sample.get("correct") for sample in group_samples.values()):
                        continue
                    paired_positions += 1
                    comparisons = {
                        "result_only_vs_sft2": comparison(
                            group_samples["sft2"],
                            group_samples["result_only"],
                        ),
                        "process_vs_sft2": comparison(
                            group_samples["sft2"],
                            group_samples["process"],
                        ),
                        "process_vs_result_only": comparison(
                            group_samples["result_only"],
                            group_samples["process"],
                        ),
                    }
                    for name, item in comparisons.items():
                        exact_reasoning[name] += int(item["reasoning_exactly_equal"])
                        exact_tools[name] += int(item["tool_sequence_equal"])
                        similarity_sums[name] += item["reasoning_token_jaccard"]

                    compact_models = {}
                    for model, sample in group_samples.items():
                        compact = compact_sample(model, sample)
                        compact_models[model] = compact
                        if not sample.get("correct"):
                            failure_type = str(sample.get("failure_type") or "unknown")
                            failure_counts[failure_type] += 1
                            failures_by_model[model] += 1
                            row = {
                                "example_index": example_index,
                                "db_id": records[model].get("db_id"),
                                "question": records[model].get("question"),
                                **compact,
                                "report": f"by_question/example_{example_index:05d}.md",
                            }
                            failed_target.write(json_dump(row) + "\n")
                            if compact["format_diagnostic"]["format_only_candidate"]:
                                format_candidates.append(row)

                    source = group_samples["sft2"]
                    for target in ("result_only", "process"):
                        if not source.get("correct") and group_samples[target].get("correct"):
                            strict_gains[target] += 1

                    paired_target.write(
                        json_dump(
                            {
                                "example_index": example_index,
                                "sample_index": sample_index,
                                "db_id": records["sft2"].get("db_id"),
                                "question": records["sft2"].get("question"),
                                "models": compact_models,
                                "comparisons": comparisons,
                                "report": f"by_question/example_{example_index:05d}.md",
                            }
                        )
                        + "\n"
                    )
                    question_groups.append(
                        {
                            "sample_index": sample_index,
                            "samples": group_samples,
                            "comparisons": comparisons,
                        }
                    )

                if question_groups:
                    report_name = f"example_{example_index:05d}.md"
                    (report_dir / report_name).write_text(
                        render_question(records, question_groups),
                        encoding="utf-8",
                    )
                    report_index.append(
                        {
                            "example_index": example_index,
                            "db_id": records["sft2"].get("db_id"),
                            "question": records["sft2"].get("question"),
                            "sample_positions_with_any_failure": [
                                group["sample_index"] for group in question_groups
                            ],
                            "report": f"by_question/{report_name}",
                        }
                    )
        finally:
            for stream in streams.values():
                stream.close()

    format_path = output_dir / "format_only_candidates.jsonl"
    format_path.write_text(
        "".join(json_dump(row) + "\n" for row in format_candidates),
        encoding="utf-8",
    )
    (output_dir / "report_index.json").write_text(
        json_dump(report_index, indent=2) + "\n",
        encoding="utf-8",
    )

    failed_total = sum(failures_by_model.values())
    summary = {
        "schema_version": "three-model-reasoning-tool-text-report-v1",
        "scope": (
            "Text and authored tool calls only. No database replay and no tool/database "
            "observations are included."
        ),
        "logic_warning": (
            "A format-only candidate means only that a JSON-shaped call is visible in raw "
            "model text despite strict parser failure. Without execution observations it "
            "does not prove logical correctness."
        ),
        "alignment_warning": (
            "sample_index is positional only; the runs are independent stochastic samples"
        ),
        "inputs": {
            model: {
                "all_jsonl": str(paths[model].resolve()),
                "sha256": file_sha256(paths[model]),
            }
            for model in MODEL_ORDER
        },
        "examples": len(examples),
        "question_reports": len(report_index),
        "paired_positions_with_any_failure": paired_positions,
        "failed_trajectories": failed_total,
        "failed_trajectories_by_model": dict(failures_by_model),
        "failure_types": dict(failure_counts),
        "strict_gains_at_positional_sample_vs_sft2": dict(strict_gains),
        "format_only_candidates": len(format_candidates),
        "exact_reasoning_count": dict(exact_reasoning),
        "exact_tool_sequence_count": dict(exact_tools),
        "mean_reasoning_token_jaccard": {
            name: value / paired_positions
            for name, value in similarity_sums.items()
        },
        "artifacts": {
            "readme": str((output_dir / "README.md").resolve()),
            "reports": str(report_dir.resolve()),
            "report_index": str((output_dir / "report_index.json").resolve()),
            "failed_index": str(failed_index_path.resolve()),
            "paired_index": str(paired_index_path.resolve()),
            "format_candidates": str(format_path.resolve()),
        },
    }
    (output_dir / "summary.json").write_text(
        json_dump(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    readme = [
        "# 三模型失败轨迹：推理文本与工具调用报告",
        "",
        "本报告不执行数据库 replay，也不展示工具或数据库返回。",
        "",
        f"- 三组失败轨迹合计：{failed_total}",
        f"- 至少一组失败的位置：{paired_positions}",
        f"- 逐题报告：{len(report_index)}",
        f"- 仅格式候选：{len(format_candidates)}（不等于逻辑正确）",
        "",
        "## 使用边界",
        "",
        "- 每份逐题报告同时展示三种模型的推理文本与工具名/参数。",
        "- `sample_index` 只是位置对齐；三次运行是独立随机采样。",
        "- strict correct 可证明该次完整轨迹通过正式判分。",
        "- format-only candidate 只表示严格解析失败时文本里仍可见 JSON 调用；",
        "  不看数据库执行结果时，不能自动证明逻辑正确。",
        "",
        "## 逐题报告",
        "",
    ]
    for row in report_index:
        readme.append(
            f"- [{row['example_index']} — {row['question']}]({row['report']})"
        )
    (output_dir / "README.md").write_text(
        "\n".join(readme) + "\n",
        encoding="utf-8",
    )
    print(json_dump(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
