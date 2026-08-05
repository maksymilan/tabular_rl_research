#!/usr/bin/env python3
"""Render one plain-text document for three matched trajectory runs.

Only model reasoning, authored tool calls, recorded errors, and final strict
evaluation outcomes are included.  The script never executes a tool, opens a
database, or renders a tool/database observation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build_three_model_reasoning_tool_report import (
    MODEL_LABELS,
    MODEL_ORDER,
    action_sequence,
    comparison,
    format_diagnostic,
    json_dump,
    jsonl_offsets_by_example,
    parsed_action,
    read_jsonl_record_at,
    tool_sequence,
    turn_reasoning,
    visible_actions,
)


DIVIDER = "=" * 100
SUBDIVIDER = "-" * 100


def final_result(sample: dict[str, Any]) -> str:
    if sample.get("correct"):
        return "CORRECT（通过正式 strict bird-set 评测）"
    return f"FAILED（{sample.get('failure_type') or 'unknown'}）"


def render_turn(turn: dict[str, Any]) -> list[str]:
    turn_number = int(turn.get("turn_index", 0)) + 1
    action = parsed_action(turn)
    recovered = [] if action is not None else visible_actions(turn)
    lines = [
        f"[Turn {turn_number}]",
        "推理内容：",
        turn_reasoning(turn) or "<无推理文本>",
        "",
    ]
    if action is not None:
        lines.extend(
            [
                "工具调用：",
                f"tool = {action['tool']}",
                "arguments =",
                json_dump(action["arguments"], indent=2),
            ]
        )
    elif recovered:
        lines.extend(
            [
                "工具调用：严格解析失败；原始文本中可见以下 JSON 调用（仅格式候选）：",
                json_dump(recovered, indent=2),
            ]
        )
    else:
        lines.extend(["工具调用：", "<未解析到工具调用>"])

    if turn.get("execution_error_type") or turn.get("execution_error"):
        lines.extend(
            [
                "记录错误：",
                json_dump(
                    {
                        "execution_error_type": turn.get("execution_error_type"),
                        "execution_error": turn.get("execution_error"),
                    },
                    indent=2,
                ),
            ]
        )
    lines.append("")
    return lines


def render_model(model: str, sample: dict[str, Any]) -> list[str]:
    diagnostic = format_diagnostic(sample)
    actions = action_sequence(sample)
    final_action = actions[-1] if actions else None
    lines = [
        SUBDIVIDER,
        f"模型：{MODEL_LABELS[model]}",
        f"轨迹 ID：{sample.get('trajectory_id') or '<无>'}",
        f"sample_index：{sample.get('sample_index')}",
        f"工具序列：{tool_sequence(sample)}",
        "",
    ]
    for turn in sample.get("turns") or []:
        lines.extend(render_turn(turn))
    if not sample.get("turns"):
        lines.extend(["<无模型 turn>", ""])

    lines.extend(
        [
            "最终结果：",
            f"result = {final_result(sample)}",
            f"strict_correct = {bool(sample.get('correct'))}",
            f"legal_termination = {bool(sample.get('legal'))}",
            f"failure_type = {sample.get('failure_type')}",
            f"steps = {sample.get('steps')}",
            f"errors = {sample.get('errors')}",
            f"format_diagnostic = {diagnostic['category']}",
            "final_visible_tool_call =",
            json_dump(final_action, indent=2) if final_action is not None else "<无>",
        ]
    )
    if diagnostic["format_only_candidate"]:
        lines.extend(
            [
                "说明：这是严格格式失败候选；文本中可见 JSON 调用，但在不查看执行结果时，",
                "不能据此断言该调用或最终答案逻辑正确。",
            ]
        )
    lines.append("")
    return lines


def render_difference_summary(group: dict[str, dict[str, Any]]) -> list[str]:
    pairs = (
        ("Result-only vs SFT2", "sft2", "result_only"),
        ("Process-RL vs SFT2", "sft2", "process"),
        ("Process-RL vs Result-only", "result_only", "process"),
    )
    lines = [
        "三模型区别摘要：",
        (
            "失败类型："
            f"SFT2={group['sft2'].get('failure_type')}；"
            f"Result-only={group['result_only'].get('failure_type')}；"
            f"Process-RL={group['process'].get('failure_type')}"
        ),
    ]
    for label, left, right in pairs:
        item = comparison(group[left], group[right])
        lines.extend(
            [
                f"{label}：",
                (
                    "  推理文本 token Jaccard = "
                    f"{item['reasoning_token_jaccard']:.4f}"
                    "（越低表示文本差异越大）"
                ),
                f"  工具序列相同 = {item['tool_sequence_equal']}",
                f"  完整工具参数序列相同 = {item['action_sequence_equal']}",
                f"  {MODEL_LABELS[left]} 工具序列 = {item['left_tool_sequence']}",
                f"  {MODEL_LABELS[right]} 工具序列 = {item['right_tool_sequence']}",
            ]
        )
    lines.extend(
        [
            "说明：以下为三条完整轨迹。文本相似度和工具差异只描述独立随机样本，",
            "不表示同一随机轨迹被训练前后因果地改写。",
            "",
        ]
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft2-dir", type=Path, required=True)
    parser.add_argument("--result-only-dir", type=Path, required=True)
    parser.add_argument("--process-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selection",
        help="Optional comma-separated example_index:sample_index positions.",
    )
    args = parser.parse_args()
    selection = None
    if args.selection:
        selection = {
            tuple(int(value) for value in item.split(":", 1))
            for item in args.selection.split(",")
        }

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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    streams = {model: paths[model].open("rb") for model in MODEL_ORDER}
    paired_positions = 0
    failed_trajectories = 0
    rendered_trajectories = 0

    with args.output.open("w", encoding="utf-8") as target:
        target.write(
            "\n".join(
                [
                    "三模型失败位置：推理内容、工具调用与最终结果",
                    "",
                    "范围：SFT2、Result-only RL、Process-RL；同一 300 题，每题 K=4。",
                    (
                        f"本文件选取 {len(selection)} 个三模型共同失败位置。"
                        if selection is not None
                        else "本文件包含所有至少一条模型失败的位置。"
                    ),
                    "本文件不执行数据库 replay，不包含任何数据库或工具返回。",
                    "只保留模型推理文本、工具调用、记录错误和正式 strict 评测结果。",
                    "sample_index 只做位置对齐；三组轨迹是独立随机采样，不是共享随机数反事实。",
                    "",
                    DIVIDER,
                    "",
                ]
            )
        )
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

                for sample_index in sorted(positions):
                    if (
                        selection is not None
                        and (example_index, sample_index) not in selection
                    ):
                        continue
                    group = {
                        model: samples[model][sample_index]
                        for model in MODEL_ORDER
                    }
                    if selection is not None and any(
                        sample.get("correct") for sample in group.values()
                    ):
                        raise ValueError(
                            f"selected position {example_index}:{sample_index} "
                            "is not a three-model common failure"
                        )
                    if all(sample.get("correct") for sample in group.values()):
                        continue
                    paired_positions += 1
                    failed_trajectories += sum(
                        not bool(sample.get("correct"))
                        for sample in group.values()
                    )
                    rendered_trajectories += len(MODEL_ORDER)
                    record = records["sft2"]
                    target.write(
                        "\n".join(
                            [
                                DIVIDER,
                                f"位置编号：{paired_positions}",
                                f"example_index：{example_index}",
                                f"sample_index：{sample_index}",
                                f"db_id：{record.get('db_id')}",
                                f"问题：{record.get('question')}",
                                (
                                    "外部知识："
                                    f"{record.get('external_knowledge') or '<无>'}"
                                ),
                                "",
                            ]
                        )
                    )
                    target.write("\n".join(render_difference_summary(group)))
                    for model in MODEL_ORDER:
                        target.write("\n".join(render_model(model, group[model])))
                    target.write("\n")
        finally:
            for stream in streams.values():
                stream.close()

        if selection is not None and paired_positions != len(selection):
            raise ValueError(
                f"rendered {paired_positions} selected positions, expected {len(selection)}"
            )

        target.write(
            "\n".join(
                [
                    DIVIDER,
                    "文档汇总",
                    f"有至少一条失败的对齐位置：{paired_positions}",
                    f"其中失败轨迹：{failed_trajectories}",
                    f"为横向比较而展示的三模型轨迹总数：{rendered_trajectories}",
                    "",
                ]
            )
        )

    print(
        json_dump(
            {
                "output": str(args.output.resolve()),
                "paired_positions": paired_positions,
                "failed_trajectories": failed_trajectories,
                "rendered_trajectories": rendered_trajectories,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
