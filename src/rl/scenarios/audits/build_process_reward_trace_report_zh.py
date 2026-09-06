#!/usr/bin/env python3
"""Build an exhaustive Chinese per-action Process RL reward report without gold SQL."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("LABEL", "ROLLOUTS_JSONL", "METRICS_JSONL"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md_text(value: Any) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    return text.replace("|", r"\|")


def short_json(value: Any, limit: int = 220) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return md_text(text)


def compact_output(value: Any, limit: int = 260) -> str:
    if value is None:
        return "无"
    if isinstance(value, dict):
        selected = {}
        for key in (
            "error_type",
            "error_code",
            "message",
            "handle",
            "row_count",
            "columns",
            "value",
        ):
            if key in value:
                selected[key] = value[key]
        value = selected or {"keys": sorted(value)}
    return short_json(value, limit=limit)


def reward_feature_text(features: dict[str, Any]) -> str:
    terms = []
    if features.get("is_terminal"):
        terms.append("终局=1")
    if float(features.get("back_slice") or 0):
        terms.append(f"依赖={float(features['back_slice']):g}")
    if float(features.get("search_reduction") or 0):
        terms.append(f"缩减={float(features['search_reduction']):g}")
    if float(features.get("tool_error") or 0):
        terms.append("执行错误")
    if features.get("adjacent_repeat"):
        terms.append("相邻重复")
    if float(features.get("legal_no_state_change") or 0):
        terms.append("合法无状态变化")
    return "；".join(terms) or "无奖励特征"


def action_detail(
    turn: dict[str, Any],
    reward_step: dict[str, Any],
) -> str:
    parsed = turn.get("parsed") or {}
    reasoning = parsed.get("think") or ""
    arguments = parsed.get("arguments")
    output = turn.get("tool_output")
    parts = [
        "<details>",
        (
            f"<summary>动作 {reward_step['action_index']}："
            f"{html.escape(str(reward_step['tool']))} 的完整参数、推理与环境反馈</summary>"
        ),
        "",
        "**参数**",
        "",
        f"<pre>{html.escape(json.dumps(arguments, ensure_ascii=False, indent=2))}</pre>",
        "",
        "**模型推理**",
        "",
        f"<pre>{html.escape(str(reasoning))}</pre>",
        "",
        "**环境反馈**",
        "",
        f"<pre>{html.escape(json.dumps(output, ensure_ascii=False, indent=2))}</pre>",
        "",
        "</details>",
    ]
    return "\n".join(parts)


def run_summary(
    label: str,
    rollouts: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    total_rewards = [
        float(row["process_reward"]["total_reward"])
        for row in rollouts
    ]
    step_rewards = [
        float(step["reward"])
        for row in rollouts
        for step in row["process_reward"]["steps"]
    ]
    positive_credit = Counter()
    applied_penalty = Counter()
    penalty_events = Counter()
    for row in rollouts:
        process_reward = row["process_reward"]
        category_raw = Counter()
        penalty_raw = Counter()
        for step in process_reward["steps"]:
            features = step["features"]
            category_raw["终局正确"] += float(bool(features.get("is_terminal")))
            category_raw["数据依赖"] += float(features.get("back_slice") or 0)
            category_raw["搜索空间缩减"] += float(features.get("search_reduction") or 0)
            values = {
                "执行错误": 0.08 * float(features.get("tool_error") or 0),
                "相邻完全相同调用": 0.06 * float(bool(features.get("adjacent_repeat"))),
                "合法但状态不变": 0.03 * float(
                    features.get("legal_no_state_change") or 0
                ),
            }
            for name, value in values.items():
                penalty_raw[name] += value
                penalty_events[name] += int(value > 0)
        if row.get("correct") and float(process_reward["positive_mass"]) > 0:
            for name, value in category_raw.items():
                positive_credit[name] += value / float(process_reward["positive_mass"])
        raw_penalty = float(process_reward["raw_penalty_mass"])
        penalty_scale = (
            float(process_reward["capped_penalty"]) / raw_penalty
            if raw_penalty
            else 0.0
        )
        for name, value in penalty_raw.items():
            applied_penalty[name] += value * penalty_scale

    return {
        "label": label,
        "metric_steps": len(metrics),
        "rollouts": len(rollouts),
        "correct": sum(bool(row.get("correct")) for row in rollouts),
        "updated_training_steps": sum(bool(row.get("updated")) for row in metrics),
        "process_update_trajectories": sum(
            bool(value)
            for row in metrics
            for value in row.get("process_update", [])
        ),
        "all_group_advantages_zero": all(
            all(float(value) == 0.0 for value in row.get("advantages", []))
            for row in metrics
        ),
        "optimization_errors": [
            {
                "step": row.get("step"),
                "error": row.get("optimization_error"),
            }
            for row in metrics
            if row.get("optimization_error")
        ],
        "total_reward": {
            "sum": sum(total_rewards),
            "mean": fmean(total_rewards) if total_rewards else None,
            "positive": sum(value > 0 for value in total_rewards),
            "zero": sum(value == 0 for value in total_rewards),
            "negative": sum(value < 0 for value in total_rewards),
        },
        "step_reward": {
            "count": len(step_rewards),
            "sum": sum(step_rewards),
            "mean": fmean(step_rewards) if step_rewards else None,
            "positive": sum(value > 0 for value in step_rewards),
            "zero": sum(value == 0 for value in step_rewards),
            "negative": sum(value < 0 for value in step_rewards),
        },
        "positive_credit": dict(positive_credit),
        "applied_penalty": dict(applied_penalty),
        "penalty_events": dict(penalty_events),
    }


def build_report(
    runs: list[tuple[str, Path, Path]],
) -> tuple[str, dict[str, Any]]:
    loaded_runs = []
    sources = []
    for label, rollout_path, metric_path in runs:
        metrics = read_jsonl(metric_path)
        if not metrics:
            raise ValueError(f"{metric_path} has no completed metric")
        max_step = max(int(row["step"]) for row in metrics)
        rollouts = [
            row
            for row in read_jsonl(rollout_path)
            if int(row["training_step"]) <= max_step
        ]
        expected = max_step * 4
        if len(rollouts) != expected:
            raise ValueError(
                f"{label}: expected {expected} rollouts through step {max_step}, "
                f"found {len(rollouts)}"
            )
        loaded_runs.append((label, rollouts, metrics))
        sources.extend(
            [
                {
                    "label": label,
                    "kind": "rollouts",
                    "path": str(rollout_path.resolve()),
                    "sha256": sha256(rollout_path),
                    "snapshot_max_step": max_step,
                },
                {
                    "label": label,
                    "kind": "metrics",
                    "path": str(metric_path.resolve()),
                    "sha256": sha256(metric_path),
                    "snapshot_max_step": max_step,
                },
            ]
        )

    summaries = [
        run_summary(label, rollouts, metrics)
        for label, rollouts, metrics in loaded_runs
    ]
    now = datetime.now(timezone.utc).astimezone()
    lines = [
        "# SFT2 简化 Process RL：逐轨迹逐动作奖励分配报告",
        "",
        f"- 生成时间：{now.isoformat(timespec='seconds')}",
        "- 快照性质：训练快照；只包含输入文件中已经完整落盘的 step。",
        "- 评分口径：`bird-set`；任务参考结果已在 rollout 前过滤为空/NULL/标量零。",
        "- 安全说明：报告不包含 gold SQL，也不把模型推理当作事实依据。",
        "",
        "## 奖励公式",
        "",
        "成功轨迹的正奖励总量固定为 `+1`，在终局动作、真实 back-slice 数据依赖动作、"
        "真实搜索空间缩减动作之间按原始正奖励质量归一化分配。惩罚为：执行错误"
        " `-0.08/次`、相邻完全相同调用 `-0.06/次`、合法但状态不变 `-0.03/次`，"
        "单轨迹惩罚最多 `-0.8`。失败轨迹没有终局正奖励；未触发局部惩罚时为0分。",
        "",
        "逐动作表中的：",
        "",
        "- `原始正质量` 是归一化前的 `g_positive`；",
        "- `实际正奖励` 是成功轨迹归一化后分到该动作的 `c_positive`；",
        "- `实际惩罚` 是惩罚封顶后分到该动作的 `capped_penalty × c_negative`；",
        "- `最终动作奖励 = 实际正奖励 - 实际惩罚`。",
        "",
        "## 汇总",
        "",
        "| 训练组 | 完成step | 轨迹数 | 正确 | 实际更新step | 正/零/负轨迹 | 平均轨迹奖励 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        total = summary["total_reward"]
        lines.append(
            f"| {md_text(summary['label'])} | {summary['metric_steps']} | "
            f"{summary['rollouts']} | {summary['correct']} | "
            f"{summary['updated_training_steps']} | "
            f"{total['positive']}/{total['zero']}/{total['negative']} | "
            f"{total['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### 优化执行审计",
            "",
            "这里的组级 `advantages` 只对应粗粒度结果奖励；Process RL 的实际更新使用"
            "逐动作 `step_rewards`。因此组级 advantage 全为0并不等于没有梯度更新。",
            "",
            "| 训练组 | 标记process_update的轨迹 | 组级advantages是否全0 | 未更新step及原因 |",
            "|---|---:|---|---|",
        ]
    )
    for summary in summaries:
        errors = "；".join(
            f"step {item['step']}: {item['error']}"
            for item in summary["optimization_errors"]
        ) or "无"
        lines.append(
            f"| {md_text(summary['label'])} | "
            f"{summary['process_update_trajectories']} | "
            f"{'是' if summary['all_group_advantages_zero'] else '否'} | "
            f"{md_text(errors)} |"
        )
    lines.extend(
        [
            "",
            "### 正奖励与惩罚归因",
            "",
            "| 训练组 | 终局正确信用 | 数据依赖信用 | 搜索缩减信用 | 执行错误惩罚 | 相邻重复惩罚 | 无状态变化惩罚 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for summary in summaries:
        positive = summary["positive_credit"]
        penalty = summary["applied_penalty"]
        lines.append(
            f"| {md_text(summary['label'])} | "
            f"{positive.get('终局正确', 0):.6f} | "
            f"{positive.get('数据依赖', 0):.6f} | "
            f"{positive.get('搜索空间缩减', 0):.6f} | "
            f"-{penalty.get('执行错误', 0):.6f} | "
            f"-{penalty.get('相邻完全相同调用', 0):.6f} | "
            f"-{penalty.get('合法但状态不变', 0):.6f} |"
        )

    for label, rollouts, metrics in loaded_runs:
        metrics_by_step = {int(row["step"]): row for row in metrics}
        lines.extend(["", f"## {label}：逐轨迹明细", ""])
        for row in sorted(
            rollouts,
            key=lambda item: (int(item["training_step"]), int(item["sample_index"])),
        ):
            training_step = int(row["training_step"])
            sample_index = int(row["sample_index"])
            metric = metrics_by_step[training_step]
            reward = row["process_reward"]
            applied = bool(metric.get("updated"))
            lines.extend(
                [
                    (
                        f"### Step {training_step} / Sample {sample_index} / "
                        f"`{md_text(row['trajectory_id'])}`"
                    ),
                    "",
                    f"- 题号：`{row['example_index']}`；数据库：`{md_text(row['db_id'])}`",
                    f"- 问题：{md_text(row['question'])}",
                    (
                        f"- 结果：正确={bool(row.get('correct'))}，合法={bool(row.get('legal'))}，"
                        f"failure_type=`{md_text(row.get('failure_type'))}`"
                    ),
                    (
                        f"- 训练应用：{'已反向传播' if applied else '未反向传播'}；"
                        f"该组 optimization_error=`{md_text(metric.get('optimization_error'))}`"
                    ),
                    (
                        f"- 奖励守恒：正质量={float(reward['positive_mass']):.6f}，"
                        f"原始惩罚={float(reward['raw_penalty_mass']):.6f}，"
                        f"封顶后惩罚={float(reward['capped_penalty']):.6f}，"
                        f"轨迹总奖励={float(reward['total_reward']):.6f}"
                    ),
                    "",
                    "| 动作 | 工具 | 参数摘要 | 执行状态 | 奖励特征 | 原始正质量 | 实际正奖励 | 实际惩罚 | 最终动作奖励 |",
                    "|---:|---|---|---|---|---:|---:|---:|---:|",
                ]
            )
            turns = row.get("turns") or []
            reward_steps = reward["steps"]
            if len(turns) != len(reward_steps):
                raise ValueError(
                    f"{row['trajectory_id']}: {len(turns)} turns != "
                    f"{len(reward_steps)} reward steps"
                )
            for turn, reward_step in zip(turns, reward_steps, strict=True):
                features = reward_step["features"]
                parsed = turn.get("parsed") or {}
                status = (
                    "成功"
                    if features.get("legal_success")
                    else f"错误：{features.get('error_type')}"
                )
                actual_positive = (
                    float(reward_step["c_positive"])
                    if row.get("correct")
                    else 0.0
                )
                actual_penalty = (
                    float(reward["capped_penalty"])
                    * float(reward_step["c_negative"])
                )
                lines.append(
                    f"| {reward_step['action_index']} | "
                    f"`{md_text(reward_step['tool'])}` | "
                    f"`{short_json(parsed.get('arguments'))}` | "
                    f"{md_text(status)} | "
                    f"{md_text(reward_feature_text(features))} | "
                    f"{float(reward_step['g_positive']):.6f} | "
                    f"{actual_positive:.6f} | "
                    f"-{actual_penalty:.6f} | "
                    f"{float(reward_step['reward']):.6f} |"
                )
            lines.append("")
            lines.append("#### 完整动作内容")
            lines.append("")
            for turn, reward_step in zip(turns, reward_steps, strict=True):
                lines.append(action_detail(turn, reward_step))
                lines.append("")

    manifest = {
        "schema_version": "process-reward-trace-report-zh-v1",
        "generated_at": now.isoformat(timespec="seconds"),
        "denotation_comparison": "bird-set",
        "contains_gold_sql": False,
        "sources": sources,
        "runs": summaries,
    }
    return "\n".join(lines).rstrip() + "\n", manifest


def main() -> int:
    args = parse_args()
    runs = [
        (label, Path(rollouts), Path(metrics))
        for label, rollouts, metrics in args.run
    ]
    report, manifest = build_report(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
