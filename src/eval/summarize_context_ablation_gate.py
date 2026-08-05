#!/usr/bin/env python3
"""Summarize paired atomic context-management ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


VERSIONS = ("version39", "version46", "version47", "version48", "version49")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def action_key(parsed: dict) -> str:
    return json.dumps(
        {"tool": parsed.get("tool"), "arguments": parsed.get("arguments") or {}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def summarize_records(
    records: list[dict],
    targets: set[str],
    controls: set[str],
    excluded_ids: set[str],
) -> dict:
    records = [
        record for record in records
        if record.get("trajectory_id") not in excluded_ids
    ]
    usage = Counter()
    tools = Counter()
    context_chars = []
    repeated_reads = 0
    read_actions = 0
    read_actions_by_task = {}
    repeated_reads_by_task = {}
    errors_by_type = Counter()
    outcomes = {}
    for record in records:
        trajectory_id = record.get("trajectory_id")
        outcomes[trajectory_id] = bool(record.get("correct"))
        usage.update(record.get("usage") or {})
        seen_reads = set()
        task_reads = 0
        task_repeated_reads = 0
        for event in record.get("error_events") or []:
            error = event.get("type") or event.get("error_type") or "unknown"
            errors_by_type[str(error)] += 1
        for turn in record.get("turns") or []:
            model_input = turn.get("model_input")
            if isinstance(model_input, list):
                context_chars.append(sum(len(str(item.get("content", ""))) for item in model_input))
            parsed = turn.get("parsed")
            if not isinstance(parsed, dict):
                continue
            tool = parsed.get("tool")
            if tool:
                tools[str(tool)] += 1
            if tool in {"read_subtable", "inspect_rows"}:
                read_actions += 1
                task_reads += 1
                key = action_key(parsed)
                if key in seen_reads:
                    repeated_reads += 1
                    task_repeated_reads += 1
                seen_reads.add(key)
        if task_reads:
            read_actions_by_task[trajectory_id] = task_reads
        if task_repeated_reads:
            repeated_reads_by_task[trajectory_id] = task_repeated_reads

    def correct_in(ids: set[str]) -> int:
        return sum(bool(outcomes.get(item)) for item in ids)

    return {
        "tasks": len(records),
        "correct": sum(bool(record.get("correct")) for record in records),
        "legal": sum(bool(record.get("legal")) for record in records),
        "targets_correct": correct_in(targets),
        "controls_correct": correct_in(controls),
        "targets_total": len(targets - excluded_ids),
        "controls_total": len(controls - excluded_ids),
        "actions": sum(int(record.get("steps") or 0) for record in records),
        "mean_actions": round(
            statistics.fmean(int(record.get("steps") or 0) for record in records), 3
        ),
        "process_errors": sum(int(record.get("errors") or 0) for record in records),
        "failure_types": dict(Counter(
            str(record.get("failure_type") or "none") for record in records
        )),
        "error_types": dict(errors_by_type),
        "read_actions": read_actions,
        "repeated_exact_read_actions": repeated_reads,
        "read_actions_by_task": read_actions_by_task,
        "repeated_exact_reads_by_task": repeated_reads_by_task,
        "tool_counts": dict(tools),
        "usage": dict(usage),
        "context_chars": {
            "requests": len(context_chars),
            "mean": round(statistics.fmean(context_chars), 2) if context_chars else 0,
            "median": round(statistics.median(context_chars), 2) if context_chars else 0,
            "p95": percentile(context_chars, 0.95) if context_chars else 0,
            "max": max(context_chars, default=0),
        },
        "outcomes": outcomes,
    }


def exact_mcnemar(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * (0.5 ** discordant)
        for index in range(min(gains, regressions) + 1)
    )
    return min(1.0, 2 * tail)


def paired(left: dict, right: dict) -> dict:
    common = sorted(set(left["outcomes"]) & set(right["outcomes"]))
    gains = [item for item in common if not left["outcomes"][item] and right["outcomes"][item]]
    regressions = [item for item in common if left["outcomes"][item] and not right["outcomes"][item]]
    return {
        "left": left["version"],
        "right": right["version"],
        "paired_tasks": len(common),
        "gains": gains,
        "regressions": regressions,
        "both_correct": sum(left["outcomes"][item] and right["outcomes"][item] for item in common),
        "both_wrong": sum(not left["outcomes"][item] and not right["outcomes"][item] for item in common),
        "exact_mcnemar_p": exact_mcnemar(len(gains), len(regressions)),
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Atomic 上下文管理 Handle Card 消融 Gate16",
        "",
        "## 结果",
        "",
        "| 版本 | 改动 | 正确 | Target | Control | Legal | Errors | Mean actions | Reads | Exact rereads | Prompt chars/request | Tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "version39": "同时间 baseline",
        "version46": "仅 handle cards",
        "version47": "+ old rows archived / latest full",
        "version48": "+ interpret-before-act",
        "version49": "active dependency rows / inactive archive",
    }
    for version in VERSIONS:
        item = summary["versions"][version]
        usage = item.get("usage") or {}
        lines.append(
            f"| {version} | {labels[version]} | {item['correct']}/{item['tasks']} | "
            f"{item['targets_correct']}/{item['targets_total']} | "
            f"{item['controls_correct']}/{item['controls_total']} | "
            f"{item['legal']}/{item['tasks']} | {item['process_errors']} | "
            f"{item['mean_actions']:.2f} | {item['read_actions']} | "
            f"{item['repeated_exact_read_actions']} | {item['context_chars']['mean']:.0f} | "
            f"{usage.get('total_tokens', 0)} |"
        )
    lines.extend([
        "",
        "version49 原始 16 题 artifact 为 9/16、15/16 legal；其中全局配对排除项 "
        "`bird_train_06299` 在 version49 中正确（21 步、2 次错误）。为避免利用 version46 的"
        "缺失结果选择性计分，上表所有版本仍统一排除该题。该题涉及超出 recent-4 的 dormant "
        "branch，实际用了 10 次读取、2 次精确重读，但没有再耗尽 30 步。",
    ])
    offline = summary.get("offline_renderer") or {}
    if offline:
        lines.extend([
            "",
            "## 离线重渲染",
            "",
            "对历史 fixed-200 的每个真实 prefix 做纯渲染替换，canonical state 和动作不变：",
            "",
            "| 版本 | 末轮平均字符变化 | 解释 |",
            "|---|---:|---|",
            f"| version46 | -{offline['version46_final_saved_mean']:.0f} | handle card 净压缩 |",
            f"| version47 | -{offline['version47_final_saved_mean']:.0f} | latest full output 抵消大部分 rows 归档收益 |",
            f"| version48 | +{-offline['version48_final_saved_mean']:.0f} | 总结提示使 prompt 净增长 |",
            f"| version49 | -{offline['version49_final_saved_mean']:.0f} | 仅归档 active closure 外的 rows |",
            "",
            f"version47 全局归档后，历史真实下一动作中有 "
            f"{offline['version47_turns_with_grounding_value_removed']} 轮、"
            f"{offline['version47_grounding_values_removed']} 个 literal 不再可见；"
            f"version49 降至 {offline['version49_turns_with_grounding_value_removed']} 轮、"
            f"{offline['version49_grounding_values_removed']} 个 literal，均需重新读取。",
        ])
    lines.extend(["", "## 配对变化", ""])
    for comparison in summary["comparisons"]:
        lines.append(
            f"- {comparison['left']} → {comparison['right']}："
            f"{len(comparison['gains'])} gains、{len(comparison['regressions'])} regressions，"
            f"exact McNemar `p={comparison['exact_mcnemar_p']:.4f}`。"
        )
        if comparison["gains"]:
            lines.append(f"  - gains: {', '.join(comparison['gains'])}")
        if comparison["regressions"]:
            lines.append(f"  - regressions: {', '.join(comparison['regressions'])}")
    baseline = summary["versions"]["version39"]
    v46 = summary["versions"]["version46"]
    v47 = summary["versions"]["version47"]
    v48 = summary["versions"]["version48"]
    v49 = summary["versions"]["version49"]
    lines.extend([
        "",
        "## 诊断",
        "",
        f"1. **Handle card 单独没有能力收益。** version46 与 baseline 在 15 个语义可比任务上"
        f"完全同 outcome，但 actions 增加 {(v46['mean_actions'] / baseline['mean_actions'] - 1) * 100:.1f}%，"
        f"tokens 增加 {(v46['usage']['total_tokens'] / baseline['usage']['total_tokens'] - 1) * 100:.1f}%，"
        f"process errors 从 {baseline['process_errors']} 增至 {v46['process_errors']}。",
        "2. **归档旧 rows 有弱 target 信号，但稳定性不合格。** version47 相对 version46 为 "
        f"2 gains / 1 regression，p=1.0；Legal 从 {v46['legal']}/15 降至 {v47['legal']}/15。"
        f"read calls 从 {baseline['read_actions']} 增至 {v47['read_actions']}，其中 "
        f"{v47['repeated_exact_read_actions']} 次是同题内精确重读。",
        "3. **强制先总结没有净增益。** version48 相对 version47 为 1 gain / 1 regression，"
        f"actions 达 baseline 的 {v48['mean_actions'] / baseline['mean_actions']:.2f} 倍，"
        f"tokens 达 {v48['usage']['total_tokens'] / baseline['usage']['total_tokens']:.2f} 倍，"
        f"精确重读进一步升至 {v48['repeated_exact_read_actions']} 次。",
        "4. **主要失败机制是 observation churn。** `bird_train_05873` 在 version47/48 中分别"
        f"产生 {v47['repeated_exact_reads_by_task'].get('bird_train_05873', 0)} / "
        f"{v48['repeated_exact_reads_by_task'].get('bird_train_05873', 0)} 次精确重读并达到 max-steps；"
        "baseline 与 version46 都正确。`bird_train_02918` 虽保持正确，也从 baseline 10 步增长到"
        "version47 21 步、version48 26 步。",
        "5. **Dependency-aware active/archive 消除了配对子集的 churn，但没有带来能力增益。** version49 在"
        f"配对任务上为 {v49['correct']}/{v49['tasks']}，与 baseline outcome 完全一致；"
        f"8/8 controls 全保留，但 0/{v49['targets_total']} targets 恢复。读取降至 "
        f"{v49['read_actions']} 次、精确重读 {v49['repeated_exact_read_actions']} 次，"
        f"`bird_train_05873` 从 version47/48 的 30 步失败恢复为 8 步正确，"
        f"`bird_train_02918` 从 21/26 步回落到 10 步。相对 baseline，actions 仅变化 "
        f"{(v49['mean_actions'] / baseline['mean_actions'] - 1) * 100:.1f}%，tokens 变化 "
        f"{(v49['usage']['total_tokens'] / baseline['usage']['total_tokens'] - 1) * 100:.1f}%。",
        "",
        "## version49 失败归因",
        "",
        "- **最终列/表示不精确（2）**：`02868` 多返回 MiddleName；`00582` 把 First/Last "
        "拼成一个 name 列。两题实体与行集合都已找对。",
        "- **擅自改变问题语义（2）**：`02438` 把 distinct status 改成每单 latest status 并多带 "
        "order_id；`06246` 无依据增加 `type='Post Office'` 限制，得到 137 而非 173。",
        "- **聚合/行粒度错误（1）**：`00074` 没有按题目要求使用全局平均订单数量，最终只保留 "
        "1 个标题而非 3 个。",
        "- **排序后单行边界错误（1）**：`01213` 找对最新作品 Henry VIII，却返回其全部 47 个"
        "角色；gold 的 `ORDER BY Date DESC LIMIT 1` 只保留 1 行。",
        "- **工具参数恢复失败（1）**：`06165` 两次 join 列命名错误后，又用非法 "
        "`read_subtable(limit=50)`，触发同类错误上限。",
        "",
        "## 决策",
        "",
        "version49 不扩到 Gate50，不进入 SFT/RL，也不替换 version39：它通过了上下文稳定性目标，"
        "但没有恢复任何 target，且合法终止少 1 题。保留 active/archive renderer 作为工程原型和"
        "后续单变量研究基线；不再扩展全局 rows 归档或 interpret-before-act 提示。下一步若继续，"
        "应针对合法链路上的语义/最终表错误，而不是继续压缩 resident rows。",
    ])
    lines.extend([
        "",
        "## 基础设施处理",
        "",
        f"- 全局语义配对子集排除：{', '.join(summary['excluded_infrastructure_ids']) or '无'}。",
        "- 只允许用 fresh retry 替换原始 `api_error`；wrong answer、工具错误和 max-steps 均不重试。",
        f"- 已替换：{', '.join(summary['infrastructure_replacements']) or '无'}。",
        "",
        "## 协议边界",
        "",
        "四个实验版本均保持 version39 工具、参数、执行、canonical state、grounding、recent-4、"
        "external knowledge 和 exact-table terminal 不变。所有轨迹均为 diagnostic-only，不进入 SFT/RL。",
        "",
        "离线重渲染与 provider 结果见同目录 JSON；本报告只在五个 16-task artifact 全部"
        "存在且 trajectory id 完全配对后生成。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument(
        "--replacement-record",
        type=Path,
        action="append",
        default=[],
        help="one-record verified.all.jsonl from a fresh infrastructure-only retry",
    )
    parser.add_argument(
        "--offline-audit",
        type=Path,
        action="append",
        default=[],
        help="offline renderer audit JSON; may be supplied more than once",
    )
    args = parser.parse_args()

    cohort = read_json(args.cohort_manifest)
    targets = set(cohort["selection"]["targets"])
    controls = set(cohort["selection"]["controls"])
    version_records = {}
    version_artifacts = {}
    for version in VERSIONS:
        all_path = args.gate_root / version / "verified.all.jsonl"
        manifest_path = args.gate_root / version / "verified.manifest.json"
        if not all_path.is_file() or not manifest_path.is_file():
            raise SystemExit(f"missing complete artifact for {version}: {all_path}")
        records = read_jsonl(all_path)
        if len(records) != 16:
            raise SystemExit(f"{version} has {len(records)} records, expected 16")
        version_records[version] = records
        version_artifacts[version] = {
            "all_path": str(all_path),
            "all_sha256": file_sha256(all_path),
            "manifest": read_json(manifest_path),
        }
    expected_ids = set(targets) | set(controls)
    for version, records in version_records.items():
        if {record.get("trajectory_id") for record in records} != expected_ids:
            raise SystemExit(f"{version} trajectory ids do not match the frozen cohort")

    replacements = []
    for path in args.replacement_record:
        rows = read_jsonl(path)
        if len(rows) != 1:
            raise SystemExit(f"replacement must contain exactly one record: {path}")
        replacement = rows[0]
        version = replacement.get("protocol_version")
        trajectory_id = replacement.get("trajectory_id")
        if version not in version_records:
            raise SystemExit(f"replacement has unknown protocol version: {path}")
        index = next(
            (
                index for index, record in enumerate(version_records[version])
                if record.get("trajectory_id") == trajectory_id
            ),
            None,
        )
        if index is None:
            raise SystemExit(f"replacement task is outside the cohort: {path}")
        original = version_records[version][index]
        if original.get("failure_type") != "api_error":
            raise SystemExit(
                f"refusing to replace non-infrastructure result {version}/{trajectory_id}"
            )
        version_records[version][index] = replacement
        replacements.append(f"{version}/{trajectory_id}")

    excluded_ids = {
        record.get("trajectory_id")
        for records in version_records.values()
        for record in records
        if record.get("failure_type") == "api_error"
    }
    versions = {}
    for version, records in version_records.items():
        result = summarize_records(records, targets, controls, excluded_ids)
        result.update({"version": version, **version_artifacts[version]})
        versions[version] = result

    summary = {
        "cohort": cohort,
        "excluded_infrastructure_ids": sorted(excluded_ids),
        "infrastructure_replacements": replacements,
        "replacement_artifacts": {
            str(path): file_sha256(path) for path in args.replacement_record
        },
        "versions": versions,
        "comparisons": [
            paired(versions["version39"], versions["version46"]),
            paired(versions["version46"], versions["version47"]),
            paired(versions["version47"], versions["version48"]),
            paired(versions["version39"], versions["version48"]),
            paired(versions["version46"], versions["version49"]),
            paired(versions["version48"], versions["version49"]),
            paired(versions["version39"], versions["version49"]),
        ],
    }
    if args.offline_audit:
        offline_summaries = [read_json(path)["summary"] for path in args.offline_audit]

        def weighted_final_mean(key: str) -> float:
            weighted = 0.0
            count = 0
            for item in offline_summaries:
                metric = item["final_turn_context_chars"][key]
                weighted += float(metric["mean"]) * int(metric["count"])
                count += int(metric["count"])
            return weighted / count

        summary["offline_renderer"] = {
            "artifacts": {
                str(path): file_sha256(path) for path in args.offline_audit
            },
            "version46_final_saved_mean": weighted_final_mean("version46_saved"),
            "version47_final_saved_mean": weighted_final_mean("version47_saved"),
            "version48_final_saved_mean": weighted_final_mean("version48_saved"),
            "version49_final_saved_mean": weighted_final_mean("version49_saved"),
            "version47_turns_with_grounding_value_removed": sum(
                item["next_action_visibility"]["version47"]["turns_with_argument_leaf_removed"]
                for item in offline_summaries
            ),
            "version47_grounding_values_removed": sum(
                item["next_action_visibility"]["version47"]["removed_argument_leaf_count"]
                for item in offline_summaries
            ),
            "version49_turns_with_grounding_value_removed": sum(
                item["next_action_visibility"]["version49"]["turns_with_argument_leaf_removed"]
                for item in offline_summaries
            ),
            "version49_grounding_values_removed": sum(
                item["next_action_visibility"]["version49"]["removed_argument_leaf_count"]
                for item in offline_summaries
            ),
        }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_out.write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))


if __name__ == "__main__":
    main()
