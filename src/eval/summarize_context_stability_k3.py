#!/usr/bin/env python3
"""Summarize the paired version39/version49 K=3 context-stability gate."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


ARMS = ("version39", "version49")
RUNS = ("run1", "run2", "run3")
READ_TOOLS = {"read_subtable", "inspect_rows"}
EXPECTED_MANIFEST_FIELDS = {
    "tool_scheme": "atomic",
    "assistant_carrier": "think-json-v1",
    "method": "external_llm_closed_loop",
    "model": "deepseek-v4-flash",
    "context_mode": "rolling-legal-history",
    "history_turns": 4,
    "rolling_prompt_variant": "full",
    "plan_policy": "optional",
    "deepseek_carrier": "json-output",
    "denotation_comparison": "bird-set",
    "database_context_profile": "catalog-v1",
    "attempts_per_example": 1,
    "max_steps": 30,
    "sft_export_eligible": False,
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest: dict, arm: str, artifact_name: str) -> None:
    for field, expected in EXPECTED_MANIFEST_FIELDS.items():
        actual = manifest.get(field)
        if actual != expected:
            raise SystemExit(
                f"{artifact_name}: {field}={actual!r}, expected {expected!r}"
            )
    if manifest.get("protocol_version") != arm:
        raise SystemExit(f"{artifact_name}: protocol version drift")
    if manifest.get("raw_attempt_records") != 8:
        raise SystemExit(f"{artifact_name}: expected eight raw attempt records")
    if manifest.get("unique_examples") != 8:
        raise SystemExit(f"{artifact_name}: expected eight unique examples")
    if manifest.get("duplicate_attempt_records") != 0:
        raise SystemExit(f"{artifact_name}: duplicate attempt records")
    if (manifest.get("counts") or {}).get("total") != 8:
        raise SystemExit(f"{artifact_name}: persisted total is not eight")


def action_key(parsed: dict) -> str:
    return json.dumps(
        {"tool": parsed.get("tool"), "arguments": parsed.get("arguments") or {}},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def summarize_attempt(record: dict) -> dict:
    seen_reads: set[str] = set()
    reads = 0
    exact_rereads = 0
    for turn in record.get("turns") or []:
        parsed = turn.get("parsed")
        if not isinstance(parsed, dict) or parsed.get("tool") not in READ_TOOLS:
            continue
        reads += 1
        key = action_key(parsed)
        if key in seen_reads:
            exact_rereads += 1
        seen_reads.add(key)
    usage = record.get("usage") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    return {
        "trajectory_id": record.get("trajectory_id"),
        "correct": bool(record.get("correct")),
        "legal": bool(record.get("legal")),
        "steps": int(record.get("steps") or 0),
        "errors": int(record.get("errors") or 0),
        "failure_type": record.get("failure_type"),
        "reads": reads,
        "exact_rereads": exact_rereads,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "reasoning_tokens": int(completion_details.get("reasoning_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def summarize_arm(attempts: list[dict], controls: set[str]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for attempt in attempts:
        by_task.setdefault(attempt["trajectory_id"], []).append(attempt)
    task_metrics = {}
    for task_id, items in sorted(by_task.items()):
        task_metrics[task_id] = {
            "attempts": len(items),
            "correct": sum(item["correct"] for item in items),
            "legal": sum(item["legal"] for item in items),
            "max_steps": sum(item["failure_type"] == "max_steps" for item in items),
            "mean_steps": round(statistics.fmean(item["steps"] for item in items), 3),
            "reads": sum(item["reads"] for item in items),
            "max_reads": max(item["reads"] for item in items),
            "exact_rereads": sum(item["exact_rereads"] for item in items),
            "max_exact_rereads": max(item["exact_rereads"] for item in items),
            "tokens": sum(item["total_tokens"] for item in items),
        }
    return {
        "attempts": len(attempts),
        "correct": sum(item["correct"] for item in attempts),
        "legal": sum(item["legal"] for item in attempts),
        "control_correct": sum(
            item["correct"] for item in attempts if item["trajectory_id"] in controls
        ),
        "control_attempts": sum(
            1 for item in attempts if item["trajectory_id"] in controls
        ),
        "steps": sum(item["steps"] for item in attempts),
        "mean_steps": round(statistics.fmean(item["steps"] for item in attempts), 3),
        "errors": sum(item["errors"] for item in attempts),
        "reads": sum(item["reads"] for item in attempts),
        "exact_rereads": sum(item["exact_rereads"] for item in attempts),
        "median_exact_rereads": statistics.median(
            item["exact_rereads"] for item in attempts
        ),
        "max_steps": sum(item["failure_type"] == "max_steps" for item in attempts),
        "failure_types": dict(Counter(
            str(item["failure_type"] or "none") for item in attempts
        )),
        "prompt_tokens": sum(item["prompt_tokens"] for item in attempts),
        "completion_tokens": sum(item["completion_tokens"] for item in attempts),
        "reasoning_tokens": sum(item["reasoning_tokens"] for item in attempts),
        "tokens": sum(item["total_tokens"] for item in attempts),
        "task_metrics": task_metrics,
    }


def markdown(summary: dict) -> str:
    baseline = summary["arms"]["version39"]
    candidate = summary["arms"]["version49"]
    step_delta = (candidate["mean_steps"] / baseline["mean_steps"] - 1.0) * 100.0
    token_delta = (candidate["tokens"] / baseline["tokens"] - 1.0) * 100.0
    read_delta = (candidate["reads"] / baseline["reads"] - 1.0) * 100.0
    baseline_prompt_per_step = baseline["prompt_tokens"] / baseline["steps"]
    candidate_prompt_per_step = candidate["prompt_tokens"] / candidate["steps"]
    prompt_per_step_delta = (
        candidate_prompt_per_step / baseline_prompt_per_step - 1.0
    ) * 100.0
    lines = [
        "# version39 / version49 上下文稳定性 K=3 Gate8",
        "",
        "每个版本包含 3 次独立运行，每题每次恰好一个 semantic attempt；不使用 pass@K、"
        "成功早停或 verifier 选择。指标为 `bird-set`。",
        "本实验验证的是 version49 active/archive renderer 的重复运行稳定性，不是对模型总体"
        "准确率的无偏估计，也不验证由模型自由撰写长期摘要的另一套协议。",
        "",
        "| 版本 | Correct | Legal | Control | Mean steps | Errors | Reads | Exact rereads | Max-step | Tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        item = summary["arms"][arm]
        lines.append(
            f"| {arm} | {item['correct']}/{item['attempts']} | "
            f"{item['legal']}/{item['attempts']} | "
            f"{item['control_correct']}/{item['control_attempts']} | "
            f"{item['mean_steps']:.2f} | {item['errors']} | {item['reads']} | "
            f"{item['exact_rereads']} | {item['max_steps']} | {item['tokens']} |"
        )
    lines.extend([
        "",
        "三轮逐轮正确数完全相同，因此 24 个配对 attempt 为 "
        f"{summary['paired_gains']} gains / {summary['paired_regressions']} regressions。"
        f"但 version49 的平均步数增加 {step_delta:.1f}%，row reads 增加 {read_delta:.1f}%，"
        f"tokens 增加 {token_delta:.1f}%。",
        f"总 prompt tokens 为 {baseline['prompt_tokens']} → {candidate['prompt_tokens']}；"
        f"即使除以语义步骤，仍由 {baseline_prompt_per_step:.0f} 增至 "
        f"{candidate_prompt_per_step:.0f}（{prompt_per_step_delta:+.1f}%）。"
        "因此总成本上升同时来自更长轨迹和未降低的单步上下文。",
        "",
        "## 逐轮",
        "",
        "| Run | v39 correct/legal | v49 correct/legal | v39 steps | v49 steps | v39 reads/rereads | v49 reads/rereads |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in RUNS:
        left = summary["run_metrics"][run]["version39"]
        right = summary["run_metrics"][run]["version49"]
        lines.append(
            f"| {run} | {left['correct']}/8 / {left['legal']}/8 | "
            f"{right['correct']}/8 / {right['legal']}/8 | "
            f"{left['mean_steps']:.2f} | {right['mean_steps']:.2f} | "
            f"{left['reads']}/{left['exact_rereads']} | "
            f"{right['reads']}/{right['exact_rereads']} |"
        )
    lines.extend(["", "## 逐题", "", "| Task | v39 correct | v49 correct | v39 steps | v49 steps | v39 reads/rereads | v49 reads/rereads |", "|---|---:|---:|---:|---:|---:|---:|"])
    task_ids = summary["cohort"]["selection"]["context_sensitive"] + summary["cohort"]["selection"]["controls"]
    for task_id in task_ids:
        left = baseline["task_metrics"][task_id]
        right = candidate["task_metrics"][task_id]
        lines.append(
            f"| {task_id} | {left['correct']}/3 | {right['correct']}/3 | "
            f"{left['mean_steps']:.2f} | {right['mean_steps']:.2f} | "
            f"{left['reads']}/{left['exact_rereads']} | "
            f"{right['reads']}/{right['exact_rereads']} |"
        )
    checks = summary["acceptance_checks"]
    lines.extend(["", "## 预注册条件", ""])
    for name, check in checks.items():
        lines.append(
            f"- {'PASS' if check['passed'] else 'FAIL'} `{name}`：{check['detail']}"
        )
    lines.extend([
        "",
        "## 轨迹审计",
        "",
        "- `bird_train_02918`：version39 三轮 reads 为 2/4/2，version49 为 7/5/20。"
        "version49 第三轮有 5 次 exact reread，并在部门历史与人员信用卡映射之间反复读取，"
        "没有推进到信用卡到期年份过滤，最终 max-steps。",
        "- `bird_train_06299`：两个版本都是 0/3，说明 active/archive 不是该题失败的唯一原因。"
        "但 version39 三轮均未 max-steps；version49 后两轮 reads 为 22/18，均跑满 30 步，"
        "反复重建同一菜单过滤/连接分支。归档策略放大了既有求解困难。",
        "- 五个稳定 controls 在两边均为 15/15。当前证据支持“简单任务行为保持”，"
        "不支持“困难长程任务上下文更可控”。",
        "",
        "## 基础设施说明",
        "",
        "version39/run1 的 `bird_train_06299` 首个 worker 请求长期无返回且未写入 terminal record；"
        "主进程中断后仅以 `--resume --start 2 --limit 1` 补齐该缺失记录。原请求没有可选择的"
        "语义结果，因此未形成 pass@K 或 verifier 选择。",
    ])
    lines.extend([
        "",
        "## 决策",
        "",
        summary["decision"],
        "",
        "全部输出均为 `diagnostic_only_pending_protocol_scale_gate`，不进入 SFT/RL。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    cohort = read_json(args.cohort_manifest)
    expected_ids = set(cohort["selection"]["context_sensitive"]) | set(
        cohort["selection"]["controls"]
    )
    controls = set(cohort["selection"]["controls"])
    artifacts = {}
    arm_attempts = {arm: [] for arm in ARMS}
    run_metrics = {}
    paired_attempts = []
    for run in RUNS:
        run_records = {}
        run_metrics[run] = {}
        for arm in ARMS:
            all_path = args.root / arm / run / "verified.all.jsonl"
            manifest_path = args.root / arm / run / "verified.manifest.json"
            if not all_path.is_file() or not manifest_path.is_file():
                raise SystemExit(f"missing complete artifact: {all_path}")
            records = read_jsonl(all_path)
            if len(records) != len(expected_ids):
                raise SystemExit(f"{arm}/{run} has {len(records)} records")
            by_id = {record.get("trajectory_id"): record for record in records}
            if set(by_id) != expected_ids:
                raise SystemExit(f"{arm}/{run} ids do not match frozen cohort")
            run_records[arm] = by_id
            manifest = read_json(manifest_path)
            validate_manifest(manifest, arm, f"{arm}/{run}")
            artifacts[f"{arm}/{run}"] = {
                "all_path": str(all_path),
                "all_sha256": sha256(all_path),
                "manifest_path": str(manifest_path),
                "manifest_sha256": sha256(manifest_path),
                "manifest": manifest,
            }
            summarized = [summarize_attempt(record) for record in records]
            arm_attempts[arm].extend(summarized)
            run_metrics[run][arm] = summarize_arm(summarized, controls)
        for task_id in sorted(expected_ids):
            left = bool(run_records["version39"][task_id].get("correct"))
            right = bool(run_records["version49"][task_id].get("correct"))
            paired_attempts.append({
                "run": run,
                "trajectory_id": task_id,
                "version39_correct": left,
                "version49_correct": right,
            })

    arms = {
        arm: summarize_arm(arm_attempts[arm], controls)
        for arm in ARMS
    }
    baseline = arms["version39"]
    candidate = arms["version49"]
    prompt_fields = (
        "teacher_canonical_prompt_sha256",
        "teacher_provider_prompt_sha256",
        "student_runtime_prompt_sha256",
        "tool_schema_sha256",
        "model_visible_tool_schema_sha256",
    )
    for field in prompt_fields:
        values = {
            artifact["manifest"].get(field)
            for artifact in artifacts.values()
        }
        if len(values) != 1 or None in values:
            raise SystemExit(f"prompt/tool contract drift in {field}: {values}")
    v49_tasks = candidate["task_metrics"]
    checks = {
        "accuracy_non_regression": {
            "passed": candidate["correct"] >= baseline["correct"],
            "detail": f"{candidate['correct']}/24 versus {baseline['correct']}/24",
        },
        "control_non_regression": {
            "passed": candidate["control_correct"] >= baseline["control_correct"],
            "detail": (
                f"{candidate['control_correct']}/15 versus "
                f"{baseline['control_correct']}/15"
            ),
        },
        "legal_at_least_95_percent": {
            "passed": candidate["legal"] >= 23,
            "detail": f"{candidate['legal']}/24",
        },
        "median_exact_rereads_zero": {
            "passed": candidate["median_exact_rereads"] == 0,
            "detail": str(candidate["median_exact_rereads"]),
        },
        "mean_steps_within_10_percent": {
            "passed": candidate["mean_steps"] <= baseline["mean_steps"] * 1.10,
            "detail": f"{candidate['mean_steps']:.3f} versus {baseline['mean_steps']:.3f}",
        },
        "tokens_within_10_percent": {
            "passed": candidate["tokens"] <= baseline["tokens"] * 1.10,
            "detail": f"{candidate['tokens']} versus {baseline['tokens']}",
        },
        "churn_tasks_no_max_steps": {
            "passed": all(v49_tasks[item]["max_steps"] == 0 for item in ("bird_train_05873", "bird_train_02918")),
            "detail": (
                f"05873={v49_tasks['bird_train_05873']['max_steps']}, "
                f"02918={v49_tasks['bird_train_02918']['max_steps']}"
            ),
        },
        "churn_tasks_at_least_two_of_three_correct": {
            "passed": all(v49_tasks[item]["correct"] >= 2 for item in ("bird_train_05873", "bird_train_02918")),
            "detail": (
                f"05873={v49_tasks['bird_train_05873']['correct']}/3, "
                f"02918={v49_tasks['bird_train_02918']['correct']}/3"
            ),
        },
        "dormant_branch_bounded_reads": {
            "passed": (
                v49_tasks["bird_train_06299"]["max_reads"] <= 15
                and v49_tasks["bird_train_06299"]["max_exact_rereads"] <= 2
            ),
            "detail": (
                f"max reads={v49_tasks['bird_train_06299']['max_reads']}, "
                f"max exact rereads={v49_tasks['bird_train_06299']['max_exact_rereads']}"
            ),
        },
    }
    passed = all(item["passed"] for item in checks.values())
    decision = (
        "version49 通过全部稳定性条件，可保留为工程 context renderer；这仍不构成准确率推广。"
        if passed
        else "version49 未通过预注册稳定性条件；不应替换 version39，也不继续叠加新的语义/终止改动。"
    )
    summary = {
        "schema_version": "context-stability-k3-summary-v1",
        "cohort": cohort,
        "artifacts": artifacts,
        "arms": arms,
        "run_metrics": run_metrics,
        "validation": {
            "artifact_manifests": len(artifacts),
            "raw_attempt_records": sum(
                artifact["manifest"]["raw_attempt_records"]
                for artifact in artifacts.values()
            ),
            "prompt_and_public_schema_hashes_identical": True,
            "no_duplicate_attempt_records": True,
            "diagnostic_only": True,
        },
        "paired_attempts": paired_attempts,
        "paired_gains": sum(
            not item["version39_correct"] and item["version49_correct"]
            for item in paired_attempts
        ),
        "paired_regressions": sum(
            item["version39_correct"] and not item["version49_correct"]
            for item in paired_attempts
        ),
        "acceptance_checks": checks,
        "passed": passed,
        "decision": decision,
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
