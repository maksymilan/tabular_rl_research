#!/usr/bin/env python3
"""Use an external LLM to rewrite recovery trajectory reasoning without changing actions.

Input trajectories should already be harness-verified. This script lets the external model improve
only `steps[*].think`; tool calls, tool outputs, final answers, step ids, and provenance are copied
unchanged. This keeps semantic data generation separated from execution authority.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from fill_think import call, load_api  # noqa: E402

FIRST_PERSON = re.compile(r"\b(I|my|me|I'll|I'm|I will|I need|I see|I should|I can|I now)\b", re.I)
BANNED = re.compile(
    r"\b(the model|a model|the agent|the assistant|gold sql|gold answer|verified answer|"
    r"clean trajectory|ground truth|dataset label|external model|annotator|already-executed|"
    r"failed prefix|failed path|verified path|verified source|corrected path|corrected evidence|"
    r"corrected|prior attempt metadata|marked as wrong|wrong answer|expected answer|known-wrong|"
    r"not successful|successful path)\b",
    re.I,
)
TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.S)
OBSERVATION_WORDS = re.compile(
    r"\b(observation|observed|shows?|showed|returned|result|rows?|values?|column|schema|"
    r"empty|0 rows|count|distinct|domain|error|mismatch|not support|does not support|rather than|"
    r"instead of|missing|irrelevant|cardinality)\b",
    re.I,
)
PIVOT_WORDS = re.compile(
    r"\b(does not support|not support|mismatch|rather than|instead|not match|not enough|"
    r"should go back|try another|choose another|different|assumption|revise|reconsider)\b",
    re.I,
)
NON_RECOVERY_WORDS = re.compile(
    r"\b(no evidence of failure|no mismatch|directly addresses|continue with (?:the )?current|"
    r"proceed with (?:this|the current|the result)|does not change my plan|working as intended|"
    r"seems plausible|already gives? me|already gave me)\b",
    re.I,
)

SYSTEM = """You rewrite reasoning for an online table-tool agent trajectory.

Role: write as the table-tool agent itself, in first person, one concise <think> string per step.
The agent is solving the question step by step. It does NOT know a gold route and it is NOT editing
a completed answer after the fact.

Hard constraints:
- Do NOT change, add, remove, or reorder tool calls. You only write reasoning text.
- Do NOT mention gold SQL, verified trajectories, labels, external annotation, or ground truth.
- Do NOT say the path is "failed", "verified", "corrected", "wrong answer", "expected answer",
  "prior attempt", or "ground truth".
- Do NOT copy unsupported claims from an earlier attempt. Use them only to understand what the
  agent might have been assuming at that moment.
- Do NOT claim knowledge of a table/column before the corresponding observation has shown it.
- For early exploratory steps, explain why the action was plausible from the catalog/question at that
  time. Do not reveal that it will later be abandoned.
- At the recovery pivot, the agent must notice a LOCAL evidence mismatch from the latest observation:
  empty result, irrelevant value domain, missing requested column/concept, wrong cardinality, or an
  execution/protocol error. The pivot should sound like:
  "This observation does not support my earlier assumption: the values I saw are rankings, while the
  question asks for acting status. I should go back to the schema evidence and try another source."
- For later steps, explain why the next source/table/column/tool follows from the observed schema or
  result, not because it is known to be right.
- For final answer_from_context, explain how the cited evidence/result answers the question.

Return ONLY JSON:
{
  "pivot_step_id": "step_3",
  "unsupported_assumption": "I was treating Rank as the status concept.",
  "pivot_evidence": "The observed Rank values are numeric rankings, not status labels.",
  "next_strategy": "Go back to schema evidence and inspect a table/column that actually represents status.",
  "thinks": ["...", "..."]
}
The `thinks` array length must equal the number of steps.
`pivot_step_id` is the step where the agent first has enough observation evidence to revise its
assumption. If the trajectory only contains a weak recovery, still choose the most local evidence
step and explain the limitation honestly.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def compact(value: Any, limit: int = 1200) -> Any:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + "...<truncated>"


def failed_model_reason(step: dict[str, Any]) -> str:
    text = ((step.get("source_rollout") or {}).get("model_output") or "").strip()
    if not text:
        return ""
    text = TOOL_CALL_RE.sub("", text).replace("<think>", "").replace("</think>", "")
    return re.sub(r"\s+", " ", text).strip()[:1200]


def step_brief(step: dict[str, Any]) -> dict[str, Any]:
    call = step.get("tool_call") or {}
    item = {
        "step_id": step.get("step_id"),
        "tool": call.get("tool"),
        "arguments": call.get("arguments"),
        "status": step.get("tool_status", "success"),
        "output": compact(step.get("tool_output"), 1800),
        "current_think": step.get("think", ""),
    }
    original = failed_model_reason(step)
    if original:
        item["earlier_reason_do_not_copy"] = original
    return item


def build_messages(traj: dict[str, Any], feedback: str = "") -> list[dict[str, str]]:
    source_candidate = (traj.get("enrichment") or {}).get("source_candidate", {})
    payload = {
        "trajectory_id": traj.get("trajectory_id"),
        "question": traj.get("question"),
        "opening_catalog": compact(traj.get("initial_state", {}).get("dataset_overview"), 2000),
        "structure_only_hint": compact({
            "selection_method": source_candidate.get("selection_method"),
            "structural_hint": source_candidate.get("structural_hint"),
        }, 1800),
        "steps": [step_brief(step) for step in traj.get("steps", [])],
    }
    user = (
        "Rewrite the reasoning as if it was produced online while the agent observes each step. "
        "Preserve the exact number of steps and write one first-person think for each step. The "
        "reasoning should show the agent noticing evidence that weakens an earlier assumption and "
        "then choosing a different next action; it must not sound like an annotator rewriting a "
        "known-wrong trajectory.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )
    if feedback:
        user += f"\n\nPrevious output was rejected for:\n{feedback}\nPlease fix and return JSON only."
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_payload(text: str, n_steps: int) -> dict[str, Any] | None:
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    thinks = payload.get("thinks") if isinstance(payload, dict) else payload
    if not isinstance(thinks, list) or len(thinks) != n_steps:
        return None
    if not all(isinstance(item, str) and item.strip() for item in thinks):
        return None
    if not isinstance(payload, dict):
        payload = {"thinks": thinks}
    payload["thinks"] = [re.sub(r"\s+", " ", item).strip() for item in thinks]
    return payload


def step_index_by_id(traj: dict[str, Any], step_id: str) -> int | None:
    for index, step in enumerate(traj.get("steps", []), 1):
        if step.get("step_id") == step_id:
            return index
    return None


def validate_payload(payload: dict[str, Any], traj: dict[str, Any]) -> tuple[list[str], list[str]]:
    thinks = payload["thinks"]
    issues = []
    warnings = []
    for index, think in enumerate(thinks, 1):
        if not FIRST_PERSON.search(think):
            issues.append(f"step {index}: not first-person")
        if BANNED.search(think):
            issues.append(f"step {index}: contains banned annotator/gold wording")
        if "<tool_call>" in think or "</tool_call>" in think:
            issues.append(f"step {index}: contains tool_call markup")
    if len(" ".join(thinks)) < 80:
        issues.append("all thinks are suspiciously short")

    pivot_step_id = str(payload.get("pivot_step_id") or "").strip()
    pivot_index = step_index_by_id(traj, pivot_step_id) if pivot_step_id else None
    if not pivot_step_id:
        issues.append("missing pivot_step_id")
    elif pivot_index is None:
        issues.append(f"pivot_step_id {pivot_step_id!r} does not match a step_id")
    else:
        pivot_tool = (traj["steps"][pivot_index - 1].get("tool_call") or {}).get("tool")
        if pivot_tool == "answer_from_context":
            issues.append("pivot_step_id points to final answer step")
        pivot_think = thinks[pivot_index - 1]
        diag_text = " ".join(
            str(payload.get(key, ""))
            for key in ("unsupported_assumption", "pivot_evidence", "next_strategy")
        )
        if not OBSERVATION_WORDS.search(pivot_think + " " + diag_text):
            warnings.append(f"{pivot_step_id}: pivot is weakly grounded in observation text")
        if not PIVOT_WORDS.search(pivot_think + " " + diag_text):
            warnings.append(f"{pivot_step_id}: pivot does not clearly revise an assumption")
        if NON_RECOVERY_WORDS.search(pivot_think + " " + diag_text):
            warnings.append(f"{pivot_step_id}: pivot continues a plausible path rather than recovering")

    for key in ("unsupported_assumption", "pivot_evidence", "next_strategy"):
        value = str(payload.get(key) or "").strip()
        if not value:
            issues.append(f"missing {key}")
        elif BANNED.search(value):
            issues.append(f"{key}: contains banned annotator/gold wording")

    failed_prefix_steps = int((traj.get("enrichment") or {}).get("n_failed_prefix_steps") or 0)
    if pivot_index is not None and failed_prefix_steps and pivot_index > failed_prefix_steps + 2:
        warnings.append(
            f"{pivot_step_id}: pivot is late for failed_prefix={failed_prefix_steps}; check whether "
            "the recovery sounds post-hoc"
        )
    return issues, warnings


def rewrite_one(
    traj: dict[str, Any],
    *,
    base: str,
    key: str,
    model: str,
    max_attempts: int,
    api_retries: int,
    api_timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rejected = []
    attempts = []
    feedback = ""
    n_steps = len(traj.get("steps", []))
    for attempt in range(1, max_attempts + 1):
        last_error = ""
        for api_try in range(1, api_retries + 1):
            try:
                text, usage = call(base, key, model, build_messages(traj, feedback), timeout=api_timeout)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                if api_try < api_retries:
                    time.sleep(min(2 * api_try, 8))
        else:
            attempts.append({"attempt": attempt, "stage": "api", "status": "rejected", "error": last_error})
            rejected.append({"attempt": attempt, "stage": "api", "error": last_error})
            feedback = f"API failed: {last_error}"
            continue

        payload = parse_payload(text, n_steps)
        if payload is None:
            feedback = "output is not JSON {'thinks': [...]} with the exact step count"
            item = {
                "attempt": attempt,
                "stage": "parse",
                "status": "rejected",
                "issues": feedback,
                "model_output": text,
                "usage": usage,
            }
            attempts.append(item)
            rejected.append(item)
            continue
        issues, warnings = validate_payload(payload, traj)
        if issues:
            feedback = "; ".join(issues)
            item = {
                "attempt": attempt,
                "stage": "validate",
                "status": "rejected",
                "issues": issues,
                "warnings": warnings,
                "model_output": text,
                "usage": usage,
            }
            attempts.append(item)
            rejected.append(item)
            continue

        out = copy.deepcopy(traj)
        for step, think in zip(out["steps"], payload["thinks"], strict=True):
            step["think"] = think
        enr = out.setdefault("enrichment", {})
        prior_generator = copy.deepcopy(enr.get("generator"))
        quality_status = "review" if warnings else "ready"
        enr["quality_status"] = quality_status
        enr["semantic_rewrite"] = {
            "status": "rewritten",
            "quality_status": quality_status,
            "model": model,
            "attempt": attempt,
            "usage": usage,
            "prior_generator": prior_generator,
            "pivot": {
                "step_id": payload.get("pivot_step_id"),
                "unsupported_assumption": payload.get("unsupported_assumption"),
                "pivot_evidence": payload.get("pivot_evidence"),
                "next_strategy": payload.get("next_strategy"),
                "warnings": warnings,
            },
            "accepted_model_output": text,
            "rejected_candidates": rejected,
            "attempt_history": attempts + [{
                "attempt": attempt,
                "stage": "accepted",
                "status": "accepted",
                "warnings": warnings,
                "model_output": text,
                "usage": usage,
            }],
        }
        enr["generator"] = {"model": model, "mode_requested": "recovery_think_rewrite"}
        return out, {
            "status": "rewritten",
            "quality_status": quality_status,
            "attempt": attempt,
            "rejections": len(rejected),
            "warnings": warnings,
            "attempt_history": enr["semantic_rewrite"]["attempt_history"],
        }

    out = copy.deepcopy(traj)
    enr = out.setdefault("enrichment", {})
    enr["semantic_rewrite"] = {
        "status": "fallback_original_think",
        "quality_status": "reject",
        "model": model,
        "rejected_candidates": rejected,
        "attempt_history": attempts,
    }
    enr["quality_status"] = "reject"
    return out, {
        "status": "fallback",
        "quality_status": "reject",
        "rejections": len(rejected),
        "attempt_history": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-timeout", type=int, default=240)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--audit-output",
        default=None,
        help="Optional JSONL path storing every external-model attempt and validator decision.",
    )
    args = parser.parse_args()

    key, base = load_api()
    records = load_jsonl(Path(args.input))
    if args.limit:
        records = records[: args.limit]

    if args.workers <= 0:
        parser.error("--workers must be positive")

    results: list[dict[str, Any] | None] = [None] * len(records)
    audits: list[dict[str, Any] | None] = [None] * len(records)
    statuses = collections.Counter()
    quality_statuses = collections.Counter()
    warning_counts = collections.Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                rewrite_one,
                traj,
                base=base,
                key=key,
                model=args.model,
                max_attempts=args.max_attempts,
                api_retries=args.api_retries,
                api_timeout=args.api_timeout,
            ): index
            for index, traj in enumerate(records)
        }
        for done_count, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            traj = records[index]
            rewritten, meta = future.result()
            results[index] = rewritten
            statuses[meta["status"]] += 1
            quality_statuses[meta.get("quality_status", "unknown")] += 1
            for warning in meta.get("warnings", []) or []:
                warning_counts[warning.split(":", 1)[-1].strip()] += 1
            audits[index] = {
                "trajectory_id": traj.get("trajectory_id"),
                "status": meta["status"],
                "attempt": meta.get("attempt"),
                "rejections": meta["rejections"],
                "attempt_history": meta.get("attempt_history", []),
            }
            print(
                f"[{done_count}/{len(records)}] {traj.get('trajectory_id')} "
                f"{meta['status']} attempts={meta.get('attempt', '-')} rejects={meta['rejections']} "
                f"warnings={len(meta.get('warnings', []) or [])}",
                flush=True,
            )

    output = Path(args.output)
    write_jsonl(output, [item for item in results if item is not None])
    summary = {
        "input": args.input,
        "output": str(output),
        "model": args.model,
        "records": len(results),
        "status_counts": dict(statuses),
        "quality_status_counts": dict(quality_statuses),
        "warning_counts": dict(warning_counts),
        "max_attempts": args.max_attempts,
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_path = Path(args.audit_output) if args.audit_output else output.with_suffix(output.suffix + ".audit.jsonl")
    write_jsonl(audit_path, [item for item in audits if item is not None])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {output}")
    print(f"-> {summary_path}")
    print(f"-> {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
