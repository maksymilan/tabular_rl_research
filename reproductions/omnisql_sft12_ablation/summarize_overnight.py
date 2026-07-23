#!/usr/bin/env python3
"""Write a compact morning summary from completed overnight artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pass1(summary: dict) -> tuple[int, int]:
    value = summary["pass_at"]["1"]
    return int(value["correct"]), int(value["total"])


def pct(correct: int, total: int) -> float:
    return 100.0 * correct / total


def main() -> int:
    q_tool = pass1(
        load(
            ROOT
            / "results/tool/qwen25-coder_sft_fixed200_tool_output_only_strict_multiset"
            / "summary.json"
        )
    )
    o_tool = pass1(
        load(
            ROOT
            / "results/tool/omnisql_sft_fixed200_tool_output_only_strict_multiset"
            / "summary.json"
        )
    )
    q_direct = pass1(
        load(ROOT / "results/direct_sql/qwen25_coder_sft_greedy_bird_set/summary.json")
    )
    omni_log = (
        ROOT / "results/omnisql_direct_sql/bird_dev1534/greedy_evaluation.log"
    ).read_text(encoding="utf-8", errors="replace")
    match = re.search(r"EX Accuracy \(greedy search\):\s*([0-9.]+)", omni_log)
    if not match:
        raise ValueError("could not parse OmniSQL official EX accuracy")
    omni_rate = float(match.group(1))
    omni_correct = round(omni_rate * 1534)

    report = f"""# OmniSQL / Qwen2.5-Coder 1K Tool-SFT Overnight Results

Protocol: `v2i-state-only-join-feedback-r2`; training: identical 1,024 records and identical
two-epoch QLoRA configuration.

| Model | Historical tool base | Historical tool after SFT | Direct SQL base | Direct SQL after SFT | Direct delta |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B-Instruct | 0/200 = 0.00% | {q_tool[0]}/{q_tool[1]} = {pct(*q_tool):.2f}% | 748/1534 = 48.76% | {q_direct[0]}/{q_direct[1]} = {pct(*q_direct):.2f}% | {pct(*q_direct) - 48.7614:+.2f} pp |
| OmniSQL-7B | stopped at user request | {o_tool[0]}/{o_tool[1]} = {pct(*o_tool):.2f}% | 983/1534 = 64.08% | {omni_correct}/1534 = {100 * omni_rate:.2f}% | {100 * omni_rate - 64.0808:+.2f} pp |

Metrics are intentionally separate: historical tool evaluation is greedy
`tool-output-only + strict-multiset` on the frozen difficulty-stratified 200 cohort. It grades only
the exact harness-owned result table cited by `answer_from_context`; model-authored answer values
and column permutation fallbacks are disabled. Direct SQL is greedy BIRD reference EX (`bird-set`)
on all 1,534 dev tasks. Direct-SQL deltas measure capability retention; tool scores measure
interface acquisition.
"""
    out = ROOT / "results/OVERNIGHT_SUMMARY.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
