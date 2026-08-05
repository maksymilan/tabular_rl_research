#!/usr/bin/env python3
"""Build a two-axis classification of version24 fixed-200 failures.

Task type describes the dominant operation requested by the question. Root
cause describes the primary audited reason that the version24 trajectory did
not pass. Both axes are mutually exclusive by design so their counts sum to
the full failure set.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TASK_TYPES: dict[str, tuple[str, ...]] = {
    "ranking_extreme_topk": (
        "bird_train_00041",
        "bird_train_00214",
        "bird_train_00796",
        "bird_train_01152",
        "bird_train_01213",
        "bird_train_01569",
        "bird_train_02052",
        "bird_train_02418",
        "bird_train_02512",
        "bird_train_02925",
        "bird_train_03042",
        "bird_train_03664",
        "bird_train_03688",
        "bird_train_03724",
        "bird_train_03969",
        "bird_train_04848",
        "bird_train_05316",
        "bird_train_05440",
        "bird_train_06324",
    ),
    "conditional_lookup_listing": (
        "bird_train_00040",
        "bird_train_00582",
        "bird_train_00715",
        "bird_train_00886",
        "bird_train_01461",
        "bird_train_02078",
        "bird_train_02438",
        "bird_train_02868",
        "bird_train_03262",
        "bird_train_04426",
        "bird_train_04822",
        "bird_train_06026",
        "bird_train_06165",
    ),
    "count_sum_aggregation": (
        "bird_train_01167",
        "bird_train_01589",
        "bird_train_02189",
        "bird_train_03636",
        "bird_train_03977",
        "bird_train_05544",
        "bird_train_06246",
        "bird_train_06299",
        "bird_train_06492",
    ),
    "percentage_difference_temporal_arithmetic": (
        "bird_train_00004",
        "bird_train_00593",
        "bird_train_01530",
        "bird_train_01692",
        "bird_train_01888",
        "bird_train_02088",
        "bird_train_04869",
        "bird_train_05147",
        "bird_train_05161",
        "bird_train_05632",
    ),
    "compound_analytic_comparison": (
        "bird_train_00074",
        "bird_train_01560",
        "bird_train_04038",
        "bird_train_05668",
    ),
}

ROOT_CAUSES: dict[str, tuple[str, ...]] = {
    "benchmark_convention_or_question_ambiguity": (
        "bird_train_00040",
        "bird_train_00214",
        "bird_train_00715",
        "bird_train_00796",
        "bird_train_01152",
        "bird_train_01213",
        "bird_train_01589",
        "bird_train_01888",
        "bird_train_02925",
        "bird_train_03688",
        "bird_train_04038",
        "bird_train_04822",
        "bird_train_04848",
        "bird_train_05632",
        "bird_train_06026",
        "bird_train_06165",
        "bird_train_06492",
    ),
    "population_grain_join_or_multiplicity": (
        "bird_train_00041",
        "bird_train_00074",
        "bird_train_00593",
        "bird_train_01167",
        "bird_train_01530",
        "bird_train_01692",
        "bird_train_02052",
        "bird_train_02078",
        "bird_train_02088",
        "bird_train_02189",
        "bird_train_02418",
        "bird_train_03977",
        "bird_train_04869",
        "bird_train_05147",
        "bird_train_05544",
    ),
    "answer_slots_or_output_representation": (
        "bird_train_00582",
        "bird_train_01569",
        "bird_train_02512",
        "bird_train_02868",
        "bird_train_03042",
        "bird_train_03262",
        "bird_train_03724",
        "bird_train_03969",
        "bird_train_04426",
        "bird_train_05316",
        "bird_train_05668",
    ),
    "semantic_mapping_operator_formula_or_literal": (
        "bird_train_00004",
        "bird_train_00886",
        "bird_train_01461",
        "bird_train_01560",
        "bird_train_03664",
        "bird_train_05161",
        "bird_train_05440",
        "bird_train_06246",
        "bird_train_06324",
    ),
    "tool_protocol_or_nontermination": (
        "bird_train_02438",
        "bird_train_03636",
        "bird_train_06299",
    ),
}

TASK_TYPE_ZH = {
    "ranking_extreme_topk": "排序、极值与 Top-K",
    "conditional_lookup_listing": "条件检索与属性列表",
    "count_sum_aggregation": "计数、求和与聚合",
    "percentage_difference_temporal_arithmetic": "百分比、差值、时间与算术",
    "compound_analytic_comparison": "复合分析与组间比较",
}

ROOT_CAUSE_ZH = {
    "benchmark_convention_or_question_ambiguity": "问题表述或基准约定不充分",
    "population_grain_join_or_multiplicity": "总体、粒度、连接或重复度错误",
    "answer_slots_or_output_representation": "答案字段或输出表示错误",
    "semantic_mapping_operator_formula_or_literal": "语义映射、算子、公式或字面量错误",
    "tool_protocol_or_nontermination": "工具协议、执行恢复或未终止",
}

MISMATCH_ZH = {
    "same_shape_numeric_mismatch": "同形状数值错误",
    "extra_columns": "多余列",
    "same_width_row_count_differs": "同列宽但行数不同",
    "same_shape_mixed_mismatch": "同形状混合值错误",
    "too_few_columns": "缺少列",
    "same_shape_text_mismatch": "同形状文本错误",
    "empty_prediction": "空预测",
    "visible_sample_equal_denotation_differs": "可见样例相同但完整集合不同",
    None: "无可评分终态",
}


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            records.extend(json.loads(line) for line in source if line.strip())
    return records


def invert(groups: dict[str, tuple[str, ...]], name: str) -> dict[str, str]:
    inverted: dict[str, str] = {}
    for label, task_ids in groups.items():
        for task_id in task_ids:
            if task_id in inverted:
                raise ValueError(f"{task_id} has duplicate {name} labels")
            inverted[task_id] = label
    return inverted


def percentage(count: int, total: int) -> str:
    return f"{count / total * 100:.1f}%"


def markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    total = len(rows)
    task_counts = Counter(row["task_type"] for row in rows)
    cause_counts = Counter(row["root_cause"] for row in rows)
    mismatch_counts = Counter(row["visible_mismatch"] for row in rows)
    lookup_labels = {"ranking_extreme_topk", "conditional_lookup_listing"}
    numeric_labels = {
        "count_sum_aggregation",
        "percentage_difference_temporal_arithmetic",
        "compound_analytic_comparison",
    }
    lookup_rows = [row for row in rows if row["task_type"] in lookup_labels]
    numeric_rows = [row for row in rows if row["task_type"] in numeric_labels]
    legal_count = sum(row["legal"] for row in rows)
    zero_error_count = sum(row["process_error_count"] == 0 for row in rows)

    lines = [
        "# Version24 fixed-200 的 55 条失败任务分类",
        "",
        "日期：2026-07-30",
        "",
        "## 口径",
        "",
        (
            "本报告分析 version24 在 frozen fixed-200、`bird-set` 指标下的 55 条失败。"
            "“任务类型”描述题目主要要求完成的操作；“失败主因”描述该次轨迹未通过的"
            "首要原因。两套标签分别互斥，因此每张主表都严格合计为 55。复合题按最能"
            "决定答案的操作归入一个主类型。失败主因是基于问题、外部知识、当次动作路径、"
            "结构化错误和终态差异做的人工审计假设，不等同于已经完成因果干预验证。"
        ),
        "",
        "## 核心结论",
        "",
        (
            f"- 检索、列表、排序与极值类共 **{len(lookup_rows)}/{total} = "
            f"{percentage(len(lookup_rows), total)}**；数值聚合、算术和复合分析类共 "
            f"**{len(numeric_rows)}/{total} = {percentage(len(numeric_rows), total)}**。"
        ),
        (
            f"- **{legal_count}/{total}** 条失败轨迹已经合法终止，"
            f"**{zero_error_count}/{total}** 条没有任何过程错误；"
            "所以主要瓶颈不是模型不会调用工具，而是合法路径中的语义、粒度和输出选择。"
        ),
        (
            "- 23 条数值/分析题中，11 条主因是总体、分母、粒度、连接或重复度；"
            "真正归为语义/公式映射的只有 4 条，工具或未终止 2 条。"
            "因此不能把同形状数值错误简单归因于算术能力。"
        ),
        (
            "- 32 条检索/排序题中，12 条是问题或基准隐含约定，10 条是答案字段或"
            "输出表示；这两类合计 22 条，明显多于关系粒度错误。"
        ),
        "",
        "## 任务类型",
        "",
        "| 任务主类型 | 数量 | 占 55 条比例 |",
        "| --- | ---: | ---: |",
    ]
    for label in TASK_TYPES:
        count = task_counts[label]
        lines.append(f"| {TASK_TYPE_ZH[label]} | {count} | {percentage(count, total)} |")
    lines.extend(
        [
            f"| **合计** | **{total}** | **100.0%** |",
            "",
            "任务类型说明：",
            "",
            "- 排序、极值与 Top-K：最高、最低、最新、最老、最多、前 N 等实体选择。",
            "- 条件检索与属性列表：按条件过滤并返回名称、字段或记录列表。",
            "- 计数、求和与聚合：COUNT、SUM、分组计数以及总量。",
            "- 百分比、差值、时间与算术：需要在一个或多个已计算量之上继续运算。",
            "- 复合分析与组间比较：均值阈值、相关性式比较、分组分布或“差值后再取极值”。",
            "",
            "## 失败主因",
            "",
            "| 失败主因 | 数量 | 占 55 条比例 |",
            "| --- | ---: | ---: |",
        ]
    )
    for label in ROOT_CAUSES:
        count = cause_counts[label]
        lines.append(f"| {ROOT_CAUSE_ZH[label]} | {count} | {percentage(count, total)} |")
    lines.extend(
        [
            f"| **合计** | **{total}** | **100.0%** |",
            "",
            "主因解释：",
            "",
            (
                "- 问题表述或基准约定不充分：模型给出的解释在公开问题文字下具有合理性，"
                "但与参考答案采用的列、并列处理、去重或隐含约定不同。此标签不表示可以"
                "事后修改指标。"
            ),
            (
                "- 总体、粒度、连接或重复度错误：分母、计数单位、distinct、连接键、"
                "一对多展开或集合对齐错误。"
            ),
            (
                "- 答案字段或输出表示错误：实体/行集基本正确，但多列、少列、拼接字段、"
                "携带排序辅助列或返回 ID/标签的表示不符。"
            ),
            (
                "- 语义映射、算子、公式或字面量错误：错误理解外部知识、自然语言谓词、"
                "目标表/列、计算公式或过滤值。"
            ),
            (
                "- 工具协议、执行恢复或未终止：没有形成可评分的合法终态，包括参数验证"
                "耗尽、重复非标量引用和达到最大动作数。"
            ),
            "",
            "## 任务类型 × 失败主因",
            "",
        ]
    )
    cause_order = list(ROOT_CAUSES)
    lines.append(
        "| 任务类型 | "
        + " | ".join(ROOT_CAUSE_ZH[label] for label in cause_order)
        + " | 合计 |"
    )
    lines.append("| --- | " + " | ".join("---:" for _ in cause_order) + " | ---: |")
    for task_label in TASK_TYPES:
        cells = [
            sum(
                row["task_type"] == task_label and row["root_cause"] == cause_label
                for row in rows
            )
            for cause_label in cause_order
        ]
        lines.append(
            f"| {TASK_TYPE_ZH[task_label]} | "
            + " | ".join(str(value) for value in cells)
            + f" | {sum(cells)} |"
        )
    lines.append(
        "| **合计** | "
        + " | ".join(f"**{cause_counts[label]}**" for label in cause_order)
        + f" | **{total}** |"
    )
    lines.extend(
        [
            "",
            "## 可观察的终态差异",
            "",
            (
                "这张表只描述预测结果与参考结果的外观差异，不把外观差异直接当成根因。"
                "例如“同形状数值错误”既可能来自算术，也可能来自错误分母或连接重复。"
            ),
            "",
            "| 终态差异 | 数量 |",
            "| --- | ---: |",
        ]
    )
    mismatch_order = [
        "same_shape_numeric_mismatch",
        "extra_columns",
        "same_width_row_count_differs",
        "same_shape_mixed_mismatch",
        "too_few_columns",
        "same_shape_text_mismatch",
        None,
        "empty_prediction",
        "visible_sample_equal_denotation_differs",
    ]
    for mismatch in mismatch_order:
        count = mismatch_counts[mismatch]
        if count:
            lines.append(f"| {MISMATCH_ZH[mismatch]} | {count} |")
    lines.extend(
        [
            f"| **合计** | **{total}** |",
            "",
            "## 失败轨迹中的工具路径特征",
            "",
            "下列特征允许重叠，只描述 55 条失败轨迹是否曾使用相应工具：",
            "",
            "| 路径特征 | 任务数 |",
            "| --- | ---: |",
        ]
    )
    tool_features = (
        ("使用 `join_tables`", "join_tables"),
        ("使用 `group_aggregate`", "group_aggregate"),
        ("使用 `extreme_value_select`", "extreme_value_select"),
        ("使用 `scalar_compute`", "scalar_compute"),
        ("使用 `set_op`", "set_op"),
    )
    for label, tool in tool_features:
        count = sum(tool in row["tools_used"] for row in rows)
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            f"| 合法终止 | {legal_count} |",
            f"| 零过程错误 | {zero_error_count} |",
            "",
            "## 逐题标签",
            "",
            "| 任务 | 任务类型 | 失败主因 | 终态差异 | 合法终止 | 问题 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(row["trajectory_id"]),
                    TASK_TYPE_ZH[row["task_type"]],
                    ROOT_CAUSE_ZH[row["root_cause"]],
                    MISMATCH_ZH[row["visible_mismatch"]],
                    "是" if row["legal"] else "否",
                    markdown_escape(row["question"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--failure-corpus", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    task_type_by_id = invert(TASK_TYPES, "task type")
    root_cause_by_id = invert(ROOT_CAUSES, "root cause")
    records = read_jsonl(args.input)
    failures = [record for record in records if record.get("correct") is False]
    failure_ids = {str(record.get("trajectory_id")) for record in failures}
    if len(records) != 200 or len(failures) != 55:
        raise ValueError(
            f"expected 200 records and 55 failures, got {len(records)} and {len(failures)}"
        )
    for labels, label_name in (
        (task_type_by_id, "task type"),
        (root_cause_by_id, "root cause"),
    ):
        if set(labels) != failure_ids:
            missing = sorted(failure_ids - set(labels))
            extra = sorted(set(labels) - failure_ids)
            raise ValueError(f"{label_name} coverage mismatch: missing={missing}, extra={extra}")

    corpus_rows = read_jsonl([args.failure_corpus])
    mismatch_by_id = {
        str(row["task_id"]): row.get("visible_mismatch")
        for row in corpus_rows
        if row.get("task_id") in failure_ids
    }
    if set(mismatch_by_id) != failure_ids:
        raise ValueError("terminal mismatch corpus does not cover all 55 failures")

    rows = []
    for record in sorted(failures, key=lambda item: int(item["example_index"])):
        task_id = str(record["trajectory_id"])
        rows.append(
            {
                "example_index": record["example_index"],
                "trajectory_id": task_id,
                "db_id": record["db_id"],
                "difficulty": record["difficulty"],
                "question": record["question"],
                "task_type": task_type_by_id[task_id],
                "root_cause": root_cause_by_id[task_id],
                "visible_mismatch": mismatch_by_id[task_id],
                "legal": bool(record["legal"]),
                "process_error_count": int(record["errors"]),
                "failure_type": record["failure_type"],
                "tools_used": sorted(
                    {
                        str(turn["parsed"]["tool"])
                        for turn in record.get("turns", [])
                        if isinstance(turn.get("parsed"), dict)
                        and isinstance(turn["parsed"].get("tool"), str)
                    }
                ),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "version24_failures55_classification.jsonl"
    report_path = args.output_dir / "version24_failures55_classification_zh.md"
    write_jsonl(jsonl_path, rows)
    write_report(report_path, rows)

    print(
        json.dumps(
            {
                "failures": len(rows),
                "task_type_counts": Counter(row["task_type"] for row in rows),
                "root_cause_counts": Counter(row["root_cause"] for row in rows),
                "visible_mismatch_counts": Counter(
                    str(row["visible_mismatch"]) for row in rows
                ),
                "jsonl": str(jsonl_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
