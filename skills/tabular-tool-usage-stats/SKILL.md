---
name: tabular-tool-usage-stats
description: Use when working in this table RL research repo to summarize tool-call frequency, trajectory length, and rare tool-mode coverage from trajectory JSONL or SFT JSONL files, especially before/after sampling datasets for SFT or eval.
---

# Tabular Tool Usage Stats

## Quick Start

Use the bundled script for deterministic statistics:

```bash
python3 skills/tabular-tool-usage-stats/scripts/tool_usage_stats.py data/trajectories/<file>.jsonl
```

For machine-readable output:

```bash
python3 skills/tabular-tool-usage-stats/scripts/tool_usage_stats.py data/trajectories/<file>.jsonl --json
```

## What To Report

Always report:

- Record count and total tool-call count.
- Step-count distribution: average, min, max, p50, p75, p90, p95, p99.
- Per-tool table: calls, call percentage, trajectories containing the tool, trajectory percentage, average calls per trajectory.
- Feature/mode coverage when relevant, such as n-way joins, value-ref filters, in-table filters, set-op variants, multi-aggregate cases, LIKE filters, and multi-table describe calls.

Use `steps[*].tool_call.tool` as the authoritative source for trajectory JSONL. For SFT/sharegpt-style JSONL, parse assistant `<tool_call>...</tool_call>` blocks only as a fallback.

## Interpretation Notes

- Tool-name coverage alone can hide important sparsity. Prefer feature/mode coverage when deciding whether a 2k sample is balanced enough.
- `answer_from_context` should usually be exactly one call per trajectory.
- `describe_table` near one call per trajectory is expected for enriched data, but consecutive single-table describes may indicate a mechanical perception pattern.
- Rare modes worth watching include `set_op_union`, `like_filter`, `whole_table_multi_agg`, `grouped_multi_agg`, `value_ref_filter`, and `in_table_filter`.
- If a file name includes `plan`, still verify whether actual `plan` tool calls appear in the steps; do not infer tool usage from the file name.

## Resource

- `scripts/tool_usage_stats.py`: standalone JSONL analyzer for this repo's trajectory and SFT data formats.
