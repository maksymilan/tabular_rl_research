# Trajectory Agent Notes

This folder stores concrete tool-use trajectories constructed from `data_sample/` cases.

Directory convention:

```text
trajectory/
  <question_type>/
    <case_id>/
      trajectory.md
      trajectory.json
```

Rules:
- Use `tool_design/tool_usage.md` as the interface contract.
- Use tool input/output formats consistent with `tool_design/tool_json_examples/`.
- `trajectory.md` records the readable step-by-step process.
- `trajectory.json` records the structured tool calls, tool outputs, state deltas, memory updates, and final answer.
- Tool history is represented in the JSON trajectory for analysis, but full tool history should not be treated as default model-visible context.
- Prefer examples from `data_sample/` and record the source sample path and record index.

Current question type folders:
- `numerical_reasoning_aggregation`: aggregation over a visible table.
- `numerical_reasoning_filter_aggregation`: filter rows first, then aggregate.
- `entity_lookup_evidence`: locate entity/evidence cells and answer from them.
