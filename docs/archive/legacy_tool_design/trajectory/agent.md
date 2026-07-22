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
- `trajectory.json` must follow `tool_design/tool_json_examples/trajectory_format.json` exactly.
- `trajectory.json.initial_state.dataset_overview` records the schema and full-table metadata that the harness provides before the first model action.
- Do not add `inspect_dataset` as a trajectory step.
- Do not add Reason/Observation summaries such as `decision_summary` to `trajectory.json`; keep them in `trajectory.md`.
- Tool history is represented in the JSON trajectory for analysis, but full tool history should not be treated as default model-visible context.
- Prefer examples from `data_sample/` and record the source sample path and record index.
- A multi-hop trajectory should have real information dependencies between steps. Do not treat schema samples or coincidentally visible head rows as proof of a full-table conclusion.
- When a conclusion depends on ordering within a filtered set, first establish the filter scope and then use an ordering-aware tool such as `extreme_value_select`.

Current question type folders:
- `numerical_reasoning_aggregation`: aggregation over a visible table.
- `numerical_reasoning_filter_aggregation`: filter rows first, then aggregate.
- `entity_lookup_evidence`: locate entity/evidence cells and answer from them.
- `entity_comparison`: retrieve multiple entities and compare their attributes.
- `temporal_comparison`: retrieve a target fact and later/earlier evidence to derive a time-based conclusion.
