# Tool Design Validation Notes

This directory stores validation utilities for the canonical formats under `tool_design/`.

Use `python3 scripts/tool_design/validate_trajectories.py` after adding or changing a trajectory.

The validator checks:
- canonical trajectory top-level fields;
- tool call, tool output, and state delta fields;
- retrieve-row mode-specific fields;
- table and column references;
- memory support field names;
- dynamic table context row/data consistency;
- final `answer_from_context` placement.

The validator checks format consistency, not whether the returned rows or answers are semantically correct. Semantic checks still require comparison with the source sample.
