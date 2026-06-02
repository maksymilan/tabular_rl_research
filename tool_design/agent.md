# Tool Design Agent Notes

This directory records the research design for a table-agent harness focused on RL training for LLM table tool use.

For concrete interface formats, always start from `tool_usage.md` and `tool_json_examples/`. Future tool design changes and tool usage should follow those formats unless the spec is explicitly versioned.

Concrete example trajectories live in `trajectory/`. Use those files to understand how a full question-answering run should record tool calls, outputs, context updates, memory updates, and final answering.

## Current Research Idea

The proposed system frames Table QA / Table Reasoning as a multi-step, observable tool-use trajectory. The model does not directly consume the full raw table or issue dataset-specific SQL/Python. Instead, an external harness maintains task state and executes abstract, data-format-agnostic tools through dataset adapters.

Core state:
- `Dynamic Table Context`: the materialized table subset currently visible to the model. It changes through retrieval and pruning actions.
- `Static Task Memory`: a structured task-level scratchpad that stores schema understanding, cross-table relations, confirmed facts, intermediate results, plans, and excluded information.
- `Tool History`: the chronological trace of tool calls, normalized inputs, outputs, and state changes. It should be recorded by the harness for analysis and reward computation, but should not be directly included in the model-visible prompt context by default.

Core tools in the first design:
- `inspect_dataset`: get dataset/table overview, columns, row counts, and sample rows.
- `retrieve_column_context`: retrieve relevant columns and a small real-data view.
- `aggregate_column`: compute numeric/statistical aggregations without exposing all rows.
- `retrieve_row_context`: retrieve rows by entity match, condition filter, or semantic match.
- `drop_context`: remove irrelevant rows or, less frequently, columns from the dynamic context.
- `add_to_memory`: add confirmed facts, intermediate results, schema facts, or plans to task memory.
- `refine_memory`: compress, remove, or update task memory entries.
- `answer_from_context`: produce final answer with cited evidence rows and columns.

Canonical JSON examples for each tool are stored in `tool_json_examples/`.

The intended RL signal is not only final-answer correctness. It should eventually reward useful information acquisition, useful removal of irrelevant data, evidence construction, and non-redundant tool use. In the current phase, concrete reward optimization is intentionally deferred; first priority is to make the tool design viable across most table QA task types.

## Current Priorities

1. Make the abstract tool set usable across many table QA cases before over-optimizing rewards.
2. Treat reward issues such as evidence intersection, pruning correctness, and multi-path credit assignment as known defects to revisit after the harness can run.
3. Use broad case coverage to refine the tool set: lookup, filtering, aggregation, comparison, ranking, fact verification, semantic judgment, and multi-table joins.
4. Keep `single-entity` and `multi-entity` as coarse early labels, not final task definitions.
5. Record tool history internally for duplicate-call checks and trajectory analysis, but avoid putting full tool history into the model prompt.

## Reward Design Direction

Important candidate rewards:
- final answer correctness;
- acquisition of answer-relevant rows/columns/cells;
- pruning of provably irrelevant rows;
- preservation of necessary evidence after pruning;
- concise tool trajectories;
- penalties for repeated or semantically redundant tool calls;
- evidence citation quality in `answer_from_context`.

One proposed search strategy is multi-path exploration: run multiple tool-use trajectories, keep paths with correct final answers, inspect their resulting sub-tables, and reward operations that consistently acquire shared relevant evidence while discarding irrelevant context. For now this is a future optimization direction, not a hard constraint on the first tool implementation.

## Potential Weaknesses And Missing Pieces

1. Credit assignment may be noisy.

Taking intersections of sub-tables from multiple correct paths can identify stable evidence, but correct paths may use different valid evidence sets. A strict intersection can under-reward alternate valid strategies, especially for aggregation, comparison, or semantic questions where many rows contribute. Keep this as a known reward-design defect for later optimization.

2. Negative reward for removed information needs ground truth.

Rewarding `drop_context` requires knowing that removed rows/columns are truly irrelevant. This is hard when answers depend on aggregates, absence of evidence, global maxima/minima, or comparison against all rows. A row that looks irrelevant locally may be necessary to prove a maximum or denominator. Treat this as another reward-design defect to revisit after the basic harness works.

3. Repeated tool-call penalties need semantic equivalence, not exact matching.

Exact duplicate calls are easy to penalize, but near-duplicates are harder: same intent with different wording, overlapping filters, or larger/smaller `top_k`. The harness should define redundancy using both normalized tool arguments and overlap in returned rows/columns. This analysis should use internal tool history rather than placing tool history in the model-visible context.

4. Column-first versus row-first heuristics are task-dependent.

The current single-entity/multi-entity split is useful as a coarse initial categorization but is under-specified. Many tasks require both global column discovery and row filtering. The tool design should be tested against broader case types and refined based on failures.

5. Tool granularity can hide important reasoning.

`retrieve_column_context` combines column retrieval and real-data materialization. This is practical, but it can make it harder to separate rewards for "selecting the right schema" from rewards for "seeing useful values." Consider logging both candidate columns and materialized rows internally.

6. Adapter capability must be measured, but adapters are not inherently in conflict with tool-use learning.

Adapters are necessary because the same abstract tools must work over CSV, DataFrame, JSON, nested JSON, markdown tables, databases, and other dataset forms. This does not conflict with studying model tool-call ability. The concern is only that a very strong adapter may over-select answer evidence, making the policy look better than it is. Log adapter settings and compare simple versus stronger adapters when possible.

7. Dynamic context as `selected_rows x selected_columns` may be too restrictive.

Some evidence is naturally cell-level, group-level, join-result-level, or derived-fact-level. The current plan is to store these higher-level evidence objects in Static Task Memory while keeping Dynamic Table Context as the materialized table view. This makes memory design central rather than auxiliary.

8. Task memory can become an uncontrolled hidden channel.

Although memory is observable, the model may write unsupported claims. Because memory is task-level only, temporal staleness is not the main issue; correctness of the memory write is. When writing memory, the model should append a justification/check field that records why the memory item is believed true, what evidence or tool output supports it, and what condition would invalidate it. Later use can reference this judgment information instead of re-validating from scratch every time.

9. Final-answer reward may dominate process rewards.

If answer correctness is much stronger than process shaping, the model may learn shortcuts or overfit to answer patterns. If process rewards are too strong, it may optimize tool behavior without solving the question. Reward scales need ablation.

10. Evaluation needs baselines and ablations.

Track at least: direct answer without tools, full-table context, SQL/Python/code-agent baseline where applicable, retrieval-only without pruning, pruning without memory, and full harness. This will clarify which component actually contributes.

## Design Recommendations

- Define evidence labels per dataset when possible: FeTaQA `highlighted_cell_ids`, TableBench answer-relevant rows/columns when derivable, and synthetic labels for controlled cases.
- Log every step with before/after context size, returned row/column ids, dropped ids, memory diffs, normalized tool signature, and output overlap with previous calls.
- Keep full tool history internal to the harness; expose only compact state summaries if the prompt needs them.
- Track adapter type and strength, but treat adapters as the required execution layer for abstract data-format-independent tools.
- Start with tasks where reward can be verified: lookup, filtering, aggregation, comparison, and simple joins.
- Treat `single-entity` and `multi-entity` as early heuristics, not final taxonomy.
- Add an explicit `context_budget` metric: tokens, rows, columns, and cells shown to the model.
- Require `answer_from_context` to cite evidence; use citation validity as a process reward.
- For pruning rewards, penalize dropping gold evidence more strongly than failing to drop irrelevant context.
- For `add_to_memory`, require fields like `type`, `content`, `supporting_step_ids`, `supporting_rows_or_columns`, `confidence_or_check`, and optional `invalidating_condition`.

## Open Research Questions

- What is the best evidence unit for reward: row, column, cell, table, operation result, or derived fact?
- Should memory writes be actions with rewards, or only auxiliary state updates?
- How should adapter capability be ablated without losing the benefit of data-format-independent tools?
- Can multi-path correct trajectories produce reliable pseudo-labels for evidence without gold annotations?
- Should search trajectories optimize for minimal sufficient context or robust redundant evidence?
- What memory schema best stores cell-level, group-level, join-result-level, and derived-fact evidence without becoming a hallucination channel?
