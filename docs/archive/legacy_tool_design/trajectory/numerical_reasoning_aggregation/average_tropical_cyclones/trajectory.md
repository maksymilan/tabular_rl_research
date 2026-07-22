# Trajectory: Average Tropical Cyclones

Source sample: `data_sample/tablebench_sample.json`, record `0`.

Question type: `NumericalReasoning/Aggregation`.

Question: What is the average number of tropical cyclones per season?

Gold answer: `10.6`.

## Steps

Initial state: The harness provides the table schema and full-table statistics for `tropical cyclones`, including `count=10`, `sum=106`, and `mean=10.6`.

1. `add_to_memory`
   - The agent stores the initialized column statistic and parse check as a memory item.
   - No retrieval action is needed because the question asks for an unfiltered full-table statistic already present in metadata.

2. `answer_from_context`
   - The agent answers `10.6`, citing the aggregate result memory and the target column.

Structured JSON: `trajectory.json`.
