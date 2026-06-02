# Trajectory: Average Tropical Cyclones

Source sample: `data_sample/tablebench_sample.json`, record `0`.

Question type: `NumericalReasoning/Aggregation`.

Question: What is the average number of tropical cyclones per season?

Gold answer: `10.6`.

## Steps

1. `inspect_dataset`
   - The agent inspects the table schema.
   - It learns that the table contains season-level storm statistics and a `tropical cyclones` column.

2. `retrieve_column_context`
   - The agent asks for columns relevant to computing the average number of tropical cyclones.
   - The harness returns `season` and `tropical cyclones` with only fixed sample rows for reference.
   - It also returns the needed numeric statistic: `mean=10.6`.

3. `add_to_memory`
   - The agent stores the column statistic and parse check as a memory item.
   - No extra `aggregate_column` call is needed because the required mean was already returned by column context.

4. `answer_from_context`
   - The agent answers `10.6`, citing the aggregate result memory and the target column.

Structured JSON: `trajectory.json`.
