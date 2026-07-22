# Trajectory: GDP For Countries With UN Budget Greater Than 2 Percent

Source sample: `data_sample/tablebench_sample.json`, record `2`.

Question type: `NumericalReasoning/FilterAggregation`.

Question: What is the total GDP (nominal) of all countries with a UN budget greater than 2%?

Gold answer: `7700143`.

## Steps

Initial state: The harness provides the country-level table schema and full-table column metadata in `dataset_overview`.

1. `retrieve_row_context`
   - The agent filters rows where `un budget > 2%`.
   - The harness returns Italy, Canada, Spain, Mexico, and South Korea.
   - The tool automatically reports `sum=7700143` for the matched rows' nominal GDP values.

2. `add_to_memory`
   - The agent stores the filtered scope and its matched-set statistic.

3. `answer_from_context`
   - The agent answers `7700143` and cites the filtered rows and GDP column.

Structured JSON: `trajectory.json`.
