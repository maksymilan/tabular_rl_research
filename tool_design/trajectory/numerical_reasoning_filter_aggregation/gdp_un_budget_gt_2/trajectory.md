# Trajectory: GDP For Countries With UN Budget Greater Than 2 Percent

Source sample: `data_sample/tablebench_sample.json`, record `2`.

Question type: `NumericalReasoning/FilterAggregation`.

Question: What is the total GDP (nominal) of all countries with a UN budget greater than 2%?

Gold answer: `7700143`.

## Steps

1. `inspect_dataset`
   - The agent inspects the country-level table.
   - It finds the relevant columns: `country`, `un budget`, and `gdp (nominal) (millions of usd) 2011`.

2. `retrieve_column_context`
   - The agent retrieves the filter column and aggregation column.
   - The harness returns only fixed sample rows for column-level awareness.

3. `retrieve_row_context`
   - The agent filters rows where `un budget > 2%`.
   - The harness returns Italy, Canada, Spain, Mexico, and South Korea.

4. `add_to_memory`
   - The agent stores the filtered rows and sum check.
   - Sum: `2198730 + 1736869 + 1493513 + 1154784 + 1116247 = 7700143`.

5. `answer_from_context`
   - The agent answers `7700143` and cites the filtered rows and GDP column.

Structured JSON: `trajectory.json`.
