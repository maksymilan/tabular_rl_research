# ReAct Trajectory: Busiest Airline In A Multi-Table Airline DB

Source sample: `data_sample/tqabench_airline_sample.json`, record `0` (tqabench `original_dataset_id = 32k_0_5_2`).

Question type: `MultiTableJoinAggregation`.

Question (authored): Which airline operates the most flights in the table, and how many flights is that?

Authored gold answer: `American Airlines Inc.: AA operated the most flights in the table, with 12.`

## Why an authored question

This record is a real tqabench airline database snapshot with three tables: `Air Carriers` (23 rows), `Airports` (96 rows), and `Airlines` (100 flight rows). tqabench pairs each question with a *sampled* snapshot but computes its native gold over the *full* database, so the native question/answer (`How many airlines land in Denver, CO: Denver International?` / `6`) is not reproducible from the provided 100-row snapshot. To keep the trajectory self-contained and verifiable, the snapshot is treated as the table-of-record and a snapshot-faithful question is authored. Every number below is computed directly from the snapshot tables.

This case exists to exercise the two new tools (`group_aggregate`, `join_tables`) and the two-tier `data_view` model, in contrast to the single-table trajectories and the manual two-`entity_match` comparison in `entity_comparison/fetaqa_andretti_comparison`.

## Steps

Initial state: The harness provides the three-table schema and `relations` (PK/FK with cardinality) in `dataset_overview`. The relation `Airlines.OP_CARRIER_AIRLINE_ID -> Air Carriers.Code (many_to_one)` is the join key the model needs.

1. Reason: Count flights per carrier across the whole `Airlines` table.
   - Action: `group_aggregate(table_name=Airlines, group_by=[OP_CARRIER_AIRLINE_ID], aggregations=[count as flight_count])`
   - Observation: A derived `data_view` table `g_flights_by_carrier` (kind=group, 21 rows) is registered. Only a 3-group head sample is returned to the model; the full group table stays harness-side.

2. Reason: The group table has carrier ids but not names; attach airline names through the declared foreign key.
   - Action: `join_tables(left=g_flights_by_carrier, right="Air Carriers", on=[OP_CARRIER_AIRLINE_ID = Code], inner)`
   - Observation: A derived `data_view` table `j_carrier_named` (kind=join, 21 rows) is registered. `join_diagnostics` reports `0` unmatched carriers and `fanout_warning=false` (many_to_one). This demonstrates joining a *derived* table with a *source* table.

3. Reason: Take the carrier with the maximum flight count.
   - Action: `retrieve_row_context(mode=extreme_value_select, table_name=j_carrier_named, search_scope=full_table, target_column=flight_count, order=max, top_k=1)`
   - Observation: Carrier `19805` = `American Airlines Inc.: AA` with `flight_count = 12`, ahead of the runner-up (`11`). This row is the only slice pulled into `dynamic_table_context`.

4. Reason: Record the verified result.
   - Action: `add_to_memory` (derived_fact)
   - Observation: Memory stores the busiest airline, count `12`, runner-up `11`, and the group→join→max method, citing the derived tables.

5. Reason: Answer from the cited evidence row and memory.
   - Action: `answer_from_context`
   - Observation: Final answer matches the authored gold.

For reference, the top carriers by flight count in this snapshot are: AA (12), Delta `19790` (11), Southwest `19393` (11), JetBlue `20409` (9), United `19977` (8).

Structured JSON: `trajectory.json`.
