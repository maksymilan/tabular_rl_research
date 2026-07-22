# Trajectory: 1982 Illinois Governor Winner And Margin

Source sample: `data_sample/fetaqa_sample.json`, record `0`.

Question type: `EntityLookup/EvidenceCells`.

Question: Who won the 1982 Illinois gubernatorial election, and how many votes was the margin?

Gold answer: Thompson prevailed in the 1982 Illinois gubernatorial election by a 5,074 vote margin.

## Steps

Initial state: The harness provides the election results schema and full-table column metadata in `dataset_overview`.

1. `retrieve_column_context`
   - The agent retrieves fixed samples for `Party_0`, `Party_1`, `Candidate`, `Votes`, and `%`.
   - The samples show that candidate rows use `Party_0="-"`.

2. `retrieve_row_context`
   - The agent uses `mode=condition_filter` on the full table with `Party_0="-"` and numeric `Votes`.
   - This explicitly constructs candidate-shaped rows without asking the adapter to invent a semantic row type.

3. `retrieve_row_context`
   - The agent uses `mode=extreme_value_select` with `search_scope=dynamic_table_context` and `top_k=2`.
   - This is a nested tool call over the candidate rows maintained in the dynamic table.
   - It gets row `1` for `James R. Thompson (incumbent)` and row `2` for `Adlai Stevenson III`.

4. `add_to_memory`
   - The agent stores the winner and computes the margin from the top two candidate rows.
   - Calculation: `1,816,101 - 1,811,027 = 5,074`.
   - The original FeTaQA annotation highlights `[1, 2]` and `[6, 3]`, but this trajectory uses a derived margin because the model should not rely on unseen summary rows.

5. `answer_from_context`
   - The agent answers with the winner surname and margin, citing the evidence rows/columns and memory item.

Structured JSON: `trajectory.json`.
