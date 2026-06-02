# Trajectory: 1982 Illinois Governor Winner And Margin

Source sample: `data_sample/fetaqa_sample.json`, record `0`.

Question type: `EntityLookup/EvidenceCells`.

Question: Who won the 1982 Illinois gubernatorial election, and how many votes was the margin?

Gold answer: Thompson prevailed in the 1982 Illinois gubernatorial election by a 5,074 vote margin.

## Steps

1. `inspect_dataset`
   - The agent inspects the table title and columns.
   - It sees an election results table with candidate, votes, percentage, and majority rows.

2. `retrieve_column_context`
   - The agent retrieves columns needed for winner and margin: `Candidate`, `Votes`, and `%`.
   - This step only returns fixed sample rows for column-level awareness, not the full table.
   - Since `Votes` is numeric after comma removal, the column context also exposes statistics such as max vote value.

3. `retrieve_row_context`
   - The agent retrieves the top two candidate rows by `Votes`.
   - It uses `mode=extreme_value_select` with `top_k=2`.
   - This avoids assuming the model already knows whether a `Majority` summary row exists.
   - It gets row `1` for `James R. Thompson (incumbent)` and row `2` for `Adlai Stevenson III`.

4. `add_to_memory`
   - The agent stores the winner and computes the margin from the top two candidate rows.
   - Calculation: `1,816,101 - 1,811,027 = 5,074`.
   - The original FeTaQA annotation highlights `[1, 2]` and `[6, 3]`, but this trajectory uses a derived margin because the model should not rely on unseen summary rows.

5. `answer_from_context`
   - The agent answers with the winner surname and margin, citing the evidence rows/columns and memory item.

Structured JSON: `trajectory.json`.
