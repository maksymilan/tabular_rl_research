# Process Reward V4: Column-Aware Grounding

## Scope

V4 repairs the deterministic false edges found in the external V2/V3 audit and evaluates the
current SFT candidate pool. Runtime reward remains model-independent; external models are used only
for offline QA.

Input: `data/trajectories/bird_verified_rolling4_full_resident_r2_candidate_pool.jsonl`.
All 316 episodes replay and match their final BIRD-train denotation.

Final artifact: `data/rl/bird_candidate316_process_reward_v4d_grounding/`.

## Implementation

- `read_subtable` column names are harness-private `observed_columns`; they do not alter historical
  model-visible outputs or plan evidence.
- Row grounding matches `(source table, source column, value)` to the predicate's target column.
  Columns must share lineage, a distinctive name, or a schema foreign-key path.
- A missing SQLite foreign-key target column resolves only when the referenced table has exactly
  one primary key. No model argument is repaired.
- `answer_from_context` no longer receives a grounding edge merely because an earlier row contains
  the same scalar.
- Explicit multi-value answers can collect several read handles. Numeric equality alone cannot add
  a secondary handle.
- Completeness diagnostics separately report unsupported final values and non-task constants that
  lack prior domain/row evidence.

## Results

| Metric | Result |
| --- | ---: |
| Replay correct | 316 / 316 |
| Structured slice | 315 / 316 |
| Final-value complete | 244 / 316 |
| Action-literal complete | 305 / 316 |
| Both deterministic gates | 235 / 316 (74.37%) |
| Multi-table final answers | 3 |
| Reward min / mean / max | 0.79 / 0.9662 / 1.00 |
| Max positive step share p50 / p90 / max | 0.272 / 0.422 / 0.625 |
| Positive / negative / zero steps | 1496 / 126 / 474 |

Reward conservation, positive-correct totals, and expected labels pass. `process_reward_ready` is
still false because 81 episodes fail deterministic completeness and the revised edges have not yet
received a fresh precision audit.

## Concrete Cases

`bird_train_00835`: the old false edge from a value `7` in `Season` to
`Match.Match_Winner = 7` is gone. The valid chain now uses `Team.Team_Id -> Match.Match_Winner` via
the declared foreign key. Its seven step rewards are approximately
`0.1766, 0.1534, 0.1766, 0.1766, 0.1163, 0.2003, 0.0`. The episode is still excluded by the strict
SFT gate because the handwritten percentage `17.105...` is not itself present in a tool result.

`bird_train_02424`: the shipping-method identifier `2` is grounded through
`cust_order.shipping_method_id -> shipping_method.method_id`, including the BIRD schema's missing
FK-target-column resolution to the unique primary key. Its customer read, shipping-method read,
filter, and aggregate all receive positive credit. No historical `2` is linked directly to the
final count.

`bird_train_04133`: the answer combines venue and winning-team values from two handles. V4 adds both
producer/read chains. Two protocol errors receive `-0.08` each; the first legal recovery and the
final legal recovery receive feedback credit. The correct episode totals `0.84`, not `1.0`, because
the two real error penalties remain audit-visible.

## SFT Gate

The first conservative split was frozen before the unique-primary-key refinement: 228 complete
episodes became 206 train / 22 validation episodes, with 1,303 / 130 single-step targets and no
episode overlap. This is a valid subset of the final 235 V4-complete episodes; training does not
need to restart merely to add seven newly admitted examples.

The Qwen/LLaMA-Factory 6400-token audit retains every final target: train 1303/1303 and eval
130/130. Do not add the remaining execution-correct episodes to SFT-1 until their unsupported
constants or final values are resolved by real harness evidence.
