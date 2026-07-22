# Recovery Pivot Smoke40 Quality Report

Date: 2026-06-29

## Goal

Construct failure-recovery trajectories where the agent notices a local reasoning error from tool
observations and then changes strategy. The target is not post-hoc rewriting of a known-wrong path.

## Generator

- Input: `data/trajectories/recovery_v10_train.jsonl`
- Script: `src/sft/rewrite_recovery_think.py`
- External model: `deepseek-v4-flash`
- Max attempts: 10
- API retries: 3
- Workers: 4

The prompt now requires the external model to output:

- `pivot_step_id`
- `unsupported_assumption`
- `pivot_evidence`
- `next_strategy`
- `thinks`

The script stores accepted and rejected external-model outputs in both the trajectory metadata and
the audit file, so validator decisions can be reviewed later.

## Files

- Raw generated 40 records:
  `data/trajectories/recovery_v10_flash_pivot_smoke40_v2.jsonl`
- Full attempt audit:
  `data/trajectories/recovery_v10_flash_pivot_smoke40_v2.jsonl.audit.jsonl`
- Reviewed 40 records:
  `data/trajectories/recovery_v10_flash_pivot_smoke40_v2_reviewed2.jsonl`
- Ready-only subset:
  `data/trajectories/recovery_v10_flash_pivot_smoke40_v2_ready2.jsonl`
- Reviewed summary:
  `data/trajectories/recovery_v10_flash_pivot_smoke40_v2_reviewed2.jsonl.summary.json`

## Results

Second-pass generation:

- 40 / 40 rewritten
- 0 fallback
- 21 records accepted on attempt 1
- 19 records required at least one rejected attempt
- 81 total external-model attempts recorded
- Rejection causes:
  - 35 validation rejects
  - 6 parse rejects
  - main validation causes: banned post-hoc wording and non-first-person steps

Reviewed quality split:

- `ready`: 22
- `review`: 18
- `reject`: 0

No accepted `think` text contains the current banned post-hoc markers:

- `gold sql`, `gold answer`, `ground truth`
- `verified path/source`
- `failed path/prefix`
- `corrected`
- `wrong answer`, `expected answer`
- `the model`, `the agent`

## Quality Decision

The 40-record smoke demonstrates that the new prompt and validator can produce usable recovery
reasoning, but quality is not guaranteed for all accepted rewrites. The safe training subset is the
22-record `ready2` file. The remaining 18 records are retained as `review`, not deleted, because some
may be salvageable and the validator may still be over-conservative.

Do not use the full `reviewed2` file for SFT without manual review or another repair pass. Use
`ready2` for high-confidence testing.

## Remaining Issues

- Some generated pivots are still weak: they describe confirmation, repeated execution, or output
format adjustment rather than a genuine failed assumption.
- Some recovery candidates themselves are not ideal because the deterministic builder may splice a
failed prefix with a clean path where the semantic delta is small.
- More robust recovery data should select candidates whose failed prefix has a concrete observable
symptom, such as empty rows, wrong value domain, scalar/list mismatch, wrong table/column concept, or
executor error.

## Next Recommendation

Before scaling beyond this smoke:

1. Select recovery candidates with stronger observable failure signals.
2. Generate with the pivot-aware DeepSeek Flash prompt.
3. Treat warning-bearing records as `review`, not `ready`.
4. Train only on ready records, while keeping audit files for validator debugging.
