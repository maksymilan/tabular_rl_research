# RL data selection modules

This package contains reusable selection policies. An experiment scenario file
chooses a policy and supplies dataset, split, K, eligibility range, exclusions,
seed, and output paths; it must not reimplement JSONL parsing or pass@k
predicates.

- `passk.py`: shared K-sample counts, infrastructure-failure filtering, mixed
  outcome detection, and deterministic selection.
- `screened_pool.py`: merges completed K=8 screening observations without
  discarding rescreen history, resolves full task payloads, and writes strict,
  historical-expanded, and known-exclusion-fresh candidate views with hashes,
  plus `retained_non_candidate_observations.jsonl` for outcomes outside the
  training band.
- `passk_artifact.py`: reduces a raw K-sample result file plus its immutable
  input tasks into a compact snapshot (identity, counts, decode parameters; no
  transcripts) and verifies that every input task is matched exactly once.

The current versioned pool is resolved through
`data/inventory/rl_training_candidates_current.json`
(`data/inventory/rl_training_candidates_1to6_current.json` is the same active
pointer for the 2026-09-16 `correct_count in [1,6]` band).  Add a new screening
source to the active snapshot's `sources.json` and rerun the thin
`scenarios/data/build_screened_rl_candidate_pool.py` entrypoint; do not rescan
all historical rollout files or hand-edit candidate counts.  Raw result files
stay on the screening host; export their compact snapshots with the thin
`scenarios/data/export_passk_screen_snapshot.py` entrypoint.

Range-specific exploratory inventories are versioned separately.  The
2026-09-12 `correct_count in [1,7]` inventory is resolved through
`data/inventory/rl_training_candidates_1to7_current.json` and, with the
2026-09-11 `[2,6]` inventory, is kept for audit only.  Neither may silently
become a training cohort.  New K=8 screening batches should record their own
immutable input/result manifests and be merged by stable `example_id` before
eligibility counts are reported.

Historical selectors remain at the old path until their complete launcher,
test, and manifest chain is migrated.
