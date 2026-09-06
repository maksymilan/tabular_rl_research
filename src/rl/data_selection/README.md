# RL data selection modules

This package contains reusable selection policies. An experiment scenario file
chooses a policy and supplies dataset, split, K, eligibility range, exclusions,
seed, and output paths; it must not reimplement JSONL parsing or pass@k
predicates.

- `passk.py`: shared K-sample counts, infrastructure-failure filtering, mixed
  outcome detection, and deterministic selection.

Historical selectors remain at the old path until their complete launcher,
test, and manifest chain is migrated.
