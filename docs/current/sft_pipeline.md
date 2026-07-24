# SFT pipeline

## Accepted sources

SFT examples come from causal student or external-teacher model↔harness episodes. A successful
episode must pass terminal denotation scoring, fresh replay, protocol/quality gates, and the rolling
single-action export checks before entering a mixture.

The denotation comparison used to admit an episode is stored in
`rollout_generation.denotation_comparison`, and fresh replay reuses that exact comparison.
Historical artifacts that predate this field require an explicit replay override; they must never
be silently replayed under a comparison that conflicts with their manifest. The current default
for every new BIRD episode, replay, grounding gate, and reward audit is `bird-set`.

The current BIRD SFT-2 construction is student-first pass@k. Teacher fallback is restricted to true
student pass@k failures. Provider attempts, transport failures, rejected trajectories, and duplicate
actions remain auditable but do not become training targets.

## Export contract

`src/sft/export_sft_dataset.py` and `src/sft/build_rolling_sft_data.py` render each supervised action
from the state immediately before that action. The input must exclude the current tool output,
future actions or observations, hidden gold SQL, and future factual provenance.

Recovered trajectories retain erroneous turns in the audit record, but rejected actions are not SFT
targets. The first later legal action can be labeled as feedback recovery when it causally uses the
structured error.

## Current entry points

- `src/sft/generate_teacher_rollouts.py`: closed-loop external-teacher generation.
- `src/sft/build_bird_sft2_dataset.py`: replay student successes and select fallback tasks.
- `src/sft/assemble_bird_sft2_mixture.py`: deterministic mixture assembly.
- `src/sft/export_sft_dataset.py`: final ShareGPT-style export.
- `src/sft/train_bird_sft2_qwen25_7b.sh`: current training launcher.
