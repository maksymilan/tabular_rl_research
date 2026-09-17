# Runtime infrastructure

Fixed Atomic version26 environment, task loading, replay, terminal scoring, and
failure normalization live here. These modules are shared by all RL workflows
and experiment scenarios.

`error_feedback.py` supplies opt-in `actionable-error-v1` rejected-action observations.
It consumes only the rejected action and existing public Harness metadata, without SQL reads,
semantic repair, or reward changes. The isolated RL environment defaults to `legacy` until
the SFT dev regression admits this candidate; the same renderer is used by the pinned-eval overlay.
