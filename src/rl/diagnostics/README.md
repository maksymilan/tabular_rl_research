# Diagnostic primitives

This package contains reusable diagnostic infrastructure.  The public modules
are split by responsibility: `io.py` for strict JSON/JSONL, hashes, and atomic
writes; `validation.py` for fail-closed field and path checks; `records.py` for
trajectory/sample extraction; `replay.py` for rollout grouping and Harness
error status preparation; `metrics.py` for distributions, AUC, paired tests,
and bootstrap intervals; `trajectory.py` for dense reward categories;
`gradient.py` for Gate60 token masks and gradient geometry; and `reporting.py`
for stable report/manifest artifacts.  `common_io.py` remains a compatibility
facade for older callers.

Experiment-specific audits and one-off analyses live in
`rl.scenarios.diagnostics`, where their provenance and scope are explicit.
New diagnostics should compose these small APIs and keep CLI orchestration and
experiment schema decisions in the scenario module.
