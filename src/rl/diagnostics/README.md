# Diagnostic primitives

This package contains only reusable diagnostic code: shared JSONL/statistics
I/O (`common_io.py`), evaluation identity checks, and semantic quality
scoring. Experiment-specific audits and one-off analyses live in
`rl.scenarios.diagnostics`, where their provenance and scope are explicit.
New diagnostics should expose small importable functions here and keep command
line orchestration in a scenario module.
