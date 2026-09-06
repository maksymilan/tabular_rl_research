# Fixed-pool infrastructure

This package contains reusable fixed-pool generation, assembly, rescoring, and
validation code. Named pool searches, cohort freezing, counterfactual builds,
and other experiment-specific operations live under `rl.scenarios.fixed_pool`.
Tests are kept under `rl.tests.fixed_pool` so the infrastructure directory does
not mix implementation with test entrypoints.
