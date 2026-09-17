# 60-task hybrid restart evidence

Diagnostic-only; not evidence of an effective RL update or accuracy improvement.

Remote root:
`/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_v26_legal_reason_hybrid_balanced60_r2_20260910`

Implementation is the composition, in order, of:

1. `implementation.tar.gz`: `f28093f2cd39d5c8fc947abf228bd67c4a9fe05e8501add9e1d39dc0a38e2898`
2. `cleanup_patch_v1.tar.gz`: `1d772ec13494831465ace0caf2982c8ce02843dafba60538cd978a0e2771a255`

`cohort.jsonl`: 60 tasks, SHA256
`b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4`.

`cpu_preflight.json` is the actual remote receipt **before the cleanup-only patch**;
the launcher's final receipt must record the patched helper hash. Protocol, cohort,
model, reward and span settings are unchanged by the patch.

`regression_with_cleanup.log` records the actual post-patch remote tests.
`pytest_dependencies_v2.tar.gz` is isolated test-only tooling, not a model/runtime
dependency change. The earlier dependency tar is retained as failed setup evidence.

Current run state, authorization and next steps:
[`LEGAL_REASON_HYBRID_RESTART_20260910.md`](../../../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md).
