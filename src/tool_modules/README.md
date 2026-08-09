# Tool module ownership

This directory is the canonical home of independently selectable non-atomic tool schemes and the
cross-scheme registry.

```text
tool_modules/
├── registry.py
├── action_block/
│   ├── protocol.py
│   ├── evaluator.py
│   ├── sft_export.py
│   ├── analyze.py
│   ├── audit_failures.py
│   └── tests/
├── relational_program/
│   ├── protocol.py
│   ├── evaluator.py
│   └── tests/
├── direct_sql_search/
│   ├── protocol.py
│   ├── audit.py
│   └── behavior_audit.py
├── iterative_sql/
│   ├── protocol.py
│   └── audit.py
├── sql_common/
│   ├── runner.py
│   └── tests/
└── native_tool_bundle/
    ├── protocol.py
    ├── provider_tools.py
    ├── audit.py
    └── tests/
```

`sql_common` is deliberately shared: the two SQL schemes have distinct prompts, action schemas,
hashes, and identities, but use the same immutable SQLite safety/execution/context/scoring loop.

The atomic protocol remains the project-wide core in `src/sft/protocol.py`, with shared harness
semantics in `src/harness/` and its general evaluator in `src/eval/rollout.py`. It is not duplicated
inside this directory because SFT, evaluation, and RL all consume that same promoted contract.

Files left at historical flat paths in `src/eval/` or `src/sft/` are import/CLI compatibility
aliases. New code must import `tool_modules.*` directly. The boundary is enforced by
`tests/test_module_boundaries.py`.
