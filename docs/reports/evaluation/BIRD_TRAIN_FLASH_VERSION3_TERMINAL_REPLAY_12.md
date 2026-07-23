# BIRD-train Flash version3 terminal-error replay (12 tasks)

Date: 2026-07-23

## Scope

Replay the 10 protocol-terminal and 2 execution-terminal version2 cases after version3 introduced:

- provider-specific `LAST TOOL ERROR` for rejected DeepSeek split carriers;
- stable exact names for projected join columns;
- quoting of exact environment-exposed names in `project`;
- a canonical three-table join example.

## Result

| Outcome | Version2 on these tasks | Version3 replay |
|---|---:|---:|
| Correct | 0 | 2 |
| Legal but wrong denotation | 0 | 8 |
| Protocol/carrier termination | 10 | 2 |
| Execution termination | 2 | 0 |

The raw version3 summary labels the final two carrier failures as `argument_validation_error`. This
was a reporting bug: the precise carrier message includes the literal JSON field `"arguments"`, and
the old string heuristic interpreted that word as an argument-schema failure. Inspection of the raw
adapter records confirms both are DeepSeek carrier failures. Version4 fixes the classifier and keeps
future carrier failures in the protocol-error budget.

Version3 therefore converted 10/12 previously nonlegal terminal failures into legal terminal
answers. Two became correct (`bird_train_02359`, `bird_train_00997`); eight exposed ordinary semantic
wrong answers that can now be audited separately from interface failures.

## Error events

| Metric | Version2 | Version3 |
|---|---:|---:|
| Total actions | 148 | 109 |
| Adapter/carrier error events | 33 | 8 |
| Execution-error events | 7 | 3 |
| Execution-terminal episodes | 2 | 0 |

All three remaining execution events are join-identifier mistakes, and all three episodes continued
to a legal terminal answer. Two occur in `bird_train_05248`; one occurs in `bird_train_05433`.
The model still prefixes a newly attached right-table key or emits `table.column`, despite the
canonical example. This confirms that join naming is now the dominant tool-call ergonomics issue.

## Interpretation

The provider feedback change works: it sharply reduces carrier errors and prevents misleading
`missing <think>` feedback. The column-name boundary also eliminates execution termination. It does
not improve the teacher's database semantics automatically, which is desirable: eight cases now
fail transparently as wrong denotations instead of being hidden under protocol/execution failures.

The next join redesign, if pursued, belongs in version5. It should replace the asymmetric
left-materialized/right-bare string convention with structured source-instance column references.
The harness should validate those references, not silently rewrite invalid strings.

## Artifacts

- Input: `data/eval_inputs/bird_train_tool_interface_ablation12_terminal_errors_version3.jsonl`
- All attempts: `data/trajectories/bird_interface_ablation12_version3_terminal_error_replay_flash_all.jsonl`
- Successes: `data/trajectories/bird_interface_ablation12_version3_terminal_error_replay_flash_success.jsonl`
- Failures: `data/trajectories/bird_interface_ablation12_version3_terminal_error_replay_flash_failures.jsonl`
- Version3 protocol hash: `ef7f3f024b18672d`
- Version4 change: carrier classification only; no version4 external replay was run in this report.
