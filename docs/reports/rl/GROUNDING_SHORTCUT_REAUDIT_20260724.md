# Grounding shortcut trajectory re-audit (2026-07-24)

## Conclusion

Manual comparison of the 14 jointly reviewer-flagged trajectories against their complete legal
tool traces, harness outputs, BIRD gold SQL, and gold denotations changes the earlier interpretation:

| Classification | Count | Trajectories |
|---|---:|---|
| Confirmed semantic shortcut | 3 | `00541`, `02918`, `03663` |
| Valid visible-cell dependency omitted by the extractor | 6 | `00739`, `03877`, `04109`, `05873`, `05875`, `06577` |
| Gold, external-knowledge, or reviewer conflict | 5 | `00525`, `00921`, `01039`, `01792`, `05527` |

The external labels remain audit evidence rather than reward targets.

## Confirmed shortcuts

- `00541` answers from `Person` after a zero-row `Credit` query that does not execute the required
  `credited='false'` constraint and uses the wrong role literal casing.
- `02918` never applies `CreditCard.ExpYear=2007`.
- `03663` observes per-store inactive counts and selects literal store `1` without executing the
  requested maximum relation. The observed value is grounded, but the `argmax` is not.

These require path-independent counterfactual suites; adding more local provenance edges cannot
prove the omitted relation.

## Repaired dependency omission

The six mapping trajectories first expose an ID in a model-visible row and then use that exact ID
in a later predicate. Their source schemas often omit the FK or use semantically different names,
such as `Ball_by_Ball.Striker -> Player.Player_Id`.

`src/harness/provenance.py` now:

- prefers declared FK/column-equivalent matches;
- otherwise accepts one unique exact visible-cell copy;
- records the consuming argument path plus source row and column indices;
- withholds direct replay binding for ambiguous cells and multi-row selections.

Literal-slot parsing and stable singleton selection are centralized in
`src/harness/observation_binding.py`. Provenance is the only producer of binding metadata;
counterfactual replay consumes that metadata and does not run a second causality heuristic.

All six trajectories now reconstruct the missing `row_observation` edge. No model-facing tool or
argument changed.

The targeted 14-trajectory reward replay completed with 14/14 `replay_correct`, 14/14 deterministic
grounding complete, 14/14 action grounding clean, and zero replay errors. Relative to the prior
packages, exactly seven row edges were added: one each for `00739`, `03877`, `04109`, `05875`, and
`06577`, plus two for the two Solution-to-Repo mappings in `05873`. No new row edge was added to
`00541` or `02918`; `03663` retains its valid value-observation edge but, because its source is a
multi-row comparison, receives no counterfactual replay binding.

## Counterfactual replay correction

The former replay fixed every authored literal. That incorrectly rejects an interactive policy
whose later action naturally changes after a changed observation. Version 2 keeps operators fixed
but rebinds only harness-proven singleton visible-cell copies. Task literals remain constant.
Multi-row choices are left fixed because the choice can contain an unexecuted `argmax` or another
relation. This behavior is covered by a two-database test where an event-to-entity ID changes from
`7` to `42`: fixed-literal replay fails and observation-bound replay returns the correct changed
entity.

The counterfactual manifest schema is therefore `process-counterfactual-suite-v2`; v1 manifests
cannot authorize process training under the changed semantics.

## Release status

Process-RL remains gated. The changed visible-copy fallback must receive a fresh independent
edge-precision audit, and counterfactual BIRD suites must cover the three confirmed shortcut
classes before a process run is launched.

Local ignored audit artifacts:
`data/results/rl_grounding_shortcut_reaudit_20260724/`.
