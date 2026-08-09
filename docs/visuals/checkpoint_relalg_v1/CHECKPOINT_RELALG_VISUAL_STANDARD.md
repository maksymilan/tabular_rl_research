# Checkpoint-RelAlg Visual Communication Standard

Version: `checkpoint-relalg-visual-standard-v1`

This standard defines a repeatable OmniGraffle workflow for diagrams of the
checkpoint-relalg protocol family. It applies to architecture maps, tool-surface maps,
state machines, causal episode flows, audit boundaries, experiment comparisons, and
future protocol extensions.

## 1. Communication Contract

Every diagram must make the following distinctions visually explicit:

1. **Task authority**: `QUESTION` and `EXTERNAL KNOWLEDGE` define the requested semantics.
2. **Database authority**: `CURRENT ENVIRONMENT STATE` and immutable relation artifacts
   define observed database facts.
3. **Working memory**: checkpoint summaries and targets guide search but are not evidence.
4. **Private model material**: provider reasoning and assistant prose are audit records,
   never executable evidence.
5. **Hidden evaluation**: gold SQL, reference rows, and correctness feedback remain behind
   a one-way no-leak boundary.

Never place literal task answers, gold SQL, reference rows, API credentials, or chain-of-
thought content in a diagram. Use abstract labels such as “Provider-private reasoning” or
“Hidden reference evaluator”.

## 2. Canvas System

- Primary canvas: `1920 × 1080 px`, landscape, white or `Canvas` background.
- Safe margin: `64 px` on every side.
- Base grid: `8 px`; major alignment grid: `24 px`.
- Minimum inter-node gap: `24 px`; minimum group gap: `40 px`.
- Default flow: left to right. Use top to bottom only for long causal sequences.
- Use orthogonal connectors. A connector may cross a group boundary only once.
- Keep a single dominant reading path and no more than two secondary paths per canvas.

Recommended canvas sequence:

1. System Architecture and Authority Boundaries
2. Unified Tool Surface and Extended Bag Relational Algebra
3. Phase Reasoning, Semantic Checkpoints, and Recovery
4. Causal Episode Loop, Safety Invariants, and Admission Boundary
5. Visual Standard and Reusable Vector Grammar

## 3. Typography

All human-readable diagram text is English and uses **Times New Roman**.

| Token | Size | Weight | Use |
|---|---:|---|---|
| `Title` | 34 pt | Bold | Canvas title |
| `Subtitle` | 16 pt | Regular | Protocol/version/status line |
| `Section` | 20 pt | Bold | Swimlane or group heading |
| `NodeTitle` | 15 pt | Bold | Process, store, or tool-group title |
| `Body` | 12.5 pt | Regular | Explanatory text and field lists |
| `EdgeLabel` | 11 pt | Italic | Connector meaning |
| `Chip` | 10.5 pt | Bold | Status, mode, version, or policy badge |
| `Footnote` | 10 pt | Regular | Constraints and provenance |

Rules:

- Use sentence case for prose and title case only for canvas/group titles.
- Use exact tool names in monospace only when the export target supports it; otherwise keep
  Times New Roman and distinguish tools with a light code-card fill.
- Never reduce body text below `10 pt` in the final PDF.
- Keep line height between `1.15` and `1.25`.

## 4. Color Tokens

| Token | Hex | Semantic role |
|---|---|---|
| `Canvas` | `#F7F8FC` | Page background |
| `Ink` | `#172033` | Primary text and high-contrast stroke |
| `MutedInk` | `#667085` | Secondary text |
| `Border` | `#CBD5E1` | Neutral node and group borders |
| `Provider` | `#6658D3` | Model/provider plane |
| `Protocol` | `#246BCE` | Validation and protocol control |
| `State` | `#138A7E` | Environment state and artifacts |
| `Execution` | `#D97706` | SQLite and execution mechanics |
| `Checkpoint` | `#8B5CF6` | Checkpoint, restore, and phase memory |
| `Success` | `#26845B` | Successful transition or admitted fact |
| `Error` | `#C2414B` | Structured error and rollback |
| `Hidden` | `#475467` | Hidden evaluator/audit zone |
| `Warning` | `#B7791F` | Diagnostic-only or optional boundary |

Fills use the semantic color at `7–12%` opacity. Text on a colored solid must meet a
minimum WCAG contrast ratio of `4.5:1`. Do not encode state by color alone: pair every
color with a label, icon, line pattern, or status badge.

## 5. Shape and Vector-Icon Grammar

Use native OmniGraffle vectors or SVG vectors. Do not use raster icons inside a source
diagram.

| Concept | Shape/icon |
|---|---|
| Process | Rounded rectangle, `12 px` radius |
| Validator | Hexagon or shield |
| Decision | Diamond |
| Source database | Cylinder |
| Relation artifact | Table card with visible header/grid |
| Observation | Eye plus note card |
| Environment state | Solid semantic container with membership chips |
| Checkpoint | Stacked snapshot plus bookmark |
| Checkpoint graph | Circular nodes and orthogonal branches |
| Terminal answer | Stadium |
| Hidden evaluator | Dark dashed secure zone with lock |
| Error | Warning diamond or octagon |
| Immutable audit record | Document stack with check mark |

Icon rules:

- Use a `24 px` icon grid and `1.75 px` rounded strokes.
- Icons inherit the semantic color of their node.
- Prefer a literal structural icon (table, cylinder, snapshot) over a decorative metaphor.
- Reuse one icon for one concept across every canvas and protocol version.
- Keep icons to the left of labels, with an `8 px` gap.

## 6. Connector Grammar

| Connector | Style | Meaning |
|---|---|---|
| Successful semantic transition | `2.25 px` solid `Protocol` | Action/state transition |
| Data or relation flow | `2.25 px` solid `State` | Schema, rows, artifact, result |
| Execution request | `2.25 px` solid `Execution` | SQL or relational operator execution |
| Structured error | `2.25 px` solid `Error` | Rejection, rollback, latest error |
| Checkpoint/restore control | `2.25 px`, `8/5` dash `Checkpoint` | Phase boundary or restore |
| Optional/retry/diagnostic | `1.75 px`, `6/5` dash `Warning` | Non-semantic retry or optional path |
| Forbidden/no-leak direction | `1.5 px`, `2/5` dot `Hidden` with stop bar | Flow that must not occur |
| Inactive branch | `1.5 px`, `3/5` dot `MutedInk` | Abandoned checkpoint path |

Place edge labels on the final segment nearest the destination. Avoid diagonal connectors,
unlabeled bidirectional arrows, and decorative arrows that do not encode a flow.

## 7. Mathematical Typesetting

Every mathematical expression must originate from valid LaTeX source.

1. Store the canonical source in `checkpoint_relalg_formulas.tex`.
2. Render with LaTeX using a Times-compatible text/math package.
3. Convert each equation to vector SVG or PDF with fonts converted to paths.
4. Place the vector on the OmniGraffle `EQUATIONS` layer.
5. Store the exact source in the object note using the prefix `latex:`.
6. Do not recreate equations with ordinary text boxes or Unicode approximations.

Formula IDs are stable (`F01`, `F02`, …). When an equation changes, update the source,
regenerate the vector, and increment the diagram manifest hash.

## 8. OmniGraffle Layer Contract

Every source document uses these layers, in order:

1. `BACKGROUND`
2. `GROUPS`
3. `CONNECTORS`
4. `NODES`
5. `ICONS`
6. `LABELS`
7. `EQUATIONS`
8. `CALLOUTS`

Lock `BACKGROUND` and `GROUPS` before editing. Keep equations and icons separate from
labels so they can be regenerated without disturbing layout. Give important objects stable
names such as `STATE.EnvironmentState`, `TOOL.aggregate`, or `BOUNDARY.no_leak`.

## 9. Construction Workflow

1. **Inventory** the current executable tools, state fields, execution modes, error
   invariants, and admission status from code and the frozen specification.
2. **Choose the canvas type**: architecture, surface, state/phase, causal sequence, or
   comparison. Do not combine unrelated questions on one canvas.
3. **Write the reading path** as a one-sentence claim before drawing.
4. **Create groups and lanes**, then nodes, then connectors, then labels.
5. **Insert vector icons** from the shared grammar and render every equation from LaTeX.
6. **Validate authority boundaries** and remove any arrow that implies gold/model feedback.
7. **Check implementation identity**: protocol, mode, carrier, tool names, status, and hashes.
8. **Export and inspect** at 100% and at thumbnail size.

## 10. Quality Gate

Before release, verify:

- [ ] All diagram text is English and Times New Roman.
- [ ] Every equation has canonical LaTeX source and a vector rendering.
- [ ] Tool names exactly match the executable registry.
- [ ] Direct, Atomic, and Hybrid mode boundaries are explicit.
- [ ] Checkpoint history is labeled working memory, not evidence.
- [ ] Failure paths visibly preserve logical state and create no artifact.
- [ ] Hidden evaluator has no return edge to the model.
- [ ] Diagnostic-only or admission status is visible.
- [ ] Rasterization has not occurred in the OmniGraffle source.
- [ ] PDF, SVG, and PNG preview exports match the source.
- [ ] The diagram manifest records source commit, standard version, and artifact hashes.

## 11. Naming and Export

Use `snake_case` filenames:

```text
<protocol>_<topic>_v<visual-version>.<graffle|svg|pdf|png>
```

Export profiles:

- SVG: preserve vectors; embed or path-convert fonts and formulas.
- PDF: vector, page-sized to artwork, no downsampling.
- PNG preview: `1920 px` wide, sRGB, for quick review only.
- OmniGraffle: editable source of record.

The editable OmniGraffle file and the LaTeX source are authoritative. PNG is never the
source of record.
