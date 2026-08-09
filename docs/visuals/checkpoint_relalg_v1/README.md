# Checkpoint-RelAlg OmniGraffle Visual System

This bundle is the editable and reproducible visual description of the
`checkpoint-relalg-v1` tool surface and causal reasoning process. All diagram labels
are English Times New Roman. Every mathematical expression is generated from canonical
LaTeX and imported as vector paths.

## Editable OmniGraffle Sources

1. `01_checkpoint_relalg_system_architecture.graffle` — authority boundaries, provider
   carriers, Harness validation, state, execution, and isolated scoring.
2. `02_checkpoint_relalg_tool_surface.graffle` — Direct, Atomic, and Hybrid modes; shared
   perception/control; immutable artifacts; extended bag relational algebra.
3. `03_checkpoint_relalg_checkpoint_reasoning.graffle` — phases, semantic checkpoints,
   restore branches, transcript reset, and structural-sharing snapshots.
4. `04_checkpoint_relalg_causal_execution_audit.graffle` — causal loop, state-preserving
   failures, resource limits, terminal scoring, fresh replay, and admission boundary.
5. `05_checkpoint_relalg_visual_standard.graffle` — reusable typography, colors, vector
   grammar, connector semantics, LaTeX rules, and release checklist.

Each canvas also has SVG and PDF vector exports plus a PNG review preview. Imported SVG
elements remain individually selectable in OmniGraffle.

## Source of Truth

- `CHECKPOINT_RELALG_VISUAL_STANDARD.md` defines the reusable drawing contract.
- `checkpoint_relalg_formulas.tex` is the canonical formula source.
- `equations/F01.svg` through `F12.svg` are path-only LaTeX renderings.
- `equations/formula_sources.json` binds each canonical formula source to its reviewed
  vector output, so unchanged equations are reused byte-for-byte across rebuilds.
- `build_checkpoint_relalg_visuals.py` deterministically regenerates vector canvases,
  previews, equation assets, and `visual_manifest.json`.
- `visual_manifest.json` records protocol identity and content hashes.

## Regeneration

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  docs/visuals/checkpoint_relalg_v1/build_checkpoint_relalg_visuals.py
```

The build requires `latex`, `pdflatex`, `dvisvgm`, and `rsvg-convert`. To refresh an
editable OmniGraffle source, open the corresponding generated SVG in OmniGraffle and save
it under the matching `.graffle` filename. Do not flatten text, icons, or equations.

## Extension Workflow

1. Start from the style-guide canvas and the written visual standard.
2. Choose one communication question per new canvas.
3. Reuse semantic shapes, colors, icons, and connector patterns.
4. Add or modify formulas only in the canonical LaTeX source and formula registry.
5. Regenerate exports, save the editable OmniGraffle document, and refresh the manifest.
6. Check the release checklist at normal size and thumbnail size.

The diagrams are descriptive and diagnostic. They do not admit trajectories to SFT or RL,
and they never expose gold SQL, reference rows, literal answers, credentials, or private
chain-of-thought.
