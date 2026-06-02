# Git Commit Conventions

This repository uses a lightweight Conventional Commits style with research-specific guardrails.

## Commit Message Format

Use:

```text
<type>(<scope>): <summary>
```

Rules:
- Keep the summary under 72 characters when practical.
- Use lowercase `type` and `scope`.
- Write the summary in imperative style, for example `add tablebench samples`.
- Add a body when the change affects data assumptions, experiment reproducibility, reward design, or tool interfaces.

## Types

- `data`: dataset downloads, sample extraction outputs, dataset metadata, or data pipeline changes.
- `feat`: new research/tooling capability.
- `fix`: bug fix.
- `exp`: experiment configuration, training/evaluation setup, or result-oriented research iteration.
- `reward`: reward function design or reward signal changes.
- `tool`: table tool schema, execution, parser, or tool-use interface changes.
- `docs`: documentation, notes, or `agent.md` updates.
- `skill`: Codex skill creation or updates.
- `chore`: maintenance with no behavior/data impact.

## Scopes

Prefer concrete scopes:
- `pipeline`
- `samples`
- `fetaqa`
- `tablebench`
- `tqabench`
- `tools`
- `rewards`
- `skills`
- `agents`

Examples:

```text
data(pipeline): add table qa dataset sampling flow
reward(cell-evidence): reward highlighted cell lookup
tool(table-api): add row filtering interface
exp(tablebench): configure pot baseline run
docs(agents): record dataset directory assumptions
```

## Data And Artifact Rules

- Do not commit full downloaded datasets under `data/`.
- It is okay to commit small inspection samples under `data_sample/`.
- Commit scripts and metadata needed to reproduce data artifacts.
- When committing generated samples, mention the script, sampling mode, count, and seed if random.
- Keep `agent.md` notes durable and factual; avoid temporary logs or guesses.

## Recommended Pre-Commit Checks

Run the checks that match the change:

```bash
bash -n scripts/data_pipeline/download_datasets.sh
python3 scripts/data_pipeline/extract_samples.py --dataset all --num 5
python3 -m json.tool data_sample/tablebench_sample.json >/dev/null
```

For future code-heavy changes, add project-specific tests here as they emerge.
