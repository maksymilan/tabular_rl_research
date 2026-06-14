---
name: tabular-dataset-case-builder
description: Use when working in this table RL research repo to download Hugging Face table QA datasets, inspect their local schema, extract small JSON samples for tool-use case construction, or update the dataset pipeline and agent notes.
---

# Tabular Dataset Case Builder

## Purpose

Use this skill to prepare table QA/reasoning datasets for LLM table-tool research. The goal is to keep full datasets reproducible while producing compact JSON samples that are easy to inspect when designing table tools, tool-call trajectories, and reward functions.

## Standard Workflow

1. Read `agent.md` at the repo root, then read the nearest directory-specific `agent.md` for the files you will touch.
2. Confirm or update the dataset list in `src/data_pipeline/download_datasets.sh`.
3. Download datasets with:

```bash
bash src/data_pipeline/download_datasets.sh
```

4. Inspect the downloaded structure with `find data -maxdepth 4 -type f -print` and dataset README files.
5. Update `DATASET_CONFIGS` in `src/data_pipeline/extract_samples.py` when new files, splits, or formats are added.
6. Generate samples:

```bash
python3 src/data_pipeline/extract_samples.py --dataset all --num 5
```

7. Verify each output in `data_sample/` is valid JSON and includes `dataset`, `source`, `selection`, and `records`.
8. Record durable context in `agent.md` files near the relevant directories.

## Local Conventions

- Full datasets live in `data/` and are ignored by git.
- Human-readable samples live in `data_sample/`.
- Pipeline code lives in `src/data_pipeline/`.
- Prefer project-relative paths over machine-specific paths such as `/data/dengyan/...`.
- Keep sampling deterministic by default. Use `--mode random --seed <seed>` only when diverse examples are needed.
- Use `duckdb` CLI for parquet sampling when Python parquet libraries are unavailable.

## Dataset Notes

Current configured datasets:
- `table-benchmark/tqabench`: parquet train shards. Its `table` and `text` fields are serialized Python literals; parse and compact them into JSON-friendly fields for inspection.
- `DongfuJiang/FeTaQA`: JSONL free-form table QA data.
- `Multilingual-Multimodal-NLP/TableBench`: JSONL core TQA examples plus DP/SCoT/TCoT/PoT instruction variants.

For additional dataset details, read `references/dataset-sampling.md` only when extending the pipeline or debugging sample extraction.
