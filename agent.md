# Project Agent Notes

This is a research project on improving LLM tool-use ability over tables with reinforcement learning. The central research focus is tool design and reward function design for table-centric reasoning.

Current data pipeline intent:
- Download public table QA/reasoning datasets into `data/`.
- Extract small, readable JSON samples into `data_sample/` for inspecting case structure and designing tool-use examples.
- Keep full datasets out of git; `.gitignore` already ignores `/data`.

Downloaded datasets:
- `table-benchmark/tqabench` -> `data/tqabench`
- `DongfuJiang/FeTaQA` -> `data/FeTaQA`
- `Multilingual-Multimodal-NLP/TableBench` -> `data/TableBench`

Useful commands:
- Download/update datasets: `bash scripts/data_pipeline/download_datasets.sh`
- Extract default samples: `python3 scripts/data_pipeline/extract_samples.py --dataset all --num 5`
- Extract deterministic random samples: `python3 scripts/data_pipeline/extract_samples.py --dataset all --num 5 --mode random --seed 20260602`

Important implementation notes:
- `tqabench` stores examples as parquet; this repo uses the `duckdb` CLI for sampling so it does not require local `pandas`, `pyarrow`, or `datasets`.
- `FeTaQA` and `TableBench` are JSONL and can be sampled with Python standard library only.
- `tqabench` has serialized Python literal strings in fields such as `table` and `text`; the extractor parses them into compact JSON-friendly `table` and `text` views for easier inspection.

Git conventions:
- Follow `docs/git_commit_conventions.md`.
- Use Conventional Commit style, e.g. `data(pipeline): add table qa dataset sampling flow`.
- Do not commit full downloaded datasets under `data/`; small inspection samples under `data_sample/` are allowed.
