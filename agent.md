# Project Agent Notes

This is a research project on improving LLM tool-use ability over tables with reinforcement learning. The central research focus is tool design and reward function design for table-centric reasoning.

Current data pipeline intent:
- Download public table QA/reasoning datasets into `data/`.
- Extract small, readable JSON samples into `data_sample/` for inspecting case structure and designing tool-use examples.
- Keep full datasets out of git; `.gitignore` already ignores `/data`.

Tool-use research design:
- `tool_design/agent.md` summarizes the current Table Agent Harness design.
- `tool_design/tool_usage.md` and `tool_design/tool_json_examples/` define the canonical tool output, intermediate state, memory, and example JSON formats for future tool design changes.
- `tool_design/trajectory/` stores complete tool-use trajectories by question type, built from `data_sample/` examples.
- The current design models table reasoning as dynamic table-context construction plus static task-memory maintenance.
- First-pass tools include dataset inspection, column retrieval, row retrieval, context pruning, memory updates/refinement, and evidence-cited final answering.
- Current priority is broad tool viability across task types; concrete reward optimization can be deferred and tracked as known defects.
- Tool history should be recorded internally by the harness for duplicate-call checks and analysis, but not included in model-visible context by default.
- Memory writes should carry supporting judgment/evidence fields so later use can avoid unsupported task-level hallucinations.

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
