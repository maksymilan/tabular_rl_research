# Data Directory Agent Notes

This directory contains downloaded full datasets and is intentionally ignored by git through `/data` in `.gitignore`.

Expected layout after running `scripts/data_pipeline/download_datasets.sh`:
- `tqabench/`: Hugging Face dataset `table-benchmark/tqabench`; train split stored as parquet shards under `data/train-*.parquet`.
- `FeTaQA/`: Hugging Face dataset `DongfuJiang/FeTaQA`; train/dev/test JSONL files.
- `TableBench/`: Hugging Face dataset `Multilingual-Multimodal-NLP/TableBench`; core TQA JSONL plus DP/SCoT/TCoT/PoT instruction JSONL files.

Do not treat these raw files as stable hand-authored repo artifacts. Prefer updating download and sampling scripts rather than editing files here.
