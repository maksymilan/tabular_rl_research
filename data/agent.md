# Data Directory Agent Notes

This directory contains downloaded full datasets and is intentionally ignored by git through `/data` in `.gitignore`.

Expected layout after running `scripts/data_pipeline/download_datasets.sh`:
- `tqabench/`: Hugging Face dataset `table-benchmark/tqabench`; train split stored as parquet shards under `data/train-*.parquet`.
- `FeTaQA/`: Hugging Face dataset `DongfuJiang/FeTaQA`; train/dev/test JSONL files.
- `TableBench/`: Hugging Face dataset `Multilingual-Multimodal-NLP/TableBench`; core TQA JSONL plus DP/SCoT/TCoT/PoT instruction JSONL files.
- `spider/`: Hugging Face dataset `xlangai/spider`; `spider/train-00000-of-00001.parquet` contains 7,000 examples and `spider/validation-00000-of-00001.parquet` contains 1,034 examples.

Spider download note:
- The Hugging Face dataset contains `db_id`, natural-language questions, target SQL, and tokenized question/query fields.
- It does not contain the original Spider SQLite database files or full database schemas. Download the upstream Spider resources separately if SQL execution against the source databases is required.

Do not treat these raw files as stable hand-authored repo artifacts. Prefer updating download and sampling scripts rather than editing files here.
