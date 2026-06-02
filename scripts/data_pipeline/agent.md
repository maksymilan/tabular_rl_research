# Data Pipeline Agent Notes

This folder owns dataset download and sample extraction.

Scripts:
- `download_datasets.sh`: downloads the configured Hugging Face datasets into `data/` by default. It supports `DATA_DIR=/custom/path`.
- `extract_samples.py`: extracts readable JSON samples into `data_sample/` by default. It supports `--dataset`, `--num`, `--mode first|random`, `--seed`, `--base-dir`, and `--out-dir`.

Operational notes:
- Use `hf download` for complete dataset snapshots.
- Use `duckdb` CLI for parquet sampling from `tqabench`.
- Keep outputs deterministic unless the user explicitly wants broader random exploration.
- When adding new datasets, extend `DATASET_CONFIGS` in `extract_samples.py` and add the Hugging Face repo/local name pair to `download_datasets.sh`.
