# Dataset Sampling Reference

## Output Shape

Each sample JSON should be an object:

```json
{
  "dataset": "local_key",
  "description": "short purpose",
  "source": {
    "repo_id": "namespace/dataset",
    "local_path": "data/...",
    "format": "jsonl",
    "split": "train"
  },
  "selection": {
    "mode": "first",
    "seed": null,
    "requested_records": 5,
    "actual_records": 5
  },
  "records": []
}
```

## Adding a Dataset

1. Add an `hf download` entry in `src/data_pipeline/download_datasets.sh`.
2. Add a `DatasetConfig` in `src/data_pipeline/extract_samples.py`.
3. If the dataset contains non-JSON table encodings, add a small postprocessor that preserves the raw field and adds a parsed JSON-friendly field.
4. Run extraction and inspect the output manually.
5. Update the nearest `agent.md` with durable path/schema notes.

For very large tables, prefer compact observation views in `data_sample/` and keep full-fidelity data in `data/`.

## Case Construction Heuristics

Good sample cases for table-tool RL usually expose at least one of:
- aggregation, comparison, filtering, sorting, joining, or cell lookup;
- multi-row or multi-column evidence;
- answer formats that can be checked by a reward function;
- table encodings that stress a proposed tool interface.
