"""Small, dependency-free utilities shared by RL data and audit tools."""

from .io import atomic_write_text, iter_jsonl, read_jsonl, sha256_file, sha256_bytes
from .stats import percentile, percentile_nearest_rank

__all__ = [
    "atomic_write_text",
    "iter_jsonl",
    "read_jsonl",
    "sha256_file",
    "sha256_bytes",
    "percentile",
    "percentile_nearest_rank",
]
