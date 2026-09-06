from pathlib import Path

from rl.shared.io import atomic_write_text, read_jsonl, sha256_bytes, sha256_file
from rl.shared.stats import percentile


def test_jsonl_digest_and_atomic_write_are_shared(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    atomic_write_text(path, '{"id": 1}\n')
    assert read_jsonl(path) == [{"id": 1}]
    assert sha256_file(path) == sha256_bytes(path.read_bytes())
    assert percentile([1, 2, 3], 0.5) == 2.0
