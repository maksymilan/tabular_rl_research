#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  printf 'usage: %s OWNER/MODEL DEST_DIR REVISION\n' "$0" >&2
  exit 2
fi

REPO_ID="$1"
DEST_DIR="$2"
REVISION="$3"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
ARIA_CONNECTIONS="${ARIA_CONNECTIONS:-8}"

[[ "$REPO_ID" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] || exit 2
[[ "$DEST_DIR" == /home/dengyan/* ]] || exit 2
[[ "$REVISION" =~ ^[A-Fa-f0-9]{40}$ ]] || exit 2
[[ "$HF_ENDPOINT" == "https://hf-mirror.com" ]] || exit 2
[[ "$ARIA_CONNECTIONS" =~ ^[1-9][0-9]*$ ]] || exit 2
(( ARIA_CONNECTIONS <= 16 )) || exit 2

for command in curl python3 aria2c; do
  command -v "$command" >/dev/null
done

mkdir -p "$DEST_DIR/.download-metadata"
TREE_JSON="$DEST_DIR/.download-metadata/tree.json"
ARIA_INPUT="$DEST_DIR/.download-metadata/aria2-input.txt"
EXPECTED_TSV="$DEST_DIR/.download-metadata/expected-files.tsv"
ARIA_LOG="$DEST_DIR/.download-metadata/aria2.log"

curl -fsSL --retry 8 --retry-delay 3 \
  "$HF_ENDPOINT/api/models/$REPO_ID/tree/$REVISION?recursive=true&expand=false" \
  -o "$TREE_JSON"

python3 - "$TREE_JSON" "$ARIA_INPUT" "$EXPECTED_TSV" \
  "$REPO_ID" "$REVISION" "$DEST_DIR" "$HF_ENDPOINT" <<'PY'
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

tree_path, aria_path, expected_path, repo_id, revision, dest_dir, endpoint = sys.argv[1:]
entries = json.loads(Path(tree_path).read_text())
files = []
for entry in entries:
    if entry.get("type") != "file":
        continue
    path = entry.get("path")
    size = entry.get("size")
    if isinstance(path, str) and path and isinstance(size, int):
        files.append((path, size))
if not files:
    raise SystemExit("repository tree contained no sized files")
with Path(aria_path).open("w") as aria, Path(expected_path).open("w") as expected:
    for path, size in sorted(files):
        parent = os.path.dirname(path)
        target_dir = os.path.join(dest_dir, parent) if parent else dest_dir
        os.makedirs(target_dir, exist_ok=True)
        url = (
            f"{endpoint}/{repo_id}/resolve/{quote(revision, safe='')}/"
            f"{quote(path, safe='/')}"
        )
        aria.write(f"{url}\n  dir={target_dir}\n  out={os.path.basename(path)}\n")
        expected.write(f"{size}\t{path}\n")
print(json.dumps({"files": len(files), "expected_bytes": sum(x[1] for x in files)}))
PY

aria2c \
  --input-file="$ARIA_INPUT" \
  --continue=true \
  --auto-file-renaming=false \
  --allow-overwrite=true \
  --max-tries=0 \
  --retry-wait=5 \
  --timeout=120 \
  --connect-timeout=20 \
  --max-connection-per-server="$ARIA_CONNECTIONS" \
  --split="$ARIA_CONNECTIONS" \
  --min-split-size=16M \
  --file-allocation=none \
  --console-log-level=notice \
  --summary-interval=20 \
  --log="$ARIA_LOG" \
  --log-level=notice

python3 - "$DEST_DIR" "$EXPECTED_TSV" "$REVISION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dest_dir, expected_path, revision = sys.argv[1:]
errors = []
files = []
for line in Path(expected_path).read_text().splitlines():
    size_text, relative = line.split("\t", 1)
    expected = int(size_text)
    target = Path(dest_dir, relative)
    if not target.is_file():
        errors.append(f"missing: {relative}")
        continue
    actual = target.stat().st_size
    if actual != expected:
        errors.append(f"size mismatch: {relative}: {actual} != {expected}")
    files.append({"path": relative, "size_bytes": actual})
partials = [str(path) for path in Path(dest_dir).rglob("*.aria2")]
errors.extend(f"incomplete: {path}" for path in partials)
if errors:
    raise SystemExit("\n".join(errors))
manifest = {
    "schema_version": "hf-mirror-snapshot-v1",
    "revision": revision,
    "files": files,
}
encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
Path(dest_dir, "snapshot_download_manifest.json").write_bytes(encoded)
print(json.dumps({
    "status": "ok",
    "files": len(files),
    "bytes": sum(item["size_bytes"] for item in files),
    "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
}))
PY
