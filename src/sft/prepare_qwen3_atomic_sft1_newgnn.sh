#!/usr/bin/env bash
set -euo pipefail

# Prepare a Qwen3-native, token-audited view of the frozen version26 SFT1 data.
# This script never changes the final target.  It removes historical reasoning
# exactly as Qwen3's official chat template does before the next generation.

PROJECT_DIR=${PROJECT_DIR:-/home/dengyan/tabular_rl_project}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/dengyan/tabular_rl_outputs}
ENV_DIR=${ENV_DIR:-/home/dengyan/miniconda3/envs/qwen3-atomic-sft}
MODEL_DIR=${MODEL_DIR:-/home/dengyan/models/Qwen3-8B-TrustSQL-baseline}
SOURCE_DIR=${SOURCE_DIR:-$OUTPUT_ROOT/data/omnisql_sft1_union_sft1_sft2_20260801/sources/sft1}
SOURCE=${SOURCE:-$SOURCE_DIR/bird_external_teacher_fixed1000_student_raw_json_audited_6400.jsonl}
SOURCE_INDEX=${SOURCE_INDEX:-$SOURCE_DIR/bird_external_teacher_fixed1000_student_raw_json_audited_6400.index.jsonl}
DATASET_DIR=${DATASET_DIR:-$OUTPUT_ROOT/data/qwen3_8b_atomic_v26_sft1_20260806}
EXPECTED_SOURCE_SHA256=39a8a298bd67f1a931ff582331cab6574185c1b8d17f4f23ac5de40cc4031310
EXPECTED_INDEX_SHA256=3c4f3ad4b01de0f9fbebfe838d79ca5648f7252fccee9f318bc33a709a219cac
EXPECTED_PROMPT_SHA256=848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316
EXPECTED_RECORDS=4471
EXPECTED_EPISODES=678

ALIGNED=$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_history_aligned.jsonl
ALIGNED_INDEX=$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_history_aligned.index.jsonl
TOKEN_AUDITED=$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_history_aligned_token_audited_6400.jsonl
TOKEN_AUDIT=${TOKEN_AUDITED%.jsonl}.token_audit.json
CANONICAL=$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_complete_6400.jsonl
CANONICAL_INDEX=$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_complete_6400.index.jsonl
TRAINING_VIEW=$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_6400_training_view.jsonl
SMOKE_VIEW=$DATASET_DIR/bird_external_teacher_qwen3_8b_atomic_v26_sft1_longest_smoke.jsonl
PREP_MANIFEST=$DATASET_DIR/preparation_manifest.json
LOG=$DATASET_DIR/preparation.log

for required in \
  "$SOURCE" \
  "$SOURCE_INDEX" \
  "$ENV_DIR/bin/python" \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/tokenizer_config.json" \
  "$PROJECT_DIR/src/sft/project_qwen3_history_view.py" \
  "$PROJECT_DIR/src/sft/rebind_sft_index_to_model_view.py" \
  "$PROJECT_DIR/src/sft/audit_rolling_lf_tokens.py" \
  "$PROJECT_DIR/src/sft/filter_sft_by_token_audit.py" \
  "$PROJECT_DIR/src/sft/project_sft_training_view.py" \
  "$PROJECT_DIR/src/sft/build_longest_record_smoke_view.py"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 2
  fi
done

if [[ -e "$DATASET_DIR" ]] && [[ -n "$(find "$DATASET_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty dataset directory: $DATASET_DIR" >&2
  exit 3
fi

actual_source_sha256=$(sha256sum "$SOURCE" | awk '{print $1}')
actual_index_sha256=$(sha256sum "$SOURCE_INDEX" | awk '{print $1}')
if [[ "$actual_source_sha256" != "$EXPECTED_SOURCE_SHA256" ]]; then
  echo "unexpected SFT1 source sha256: $actual_source_sha256" >&2
  exit 4
fi
if [[ "$actual_index_sha256" != "$EXPECTED_INDEX_SHA256" ]]; then
  echo "unexpected SFT1 index sha256: $actual_index_sha256" >&2
  exit 5
fi

mkdir -p "$DATASET_DIR"
cd "$PROJECT_DIR"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

{
  echo "[1/6] project Qwen3-native history and prove HF/LLaMA-Factory parity"
  "$ENV_DIR/bin/python" src/sft/project_qwen3_history_view.py \
    --input "$SOURCE" \
    --out "$ALIGNED" \
    --audit-model "$MODEL_DIR" \
    --audit-template qwen3 \
    --audit-cutoff-len 6400

  echo "[2/6] rebind index hashes to the actual Qwen3 model view"
  "$ENV_DIR/bin/python" src/sft/rebind_sft_index_to_model_view.py \
    --view "$ALIGNED" \
    --canonical "$SOURCE" \
    --source-index "$SOURCE_INDEX" \
    --out "$ALIGNED_INDEX" \
    --projection qwen3-history-think-strip-v1

  echo "[3/6] exact LLaMA-Factory token audit"
  "$ENV_DIR/bin/python" src/sft/audit_rolling_lf_tokens.py \
    --input "$ALIGNED" \
    --out "$TOKEN_AUDITED" \
    --model "$MODEL_DIR" \
    --template qwen3 \
    --enable-thinking \
    --cutoff-len 6400 \
    --filter-policy full-prefix

  echo "[4/6] complete-episode admission"
  "$ENV_DIR/bin/python" src/sft/filter_sft_by_token_audit.py \
    --input "$ALIGNED" \
    --index "$ALIGNED_INDEX" \
    --token-audit "$TOKEN_AUDIT" \
    --out "$CANONICAL" \
    --index-out "$CANONICAL_INDEX" \
    --dataset-name bird_external_teacher_qwen3_8b_atomic_v26_sft1_complete_6400 \
    --episode-policy keep-complete-only

  echo "[5/6] schema-stable trainer view"
  "$ENV_DIR/bin/python" src/sft/project_sft_training_view.py \
    --input "$CANONICAL" \
    --index "$CANONICAL_INDEX" \
    --out "$TRAINING_VIEW" \
    --dataset-name bird_external_teacher_qwen3_8b_atomic_v26_sft1_6400_training_view

  echo "[6/6] longest-record distributed smoke view"
  "$ENV_DIR/bin/python" src/sft/build_longest_record_smoke_view.py \
    --training-view "$TRAINING_VIEW" \
    --index "$CANONICAL_INDEX" \
    --token-audit "$TOKEN_AUDIT" \
    --out "$SMOKE_VIEW" \
    --dataset-name bird_external_teacher_qwen3_8b_atomic_v26_sft1_longest_smoke \
    --repeats 32
} > "$LOG" 2>&1

"$ENV_DIR/bin/python" - \
  "$SOURCE" "$SOURCE_INDEX" "$ALIGNED" "$ALIGNED_INDEX" "$TOKEN_AUDIT" "$CANONICAL" \
  "$CANONICAL_INDEX" "$TRAINING_VIEW" "$SMOKE_VIEW" \
  "$PREP_MANIFEST" "$EXPECTED_PROMPT_SHA256" "$EXPECTED_RECORDS" \
  "$EXPECTED_EPISODES" <<'PY'
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

source = Path(sys.argv[1])
source_index = Path(sys.argv[2])
aligned = Path(sys.argv[3])
aligned_index = Path(sys.argv[4])
token_audit_path = Path(sys.argv[5])
canonical = Path(sys.argv[6])
canonical_index = Path(sys.argv[7])
training_view = Path(sys.argv[8])
smoke_view = Path(sys.argv[9])
manifest_path = Path(sys.argv[10])
expected_prompt_sha = sys.argv[11]
expected_records = int(sys.argv[12])
expected_episodes = int(sys.argv[13])

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

source_rows = read_jsonl(source)
index_rows = read_jsonl(source_index)
canonical_rows = read_jsonl(canonical)
canonical_indexes = read_jsonl(canonical_index)
training_rows = read_jsonl(training_view)
smoke_rows = read_jsonl(smoke_view)
token_audit = json.loads(token_audit_path.read_text(encoding="utf-8"))
selection = json.loads(canonical.with_suffix(".manifest.json").read_text(encoding="utf-8"))
projection = json.loads(training_view.with_suffix(".manifest.json").read_text(encoding="utf-8"))
alignment_manifest = Path(str(aligned) + ".manifest.json")
alignment = json.loads(alignment_manifest.read_text(encoding="utf-8"))
parity = alignment.get("prefix_parity_audit") or {}
index_projection_manifest = aligned_index.with_suffix(".manifest.json")
index_projection = json.loads(index_projection_manifest.read_text(encoding="utf-8"))

if not all(len(rows) == expected_records for rows in (
    source_rows, index_rows, canonical_rows, canonical_indexes, training_rows,
)):
    raise SystemExit("record-count gate failed")
if selection["source_episodes"] != expected_episodes or selection["complete_episodes"] != expected_episodes:
    raise SystemExit("complete-episode gate failed")
if selection["dropped_records"] != 0:
    raise SystemExit("token admission dropped records")
if token_audit["kept_by_filter_policy"] != expected_records:
    raise SystemExit("full-prefix token gate failed")
if token_audit["preserve_thinking"] is not False:
    raise SystemExit("Qwen3 history policy mismatch")
if token_audit.get("enable_thinking") is not True:
    raise SystemExit("Qwen3 thinking-mode gate failed")
if token_audit["encoded_lengths"]["max"] > token_audit["cutoff_len"]:
    raise SystemExit("encoded record exceeds cutoff")
if parity.get("exact_prefix_token_sequences") != expected_records:
    raise SystemExit("Qwen3 prefix parity gate failed")
if parity.get("exact_final_target_fields") != expected_records:
    raise SystemExit("final-target identity gate failed")
if alignment.get("final_targets_exactly_preserved") != expected_records:
    raise SystemExit("alignment projection changed final targets")
if index_projection.get("records") != expected_records:
    raise SystemExit("derived-index record gate failed")
if index_projection.get("canonical_to_view_model_input_hashes_changed") != alignment.get(
    "records_with_historical_assistant"
):
    raise SystemExit("derived-index model-input hash gate failed")
if index_projection.get("target_hashes_changed_by_projection") != 0:
    raise SystemExit("derived-index target hash gate failed")
if index_projection.get("source_index_model_input_hashes_matching_canonical") != 0:
    raise SystemExit("unexpected historical source-index input-hash state")
if index_projection.get("source_index_target_hashes_matching_canonical") != 0:
    raise SystemExit("unexpected historical source-index target-hash state")
if len(smoke_rows) != 32:
    raise SystemExit("longest-record smoke view count mismatch")

prompt_hashes = Counter(
    hashlib.sha256(row["system"].encode("utf-8")).hexdigest() for row in source_rows
)
if prompt_hashes != Counter({expected_prompt_sha: expected_records}):
    raise SystemExit(f"student prompt hash gate failed: {prompt_hashes}")
files = [
    source, source_index, aligned, alignment_manifest,
    aligned_index, index_projection_manifest,
    token_audit_path, canonical, canonical_index,
    canonical.with_suffix(".manifest.json"), training_view,
    training_view.with_suffix(".manifest.json"),
    smoke_view, smoke_view.with_suffix(".manifest.json"),
    training_view.parent / "dataset_info.json",
    Path.cwd() / "src/sft/project_qwen3_history_view.py",
    Path.cwd() / "src/sft/rebind_sft_index_to_model_view.py",
    Path.cwd() / "src/sft/audit_rolling_lf_tokens.py",
    Path.cwd() / "src/sft/filter_sft_by_token_audit.py",
    Path.cwd() / "src/sft/project_sft_training_view.py",
    Path.cwd() / "src/sft/build_longest_record_smoke_view.py",
    Path.cwd() / "src/sft/prepare_qwen3_atomic_sft1_newgnn.sh",
]
payload = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment": "qwen3-8b-historical-atomic-version26-sft1-qlora",
    "protocol_boundary": (
        "frozen historical version26 single-action think-json-v1; not version54 "
        "native-tool-bundle and not TRUST-SQL's tool protocol"
    ),
    "records": expected_records,
    "complete_episodes": expected_episodes,
    "feedback_recovery_targets": selection["feedback_recovery_targets"],
    "prompt_sha256": expected_prompt_sha,
    "history_policy": "Qwen3 official template: omit prior think, retain prior raw JSON action",
    "final_targets_changed": 0,
    "qwen3_prefix_parity_records": parity["exact_prefix_token_sequences"],
    "canonical_to_qwen3_model_input_hashes_changed": index_projection[
        "canonical_to_view_model_input_hashes_changed"
    ],
    "historical_source_index_hash_note": (
        "the frozen source index predates the tagged-to-raw carrier migration; its stored "
        "input/target hashes match 0 current canonical rows. The derived index preserves "
        "those values as source_index_* lineage and binds fresh canonical/view hashes."
    ),
    "encoded_lengths": token_audit["encoded_lengths"],
    "target_lengths": token_audit["target_lengths"],
    "files": {str(path): sha256(path) for path in files},
    "git_head": subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    ).stdout.strip(),
}
manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

echo "prepared=$DATASET_DIR"
echo "manifest=$PREP_MANIFEST"
echo "log=$LOG"
