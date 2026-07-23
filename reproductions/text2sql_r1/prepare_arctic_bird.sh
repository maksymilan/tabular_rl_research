#!/usr/bin/env bash
set -euo pipefail

REPRO_ROOT="${REPRO_ROOT:-/home/dengyan/tabular_rl_outputs/text2sql_reproduction}"
ARCTIC_CODE="${ARCTIC_CODE:-$REPRO_ROOT/third_party/ArcticTraining/projects/arctic_text2sql_r1}"
BIRD_ROOT="${BIRD_ROOT:-/home/dengyan/tabular_rl_project/data/bird/dev_20240627}"
PROCESS_PYTHON="${PROCESS_PYTHON:-$REPRO_ROOT/envs/arctic_process/bin/python}"
JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-11-openjdk-amd64}"
NLTK_DATA="${NLTK_DATA:-$REPRO_ROOT/envs/nltk_data}"
INDEX_ROOT="${INDEX_ROOT:-$REPRO_ROOT/data/bird_dev_contents_index}"
TEMP_ROOT="${TEMP_ROOT:-$REPRO_ROOT/data/arctic_index_tmp}"
FULL_OUTPUT="${FULL_OUTPUT:-$REPRO_ROOT/data/dev_bird_arctic.json}"
SMOKE_OUTPUT="${SMOKE_OUTPUT:-$REPRO_ROOT/data/dev_bird_arctic_smoke8.json}"
SMOKE_GOLD="${SMOKE_GOLD:-$REPRO_ROOT/data/bird_dev_smoke8.json}"

mkdir -p "$REPRO_ROOT/data" "$NLTK_DATA"
export JAVA_HOME NLTK_DATA
export PATH="$(dirname "$PROCESS_PYTHON"):$PATH"

if [[ ! -f "$INDEX_ROOT/.complete" ]]; then
  if [[ -d "$INDEX_ROOT" ]] && find "$INDEX_ROOT" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite a non-empty incomplete index: $INDEX_ROOT" >&2
    exit 2
  fi
  (
    cd "$ARCTIC_CODE"
    "$PROCESS_PYTHON" data_preprocessing/build_contents_index.py \
      --db-root "$BIRD_ROOT/dev_databases" \
      --index-root "$INDEX_ROOT" \
      --temp-dir "$TEMP_ROOT" \
      --threads 16
  )
  touch "$INDEX_ROOT/.complete"
fi

if [[ ! -s "$FULL_OUTPUT" ]]; then
  (
    cd "$ARCTIC_CODE"
    "$PROCESS_PYTHON" data_preprocessing/process_dataset.py \
      --input_data_file "$BIRD_ROOT/dev.json" \
      --output_data_file "$FULL_OUTPUT" \
      --db_path "$BIRD_ROOT/dev_databases/" \
      --tables "$BIRD_ROOT/dev_tables.json" \
      --source bird \
      --mode dev \
      --value_limit_num 2 \
      --db_content_index_path "$INDEX_ROOT"
  )
fi

"$PROCESS_PYTHON" -c \
  'import json,sys; processed=json.load(open(sys.argv[1])); raw=json.load(open(sys.argv[2])); seen=set(); idx=[]; [(idx.append(i),seen.add(x["db_id"])) for i,x in enumerate(raw) if x["db_id"] not in seen and len(idx)<8]; json.dump([processed[i] for i in idx],open(sys.argv[3],"w"),indent=2,ensure_ascii=False); json.dump([raw[i] for i in idx],open(sys.argv[4],"w"),indent=2,ensure_ascii=False); print("prepared",len(processed),"full and stratified smoke indices",idx)' \
  "$FULL_OUTPUT" "$BIRD_ROOT/dev.json" "$SMOKE_OUTPUT" "$SMOKE_GOLD"
