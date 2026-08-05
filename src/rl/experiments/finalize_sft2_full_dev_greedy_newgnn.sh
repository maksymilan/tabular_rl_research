#!/usr/bin/env bash
# Merge the two complete SFT2 full-dev greedy shards and build one official metric artifact.
set -euo pipefail
O=/home/dengyan/tabular_rl_outputs
R=$O/eval_runtime_version36_20260728
PY=/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python
S0=$R/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_evenodd0_20260801
S1=$R/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_evenodd1_20260801
OUT=$R/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801
STATUS=$O/logs/sft2_full_dev_greedy_finalize_20260801.status
timestamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
set_status() { printf '%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" >"$STATUS"; }
exec 9>"$O/logs/sft2_full_dev_greedy_finalize_20260801.lock"
flock -n 9 || exit 0
if [[ -f "$OUT/accuracy.json" && "$(wc -l <"$OUT/all.jsonl" | tr -d '[:space:]')" -eq 1534 ]]; then
  set_status complete "rows=1534 result=$OUT"
  exit 0
fi
for parity in 0 1; do
  f="$O/logs/sft2_version36_dev1534_greedy_t0_logprobs20_evenodd${parity}_20260801.status"
  if ! grep -q $'\tcomplete\t' "$f" 2>/dev/null; then
    set_status waiting_shards "parity=$parity status=$f"
    exit 0
  fi
done
if [[ -e "$OUT" ]]; then
  set_status failed "partial merged output exists; audit before retry path=$OUT"
  exit 3
fi
set_status merging "shards=2 expected=1534"
cd "$R"
"$PY" src/eval/merge_disjoint_eval_shards.py \
  --shard "$S0" --shard "$S1" --output-dir "$OUT" --expected 1534
"$PY" src/eval/build_experiment_eval_metrics.py "$OUT"
"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
rows=[json.loads(line) for line in (p/'all.jsonl').open() if line.strip()]
assert len(rows)==1534
assert {int(row['example_index']) for row in rows}==set(range(1534))
assert all(len(row.get('samples') or [])==1 for row in rows)
assert all(float(row['temperature'])==0.0 and float(row['top_p'])==1.0 for row in rows)
assert all(row['protocol_version']=='version36' for row in rows)
accuracy=json.loads((p/'accuracy.json').read_text())
assert accuracy['pass@1']['total']==1534
PY
set_status complete "rows=1534 result=$OUT"
