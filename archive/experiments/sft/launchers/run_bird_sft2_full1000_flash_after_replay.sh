#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}
LOG=${LOG:-/tmp/bird_sft2_full1000_flash_after_replay.log}
POSTPROCESS_LABEL=com.tabularrl.bird-sft2-full1000-postprocess
PREPARE_MANIFEST=data/trajectories/bird_sft2_student_full1000_prepare.manifest.json
TASKS=data/eval_inputs/bird_sft2_flash_fallback_full1000_all368.jsonl
OLD_PREFIX=data/trajectories/bird_sft2_flash_fallback_all53
NEW_PREFIX=data/trajectories/bird_sft2_flash_fallback_full1000_all368

cd "$PROJECT_DIR"
exec >>"$LOG" 2>&1
echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] waiting for SFT2 replay manifest"
while [ ! -f "$PREPARE_MANIFEST" ]; do
  if launchctl print "gui/$(id -u)/$POSTPROCESS_LABEL" 2>/dev/null | grep -q 'state = not running'; then
    code=$(launchctl print "gui/$(id -u)/$POSTPROCESS_LABEL" | awk '/last exit code =/{print $NF; exit}')
    echo "postprocess stopped before manifest, exit=$code"
    exit 1
  fi
  sleep 60
done

.venv/bin/python -c 'import json,sys; from pathlib import Path; m=json.load(open(sys.argv[1])); rows=[x for x in Path(sys.argv[2]).read_text().splitlines() if x.strip()]; assert m["teacher_fallback_tasks"]==368, m["teacher_fallback_tasks"]; assert len(rows)==368, len(rows)' "$PREPARE_MANIFEST" "$TASKS"

if [ ! -e "${NEW_PREFIX}_all.jsonl" ]; then
  .venv/bin/python -c 'import json,sys; from pathlib import Path; tasks={str(x.get("trajectory_id") or x.get("example_id") or "rollout_train_"+str(x["example_index"])) for x in map(json.loads,Path(sys.argv[1]).read_text().splitlines())}; old=[json.loads(x) for x in Path(sys.argv[2]).read_text().splitlines() if x.strip()]; ids={x["trajectory_id"] for x in old}; assert len(old)==len(ids)==53,(len(old),len(ids)); assert ids<=tasks,sorted(ids-tasks)[:5]' "$TASKS" "${OLD_PREFIX}_all.jsonl"
  cp "${OLD_PREFIX}_all.jsonl" "${NEW_PREFIX}_all.jsonl"
  cp "${OLD_PREFIX}_success.jsonl" "${NEW_PREFIX}_success.jsonl"
  cp "${OLD_PREFIX}_failures.jsonl" "${NEW_PREFIX}_failures.jsonl"
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] seeded 53 prior Flash attempts; 315 tasks remain"
fi

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] replay accepted; resuming Flash fallback to 368"
.venv/bin/python -u src/sft/rollout_external_data.py \
  --split train --examples-file "$TASKS" --start 0 --limit 368 \
  --model deepseek-v4-flash \
  --out data/trajectories/bird_sft2_flash_fallback_full1000_all368_success.jsonl \
  --failures-out data/trajectories/bird_sft2_flash_fallback_full1000_all368_failures.jsonl \
  --all-out data/trajectories/bird_sft2_flash_fallback_full1000_all368_all.jsonl \
  --workers 4 --max-steps 30 --max-errors-per-type 3 --attempts-per-example 1 \
  --max-tokens 2048 --api-timeout 180 --api-retries 3 --table-output-rows 0 \
  --context-mode rolling-legal-history --history-turns 4 --rolling-prompt-variant full --resume
echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] Flash fallback 368 complete"
