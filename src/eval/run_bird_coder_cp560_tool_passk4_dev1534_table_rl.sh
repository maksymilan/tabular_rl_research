#!/usr/bin/env bash
set -euo pipefail

# One four-sample run computes matched pass@1, pass@2, and pass@4 metrics
# without duplicating the first two samples in a separate K=2 experiment.

PROJECT_DIR=${PROJECT_DIR:-/Users/hudou/Research/tabular_rl_research}

export BASE_MODEL=/home/dengyan/models/Qwen2.5-Coder-7B-Instruct
export ADAPTER=/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-student-raw-json-6400-qlora/checkpoint-560
export SERVED_MODEL=qwen25-coder7b-rawjson-cp560-tool-dev1534-passk4
export GPU_ID=1
export REMOTE_PORT=8036
export LOCAL_PORT=18036
export RESULT_DIR=data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_passk4_bird_set
export N=1534
export N_SAMPLES=4
export PASS_K=1,2,4
export MAX_STEPS=30
export MAX_TOKENS=1024
export TEMPERATURE=0.7
export TOP_P=0.95
export WORKERS=2
export SAMPLE_WORKERS=4
export MAX_INFLIGHT=8
export HISTORY_TURNS=4
export EVAL_ENABLE_THINKING=0
export EXISTING_SERVICE_POLICY=reject
export SERVER_CONFIG_ID=vllm-generation-config-vllm-cp560-tool-dev1534-passk4
export VLLM_LOG=/home/dengyan/tabular_rl_outputs/logs/qwen25_coder7b_cp560_tool_dev1534_passk4.vllm.log
export VLLM_PID_FILE=/home/dengyan/tabular_rl_outputs/logs/qwen25_coder7b_cp560_tool_dev1534_passk4.vllm.pid
export LOCAL_EVAL_LOG=/tmp/qwen25_coder7b_cp560_tool_dev1534_passk4.eval.log
export LOCAL_TUNNEL_LOG=/tmp/qwen25_coder7b_cp560_tool_dev1534_passk4.tunnel.log

cd "$PROJECT_DIR"
exec bash src/eval/run_bird_lora_tool_passk_table_rl.sh
