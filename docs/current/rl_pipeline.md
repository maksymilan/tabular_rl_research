# Atomic version26 RL 运行契约

更新时间：2026-09-08

启动前同时读取[`rl_performance.md`](rl_performance.md)：区分CUDA Graph执行优化与
sampling-score算法ablation，检查8B launcher覆盖、checkpointing和实际runtime身份。

当前只保留一条 RL 方案：**result-only four-level reward + correctness-primary-clean-secondary
advantage + SAAM asymmetric-error credit +
reason/tool 加权 full-response policy loss**。它是最终执行方案；新 cohort 正式运行和 matched
promotion 完成前，不把它写成已验证的 accuracy 提升。

## 固定身份

- environment：`src/rl/runtime/tool_environment_v26.py`
- protocol runtime：Atomic version26，commit `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de`
- protocol hash：`4da19387399bd3a5`
- student prompt SHA：`848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`
- model：Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`
- initial adapter：cumulative SFT `checkpoint-6380`（`checkpoint-560` 仅为小样本 RL 可行性验证）
- action carrier：`think-json-v1`
- causal history：recent 4 legal turns；max 30 agent steps；max 4,096 new tokens per turn

## Reward

Reward 只读取 Harness 的 terminal denotation 和结构化 error bit，不读取 reasoning、gold
SQL 或 gold trajectory：

| terminal | 无 Harness error | 有 Harness error |
|---|---:|---:|
| correct | +1.5 | +1.0 |
| incorrect | -0.5 | -1.0 |

timeout 属于有结构化 Harness error 的 policy action；即使后续恢复并答对，timeout action
也要按局部负向惩罚计入 SAAM credit，而不是 zero credit。

实现：`src/rl/runtime/terminal_reward.py` 的 `four-level` profile。timeout 按当前 transition
contract 处理，不把模型文字解释当作额外 reward 事实。

组内 advantage 不直接标准化四等级数值。使用 `correctness-primary-clean-secondary`，
`clean_advantage_weight=0.25`：正确/错误决定符号；无 Harness error 与有 error 只使用有界
倍率调节幅度。全对和全错组仍为零 advantage。terminal four-level reward 仍完整记录，
但不直接作为组内 advantage 的符号来源。

## SAAM asymmetric-error credit

实现：`src/rl/frameworks/trl/state_action_ambiguity.py` 和 TRL transition trainer。

1. 对每个完整 `(state, full action)` 建立跨结果组的匹配。
2. 同一动作同时出现在正确和错误轨迹时，正确组保留其正优势，错误组该动作优势置零，避免
   结果标签把共享前缀误归因给 actor。
3. 仅局部错误动作使用 `-max(abs(advantage), lambda_error)`，当前
   `lambda_error=1.0`；timeout action 同样使用 `-max(abs(advantage), lambda_error)`。
4. 其余动作沿 vanilla trajectory advantage；所有 mask/count 必须写入 transition manifest。

policy reduction 为 `trajectory_token_mean`；完整 response 使用 reason/tool 各 0.5 的
`span_balance_alpha=0.5` 加权梯度。不启用 tool-only、PCGrad 或 rank loss。当前正式 arm 的
KL 为 `kl_beta=0`；后续必须用独立匹配 KL arm 验证是否带来增益。该 arm 只能改变
`kl_beta`，固定 `checkpoint-6380`、screened cohort、decode/runtime、GPU 拓扑、optimizer
updates 和评测口径，并写入独立 output root。系数/调度在启动前预注册，结果出来后不得回挑。

## 正式预算

| 参数 | 固定值 |
|---|---:|
| cohort | 目标约500题；每题K=8且正确数2–6；只使用已筛选RL数据 |
| prompts/update | 30 |
| rollouts/prompt | K=8 |
| optimizer updates | 200（四轮覆盖） |
| optimizer | AdamW，LR `4e-7`，weight decay `0.1`，clip `0.2` |
| PPO iterations | 1 |
| agent limits | max steps 30，history 4，`max_new_tokens=4096`/turn，temperature 0.8，top-p 1 |
| actor | Qwen3-8B full trainable |

配置：`src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml`。

## A100 运行拓扑

- 一张 A100 做 replicated BF16 actor trainer（`world_size=1`）。
- 另一张不同的 A100 起 online vLLM；两张卡只能从 GPU 0–3 中选择。
- 正式动态 transition packing 上限：先以 `max_rows=4`、`token_budget=16384` 完成单卡 live
  gate；gate 通过后才可使用 `max_rows=8`、`token_budget=32768` 的正式上限。实际 GPU id、
  master/vLLM port 和 runtime path 由 launcher 写入 manifest，不在文档中永久绑定。trainer
  为单进程 `world_size=1`，不初始化 NCCL；actor 基座保持 BF16，AdamW 状态使用 FP32。显存不足时
  只能显式切换 `OPTIMIZER_NAME=paged_adamw_8bit`，并在独立 manifest 中标注为降级路径。
- canonical launcher：
  `src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`

launcher 必须 fail-closed：preflight cohort/config/adapter，检查目标 GPU/端口，训练完成后
校验 run manifest、implementation lock、precision audit、checkpoint 和 replay/audit。

## 3090 降级拓扑

如果 `a100` 不可用，目标是在 `table_rl` 或 `NewGNN` 以相同协议运行单卡 replicated trainer，并
用另一张独立 3090 运行 online vLLM。3090 路径必须使用独立 launcher、output root 和 manifest；默认从
`max_rows=1`、`token_budget=8192` 起步，显存不足时使用显式的 4-bit base 或
`OPTIMIZER_NAME=paged_adamw_8bit`，并记录降级项。它保持相同 reward、credit、cohort、update
和 matched evaluation 口径，不得与 A100 结果直接合并。无法同时获得两张空闲 3090 时，
launcher 必须 fail-closed。当前 A100 launcher 不自动接受 3090；3090 正式 launcher 通过
同等 preflight 和 live gate 后才能启动。

## 准入和评测

1. 正式实验启动前，先向用户列出完整配置和身份清单，至少包括配置及哈希、模型/checkpoint、
   cohort及哈希、protocol/prompt/runtime、reward/credit、rollout预算、optimizer、GPU拓扑、
   输出目录和评测计划；只有用户明确确认后才能启动。
2. 先在 A100 的 GPU 0–3（或 3090 服务器的实际空闲 GPU 对）完成 1-update、30题/update live
   gate，确认 replicated trainer、vLLM、权重同步和 transition loss 无 OOM/NCCL 错误。
3. 再完成约500题/200-update formal run；输出必须完整且可 fresh replay。
4. 训练后在 `table_rl` 或 `NewGNN` 做同协议 matched candidate-vs-baseline greedy eval，
   题号覆盖、API error、runtime/prompt/checkpoint identity 全部通过才可比较。
5. 只有 matched gate 通过后才能决定是否 promotion；candidate-only 结果不能当作提升。

评测使用 `evaluation.md` 的统一 handoff 规则。KL 对照要求和待登记的系数/调度见
`decision_register.md`；在对照完成前不对 KL 的收益或损失下结论。

## 明确不采用

binary-only、execution-ladder、tool-only/fixed-span reward、PCGrad、rank loss、checkpoint-
relalg、Atomic v24/v39/v51/v54、Direct/Hybrid/iterative-SQL 和 projection/rewrite/delete
均为历史诊断；保留用于审计，但不得进入当前 RL config、cohort 或 output root。

## 流水线与评测启动检查清单（2026-09-17 事故后新增，启动前必读）

2026-09-17 的"训练成功但评测全灭"事故由三个原因叠加造成（详见
`project_records/experiments.md` 与 `decisions.md`）。以下条目是**下次启动前必须逐条核对**的清单，
不是历史记录。

### A. 多阶段无人值守流水线（训练 → 评测）

1. **`PYTHONPATH` 必须显式导出**：任何调用
   `formal_v26_rollout_passk.py` / `make_eval_shards.py` / `merge_eval_shards.py` 的脚本都要
   `export PYTHONPATH=<project>/src`。缺失时 controller 在启动第一秒就
   `ModuleNotFoundError: No module named 'rl'`（本次两次评测因此秒退，训练白跑一晚）。
2. **阶段失败不得吞掉后续阶段**：后置阶段（如已完成臂的评测）必须在前置阶段失败时仍然执行；
   用 `set +e; ...; rc=$?; set -e` 显式接管返回码，禁止依赖 `set -e` 让整条链中止。
3. **阶段之间等待 GPU 空闲**：训练 launcher 释放显存有延迟，评测 launcher 是 fail-closed
   （>512 MiB 或有 compute 进程即退出）。加 30 分钟上限的轮询等待。
4. **输出目录必须全新**：评测 launcher 遇到已存在目录直接
   `ERROR: results directory already exists; resume is forbidden`（rc=2）。每次重试换新后缀，
   不要复用失败目录，否则会伪装成"又失败了一次"。
5. **端口先检查再启动**，两个 shard server 端口必须不同且空闲；训练与评测端口段分开。
6. **不要用会匹配到自身命令行的 `pkill -f <pattern>`**（会把 ssh 远端 shell 一起杀掉）；
   用 `pkill -f '[v]llm...'` 或按 PID。
7. pipeline status/log 放在固定目录，脚本自身把输出追加进 log；每一步写 `stageX_*` 状态，
   便于隔夜后定位"卡在哪一步"。
8. 启动前用 `bash src/rl/scenarios/diagnostics/preflight_eval_environment.sh <project_src> <gpu0> <gpu1>`
   做一次环境预检（见 C）。

### B. 评测用 vLLM 启动（table_rl / NewGNN 通用）

1. **必须导出 `TRITON_LIBCUDA_PATH=$PYTHON_ENV/var/triton-libcuda`**（评测 launcher 现已内置
   推导与存在性检查）。缺失时会出现两个签名，**它们是同一个原因**：
   - `RuntimeError: Bytes object is corrupted, checksum does not match`（triton 复用坏掉的
     shim 缓存）；
   - `CalledProcessError: ['/usr/bin/gcc','/tmp/.../cuda_utils.c',...,'-lcuda',...]`
     （重建 shim 时 `-lcuda` 链接失败；驱动目录只有 `libcuda.so.1`，没有链接可见 `libcuda.so`）。
   反证：同环境 `torch.compile` 与 `triton_backend()` 正常；设上该变量后 vLLM 能
   `Capturing CUDA graphs` 并 `Application startup complete`。
2. **不要用 `--enforce-eager` 绕过**：`evaluation_config.json` 会记录 `vllm.enforce_eager`，
   与 baseline（0）不一致，评测不再 matched。它只能作为一次性诊断手段。
3. 启动成功的判据是 `logs/vllm_gpu{0,1}.log` 出现 **`Application startup complete`** 且 worker
   日志出现 "loaded N examples from .../shards/gpuX.jsonl"；`vllm_pids.txt` 存在**不算**成功。
4. 失败会留下 `owned_process_cleanup.json` 与残留进程；重试前确认无残留 vLLM/eval 进程、
   显存已回到 ≤512 MiB。
5. 每次评测必须核对身份：`error_feedback_version=actionable-error-v1`、输入 SHA（BIRD-dev
   `8bf5a8bf…`）、runtime 树 SHA、protocol/prompt、`temperature=0`、`max_tokens=2048`、
   `max_steps=30`、2 卡 even/odd 分片、24 workers。
6. 新机/新快照首次评测前，先确认该 project 快照里的评测 launcher 已包含本次修复
   （`grep -c TRITON_LIBCUDA_PATH .../run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh` ≥ 1）。

### C. 一条命令的预检

```bash
bash src/rl/scenarios/diagnostics/preflight_eval_environment.sh \
  /home/dengyan/tabular_rl_outputs/<run>/project/src 0 1
```

它会检查：python 环境与 triton shim 存在、`triton_backend()` 可用、`import rl` 可用（带
`PYTHONPATH`）、目标 GPU 是否空闲（≤512 MiB 且无 compute 进程）、给定端口是否空闲。任一项失败
即非零退出。
