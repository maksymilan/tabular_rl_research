# Qwen3-8B 原子工具 SFT1：同协议 base / QLoRA 评测

这个目录提供一条隔离的评测路径：用完全相同的历史 `version26` 原子工具协议，分别评测
Qwen3-8B base 和由外部教师 SFT1 训练出的 QLoRA adapter。这样得到的两项结果可以回答
“SFT1 在当前工具协议上，相对同底模起点提升了多少”。

它不是当前协议的开发分支。运行时代码只从 commit
`4cd47c957fc6ae791e76a10594c8cd22f4d3b6de` 用 `git archive` 导出，不会导入工作区当前的
version39/version54 模块。该历史边界只用于复现已经冻结的 SFT1 数据协议，不应成为后续功能开发
的起点。

## 锁定的评测合同

- 工具协议：atomic `version26`，carrier=`think-json-v1`；
- rolling full system prompt SHA-256：
  `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`；
- 数据：本地固定的 BIRD-Dev 2024-06-27，共 1,534 题；
- greedy：每题 1 条轨迹，`temperature=0`、`top_p=1`；
- `max_steps=30`、`max_tokens>=2048`；
- `rolling-legal-history`、`history_turns=4`、full prompt、resident observation；
- 评分：`bird-set`；
- 官方 Qwen3 tokenizer/chat template，逐请求传
  `chat_template_kwargs={"enable_thinking": true}`；
- vLLM 不启用 reasoning parser，客户端读取原始 `message.content`，因此
  `<think>...</think>` 仍由 version26 strict parser 校验。

正式的 all-NewGNN 路径在运行前执行 fail-closed gate：

1. `verify_version26_runtime.py` 校验导出树的 101 个文件、关键文件哈希、隔离 import 路径、
   protocol version、prompt hash，以及不存在 reasoning parser；
2. `prepare_remote_eval_inputs.py` 只把 1,534 条输入的 `db_path` 确定性映射到 NewGNN，保持其余
   字段不变，并锁定源 JSONL、派生 JSONL、manifest 和 11 个 SQLite 数据库的 SHA-256；
3. `verify_remote_eval_assets.py` 复核 Qwen3 固定 revision、五个模型 shard，以及 adapter 臂固定的
   `checkpoint-560`、`global_step=560` 和 adapter 权重/config/trainer-state 哈希。

## 1. 正式推荐路径：all-NewGNN detached evaluation

正式长评测使用 `launch_detached_remote_newgnn.sh`。启动时 SSH 只负责上传自校验 controller bundle
和提交任务；之后 vLLM、历史 version26 harness、BIRD 数据库和结果写入全部在 NewGNN 上完成，二者
通过 `127.0.0.1` 通信。supervisor 使用 `nohup + setsid` 脱离 SSH，因此 Mac 休眠或 SSH 断开不会
中断评测。

当前正式配对配置为：

- base：NewGNN 物理 GPU 5，localhost port `8020`；
- adapter：NewGNN 物理 GPU 6，localhost port `8021`；
- 两臂均为单卡 TP=1，`MAX_MODEL_LEN=16384`、`MAX_NUM_BATCHED_TOKENS=16384`、
  `MAX_NUM_SEQS=4`、`GPU_MEMORY_UTILIZATION=0.90`、`WORKERS=4`、`MAX_INFLIGHT=4`；
- adapter 固定为
  `/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560`。

先执行无 GPU、无 SSH 的本地 dry-run：

```bash
MODE=base RUN_ID=qwen3_base_atomic_v26_remote_20260807 \
  bash reproductions/trust_sql/qwen3_8b_atomic_sft1/launch_detached_remote_newgnn.sh --dry-run
```

提交 base 臂：

```bash
MODE=base RUN_ID=qwen3_base_atomic_v26_remote_20260807 \
  bash reproductions/trust_sql/qwen3_8b_atomic_sft1/launch_detached_remote_newgnn.sh
```

提交 adapter 臂：

```bash
MODE=adapter RUN_ID=qwen3_adapter_atomic_v26_sft1_remote_20260807 \
  bash reproductions/trust_sql/qwen3_8b_atomic_sft1/launch_detached_remote_newgnn.sh
```

两臂分别写入全新的独立目录：

```text
/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26/qwen3_base_atomic_v26_remote_20260807
/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26/qwen3_adapter_atomic_v26_sft1_remote_20260807
```

状态文件分别是：

```text
/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26/qwen3_base_atomic_v26_remote_20260807/status.json
/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26/qwen3_adapter_atomic_v26_sft1_remote_20260807/status.json
```

`status.json` 的 `state` 明确区分 `starting`、`running`、`completed` 和 `failed`；失败时保留 stage、
异常和清理结果；完成时必须通过 1,534 个唯一题号的覆盖检查，且任一 sample 出现
`failure_type=api_error` 或 `ChatAPIError` 都会 fail closed，完成状态明确记录 `api_error_count=0`。
每个 `result/` 还保存
`version26_runtime_gate.json` 和 `qwen3_model_gate.json`；两臂的 runtime gate 必须逐字节相同。

GPU 和 port 可以显式覆盖，但正式配对应保持上述 GPU5/GPU6 计划，不应在看过结果后改变服务或并发
参数。每个 `RUN_ID` 只能使用一次；launcher 和 supervisor 都会拒绝复用已有目录。

## 2. 历史 runtime 的本地导出与测试

```bash
bash reproductions/trust_sql/qwen3_8b_atomic_sft1/prepare_version26_runtime.sh
```

默认导出到被 Git 忽略的：

```text
tmp/version26-runtime-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de
```

脚本对 dirty worktree 不敏感，因为只读取指定 Git object。目标已存在时只做完整校验，不覆盖。

正式 NewGNN runtime 已锁定在：

```text
/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de
```

## 3. 旧 tunnel launcher 的适用边界

`run_greedy_bird_dev1534_table_rl.sh` 保留用于短时 smoke/诊断和本地 boundary dry-run，不再作为
1,534 题正式长评测方案。它让 Mac 上的 harness 通过 SSH local-forward 访问远端 vLLM；Mac 休眠、
网络切换或 SSH tunnel 断开时，尚未完成的请求会被历史 runner 记成 `api_error`。之后即便 tunnel
恢复，已经写入结果目录的错误记录仍会污染汇总，不能视作模型真实性能。

因此正式 base/adapter 配对不得从该旧 launcher 的长评测结果继续 `--resume`，也不得把 tunnel
故障产生的 `api_error` 与 all-NewGNN 结果合并。需要短诊断时仍可执行：

```bash
bash reproductions/trust_sql/qwen3_8b_atomic_sft1/run_greedy_bird_dev1534_table_rl.sh --dry-run
```

## 结果如何解释

最关键的配对比较是：

```text
同一个 Qwen3-8B revision + 同一个 atomic version26 环境
base greedy EX  →  SFT1 QLoRA greedy EX
```

这能把底模、chat template、工具集、上下文和 scorer 固定住，形成当前 SFT 工作的可信起点。
但它不能直接等同于论文的 TRUST-SQL 47.9% raw baseline：论文使用 SQL exploration/proposal 工具和
完整 transcript，而这里使用 resident-state 原子关系工具。若要与论文结果并列，应同时报告已完成的
TRUST-SQL 作者协议复现（709/1534，46.22%）以及这里的 atomic base/adapter 配对结果，不能把两种
protocol 的绝对 EX 当成同一实验。

### 2026-08-08 正式结果

两臂均完成 1,534/1,534 个唯一题号，且无 API/断线错误：base 为 270/1534（17.60%），
checkpoint-560 SFT1 QLoRA 为 838/1534（54.63%）。逐题共有 600 个 accuracy gain、32 个
regression，净增 568 题（+37.03 个百分点），双侧 exact McNemar `p=8.50977e-137`。合法终止由
586/1534（38.20%）提升到 1317/1534（85.85%）。

严格分析、完整训练合同、哈希和解释边界见
`docs/reports/sft/QWEN3_8B_ATOMIC_V26_SFT1_BASELINE_20260808_ZH.md`；最终只读归档为
`data/results/qwen3_8b_atomic_v26_sft1_artifact_manifest.json`，状态为 `ok`、0 errors、0 incomplete。

## 本地测试

```bash
python3 -m unittest \
  reproductions.trust_sql.qwen3_8b_atomic_sft1.test_version26_eval_boundary \
  reproductions.trust_sql.qwen3_8b_atomic_sft1.test_remote_newgnn_eval \
  reproductions.trust_sql.qwen3_8b_atomic_sft1.test_archive_experiment_artifacts
```

测试只在临时目录导出、篡改或重映射固定输入，验证 gate 会拒绝漂移，并执行 launcher dry-run；
不会发起 SSH、模型加载或评测。

## 后验只读归档

训练或评测仍在进行时，可以生成明确标记为不完整的单文件草稿：

```bash
python3 reproductions/trust_sql/qwen3_8b_atomic_sft1/archive_experiment_artifacts.py \
  --mode draft \
  --output /tmp/qwen3_8b_atomic_v26_sft1_artifacts.draft.json
```

草稿模式不会重读五个大 base shard，以免给运行中的任务增加整模型规模的磁盘 I/O；它保留 launch
manifest 中固定的 shard 声明，并以 `base_shard_rehash_deferred` 明确报告尚未完成的实际复核。若确有
需要，可显式传 `--rehash-base-shards-in-draft`。

训练及 base/adapter 两个 1534-task 评测全部完成后，使用 fail-closed 最终模式：

```bash
python3 reproductions/trust_sql/qwen3_8b_atomic_sft1/archive_experiment_artifacts.py \
  --mode final \
  --output data/results/qwen3_8b_atomic_v26_sft1_artifact_manifest.json
```

最终模式会实际重算固定五个 base shard，以及 checkpoint-560、训练 launch/status/log、数据、两个评测
目录和环境版本的哈希；任一文件缺失/变化、`global_step != 560`、评测不足 1534 条或协议/版本漂移都会
写出 `status=failed` 并返回非零状态。脚本只读源产物；唯一写入是用户指定的单个 JSON 输出，已存在的
输出默认拒绝覆盖，只有显式 `--overwrite` 才会原子替换。

全 NewGNN 运行同步回来的结果目录若含 `remote_eval_asset_gate.json`，归档器还会固定校验原始 BIRD
输入哈希 `8bf5a8...23e0`、仅改写 `db_path` 后的输入哈希 `636e09...21f5`、1534 条记录、
`only_db_path_changed=true` 以及 base/adapter 模式，并归档 gate 自身 SHA。两臂远端 run dir 不同，故
这类结果以相同的 derived-input SHA 证明数据同一性，不比较 manifest 中必然不同的绝对路径；旧式本地
两臂都没有该 gate 时，仍兼容原来的 manifest dataset 路径比较。只出现单臂 gate 不视为完整配对。
