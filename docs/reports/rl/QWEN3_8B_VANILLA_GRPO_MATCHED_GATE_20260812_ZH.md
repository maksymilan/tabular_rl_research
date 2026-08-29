# Qwen3-8B vanilla GRPO 严格 matched gate

日期：2026-08-12  
范围：只定义评测与晋级口径；不修改训练、rollout，也不干预正在运行的 NewGNN
checkpoint-4 全量评测。

## 当前证据不能证明负迁移

当前可完成配对的 checkpoint-2 结果是：

| arm | correct | accuracy | legal |
|---|---:|---:|---:|
| SFT1 checkpoint-560 历史全量结果 | 838/1534 | 54.628% | 1317/1534 |
| GRPO checkpoint-2 | 827/1534 | 53.911% | 1303/1534 |

checkpoint-2 相对历史 SFT1 有 78 个 gain、89 个 regression，净值 -11（-0.717 个百分点），
exact two-sided McNemar `p=0.4391`。legal 有 91 个 gain、105 个 regression，净值 -14
（-0.913 个百分点），`p=0.3531`。这是统计不显著的轻微下降，不能称为已证实的负迁移。

而且两臂不是严格 matched：SFT1 是 2026-08-07 的旧 runtime，
`max_inflight_requests=4`；checkpoint-2 是 2026-08-12 的冻结 runtime，最终并发 24。
已有 `unified_vs_sft1.json` 中两臂的 `evaluation_identity` 还都是 `null`：candidate 的
sidecar 位于 `run_root/evaluation_identity.json`，旧 analyzer 只寻找
`run_root/result/evaluation_identity.json`；历史 SFT1 本身也没有该 sidecar。因此这组结果只能作
diagnostic，不能进入严格晋级结论。

## 冻结评测设计

下一轮 vanilla GRPO 在训练前声明四个评测 arm：

1. `sft1`：Qwen3-8B SFT1 checkpoint-560，必须重新评测，不能复用并发 4 的历史结果；
2. `checkpoint2`：中期诊断；
3. `checkpoint4`：中期诊断；
4. `final`：训练前指定的唯一 primary candidate。

所有 arm 都运行 BIRD-dev 1534 个完全相同的 `example_index`，并逐项固定：

- Qwen3-8B base model 路径和 revision；
- SFT/RL adapter 文件 SHA-256 与 `adapter_config.json` SHA-256；
- 输入 JSONL SHA-256 和去除 DB 路径后的任务身份 SHA-256；
- atomic `version26` 与协议 hash `4da19387399bd3a5`；
- 同一只读 runtime 路径和 runtime 内容 SHA-256；
- 完全相同的 vLLM serving 参数；
- `workers=max_inflight_requests=max_num_seqs=24`；
- `temperature=0`、`top_p=1`、`max_tokens=2048`、thinking enabled；
- `max_steps=30`、rolling legal history 4、`bird-set`；
- SQLite tool timeout 10 秒。

GPU 编号和 served-model 名称可以不同；base revision、runtime 内容、serving 配置、并发、输入、
agent 和 decode 合约不能不同。每个 result arm 都必须在写入第一条结果前生成不可变 identity
sidecar。下一轮 sidecar 除现有字段外还必须补充：

- `base_model_revision`；
- `adapter_config_sha256`；
- `runtime_sha256`；
- `serving`（完整的 vLLM dtype、TP、model length、batched tokens、generation config 等）。

## Fail-closed analyzer

`analyze_evaluation_results.py` 的严格模式必须显式传入 sidecar，不跨目录猜测身份。标准调用的
关键部分如下；每个 checkpoint 都要增加对应 `--arm`、`--identity` 和 `--adapter-sha`：

```bash
python src/rl/diagnostics/analyze_evaluation_results.py \
  --examples /frozen/bird_dev1534.jsonl \
  --arm sft1=/eval/sft1/result/all.jsonl \
  --arm checkpoint2=/eval/checkpoint2/result/all.jsonl \
  --arm checkpoint4=/eval/checkpoint4/result/all.jsonl \
  --arm final=/eval/final/result/all.jsonl \
  --identity sft1=/eval/sft1/evaluation_identity.json \
  --identity checkpoint2=/eval/checkpoint2/evaluation_identity.json \
  --identity checkpoint4=/eval/checkpoint4/evaluation_identity.json \
  --identity final=/eval/final/evaluation_identity.json \
  --adapter-sha sft1=SFT1_SHA256 \
  --adapter-sha checkpoint2=CP2_SHA256 \
  --adapter-sha checkpoint4=CP4_SHA256 \
  --adapter-sha final=FINAL_SHA256 \
  --require-identities --require-distinct-adapters \
  --match-identity-field base_model \
  --match-identity-field base_model_revision \
  --match-identity-field adapter_config_sha256 \
  --match-identity-field dataset \
  --match-identity-field input_sha256 \
  --match-identity-field task_identity_sha256_without_db_path \
  --match-identity-field protocol_version \
  --match-identity-field protocol_hash \
  --match-identity-field runtime \
  --match-identity-field runtime_sha256 \
  --match-identity-field serving \
  --match-identity-field concurrency \
  --match-identity-field decode \
  --match-identity-field agent \
  --match-identity-field tool_execution_timeout_seconds \
  --compare checkpoint2:sft1 \
  --compare checkpoint4:sft1 \
  --compare final:sft1 \
  --expected-count 1534 \
  --protocol-version version26 \
  --protocol-hash 4da19387399bd3a5 \
  --temperature 0 --top-p 1 \
  --denotation-comparison bird-set \
  --output /eval/vanilla_grpo_unified.json
```

任何 missing/duplicate example、非单样本、row-level 协议不一致、identity 缺失、adapter SHA
不符、两臂误载同一 adapter，或者任一 matched 字段不同，都必须在统计前报错。

## 晋级阈值

运行可执行 gate：

```bash
python src/rl/diagnostics/audit_vanilla_grpo_matched_gate.py \
  --analysis /eval/vanilla_grpo_unified.json \
  --baseline sft1 \
  --candidate checkpoint2 \
  --candidate checkpoint4 \
  --candidate final \
  --primary-candidate final \
  --expected-count 1534 \
  --min-accuracy-gain-pp 1.0 \
  --alpha 0.05 \
  --output /eval/vanilla_grpo_matched_gate.json
```

只有 `final` 同时满足以下条件才可晋级：

1. 全 1534 条完成，身份/运行合约全部 matched；
2. 相对 fresh SFT1 的 accuracy 净提升至少 `+1.0` 个百分点，即至少净增 16 题；
3. paired gain/regression 的 exact two-sided McNemar `p<0.05`；
4. legal termination 净值不下降。

`checkpoint2` 和 `checkpoint4` 只用于观察学习曲线与定位退化，不得在看完结果后挑最高者替代
预注册的 `final`。若未来确实要“任一 checkpoint 可晋级”，必须在运行前声明多重比较校正；
当前 gate 有意采用 primary-only，避免 checkpoint cherry-picking。

解释口径：通过四项才是可报告的 RL 提升；正增益但低于幅度阈值或 `p>=0.05` 统一表述为“未检测
到可靠提升”；只有严格 matched 的 paired 结果显示显著负向差异时，才表述为“负迁移”。
