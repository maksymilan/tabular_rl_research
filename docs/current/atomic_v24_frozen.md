# Atomic v24 frozen baseline

日期：2026-08-12  
状态：实现与行为基线冻结；允许 scheme-local SFT candidate 导出，RL 仍未准入

## 冻结版本

新的稳定 Atomic profile 为 `atomic-v24-frozen-v1`。它用于后续 Atomic 工具设计的固定基线，
不继续继承 checkpoint-relalg 中 v5 输出提示或任何 checkpoint/restore 触发实验。

该 profile 不是对历史代码或结果的物理删除。`micro-v1`、`semantic-v2`、`semantic-v3-v24`、
`semantic-v4-v24-interface`、`semantic-v5-v24-output` 继续保留，只用于 frozen replay、历史比较与
审计；不得把它们的轨迹重标为本 profile。

## 来源与保留项

冻结基线以 Atomic version24 的工具设计方法和后续已验证的接口修复为来源：

- 使用任务级语义原子工具，而不是九个机械 relational micro operators；
- 使用 `describe_table`、`inspect_column`、`read_rows` 进行感知；
- 使用 `filter_rows`、`shape_rows`、`join`、`group_aggregate`、`scalar_compute`、
  `rank_select`、`set_operation` 构造关系；
- 使用一个 Harness-owned relation artifact 作为 `answer`；
- 保留 Text-JSON 单 action carrier、recent-4 causal history 和 v24 compact contract；
- 保留 typed predicate/expression、immutable relation artifacts、strict artifact audit、fresh replay；
- 保留 `semantic-v4-v24-interface` 的确定性执行修复：`shape_rows` / `rank_select` 省略 `as`
  时精确保留已有 logical column name，包括 dotted name 与空格；显式 `as` 仍只能是 simple identifier。

## 舍弃项

- 模型工具面不含 `commit_checkpoint` 或 `restore_checkpoint`；
- runtime 强制 `max_checkpoints=0`、`max_restores=0`；
- model-visible context 不渲染 `CURRENT PHASE TARGETS` 或 `CHECKPOINT HISTORY`；
- 不包含 `semantic-v5-v24-output` 的 prompt-only 输出标签提示；
- 不包含 milestone、initial-target、model-choice、stress、restore 等教师触发策略；
- 不包含 producer-count 或 ordinal checkpoint Harness gate。

CheckpointStore、历史 prompt/profile 和 replay 代码继续存在，以保证旧实验精确审计；它们不属于
`atomic-v24-frozen-v1` 的 public capability surface。

## 冻结身份

- mode：`atomic`
- carrier：`text-json`（A）
- operator profile：`atomic-v24-frozen-v1`
- required guidance identity：`checkpoint-disabled-v1`（只作为运行配置身份，不向模型插入 checkpoint 文案）
- provider history：`recent-4-turns-v1`
- schema delivery：`v24-frozen-no-checkpoint-contract-v1`
- rollout admission：`diagnostic-only`（runner 原始记录保持不变）
- SFT candidate admission：
  `checkpoint-relalg-atomic-v24-frozen-bird-set-sft-candidates-v1`

当前实现生成：

- tool schema SHA256：`bc997223394700b9f87903b9ccd4497976246e801b3d8a493044641d8409f926`
- student prompt SHA256：`dbba671ce36c8e9c131236002b39b6a2d25a073527c119409e9f4abacfb3f54d`
- teacher prompt SHA256：`947445f4959d3ee9201742495af664caa7f13a36197817411224058767b4deda`
- carrier protocol SHA256：`05716470b0865cbb20bca360b42afa35b4bcd607499e5a3104a7ef0ff7d69313`

这些 hash 是 executable identity，不得在不改 profile 名或版本的情况下漂移。

## 使用

正式 runner 必须显式提供：

```text
--mode atomic
--atomic-operator-profile atomic-v24-frozen-v1
--carrier text-json
--experiment-arm A
--checkpoint-guidance-profile checkpoint-disabled-v1
--max-checkpoints 0
--max-restores 0
```

当前 profile 仍不是 RL promotion。按照 2026-08-13 的数据构造决定，训练集 episode 在同时满足
official DeepSeek identity、隐藏 gold、`bird-set` terminal correct、legal termination、structure/no-leak
和逐 record fresh replay 后，可以通过专用 exporter 进入 **scheme-local SFT candidate**。Strict artifact
和 schema match 作为诊断标签保留，不再淘汰 terminal denotation 已正确的 episode。错误 action 只保留在
因果 prefix，不作为监督 target；产生零行 intermediate artifact 的 action 同样不作为 target。不得使用
旧 Atomic exporter，也不得把这些候选直接混入其他 tool scheme 或 RL。

专用 exporter 为 `src/tool_modules/checkpoint_relalg/sft_export.py`。它保留原生
`reasoning_content + content` 双通道消息，不静默改写为 inline `<think>`；具体学生模型的模板投影必须
另行显式实现和审计。

## 独立 Train200 行为基线

2026-08-12 在 teacher1500 v2 nonempty positions 320--519 上完成了 200 个官方
`deepseek-v4-flash` causal episode：`bird-set` 147/200（73.5%），strict artifact 63/200，
schema match 80/200，legal 192/200；使用 9,344,374 tokens。四个分片的 structure、provider
history、cohort、预算与 batch 审计全部通过；fresh replay 199/200，唯一失败来自错误轨迹中 live
SQL timeout 与 replay 成功的 wall-time 边界差异，147 条正确轨迹全部通过 fresh replay。

该结果建立冻结 profile 的绝对行为基线，但不是与历史 fixed-200 version24 的同题 paired comparison。
在 147 条 `bird-set` 正确记录中，63 条 strict；额外差异主要来自 reference SQL cursor column label 和
duplicate multiplicity。依据上述 SFT candidate admission，147 条正确且通过逐 record replay 的 episode
均可进入本 scheme 的候选池；strict/schema 不再是候选淘汰门。RL 仍保持不准入。
完整结果见
`../reports/evaluation/CHECKPOINT_RELALG_ATOMIC_V24_FROZEN_TRAIN200_RESULT_20260812_ZH.md`。
