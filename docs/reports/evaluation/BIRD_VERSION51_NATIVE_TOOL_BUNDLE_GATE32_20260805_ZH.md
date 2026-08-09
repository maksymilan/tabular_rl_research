# BIRD Version51 原生 Tool Bundle Gate32

状态：**已完成并通过扩大门禁。** 本结果授权新的固定 100/200 题 paired gate，不授权
SFT/RL。

## 变更与假设

Version50 强制每回合恰好一个 native tool call，并把多调用或非空 assistant content 判为
协议错误。Fixed-200 中因此出现 124 个 carrier 错误。Version51 将同一 provider assistant
消息中的 1..8 个直接原子调用视为一个真实模型回合，并为每个 call id 返回一个工具结果。
所有调用先按同一 bundle pre-state 校验，再依 provider 顺序执行；终止调用必须单独出现。

主假设：接受官方多调用语义会恢复 version50 的 carrier-failure 题，同时不会显著破坏无
carrier 错误的正确控制题，并减少无效重试、过程错误和 token。

## 冻结 cohort

- 来源：`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`
- 来源 SHA-256：`6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`
- 追踪的冻结索引：
  `docs/reports/evaluation/cohorts/bird_version51_native_bundle_gate32_indices.json`
- 索引 SHA-256：`b7415f9ec6c42fc169e0ff033debb52cb27c2bcf9f5a19a1a18aff25233e8fa4`
- 目标 16 题：version50 错误，且至少出现一次双/三调用或非空 content carrier rejection；
- 控制 16 题：version50 正确，且没有任何 carrier rejection；
- 两组都按 `example_index` 排序后取前 16，结果产生后不得换题。

冻结 version50 同题基线：16/32 correct、26/32 legal、37 errors、234 primitive actions、
2,426,690 tokens；失败为 10 个 wrong-answer、6 个 protocol-error。

## 固定运行条件

- 官方 `https://api.deepseek.com/chat/completions`，`deepseek-v4-flash`；
- K=1，32 题，`bird-set`，catalog-v1，optional plan；
- recent-4 provider assistant turns；
- `thinking={"type":"enabled"}`、`reasoning_effort=high`、`tool_choice=auto`；
- completion budget 2,048，最多 30 model turns，每类错误上限 3；
- `version51`、`native-tool-bundle`、diagnostic-only；
- 不发送思考模式会忽略的 `temperature=0`。

## 预注册门槛

同时满足才允许扩大到新的固定 100/200 题 paired gate：

1. 目标恢复至少 6/16；
2. 控制保留至少 13/16；
3. 总正确至少 19/32（相对冻结 version50 净增至少 3）；
4. legal 至少 30/32；
5. 多调用或非空 assistant content 不再产生 carrier protocol error；
6. 没有调用读取同批前序调用新建的未见句柄，没有拒绝调用引起状态变更；
7. 总 token 不高于同题 version50 的 1.25 倍；
8. 所有正确轨迹 fresh replay，且结构/no-leak/provider-history 审计通过。

未达门槛时仍保留 version51 实现作为后续修复基线，但不得扩大、导出 SFT 或用于 RL。

## 冻结实现哈希

- `atomic_version51.py`: `2cd43f8b7f6124ba1264537bcd0bd04da864aeb341004ca2509c8b30c0f7b10a`
- `deepseek_native_tools.py`: `b9e70d6b0885cba1414b7ebff7fd61ec6296aa6412c2d78bac2fcd2349ad8333`
- `provider_adapter.py`: `1a2e9e46e5efa6b74ee67f12225cbf3a23dc68da8cfaae0f9e323f88eccc17ee`
- `generate_teacher_rollouts.py`: `81d3e9b62625a17ee8e45a869fb1a20ac023936da65864a7dd6f2e9761f4b572`
- `tool_schemes.py`: `f644abc1a6900dc94ac04acd2b566667617cf33988f286df1dc8ceb7ffed79d1`

## 结果

Version51：**22/32 correct、32/32 legal、11 errors、232 model turns、271 primitive
actions、2,631,916 tokens**。冻结 version50 同题为 16/32、26/32、37 errors、234 atomic
actions、2,426,690 tokens。

- 目标恢复：**6/16**；
- 控制保留：**16/16**；
- paired accuracy：6 gains、0 regressions，exact `p=0.03125`；
- paired legal：6 gains、0 regressions，exact `p=0.03125`；
- token：version50 的 **1.0846x**，低于 1.25x 上限；
- primitive actions 比旧 atomic actions 多 15.8%，但真实模型回合为 232，较 version50 的
  234 回合略少；这两个单位不得混报；
- 232 个 provider turns 中有 37 个多调用回合（35 个双调用、2 个三调用），92 个回合有
  非空 assistant content；两者均产生 **0 carrier rejection**；
- 11 个过程错误全部是参数/状态校验错误，无 provider/api/protocol 终止；
- 22/22 正确轨迹 fresh replay 通过；
- 22 条正确轨迹结构审计 0 issues；
- 全 32 题 provider 审计覆盖 271 个 calls：call id/顺序、原始参数→执行参数 lowering、
  reasoning history、每调用一个结果、拒绝状态不变、bundle-pre-state 未见句柄边界和 model
  input gold SQL 泄漏均为 0 issues。

产物（结果目录不作为 SFT 源）：

- `data/trajectories/version51_native_tool_bundle_20260805/gate32/all.jsonl`
- `data/trajectories/version51_native_tool_bundle_20260805/gate32/verified.jsonl`
- `data/trajectories/version51_native_tool_bundle_20260805/gate32/structural_audit.json`
- `data/trajectories/version51_native_tool_bundle_20260805/gate32/native_bundle_audit.json`

## 决策

八项预注册条件全部满足。Version51 成为后续 DeepSeek 原生工具调用实验基线，下一步可在
新的固定 cohort 上做 paired 100/200，而不是回到 version26 或继续修补已淘汰的 version50。
它仍是 diagnostic-only：在大门禁、scheme-aware exporter 和训练前质量门通过前，不得把
multi-call turn 展开为 atomic SFT 样本，也不得进入 RL。

执行后只补充了通用 BIRD `source.db_path` replay 支持、native-bundle 审计器和 manifest
history 标签纠正；这些不改变本 Gate32 已执行的 provider prompt、调用校验或 harness 语义。
