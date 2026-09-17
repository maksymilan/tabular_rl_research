# 因果数据生成契约

更新时间：2026-09-12

所有新 SFT/评测诊断数据必须由真实的 model↔Harness loop 产生。不得从 gold SQL 编译完整
trajectory，也不得用后续步骤改写早期 reasoning、observation 或 action。

## 每轮可见信息

provider/teacher 只接收：

1. 用户问题和允许的外部知识；
2. 当前阶段需要完成的目标；
3. version26 rolling legal history 最近 4 轮；
4. 当前 Harness resident state、relation artifacts 和最新 error/observation。

模型的 terminal 答案必须引用 Harness 生成的 exact result artifact。Gold SQL、gold result
和 privileged gold path 只在 Harness 内用于兼容和 denotation 检查，不能出现在 provider
request、student prompt、SFT target 或 reward 中。

## 准入门禁

- action schema/carrier/protocol/prompt identity 与 version26 一致；
- episode 真实执行且 terminal denotation 正确；
- fresh replay 完整复现；
- no-leak、结构、provider-history、cohort 和 token budget audit 通过；
- dataset source/index/runtime/tokenizer 哈希写入 manifest；
- 任一门禁失败就不进入 SFT/RL，失败记录保留审计但不得“修复后冒充生成”。

## 当前数据范围

- 冻结 SFT1：fixed-1000 来源中 678 causal episodes / 4,471 targets。
- RL cohort：目标约500题；每题 K=8 且正确轨迹数1–6（2026-09-16 起）。合并原始700题、
  后续600题、848题和新的3000题筛选后，strict candidate为1209题、expanded为1329题、
  known-exclusion fresh为776题；统一入口见
  `data/inventory/rl_training_candidates_1to6_current.json`（同时重指向
  `data/inventory/rl_training_candidates_current.json`），详细证据见
  `docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_1TO6_20260916.md`。区间外的0与7/8观察保留在
  `retained_non_candidate_observations.jsonl`，仅供记录和后续复用。候选池仍需在正式启动前
  按 admission policy 冻结 cohort，并通过 fresh replay、manual audit 和 manifest 门禁。
- 2026-09-12 起新增探索性筛选口径：完整 K=8 观察中正确轨迹数允许为1–7。现有历史观察按
  稳定 `example_id` 去重后形成strict 896题、expanded 1025题、fresh 353题的版本化库存，
  入口为 `data/inventory/rl_training_candidates_1to7_current.json`，报告见
  `docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_1TO7_20260912.md`。该探索性库存保留作审计；
  正式准入范围已在2026-09-16改为1–6，见上一条。
- 新 teacher generation 当前暂停。恢复时只用官方 DeepSeek
  `https://api.deepseek.com/chat/completions`，不允许第三方代理或隐式 fallback。

## 运行入口

- SFT：`src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`
- RL cohort preflight：`src/rl/scenarios/diagnostics/audit_qwen3_v26_saam700.py`
- 正式 RL：`src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`

历史 SFT2、projection/rewrite/delete、checkpoint-relalg 和其他工具版本均为 archive/reports
中的诊断，不得作为当前数据扩增路径。
