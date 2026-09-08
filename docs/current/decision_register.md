# 当前决策登记表

更新时间：2026-09-08。这里把已确认事项和仍需用户拍板的事项分开，避免文档再出现多套
“当前方案”。用户回复后，只修改本表和 `final_project_contract.md`，再同步对应 manifest。

## 已确认

| 事项 | 当前答案 | 证据/落点 |
|---|---|---|
| 工具版本 | Atomic version26 | `docs/current/final_project_contract.md` |
| A100 训练方式 | 单卡 replicated trainer + 独立单卡 online vLLM | update-40 run manifest；world size=1 |
| 服务器职责 | A100 优先做 RL 主实验；A100 不可用时，table_rl/NewGNN 可按降级拓扑做 RL；三机均可评测 | 用户于 2026-09-06 确认；`server_resources.md` |
| A100 serving | 另用一张 A100 跑 online vLLM | 当前 live gate 进程和 formal launcher |
| RL credit | `saam-asymmetric-error` | `src/rl/frameworks/trl/state_action_ambiguity.py` |
| RL reward | four-level result reward | `src/rl/runtime/terminal_reward.py` |
| RL advantage adapter | `correctness-primary-clean-secondary`, `clean_advantage_weight=0.25`; correctness determines the sign, cleanliness only changes magnitude | 2026-09-08 A100 rollout audit: raw four-level normalization produced positive advantages for 331 all-wrong legal trajectories and negative advantages for 69 correct-with-error trajectories |
| RL 梯度更新 | reason/tool 各 0.5 的加权 full-response 梯度 | `span_balance_alpha=0.5` |
| timeout | 视为 policy error，timeout action 使用局部负向惩罚 | SAAM asymmetric-error |
| RL 起点 | cumulative SFT `checkpoint-6380` | A100 formal/live launcher |
| RL 数据与批次 | 目标约500题；每题K=8且正确数2–6；30题/update；从checkpoint-6380启动 | 2026-09-06跨服务器筛选盘点 |
| GPU 选择 | A100 仅允许 GPU 0–3；动态选择一张 trainer 与一张不同的 vLLM 卡 | 用户确认；launcher 已加入 allowlist |
| 旧代码 | 全部移入 `archive/`，暂不删除；`src/` 不保留 compatibility stub | `archive/code/legacy_migration.md` |
| KL 验证 | 必须通过独立匹配对照验证是否有增益；当前正式 arm 仍为 `kl_beta=0` | `docs/current/rl_pipeline.md` |
| A100 资源约束 | 单卡拓扑；trainer/vLLM 必须不同卡；只允许 GPU 0–3 | 用户于 2026-09-06 确认 |
| 单卡资源入口 | 作为当前主线；默认 BF16 冻结基座、FP32 AdamW、独立 output root | `qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml` |
| rollout 生成预算 | 当前主线及后续新配置默认 `max_new_tokens=4096`（单轮）；历史配置保留其 manifest 中的显式值，不回写历史实验 | 用户于 2026-09-08 根据 A100 5,600 条 rollout 中 802 条（14.32%）触发 2,048 上限后的确认 |
| 正式实验启动确认 | 启动前必须先列出并提交用户确认：配置/配置哈希、模型与 checkpoint、cohort/数据哈希、protocol/prompt/runtime identity、reward/credit、rollout、optimizer、GPU 拓扑、输出目录和评测计划；未获确认不得启动 | 用户于 2026-09-08 明确要求 |
| 3090 降级入口 | 两张独立 3090：单卡 replicated trainer + 单卡 online vLLM；从 1 row / 8,192 tokens 起步 | 用户于 2026-09-06 确认；需独立 launcher、manifest/output root |
| Qwen2.5-7B SFT 对照 | 在 NewGNN 用四卡训练 `Qwen/Qwen2.5-7B-Instruct`，复用 Qwen3 SFT1 的同一 4,471-record model-visible 数据；只作为模型族对照，不改变 Qwen3/RL 主线 | 用户于 2026-09-04 明确要求；run `qwen25_7b_atomic_v26_sft1_same4471_4gpu_full_20260904_203529` 已以 exit 0 完成，checkpoint-560 已保存 |
| 今晚诊断对照实验 | 注册三项仅用于审计/对照的 version26 降级 RL：原版 SMC+原始 `saam_gate60` 数据、原版 SAAM+four-level+span-balanced+新版 `smc_balanced60` 数据、以及经审计后固定的 SMC 优化诊断版本；统一从 checkpoint-6380 启动，独立 output root，完成后在 NewGNN matched greedy 评测；结果不得替换正式 SAAM 主线或宣称 accuracy promotion | 用户于 2026-09-06 明确要求；table_rl 仅作诊断资源，所有配置、cohort、概率审计和评测结果分别记录 |
| Harness-conditioned SMDP/IQL 方向 | 先做离线 transition/可识别性诊断，不训练 actor；transition 只保留 model-visible state、typed action 和 Harness four-level terminal reward，拒绝 gold/ref 字段。只有“同一精确 state 存在不同 action 且 outcome 也不同”才进入 critic 小规模训练门禁 | 用户于 2026-09-07 要求尝试；实现落在 `src/rl/frameworks/trl/{mdp_critic,iql}.py` 与 `src/rl/scenarios/diagnostics/audit_harness_smdp_iql.py`，manifest 标记 `diagnostic_only` |

## 待用户确认

### 2026-09-08 大样本静态语义审计（不改训练）

用户要求先观察 a100 主实验及 table_rl 既有 K=8 rollout，不凭启发式猜测错误等级。
当前 distance/format 原型仅保留为失败诊断：不能产生 actor reward、可信度或因果归因；
不得接入训练，也不自动剔除 cohort。共享统计入口为 `rl.diagnostics.rollout_corpus`，
薄 CLI 为 `scenarios/diagnostics/audit_rollout_corpus.py`。跨 run 按数据库和题干建立题目
映射，不能直接连接 cohort 内的 example_index。正式 reward、GPU 进程及评测评分不变。
观察集和按题保留集隔离，Codex 逐条审阅不冒称独立人工标注；详细证据见
`docs/reports/rl/ROLLOUT_CORPUS_SEMANTIC_AUDIT_20260908_ZH.md`。

### 2026-09-08 用户批准的 span alpha 对照

用户要求在 table_rl 启动 `span_balance_alpha=0.25`，训练完成后在同机双卡评测。
此项为独立诊断，不修改正式 arm 的 alpha=0.5。复用
`qwen3_v26_saam_fourlevel_spanbalanced_smc_balanced60_20260907_retry2` 的实际设置：
checkpoint-6380、相同新版60题（SHA-256
`b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4`）、30题/update、K=8、
4 optimizer updates、seed=20260901、LR=4e-7、four-level/SAAM、4-bit frozen base + FP32 LoRA/AdamW。
训练完成后评测 BIRD-dev1534 greedy，自动释放本次拥有的 vLLM 进程；比较对象为
checkpoint-6380 SFT 和该 alpha=0.5 run 的 table_rl 全量评测。
评测步数必须从 trainer_state 验证；历史评测的 checkpoint10 标签不能当作真实训练步数。
入口 `src/rl/scenarios/diagnostics/run_saam_span_alpha025_table_rl.sh` 显式使用已归档的远端
训练实现，单独记录控制代码与训练实现身份；不把本地重构后的 trainer 混入该对照。

### 2026-09-07 审计修正

已发现并记录一个概率审计实现缺陷：旧诊断 runtime 在每个 question group 调用
`smc_mode_concentration_advantages` 时重置了内存中的审计缓存，因此一个 update 的 30 个
题组最终只落盘最后一个题组。历史 `audit3` 和原始 `saam_gate60` 重跑中的 4 行日志只代表
4 个 update 的各一个题组，不能作为完整概率审计。优势选择逻辑本身未改变；修复后要求每个
4-update run 写入 120 个题组记录，并通过覆盖率测试后才允许进入 SCM 优化决策。

修复后原始 cohort 已通过 120 组/960 候选的完整字段、rollout 连接和选择一致性检查；
完整概率审计门禁已通过。新版 cohort 仍需补跑完整概率审计；已完成的 SAAM 训练不重跑。
中间评测与衔接证据：`docs/reports/rl/SCM_AUDIT_PROGRESS_20260907_ZH.md`。不改变正式 RL
方案，也不预先宣称 SCM 优化版本或 accuracy promotion 已准入。

### 2026-09-07 MDP critic 预注册门禁

本方向不引入独立 reward model：reward 仍来自 Harness 的 terminal scorer，IQL 只学习
`Q(state, action)` 与 expectile `V(state)`。第一阶段只从完整 Atomic v26 rollout 构造
semantic-step SMDP transition 并做可识别性审计；不更新 actor、不替换 SAAM/GRPO。审计必须
同时观察到同一精确 model-visible state 的 action variation 和 terminal outcome variation，
否则只能说明数据可拟合 trajectory-success predictor，不能说明 critic 能把 pass@4 集中到
greedy。审计产物需记录 checkpoint-6380、protocol/prompt hash、source hash、IQL 超参和
`diagnostic_only` 状态。

1. **KL 对照系数/调度**：是否采用固定 beta（例如 `0.01`）或预注册的 beta sweep，待
   对照启动前登记；不得在看到结果后再挑选系数。

2. **新 cohort 最终规模**：当前跨服务器去重后确认 430 道满足 2–6 的题目；补充筛选完成后再冻结约500题 manifest。

## 未决前的执行规则

- 新 RL 只能从 version26 + 当前 four-level/SAAM + reason/tool 加权 config 启动；约500题
  cohort manifest 和 matched eval 完成前，不宣称 accuracy promotion。
- A100 update-40 运行已同步：checkpoint-40 可审计，step49 因 OOM 退出；它使用旧700题/14题每update配置，只保留作历史诊断。
- 不启动新的外部 DeepSeek teacher batch。
- Qwen2.5-7B 四卡 SFT 对照使用 Qwen2.5 原生 `qwen` chat template；数据内容 SHA-256
  固定为 `c19742ec55d99980075e51a52e79285e4322f89a1d1991dea592748c279e9a14`。该运行不是
  Qwen3 SFT anchor，也不得直接作为当前 RL 起点；正式 run 已完成，最终 train loss 为
  `0.5573`，global step 为 `560`。
- 评测优先放在 table_rl/NewGNN；A100 可用时承担主 RL，A100 不可用时允许在两张空闲 3090 上
  运行降级 RL 及其 rollout serving。
- KL 是否有增益已确定要通过实验回答；在对照完成前，正式 arm 继续使用 `kl_beta=0`，且
  任何 KL 对照必须与主 run 使用同一 checkpoint、cohort、decode、runtime、GPU 拓扑和预算，
  并使用独立 output root。

## 归档迁移说明

旧 scheme 的实现目标目录是 `archive/code/`，旧 launcher/config/adapter/trajectory 目标目录
是 `archive/experiments/` 或对应历史结果目录。当前 `src/tool_modules/registry.py` 已收敛为
Atomic-only；旧 scheme 包、薄别名和仅支持旧方案的 runner 已迁入 archive，不能被默认 config
或 launcher 导入。
