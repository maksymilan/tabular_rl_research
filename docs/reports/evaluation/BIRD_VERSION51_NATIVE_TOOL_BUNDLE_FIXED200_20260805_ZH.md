# BIRD Version51 原生 Tool Bundle Fixed-200

状态：**已完成并通过预注册行为门禁。** Version51 成为 version52+ 的 forward
provider-tool-call 实验基线；本实验仍是 diagnostic-only，不授权把多调用轨迹展开为 atomic
SFT 样本或用于 RL。

## 实验问题

Version51 在冻结 Gate32 上通过了扩大门禁。本次在完整固定 200 题 cohort 上检验：接受官方
DeepSeek 一个 assistant turn 返回多个 `tool_calls` 的语义，并按同一 bundle pre-state 预校验、
逐调用执行和逐 call id 回传结果，能否稳定修复 version50 的 carrier 失败，同时改善正确率、
合法终止和过程可靠性。

## 冻结 cohort 与对照

- 来源：`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`
- 题数：200，固定原顺序，全量运行（显式 `--limit 0`）
- 来源 SHA-256：
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`
- 主要配对对照：同 cohort version50 fixed-200，133/200 correct、183/200 legal、166 errors、
  1,487 actions、16,158,278 tokens
- 历史工程对照：同 cohort version24 fixed-200，145/200 correct、197/200 legal、29 errors、
  1,490 actions、8,420,861 tokens

题目、顺序、模型、最大步数、completion budget、历史窗口、数据库上下文、denotation metric
和 whole-episode K 均不得在看到结果后改变。

## 固定运行条件

- 官方 `https://api.deepseek.com/chat/completions`，`deepseek-v4-flash`
- `version51` / `native-tool-bundle` / registry v5
- K=1，`bird-set`，exact cited table terminal
- `catalog-v1`，optional plan，full prompt
- 最近 4 个真实 provider assistant turns；每个 turn 后保留其全部 `role=tool` 结果
- `thinking={"type":"enabled"}`、`reasoning_effort=high`、`tool_choice=auto`
- 每次 completion budget 2,048；最多 30 个 model turns；每类错误上限 3
- 每 assistant turn 最多 8 个调用；所有调用按 bundle pre-state 预校验
- workers=4，API timeout=300 秒，transport retries=10
- diagnostic-only，`sft_export_eligible=false`

## 预注册判断门槛

本次 200 题无论中途分数如何都跑完；以下门槛只决定结论，不改变 cohort 或补跑策略。

1. 正确数至少 140/200，且相对 version50 配对净增至少 5；
2. 合法终止至少 195/200；
3. 多调用或非空 assistant content 导致的 carrier rejection 为 0；
4. 过程错误少于 version50 的 166，且不出现系统性 provider/API 终止；
5. 总 token 不超过 version50 的 1.25 倍，即 20,197,848；
6. 所有正确轨迹 fresh replay 通过；
7. 结构审计与全量 native-bundle 的 no-leak、call/result 顺序、参数 lowering、reasoning
   history、拒绝状态不变和 bundle-pre-state 审计均为 0 issues。

即使全部通过，version51 仍只成为后续 version52+ 的实验基线；训练准入还需要独立的
scheme-aware exporter 和训练数据门禁。若正确数达到 150/200，也不能绕过该要求。

## 冻结实现哈希

- `atomic_version51.py`：
  `2cd43f8b7f6124ba1264537bcd0bd04da864aeb341004ca2509c8b30c0f7b10a`
- `deepseek_native_tools.py`：
  `b9e70d6b0885cba1414b7ebff7fd61ec6296aa6412c2d78bac2fcd2349ad8333`
- `provider_adapter.py`：
  `1a2e9e46e5efa6b74ee67f12225cbf3a23dc68da8cfaae0f9e323f88eccc17ee`
- `generate_teacher_rollouts.py`：
  `e86beb850c7e1a93618268e2b77a555acc23dc2c5344b943488def1e93805720`
- `tool_schemes.py`：
  `f644abc1a6900dc94ac04acd2b566667617cf33988f286df1dc8ceb7ffed79d1`
- `rollout.py`：
  `115cb81457c47cc10df2551075a4f82863e86685105b1f2bc29d75456eb79304`
- `audit_native_tool_bundle.py`：
  `affa20a02059d406b76f59f1ce6a9e5c508632dd85ef1e84545795b5ae2161cd`
- 公共 tool schema SHA-256：
  `4d9e6cae7652ba958f316177367f18c8b68a48c4cc7e447e1b11eb3d13487125`
- native tools SHA-256：
  `ece0de3dbcbc7b6017abdf49a733468bf3ca1d51d8582513b9440ca78f6cdadd`

运行前官方 `/models` 鉴权检查成功，并返回 `deepseek-v4-flash` 与 `deepseek-v4-pro`。仓库
preflight 在 fresh fetch 后通过：HEAD `559d895c36cb57fb54827461bc2a95bd5d676134`，
`master` 相对 `origin/master` ahead 60、behind 0；工作树已有 version50/version51 改动，
评测期间原样保留。

## 运行命令

```bash
.venv/bin/python -u src/sft/generate_teacher_rollouts.py \
  --split train \
  --examples-file data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl \
  --limit 0 \
  --model deepseek-v4-flash \
  --out data/trajectories/version51_native_tool_bundle_20260805/fixed200/verified.jsonl \
  --all-out data/trajectories/version51_native_tool_bundle_20260805/fixed200/all.jsonl \
  --failures-out data/trajectories/version51_native_tool_bundle_20260805/fixed200/failures.jsonl \
  --workers 4 --max-steps 30 --max-errors-per-type 3 \
  --attempts-per-example 1 --max-tokens 2048 \
  --api-timeout 300 --api-retries 10 --table-output-rows 0 \
  --context-mode rolling-legal-history --history-turns 4 \
  --rolling-prompt-variant full \
  --atomic-protocol-version version51 \
  --policy-prompt-variant canonical --plan-policy optional \
  --deepseek-carrier native-tool-bundle \
  --denotation-comparison bird-set \
  --diagnostic-only --database-context-profile catalog-v1
```

## 结果

### 总体结果

| 指标 | version51 native bundle | version50 single native | 历史 version24 |
|---|---:|---:|---:|
| correct | **147/200 (73.5%)** | 133/200 (66.5%) | 145/200 (72.5%) |
| legal | **200/200 (100%)** | 183/200 (91.5%) | 197/200 (98.5%) |
| process errors | **54** | 166 | 29 |
| model turns | **1,414** | 1,487 atomic turns | 1,490 atomic turns |
| primitive calls | **1,650** | 1,487 | 1,490 |
| total tokens | **16,165,179** | 16,158,278 | 8,420,861 |
| wall time | **1,855.446 s** | 1,839.026 s | 1,087.953 s |

53 个失败全部是合法终止后的 `wrong_answer`；没有 provider/API/carrier 终止。147 条正确轨迹
中 118 条 clean success、29 条 recovered success。按难度分别为 easy 62/80、medium 43/60、
hard 42/60。

Version51 相对 version50 减少 4.9% 真实模型回合，但增加 11.0% primitive calls；同一回合
的并列调用不能被误报成多个已观察反馈的模型回合。总 token 仅增加 **0.043%**，低于 1.25x
上限；相对 version24 则仍为 **1.920x**。因此它修复了 native carrier 的可靠性缺陷，但没有
解决 native schema/history 相对历史 JSON-Output 路径的 token 开销。

### 逐题配对

相对 version50：127 题共同正确、**20 gains、6 regressions**、47 题共同错误，净增 14，
exact 双侧 `p=0.0093553`。Legal 为 **17 gains、0 regressions**，`p=0.00001526`。

Version51-only gains（20）：`bird_train_00454`、`bird_train_00541`、`bird_train_00600`、
`bird_train_00858`、`bird_train_01018`、`bird_train_01148`、`bird_train_01730`、
`bird_train_01787`、`bird_train_01926`、`bird_train_02438`、`bird_train_02682`、
`bird_train_02935`、`bird_train_03042`、`bird_train_03312`、`bird_train_03505`、
`bird_train_03925`、`bird_train_05558`、`bird_train_06165`、`bird_train_06246`、
`bird_train_06299`。

Version51 regressions（6）：`bird_train_02408`、`bird_train_02868`、`bird_train_03688`、
`bird_train_05053`、`bird_train_06324`、`bird_train_06336`。

相对历史 version24：132 题共同正确、**15 gains、13 regressions**、40 题共同错误，净增 2，
exact 双侧 `p=0.8505540`；这是持平而非显著准确率提升。Legal 为 3 gains、0 regressions，
`p=0.25`。

Version51 对 version24 的 gains（15）：`bird_train_00582`、`bird_train_00593`、
`bird_train_02088`、`bird_train_02438`、`bird_train_02512`、`bird_train_03042`、
`bird_train_03636`、`bird_train_03969`、`bird_train_03977`、`bird_train_04869`、
`bird_train_05316`、`bird_train_05544`、`bird_train_06165`、`bird_train_06246`、
`bird_train_06299`。

Version51 对 version24 的 regressions（13）：`bird_train_00362`、`bird_train_01002`、
`bird_train_01535`、`bird_train_02408`、`bird_train_02901`、`bird_train_03131`、
`bird_train_03216`、`bird_train_04244`、`bird_train_05053`、`bird_train_05760`、
`bird_train_06336`、`bird_train_06454`、`bird_train_06489`。

### Provider 与过程行为

- 1,414 个真实 provider assistant turns，1,650 个 function calls；
- 216 个多调用回合：194 个双调用、22 个三调用，没有超过 3 个；
- 537 个回合带非空 assistant content，全部只作审计，**0 个因此被拒绝**；
- 2 个 `missing_tool_calls` provider-turn protocol errors，在同一 episode 内恢复；
- 其余过程错误为 51 个 `argument_validation_error` 和 1 个 `execution_error`；
- transport/context/carrier retry 均为 0；48 个长度相关 completion retry 是客户端事件，未送入
  harness；
- 1,462 次 API request attempts，prompt/completion/reasoning tokens 分别为
  15,523,315 / 641,864 / 484,104。

与 version50 相比，protocol errors 从 124 降到 2，证明主要 carrier 摩擦已消除；但参数错误
从 41 增到 51，说明下一阶段应优化语义决策、参数/状态理解和 token，而不是继续围绕
多调用/content 拒绝打补丁。

### Replay 与独立审计

- fresh deterministic `bird-set` replay：**147/147 pass**；
- `audit_verified_rollouts.py`：147 verified、1,100 legal steps、0 structural issues，full-prompt
  gate pass；
- `audit_native_tool_bundle.py`：全 200 题、1,414 turns、1,650 calls，call/result 顺序、参数
  lowering、reasoning history、bundle-pre-state、拒绝状态不变和 model-input no-leak 均为
  **0 issues**；
- final fresh-fetch preflight：HEAD/branch/upstream 不变，ahead 60、behind 0；
- 审计器最初把两个没有 primitive call 的 `missing_tool_calls` provider-turn errors 错算为
  primitive error，导致最后一条长轨迹出现假 step-id gap。修复只让 native-bundle 的
  `action_index=null` provider-turn error 不进入 primitive count；轨迹未修改。新增 2 个回归
  测试，完整 SFT 测试 **212/212 pass**，结构审计随后 0 issues。

### 产物与哈希

- `data/trajectories/version51_native_tool_bundle_20260805/fixed200/all.jsonl`：
  `907fd021c8050caebd0c3ab58a7c2933d84ef391f841d91d6cdd72b3ba2b5a96`
- `.../verified.jsonl`：
  `beb4965140d3e8382c0dd969e321a46a40d7d5b189d0d068a68d55819fc6ac8f`
- `.../failures.jsonl`：
  `c871d9e0f05f1aa996fa646d11d04999f6f0856498d28d24cf662d27acc1687a`
- `.../verified.manifest.json`：
  `07cab0e363ab0a84e0b794e5b8f8023d97a383991b74385764e9ebc2a696352f`
- `.../structural_audit.json`：
  `cdacc48da7b57b6aaed1464643a7f32d3ba8ba3d064d2a127a7f99eb6fd05eef`
- `.../native_bundle_audit.json`：
  `485324cc84a3548110bccc2494c8ed8f52053f6072a17b68daed2b3a1c4bc173`

Manifest protocol hash 为 `7c260d82b09d9c88`；canonical teacher/provider/student prompt SHA-256
分别为 `677135c9...3571`、`b0f3ce7a...3b3`、`81329794...bd54`，完整值保留在 manifest。

## 决策

七项预注册门槛全部通过。Version51 相对被淘汰的 version50 有显著正确率和合法率提升，并把
carrier protocol errors 从 124 降到 2；后续 provider-tool-call 研究应从 version51 增量到
version52，而不是回到 version26 或继续维护 version50 的单调用拒绝规则。

这不是对历史 version24 的准确率晋级：147 与 145 的配对差异不显著，过程错误仍为 54 对
29，token 仍接近 1.92x。Version51 保持 `diagnostic_only_pending_protocol_scale_gate` 和
`sft_export_eligible=false`；在 scheme-aware exporter、因果 last-turn target、multi-call credit
边界及训练前审计完成前，不得进入 SFT/RL。
