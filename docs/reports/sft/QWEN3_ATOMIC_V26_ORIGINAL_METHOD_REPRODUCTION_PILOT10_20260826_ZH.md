# Qwen3 Atomic version26 原方法复现 Pilot10（2026-08-26）

## 结论

历史 SFT1 的生成代码、教师 prompt、工具协议、provider carrier、recent-4 历史和
version26 学生投影均已精确复现；同题回答能力基本保持，但当前
`deepseek-v4-flash` 的 reasoning 长度和重复度与 2026-07 原数据存在明显漂移。

因此，9K 去除原 fixed-1000 后的 8,484 题任务清单已经冻结，但大批量付费生成暂不
放行。当前 pilot 产物均为 diagnostic-only，不自动并入训练。

## 冻结身份

- 教师生成运行时 commit：`ae3bed19b1aeeab2c69f4dcd0c658300f848c94f`
- 教师模型：`deepseek-v4-flash`
- 官方 endpoint：`https://api.deepseek.com`
- provider system prompt SHA-256：
  `d211b9d7e2c8f03a68fbf37efb846844cebc7028072d5eeed75302bdcc4f4cf0`
- provider protocol hash：`25ac4c10ef96365c`
- Atomic teacher runtime：`version24`
- provider 请求：thinking enabled、reasoning effort high、JSON Output
- rolling history：最近四个合法 assistant/Harness 对
- 单轮 completion 上限：2,048；最多 30 个工具步；单题一次语义尝试
- 学生投影运行时 commit：`4cd47c957fc6ae791e76a10594c8cd22f4d3b6de`
- 学生协议：Atomic `version26`、`think-json-v1`
- 学生 prompt SHA-256：
  `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`
- 学生工具 schema SHA-256：
  `e1533d05dcacc028c57d55d358dae109bd980e88a6c1d58c2fb456d68379aad6`
- 学生 base protocol hash：`4da19387399bd3a5`

历史 827 条保存了 model input 的原始轨迹，其 system prompt 均为同一个 16,400 字符
字符串。隔离运行时重新构造出的 prompt 与其 SHA-256、长度和 protocol hash 全部精确
相同。

## 配置漂移的首批尝试

首批十题误显式传入 `max_tokens=1024`，而历史冻结 manifest 是 2,048。该批虽然产生
1/10 正确结果，但出现 29 次 completion length retry，已写入
`CONFIGURATION_DRIFT_INVALID.json` 并永久标为 audit-only，不用于一致性结论、训练或后续
任务去重。

## 精确配置的历史失败题 Pilot10

从 fixed-1000 中排除原 704 条 verifier-correct episode 和上述无效 pilot 的十个题号，
按原顺序选择下十题。该样本本身是历史失败题，不能用于估计总体准确率。

- 3/10 `bird-set` 正确；10/10 合法终止；7/10 为 `wrong_answer`
- 83 个模型 turn；89 次 API attempt；582,927 tokens
- 3 条成功 episode、33 个 next-action target
- 3/3 fresh replay 通过；结构审核 3/3 通过
- 精确 Qwen3 6,400-token 完整前缀门：2/3 episode 通过；30/33 record 不截断

中难度旧 target 的 reasoning 中位数为 430 字符、均值 708；该批 33 个 target 的
中位数为 2,418、均值 3,841。因为候选来自历史失败集，该差异随后通过同题成功集对照
复核。

## 原成功题同题复现 Pilot10

从原 704 条成功 episode 中按原任务顺序冻结 4 easy、4 medium、2 hard。当前模型看不到
旧动作、旧 reasoning 或旧答案，重新独立执行真实模型与 Harness 的因果交互。

| 指标 | 原轨迹 | 当前复现 |
|---|---:|---:|
| `bird-set` correct | 10/10 | 9/10 |
| legal | 10/10 | 10/10 |
| 成功轨迹工具步数 | 71（十题总计） | 76（十题总计） |
| 成功 episode fresh replay | 历史已验证 | 9/9 |
| version26 targets | 66（双方都正确九题） | 69 |
| 精确 6,400-token 完整 target | 冻结 SFT1 规则 | 69/69 |

当前九条成功中，8 条为 clean success，1 条从一次 protocol error 中恢复；结构审核和
fresh replay 均无问题。说明工具流程和答案能力基本可复现，但不是确定性复制。

### 双方都正确九题的 reasoning 对比

| reasoning 字符数 | 原轨迹 | 当前复现 |
|---|---:|---:|
| mean | 685.1 | 1,225.6 |
| median | 521.5 | 802 |
| p90 | 1,414 | 2,528 |
| max | 2,438 | 5,894 |

当前 reasoning 平均长度为原来的 1.79 倍，中位数为 1.54 倍。简单重复启发式还发现：

- exact repeated sentence：原 1/66，当前 5/69；
- 同一 8-token gram 至少出现三次：原 0/66，当前 2/69；
- `gold sql`、`gold_sql`、`reasoning_content`、`response_format`、benchmark answer 等禁用
  目标文本：当前 0/69；
- 69/69 target 均通过 version26 strict parser，carrier 形状正确。

因此问题不在工具、prompt 拼接或学生投影，而在同一公开模型别名当前返回的 reasoning
分布发生了漂移。继续原样扩到数千题会把较长、较重复的思维文本重新引入训练集。

## 9K 去重结果

冻结 9K：
`data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.tasks.jsonl`
（SHA-256 `a18b8987591f0f5ab71db6cfd8e93c3422efabc70e67dbad9fac55542df7c911`）。

原 fixed-1000：
`data/eval_inputs/bird_train_external_teacher_fixed1000.jsonl`
（SHA-256 `37e23be8472c2dc2ccdd6165fb72a5003bf176f5b469de53127ffd00ecc7fea5`）。

按 `example_id` 去重：

- 两集合实际重叠 516 题；原 1,000 中另 484 题本来不在 9K；
- 剩余 8,484 题；
- BIRD 2,634、Spider 2,700、SynSQL 3,150；
- 输出 SHA-256：
  `a4c24a95384f380d2c82909280f32e37129c1378cd7f2717975548d54e6c3708`；
- 顺序保持、8,484 个 `example_id` 唯一。

该任务清单已冻结，但 manifest 状态为
`frozen_selection_generation_blocked_pending_reasoning_consistency`。在 reasoning 分布问题有
明确处理策略前，不启动大规模付费生成，也不修改已冻结的原 SFT1 数据。
