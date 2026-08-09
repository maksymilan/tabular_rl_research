# BIRD iterative-SQL v6 Target Gate20 预注册

日期：2026-08-05  
状态：已按本预注册完成；数值门全部通过，但 combined-control 语义前提失效

> 最终结果见 `BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_RESULT_20260805_ZH.md`。V6 为 15/20，v5
> 为 10/20，5 gains / 0 regressions；所有数值门通过。预注册的 `field1+field2` control
> 实际仍由 reference 要求分列，因此不能用于验证真正的单字符串反向能力。

## 目的

前一个 baseline300 第 51–70 题 Gate20 中没有多字段姓名映射目标题，也没有明确拼接控制题，
无法直接验证 `iterative-sql-v6` 的唯一 prompt 变化。本门只测试这项目标能力及其回归边界，
不作为代表性 baseline 准确率。

## Cohort 构造

- source：`data/eval_inputs/bird_train_tool_compatible.jsonl`；
- 排除：完整 active baseline300 和历史 fixed-200；
- 与上述两个已消费 cohort overlap：0；
- 任务数：20；数据库数：16；
- output SHA-256：`12c5aa8611a321a17c73ca9dd66b51c81cbb1e3e2bd57e32c35ba3c8439f8a65`；
- 选择只使用 `example_id/instance_id/example_index/db_id/question/external_knowledge`；
- `gold_sql/query/gold rows/gold path/gold sample/gold row count` 明确禁止参与选择；
- 单测验证选择对 gold 字段变化不敏感。

四个预注册类别：

| 类别 | 数量 | 作用 |
|---|---:|---|
| multi-field separate targets | 6 | `full name` 映射到 2–3 个字段，必须按声明顺序分列 |
| explicit combined controls | 4 | external knowledge 使用 `field1 + field2...` 明确要求组合，防止 v6 过度分列 |
| single-field name controls | 4 | `full name` 映射到一个现成字段，必须保持单列 |
| ordinary controls | 6 | 不涉及姓名映射，检查一般行为与成本回归 |

精确 task ids、类别顺序、source/exclusion hashes 与选择算法见
`data/eval_inputs/iterative_sql_v6_output_shape_target_gate20_v1.manifest.json`。

## 冻结运行设置

- paired K=1：同一 20 题分别运行冻结 v5 与 v6；
- model/provider：`deepseek-v4-flash`，官方 `https://api.deepseek.com`；
- lazy catalog；recent-4；30 max steps；每类最多 3 errors；
- 2,048 max tokens；20 preview rows；20 秒 SQLite deadline；`bird-set`；
- 除 protocol/interface/prompt 外，其余任务、顺序、数据库、预算、并发与 scorer 相同；
- 正确轨迹必须 fresh replay；所有记录必须通过结构和 no-hidden-input-key 审计。

## 预注册推广门

必须同时满足：

1. 6 个 separate targets 上，v6 相对 v5 至少 2 个 paired gains、0 regressions；
2. 4 个 explicit-combined controls 上，v6 不低于 v5，且保留全部 v5-correct controls；
3. 其余 10 个 single-field/ordinary controls 上最多 1 个 regression，并保留至少 90% 的
   v5-correct controls；
4. 全体 20 题净增益至少 +2；
5. v6 legal 不低于 v5，且至少 19/20；
6. v6 process errors 不超过 v5 + 2；
7. v6 total tokens 不超过 v5 的 1.10 倍；
8. 两臂均 20/20 structural pass，所有 recorded-correct 均 fresh replay 一致。

任一门失败都不推广到更大规模或 SFT/RL。该小门即使通过，也只证明输出形状规则值得进入
新的代表性 paired gate，不直接构成训练准入。

## 外部数据边界

外部模型只可接收冻结 20 题的 question、external knowledge、catalog/schema 和逐步只读工具
反馈。Gold SQL、gold rows、gold sample、gold row count 和隐藏 verifier 数据必须留在本地。
