# Direct-SQL + Value Search 双工具诊断

## 结论

Direct-SQL-search 是一个与 atomic、action-block、relational-program 互斥的第四种 tool
scheme。模型每轮只看到两个工具：

1. `search_values(query, table?, column?, limit?, offset?)`；
2. `execute_sql(sql, mode)`，其中 `mode="inspect"` 返回执行反馈，`mode="final"` 终止并
   对完整结果做隐藏 `bird-set` 判分。

冻结的 `direct-sql-search-v1` 在工程上可行，但 Gate16 **不能替代 atomic version39**：同一批题、同一
DeepSeek v4 Flash、同一 recent-4/30-step/单次语义尝试设置下，新 scheme 为 **7/16**，
atomic 为 **10/16**。新 scheme 的 7 条成功全部通过 fresh replay 与结构审计，但只调用了
2 次 value search，且两次都发生在最终跑满 30 步的失败 episode 中。因此当前证据支持把
它保留为低 token 的 SQL 控制和接口研究平台，不支持扩量或进入 SFT/RL。

当前诊断入口是 `direct-sql-search-v2` / `search-values-execute-sql-v2`。它不改变两个工具
或参数，只补齐 v1 相对 atomic 明显不足的外部教师 prompt、结构化错误、事实型 preview
反馈、resident context 去重和 immutable-database no-progress 检查。v1 仍可显式选择并保持
原 system prompt/protocol hash 以复现实验。其未重叠 Holdout Gate15 在用户明确授权外部
DeepSeek 数据范围后完成：v2 为 **6/15**，fresh atomic version39 为 **10/15**；两者均
15/15 legal。v2 通过工程稳定门，但低 4 题，未通过扩大公平比较或无 accuracy deficit 门，
因此仍无 accuracy promotion。

## 为什么 search 不是“SQL 理论上做不到”

普通 SQL 可以用 equality、`LIKE`、FTS extension 或自定义 UDF 实现部分字符串检索，
所以不能声称 value search 在计算能力上超越 SQL。这里新增的是一个数据库服务边界：

- 跨全库或指定表/列搜索真实 distinct stored values；
- 用 NFKC/casefold、token、substring、trigram 和 edit-similarity 做确定性 lexical ranking；
- 每列先用 SQLite exact/no-case-exact 与稳定 token/trigram anchor 召回，最多 4,096 个候选；
- 返回 exact stored literal、来源表列、frequency、match type、score、候选范围和截断信息；
- 不让模型编写 vendor-specific FTS/UDF，也不让一次模糊查询扫描全部 distinct value 后才排序。

因此它的正确定位是 **out-of-band bounded retrieval**，而不是新的关系代数算子。它不创建
relation、不选择答案行、不推断同义词、不修改数据库，也不能直接作为 terminal evidence。

## v2 公平性优化

对冻结 v1 Gate16 的 label-blind 审计显示，主要缺口不在 SQL 表达力：154 个动作含 46 次
schema inspect、90 次 data inspect；两个 max-step episode 各只有 20 条独特 inspect SQL，
却分别回头执行 9 和 8 条旧查询。11/16 个 episode 反复争论 `DISTINCT` 等输出语义。旧
recent-4 observation 和最近 8 个 workspace entry 还重复序列化相同的行值，而 error state
只有异常字符串。

v2 保持公开 schema hash 不变，增加：

- 与 atomic 外部教师同层级的 answer unit、row grain、eligible population、exact output
  slots、aggregation/duplicate semantics 和 final slot audit；
- `search_values` 的使用边界：只有 required literal 的存储拼写或位置未知时才搜索；
- `CURRENT DIRECT SQL STATE`：最近 6 个完整输出、此前 6 个事实卡和 archived action count；
  recent-4 observation 只指向 resident state，避免行值双份出现；
- 对任一已成功精确 action 的全 episode no-progress rejection。SQLite 连接只读，精确重跑
  不会产生新事实；错误指出 prior success step，且不触发同类三次早停；
- 带 code、attempted action、`successful_state_changed=false` 的结构化错误；
- 不二次执行 SQL 的 bounded preview profile：column count、complete/lower-bound row count、
  可见 duplicate rows、可见 NULL cells 和 final eligibility。

v1 DeepSeek system prompt SHA-256 仍为
`ab062a777f99d0d62f24f65450bf6fb11423ec0320e5f2ad0d63569ebc57b430`，协议 hash 仍为
`e3185245c5ba7c6e`。预登记、授权与完成结果见
`docs/reports/evaluation/BIRD_DIRECT_SQL_SEARCH_V2_OPTIMIZATION_HOLDOUT_GATE15_20260801_ZH.md`。

## 两工具边界

### `search_values`

```json
{"tool":"search_values","arguments":{
  "query":"Sankee",
  "table":"restaurant",
  "column":"label",
  "limit":10,
  "offset":0
}}
```

- `query` 必填，非空且不超过 256 字符；
- `table` 可省略；省略时搜索 opening catalog 中全部 source tables；
- `column` 只能与 `table` 一起使用；
- `limit` 为 1..20，`offset` 为 0..200；
- 全库结果按 match tier、score、frequency、table、column、value 稳定排序；
- executor 复用 atomic version45 的 `bounded-v1` 单表候选召回，但 scheme 自己负责跨表合并。

### `execute_sql`

```json
{"tool":"execute_sql","arguments":{
  "sql":"SELECT county FROM restaurant WHERE label = 'sankee'",
  "mode":"inspect"
}}
```

`inspect` 接受一条只读 `SELECT`、`WITH`、`PRAGMA` 或 `EXPLAIN QUERY PLAN`，返回列名与最多
20 行。`final` 只接受 `SELECT`/`WITH`，并且同一 SQL（忽略首尾空白和一个结尾分号）必须
已经在本 episode 的 inspect mode 成功执行。这样 final 仍属于第二个工具，同时强制最终
SQL 经历真实环境反馈，不能退化为 one-shot submission。

连接启用 `PRAGMA query_only=ON`；多语句和 INSERT/UPDATE/DELETE/DDL/ATTACH 均被拒绝。
每轮仍使用严格 `think-json-v1` carrier，无 parser repair。

## 实现位置

- 协议、公开 schema、strict parser：`src/sft/direct_sql_search_protocol.py`；
- 执行、provider loop、隐藏判分：`src/eval/iterative_sql.py --interface
  search-values-execute-sql-v2`；v1 为显式 replay/复现兼容入口；
- scheme registry：`src/sft/tool_schemes.py` (`tool-scheme-registry-v4`)；
- 统一 evaluation / teacher launcher：`src/eval/run_tool_scheme.py` 与
  `src/sft/generate_tool_scheme_rollouts.py`；
- fresh replay / structural audit：`src/eval/audit_direct_sql_search.py`。

推荐入口：

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python -u src/sft/generate_tool_scheme_rollouts.py \
  --tool-scheme direct-sql-search -- \
  --tasks-json TASKS.jsonl --result-dir RESULT_DIR \
  --n 16 --workers 8 --max-steps 30 --api-retries 10 \
  --denotation-comparison bird-set
```

## 冻结 Gate16

### 设置

- task source：`data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.jsonl`
  的前 16 条，恰好包含 8 个原 target 与 8 个 control；
- task SHA-256：`0993665a2df7bc6447ee1b558119fa41a674251262b90c4130f5f22bc9a79cb8`；
- model：`deepseek-v4-flash`；
- context：lazy catalog + rolling legal history recent 4；
- max steps / same-type errors：30 / 3；
- max tokens：2,048；provider retries：10；
- 每题一条语义轨迹，不对错误答案做第二次解释；
- terminal metric：`bird-set`；gold 只在 hidden verifier 中使用；
- direct scheme protocol hash：`e3185245c5ba7c6e`；
- provider system prompt SHA-256：
  `ab062a777f99d0d62f24f65450bf6fb11423ec0320e5f2ad0d63569ebc57b430`；
- public tool schema SHA-256：
  `4b67e8b88d26c9cf64389888b007a522e1087448d98d8d3daca4bb9c5f963d57`；
- atomic version39 protocol hash：`28201a10377ad567`。

### 结果

| 指标 | direct-sql-search | atomic version39 |
|---|---:|---:|
| verifier-correct | **7/16 (43.75%)** | **10/16 (62.50%)** |
| legal terminal | 14/16 | 16/16 |
| clean / recovered success | 7 / 0 | 8 / 2 |
| model actions | 154 | 141 |
| mean actions | 9.625 | 8.8125 |
| process errors | 2 | 4 |
| total tokens | 538,791 | 1,359,421 |
| value-search calls / episodes | 2 / 2 | N/A |
| max-step failures | 2 | 0 |

direct scheme 用了 atomic 的 39.6% tokens（减少 60.4%），但动作数增加 9.2%，正确率低
3 题。配对表为：both correct 6、direct only 1、atomic only 4、neither 5；exact two-sided
McNemar/binomial `p=0.375`。样本很小，不能据此估计总体差距，但方向不支持扩量。

唯一 direct-only gain 是 target 6489；4 个 regression 是 593、3969、4189、6165。两次
`search_values` 出现在 593 和 6165，二者均跑满 30 步；atomic 则分别在 26 和 18 步答对。
因此唯一 gain 不能归因于 search，观察到的 search 使用也没有产生成功轨迹。

### 审计

- direct 的 16/16 记录通过 strict reparse、两工具闭包、model-input hidden-key、prior-inspect
  terminal grounding 和可重放 tool-output 结构检查；
- 7/7 recorded successes 在 fresh database replay 下仍为 `bird-set` correct；
- atomic 的 10/10 verified episodes 通过现有 `audit_verified_rollouts.py` structural gate，
  其中 8 clean、2 recovered；
- direct artifact 出现 5 次 transport retry，但无 carrier retry；atomic 无 transport/carrier
  retry。transport retry 不是 semantic action。

产物：

- `data/trajectories/direct_sql_search_20260801/gate16_r1/`；
- `data/trajectories/direct_sql_search_20260801/atomic_version39_gate16_r1/`。

## 是否可行

需要区分三种“可行”：

1. **工程闭环可行：是。** v2 两工具 strict protocol、真实反馈、hidden scoring 和 fresh
   replay 均工作，15/15 合法终止，0 max-step，6/6 success replay-correct。
2. **低 token SQL 控制可行：是。** 它明显降低上下文 token，可作为模型/接口诊断和
   one-shot direct SQL 之外的交互控制。Holdout v2 使用 327,401 tokens，而 atomic 使用
   854,227，减少 61.7%；actions 也减少 23.5%。
3. **作为当前 atomic 的替代主线：否。** Holdout v2 为 6/15，atomic 为 10/15；配对为
   both 5、Direct-only 1、atomic-only 5、neither 4，`p=0.21875`。v2 仅搜索一次且发生在失败
   episode。停止扩量，不建立 SFT exporter 或 RL environment。

Holdout 行为审计还显示 v2 已消除重复行动和 max-step，但 88 turn 中仍有 56 个 uncertainty
turn、29 个 uncertainty+`DISTINCT` turn，只有 17 个明确 output-audit turn。9 个失败全部是
合法 final 后的 silent semantic wrong answer；因此剩余差距主要不在 SQL 表达力或错误恢复，
而在 raw SQL 把 population、grain、aggregation 和 output shape 隐藏在一个不可分解字符串中。
完整结果与四层诊断见
`docs/reports/evaluation/BIRD_DIRECT_SQL_SEARCH_V2_OPTIMIZATION_HOLDOUT_GATE15_20260801_ZH.md`。

所有输出保持 `diagnostic_only`，成功轨迹数不等于训练准入数；本实验训练准入数为 0。
