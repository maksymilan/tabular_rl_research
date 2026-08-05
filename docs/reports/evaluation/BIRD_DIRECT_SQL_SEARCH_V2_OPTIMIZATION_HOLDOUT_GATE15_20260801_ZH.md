# Direct SQL Search v2 优化与未重叠 Holdout Gate15

## 状态

本报告在 provider rollout 之前登记选择、比较和判定规则。实验只用于诊断；无论结果如何，
均不进入 SFT 或 RL。

## 为什么需要 v2

`direct-sql-search-v1` 的首批 Gate16 为 7/16，配对 atomic version39 为 10/16。但这不是一
个充分公平的能力比较：atomic 已积累多轮 prompt、事实反馈、resident state 和错误恢复优化，
而 v1 只有一个简短 SQL prompt、重复的 history/workspace 行载荷和一句异常字符串。

对 Gate16 的 label-blind 行为审计得到：

- 154 个模型动作中有 46 个 schema inspect、90 个 data inspect、14 个 final 和 2 个 search；
- 两个 max-step episode 各只有 20 条独特 inspect SQL，却分别重跑 9 和 8 条旧 SQL；
- 11/16 个 episode 的 reasoning 出现不确定状态下的 `DISTINCT` 争论，共 41 turn；
- 旧 model input 平均 9,383 字符、最大 22,660 字符；recent-4 observation 与最近 8 个
  workspace 结果重复携带行值；
- 两个 public protocol error 都是合法 carrier 中的工具名拼写错误；旧反馈未提供 error
  code、attempted action 或成功状态未变化这一边界。

这些统计不读取或输出 gold SQL、gold rows 或 verifier reference values。

## 单一优化包

`direct-sql-search-v2` 保持两个工具及其参数完全不变，只改变外部教师决策与事实反馈：

1. prompt 加入答案单位、行粒度、eligible population、精确输出槽、聚合/重复语义和终止前
   slot audit；明确外部知识映射优先，禁止为得到 plausible singleton 而发明限制；
2. `search_values` 只用于存储拼写或位置确实未知的 literal；显式映射值优先直接 SQL 验证；
3. CURRENT DIRECT SQL STATE 保留最近 6 个完整结果、此前 6 个事实卡和更老动作计数；
   recent-4 observation 只保留 state pointer，行值只出现于 authoritative state；
4. 任一已经成功的精确 action 在只读数据库上再次提交时返回结构化
   `no_progress_error/successful_action_repeated`，指出 prior step；该错误不触发同类三次早停；
5. 所有 v2 错误携带 code、fact-only details、`successful_state_changed=false`，解析成功时还
   携带完整 attempted action；
6. SQL preview 不二次执行查询，但增加 column count、结果是否完整、最小结果行数、可见
   duplicate rows、可见 NULL cells 和 final eligibility；
7. DeepSeek provider history 继续保留精确 action、默认不回灌旧 reasoning；非 split provider
   仅保留最新 reasoning 全文，较旧 reasoning 标记省略。完整原始回复仍进入 audit artifact。

v1 保持可选兼容入口，其 DeepSeek system prompt SHA-256 仍为
`ab062a777f99d0d62f24f65450bf6fb11423ec0320e5f2ad0d63569ebc57b430`，协议 hash 仍为
`e3185245c5ba7c6e`。

## 未重叠选择

- source：`data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.jsonl`；
- source SHA-256：`0993665a2df7bc6447ee1b558119fa41a674251262b90c4130f5f22bc9a79cb8`；
- discovery：位置 0..15；位置 16 已用于一次 provider smoke；
- holdout：位置 17..31，共 15 条，未用于 v1 Gate16、实现调试或 provider smoke；
- 每个 scheme 每题一个 semantic attempt；只允许同一 live episode 内依据公开错误修复；
- model：`deepseek-v4-flash`；lazy catalog；recent-4；max steps 30；max tokens 2,048；
  provider retries 10；`bird-set`；
- paired control：fresh atomic `version39`，相同 15 条、模型、预算和 terminal metric。

不在同一 holdout 上再跑 v1，避免把同一批题同时用作策略调参与 v1/v2 准确率选择。因此本门
只回答“充分优化后的 Direct SQL 相对 atomic 是否仍有明显差距”，不把 discovery 与 holdout
的非同题差异宣称为 v1→v2 accuracy uplift。

## 预登记判定

- 工程稳定：Direct SQL v2 legal terminal 至少 14/15，max-step 不超过 1；
- 值得扩大公平比较：v2 verifier-correct 不低于 atomic version39 超过 1 题，即
  `direct_correct >= atomic_correct - 1`；
- 本小样本上“未观察到准确率劣势”：`direct_correct >= atomic_correct`；
- 若低于 atomic 2 题及以上，继续保留为 SQL control，但不把当前双工具模式称为 atomic 的
  可替代工作模式；
- 单独报告 paired both/direct-only/atomic-only/neither、exact McNemar/binomial p、actions、
  errors、no-progress、search、tokens、legal 和 max-step；
- 成功数只是诊断成功轨迹数；training admission 固定为 0。

## 结果

### 基础设施状态（不计作语义结果）

首次沙箱内启动的 15/15 记录都在第一个 provider 请求、任何模型 action 形成之前以
`api_error` 结束；每条均为 `steps=1, errors=0, legal=false`。该 transport-only 产物保留在：

`data/trajectories/direct_sql_search_20260801/holdout15_v2_r1/`

因为没有形成语义动作，它不消耗预登记的每题一次 semantic attempt，也不进入任何正确率或
legal 统计。随后申请在可联网执行环境以新目录重跑时，安全审查要求用户明确知情授权：请求会
把自然语言 question、external knowledge、数据库 catalog 和逐步只读工具反馈发给 `api.md`
配置的外部 DeepSeek API；gold SQL、gold rows 和 hidden verifier 数据不会进入 model input。

用户随后明确授权把这 15 条任务的自然语言问题、external knowledge、数据库 catalog/schema
和只读工具反馈发送给外部 DeepSeek API，并明确禁止发送 gold SQL、gold rows 或隐藏验证器
数据。授权后使用新目录完成 Direct SQL v2 与 fresh atomic version39；两边都没有 transport
retry、carrier retry 或 provider failure，因此 15 对任务全部形成一个语义尝试。

### 主结果

| 指标 | Direct SQL v2 | atomic version39 |
|---|---:|---:|
| verifier-correct / retained diagnostic success | **6/15** | **10/15** |
| legal terminal | 15/15 | 15/15 |
| clean / recovered success | 5 / 1 | 10 / 0 |
| model actions | 88 | 115 |
| mean actions | 5.867 | 7.667 |
| process errors | 1 execution | 1 argument validation |
| max-step failures | 0 | 0 |
| total tokens | 327,401 | 854,227 |
| provider transport retries | 0 | 0 |
| value-search calls | 1 | N/A |

Direct 使用 atomic 的 38.33% tokens，减少 61.67%；actions 减少 23.48%。但正确率少 4 题。
配对表为 both correct 5、Direct-only 1、atomic-only 5、neither 4，exact two-sided
McNemar/binomial `p=0.21875`。样本不足以给总体差距置信结论，但方向和预登记门限都不支持
把当前 Direct SQL 替换 atomic。

唯一 Direct-only 是 `bird_train_05544`；atomic-only 是 `bird_train_00600`、
`bird_train_01050`、`bird_train_02507`、`bird_train_04869`、`bird_train_05873`。唯一一次
`search_values` 出现在错误任务 `bird_train_05161`，因此本门仍没有 search-used success。

### 审计

- Direct 的 15/15 记录通过 strict reparse、两工具闭包、model-input hidden-key、
  prior-inspect terminal grounding 和 tool-output fresh replay；6/6 recorded successes fresh
  replay 后仍为 `bird-set` correct；
- atomic 的 10/10 verified episodes 通过现有 independent structural gate 和 full prompt
  variant gate；
- 两边 task IDs、模型、max steps、max tokens、history turns、单次语义尝试和
  denotation metric 相同；result directories、protocol hashes 和 tool schemas 隔离；
- Direct protocol 为 `direct-sql-search-v2` / `1767711f3e087d9e`，public schema hash 为
  `4b67e8b88d26c9cf64389888b007a522e1087448d98d8d3daca4bb9c5f963d57`；atomic 为
  `version39` / `28201a10377ad567`。

### 四层行为诊断

#### Prompt

v2 已明确要求 answer unit、row grain、eligible population、aggregation/duplicate semantics
和 exact output slots，但这些仍是 `<think>` 中的自然语言约束，不是 harness 可验证的状态。
raw SQL 把 population、join、aggregation、ranking 和 projection 压进一个字符串；一条 SQL
可以完全合法且 preview 看似合理，却同时绕过多个 prompt 约束。行为审计中 88 个 turn 有 56
个 uncertainty turn、29 个 uncertainty+`DISTINCT` turn，后者分布在 10/15 个 episode；明确
output-audit 的 turn 只有 17 个。说明 prompt 增强被部分采用，但没有形成稳定的逐槽终止检查。

#### 错误反馈

结构化错误按设计工作：唯一 Direct execution error 含 attempted action 和
`successful_state_changed=false`，随后形成 recovered success。问题是其余 9 个失败全部是
合法 final 后的 wrong answer；SQL parser、SQLite 和 preview 都没有错误可反馈。当前 preview
只证明列数、可见行数/重复/NULL 和 final eligibility，不能验证问题语义中的 population、grain、
formula 或 field mapping。错误反馈层因此修复了 protocol/execution recovery，却无法触达主导
的 silent semantic error。

#### 上下文管理

v2 的 repeat guard 和 resident-pointer state 达到工程目标：72 条 inspection SQL 全部唯一，
0 repeated action、0 repeated inspection、0 adjacent repeat、0 max-step。Direct 也显著少于
atomic 的 actions/tokens。不过当前 model input 仍平均 12,439 字符、最大 24,691，最近 user
state 中一条 prior SQL 平均出现 3.65 次。更关键的是 resident state 保存的是 query/action
事实，不保存 harness-owned 的“当前答案单位、population、grain、required slots”语义账本；
模型不再重复 SQL，不代表它在各轮保持了同一个问题解释。v1 Gate16 与本 Holdout 任务不同，
不能用 9,383→12,439 字符宣称 v2 上下文变差或变好，只能确认本轮没有 loop/churn。

#### 模型实际回复

Direct 的错误 episode 平均 5.78 步，正确 episode 6.00 步；atomic 分别为 7.80 和 7.60 步。
Direct 更早形成可执行 final，但更高比例是 silent semantic failure。工具分布也显示它主要做
35 次 schema inspect、37 次 data inspect、15 次 final，只搜索 1 次；优化后的模型没有陷入
重复探索，却经常在 semantic bindings 尚未稳定时终止。atomic 的 typed intermediate
relations、显式 producing handles 和 exact-table terminal 给模型更多分解检查点；本小样本不能
把因果贡献精确分配给某一个 atom，但结果支持“typed decomposition/grounded state，而非仅更多
prompt prose”是其表现更好的主要候选原因。

### 预登记判定

- 工程稳定（legal ≥14/15 且 max-step ≤1）：**通过**；
- 值得扩大公平比较（Direct ≥ atomic−1）：**失败**，6 < 9；
- 未观察到 accuracy deficit（Direct ≥ atomic）：**失败**；
- Direct 低 2 题及以上：**触发**，实际低 4 题。

因此 Direct SQL v2 的结论是：**工程闭环和低成本 SQL control 可行，但作为 atomic
version39 的替代工作模式不可行。** v2 已充分修复首轮可见的 prompt、错误、上下文重复和
no-progress 公平性问题；剩余差距主要是合法 SQL 内部的 silent semantic decisions，而不是
协议不稳或缺少 SQL 表达力。不得在本 Holdout 上继续调参或重跑；后续若研究 v3，应在新
discovery cohort 上测试 harness-owned semantic checklist 或 typed SQL plan/checkpoint，再使用
新的未重叠 holdout。

### 产物

- Direct：`data/trajectories/direct_sql_search_20260803/holdout15_v2_r1_authorized/`；
- atomic：`data/trajectories/direct_sql_search_20260803/atomic_version39_holdout15_r1_authorized/`；
- paired no-leak summary：
  `data/trajectories/direct_sql_search_20260803/paired_holdout15_summary.json`；
- Direct behavior audit：
  `data/trajectories/direct_sql_search_20260803/holdout15_v2_r1_authorized/behavior_audit.json`。

所有结果仍为 diagnostic-only，training admission 固定为 **0**。
