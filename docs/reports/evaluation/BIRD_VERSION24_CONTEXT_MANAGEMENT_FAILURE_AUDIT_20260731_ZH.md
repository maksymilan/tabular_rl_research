# BIRD version24 fixed-200 上下文管理与长程失败审计

日期：2026-07-31
范围：DeepSeek v4 Flash、version24 fixed-200、`bird-set`，145/200；重点分析 55 条失败轨迹。

## 结论

当前上下文管理的主要问题不是“recent-4 把事实删掉了”，而是 **resident state 的生命周期
和显著性管理仍然偏粗糙**：事实保存得较完整，但 source schema、row reads、所有派生 handle
和 derivation 以平铺、累积方式同时出现，缺少 active branch 与 inactive branch 的区别。
长轨迹中，旧分支、零行分支、异常解释和当前主线会竞争注意力。

证据分为三层：

1. **没有硬事实丢失证据。** 200 题没有 context overflow 或 context retry；没有任何任务
   因每表最多保留四个 row reads 而淘汰已观察事实。长轨迹仍能在第 17、24、30 步复述早期
   ID、行值和表关系。
2. **存在局部冗余和循环证据。** 55 个失败中，30 个在终止前仍携带不属于最终 evidence
   dependency closure 的派生 handle；3 个出现完全相同动作的重复，其中两个是 22/30 步
   的最长失败；7 个保留零行 filter handle。
3. **还没有“扩大历史能提升能力”的因果证据。** version37 的 head-5+tail-5 比 recent-4
   使用更多动作和 token，却从 5/7 降到 4/7；version40 保留全部 reasoning 并大幅缩短
   prompt，仍在 Gate50 从 42/50 降到 37/50。两项实验均不能支持简单扩大 transcript。

因此，上下文仍值得作为工具优化方向，但下一步应优化 **model-visible resident-state
rendering、branch liveness 和异常分支折叠**，而不是把 recent-4 改成 recent-10、保留整段
失败 reasoning，或一次性塞入完整数据库上下文。

## 当前上下文管理方式

### 1. 初始任务上下文

默认 `catalog-v1` 只提供：

- 表名与行数；
- 外键关系；
- 问题；
- optional external knowledge。

列结构和值域通过 `describe_table`、`inspect_column` 和 `read_subtable` 按需获取。完整 schema
不会默认放入开局上下文。

### 2. recent-4 legal history

每轮输入为：

```text
system: protocol / tool contract
user: catalog + question + optional external knowledge
[最多四对成功 assistant action + compact observation]
user: latest compact observation + CURRENT ENVIRONMENT STATE + optional LAST TOOL ERROR
```

- 只保留最近四个成功 action/observation 对；
- 历史 observation 中的大型 schema、rows、frequent values 和 scalar sample 被替换为
  resident-state pointer；
- 被拒绝的 assistant reasoning 不进入默认合法历史；
- 最新失败调用、参数与完整结构化错误进入 `LAST TOOL ERROR`；
- 一旦下一步成功，`last_error` 清空。

recent-4 负责决策连续性，不承担事实存储。

### 3. resident state

Harness 持久保存：

- 已访问 source table 的 schema；
- inspected columns 与 frequent values；
- 每表最近四种 row reads；
- 所有 derived handles、列、row count、creator step；
- 每个派生关系的 `relation-derivation-v1`；
- scalar-producing steps；
- plan control state。

source table 若从未被访问，不进入 resident state。相同 read key 会被覆盖，而不是无限追加。
当前 version39 进一步只修改 model-visible rendering：

- 完全等价的 reads 合并并保留所有来源 step；
- 同一输入、同一输出形状的多个未引用零行 filter 被折叠为 fact-only summary；
- plan evidence 只显示 `step_id + tool`，完整事实仍由 tables/values 提供。

Canonical harness state、执行和 replay 不因此改变。

## version24 fixed-200 定量结果

### 1. 轨迹长度与正确率

| episode 动作数 | 任务数 | 正确 | 正确率 | 平均末轮上下文字符 |
|---|---:|---:|---:|---:|
| 1–4 | 25 | 21 | 84.0% | 20,410 |
| 5–8 | 125 | 100 | 80.0% | 23,329 |
| 9–12 | 36 | 17 | 47.2% | 25,864 |
| 13–16 | 10 | 7 | 70.0% | 27,604 |
| 17+ | 4 | 0 | 0.0% | 32,240 |

长轨迹表现明显更危险，但关系不是单调的：13–16 步仍有 70%，而 9–12 步只有 47.2%。
episode 长度同时受题目难度、关系复杂度和错误循环影响，不能把这张表解释成“上下文每增加
一轮就使能力下降”的因果证据。最值得注意的是 17+ 只有四题且全部失败，它们都已进入
具体轨迹审计。

55 条失败的动作数中位数为 8、均值 9.2、最大 30；19 条至少 10 步，4 条至少 15 步。

### 2. 末轮上下文组成

55 条失败轨迹：

| 组成 | 均值字符 | 中位数字符 | 最大字符 |
|---|---:|---:|---:|
| 初轮完整输入 | 18,526 | 17,755 | 26,214 |
| 末轮完整输入 | 25,515 | 24,643 | 36,768 |
| 末轮 resident-state section | 5,547 | 5,015 | 17,195 |
| 末轮 retained assistant history | 551 | 549 | 966 |

version24 system prompt 固定为 16,400 字符。失败轨迹末轮中，resident state 平均占
20.9%，retained assistant history 只占 2.2%。这说明 recent-4 assistant transcript 并不是
上下文膨胀的主体；主要增长来自 resident state，固定 system/tool prompt 则是最大的单块。

把失败终态中的各字段逐一从同一 JSON 中移除后，得到其对 resident state 的平均边际字符
贡献：

| resident-state 字段 | 平均边际字符 | 约占 resident state |
|---|---:|---:|
| relation derivation | 2,041 | 36.8% |
| source schema | 1,595 | 28.8% |
| row reads | 798 | 14.4% |
| inspected columns/values | 136 | 2.4% |
| plan | 5 | 0.1% |

因此，旧 row reads 的常驻是部分长轨迹的真实噪声源，但不是总体最大的状态组成。全删旧
reads 对 55 条失败末轮平均只减少约 798 字符，即完整末轮输入的 3.1%。更高杠杆的渲染优化
是把 verbose relation derivation 改为 Harness 生成的单行 handle card，并同时压缩旧 reads。
组件占比不会加总到 100%，因为 state 仍包含 table/step 索引、columns、row counts 等公共骨架。

reads 在少数任务上仍很重：`06165`、`06246`、`02868` 的末轮 read payload 分别约为
4,702、4,211、3,739 字符。这些任务适合作为“仅保留最新完整 read、旧 read 按需重开”的
定向诊断集。

从初轮到末轮，失败轨迹平均增加 6,989 字符，中位增加 6,812 字符。

### 3. 正确与失败轨迹的状态规模

| 指标 | 145 条正确 | 55 条失败 |
|---|---:|---:|
| 平均动作 | 6.79 | 9.20 |
| 平均末轮上下文字符 | 23,167 | 25,515 |
| 平均末轮 state 字符 | 3,632 | 5,547 |
| 平均派生 handle | 3.43 | 4.98 |
| 平均非最终依赖 handle | 1.00 | 1.64 |
| 含非最终依赖 handle 的任务 | 70/145 | 30/55 |
| 含完全重复动作的任务 | 4/145 | 3/55 |
| 含零行 filter handle 的任务 | 9/145 | 7/55 |

失败轨迹确实更长、state 更大，但在相同动作区间内差距明显缩小；13–16 步中正确轨迹的
state 甚至略大。非最终依赖 handle 在正确轨迹里也很常见。因此“state 大”是复杂度与分支
数量的指标，不是独立根因。

### 4. 没有观察到的丢失

- 200/200 没有 `context_overflow`；
- API context retry 为 0；
- 200/200 没有每表超过四种 unique row reads，因此没有事实因容量上限离开 resident state；
- 55 条失败终态没有 model-visible equivalent read duplicate；
- compact rolling observation 已避免把完整 rows/schema 同时放入历史 observation 和 state。

这否定了“当前主要是 raw transcript 重复四遍”或“早期 row read 被简单截断”两个假设。

## 长程失败逐例观察

### `bird_train_06299`：30 步，明确的状态膨胀与循环

- 先在 `MenuPage` 找到两组 UUID 对应的 menu IDs；
- 随后 `Menu` join/filter 持续返回零行；
- 模型对相同 ID 集合重复执行多种 `condition_filter`；
- 终态有 19 个 derived handles，其中 13 个是零行 filter；
- 有 8 次完全重复动作；
- 最后仍能准确复述第 2/4 步得到的 ID，说明事实没有遗忘；
- 失败原因是异常分支无法收敛，达到 30 steps，而不是早期 ID 消失。

用当前 version39 renderer 离线重渲染该终态，可通过零行分支折叠减少约 4,082 个 state
字符。它能缓解注意力噪声，但不能保证模型从数据异常中选出正确路径。

### `bird_train_04822`：22 步，重复 perception 与零行分支

- 反复检查同一 `TEAM`/`LEAGUE` 列；
- 三次读取同一 15 行表；
- 多次构造语义相同或相近的零行条件分支；
- 有 5 次完全重复动作、6 个零行 filter、8 个不在最终 evidence closure 的 derived handles；
- 最终以空表合法终止。

这里 recent-4 确实不再展示十几步前的 exact call，但 resident state 仍保存 inspected column
与 read rows。更准确的诊断是“事实可见但显著性不足/模型重新探索”，不是硬丢失。

### `bird_train_06246`：24 步，状态很大但主因是语义改写

- 末轮 state 17,195 字符，为 55 条失败最大；
- 保留 12 个 derived handles、102 行 read sample，8 个 handle 不属于最终 evidence closure；
- 模型从 Delaware state/county 多个解释分支中不断切换；
- 最后自行增加 `residential_mailboxes > 0` 并把 Delaware 解释为 NY 的 county。

轨迹没有重复 exact action，早期 state/county 事实仍在。上下文噪声可能放大了分支竞争，
但首要可见错误仍是模型对“residential area / Delaware”的语义改写。

### `bird_train_06165`：17 步，反证“早期事实被遗忘”

最终 reasoning 能逐项复述第 5–12 步的五个 paper ID、Title NULL 模式、author join 匹配和
left join 行数，说明 resident state 对早期事实的持久化有效。模型基于这些异常数据选择空表，
与 benchmark 不符；这是数据/基准解释问题，不是 recent-4 丢失。

### `bird_train_00593`：12 步，事实完整但错误 singleton 假设

第 3–4 步已经观察到两条同时满足患者、药物和病因的用药记录。模型后续选择其中一条计算
11 天并终止，遗漏另一条 18 天记录。已观察的两行仍在 resident state；问题是无题目依据地
选择 singleton，而不是第二行离开上下文。

### `bird_train_03636`：被拒绝 reasoning 连续性候选

模型连续三次提交完全相同的非标量 `value_ref`，每次都收到结构化
`ScalarGroundingError` 和 attempted action。默认历史不保存 rejected reasoning，但
`LAST TOOL ERROR` 始终存在且 state 未改变。这里可以怀疑 reasoning continuity 不足，
但错误反馈并未丢失；模型仍重复，说明仅保留失败文本未必足够。

## 已有对照实验怎么解释上下文假设

### 1. 更长 legal history 没有收益

version37 七题配对：

| 历史策略 | 正确 | 合法动作 | 教师错误 | 总 token |
|---|---:|---:|---:|---:|
| recent-4 | 5/7 | 76 | 3 | 709,739 |
| head-5+tail-5 | 4/7 | 93 | 7 | 1,051,252 |

样本很小，但方向明确：保留更多 exact history 没有修复语义失败，反而增加动作、错误和 token。

### 2. 保留全部 reasoning 也没有已隔离的收益

version40 同时改变 prompt、plan、observer 命名和 reasoning history，因此不是纯 history
消融。但它保留全部成功/失败 reasoning 后，Gate50 仍从 version24 的 42/50 降到 37/50，
新增五个回归全部是输出槽位/表示错误。它至少说明“reasoning 全留”不是一个已经验证的
通用修复。

### 3. 完整数据库上下文主要提升效率，不稳定提升正确率

完整 schema + BIRD 描述 + 每列两个样例的 Gate32 从历史 version38 的 18/32 到 19/32，
动作减少 32.4%，但过程错误增加，净正确率变化不显著；保留 `inspect_column` 的完整上下文
方案为 17/32。完整上下文会减少 schema 探索，也可能让模型看到表面匹配列后过早收敛。

### 4. 关系 derivation 有明确 token 成本

version24 相对 version20 的 prompt tokens 增加 5.4%，总 tokens 增加 5.8%。报告已指出最
可能来源是 table-bound derivation 在 resident state 中累积。它没有造成系统性准确率下降，
但也没有形成显著提升。

### 5. version39 compaction 是窄修复

将当前 version39 state renderer 离线应用到 55 个 version24 失败终态：

- 只有 3 条状态发生变化；
- 55 条平均只减少约 95 字符；
- 最大收益是 `06299` 的 4,082 字符；
- `06246`、`04822` 分别减少约 580、561 字符。

因此 version39 正确瞄准了最严重的零行循环，但不是全局上下文压缩方案。

## 判断：哪里粗糙，哪里并不粗糙

### 已经比较稳健

- recent-4 有明确边界，不无限增长；
- 大型历史 observation 不重复事实 payload；
- schema、inspected values、row reads、scalar 和 derivation 都由 Harness 持久化；
- rejected call 的完整结构化错误可见；
- canonical state 与 model-visible compaction 分离，replay/grounding 不受影响；
- 当前轨迹没有事实容量淘汰或 context overflow。

### 仍然粗糙

1. **所有 derived handles 平铺常驻。** 没有 active、inactive、abandoned branch 生命周期。
2. **derivation 与 row samples 按表累积。** 即使 handle 已与当前主线无关，仍与活跃结果同权。
3. **状态没有 dependency-first 排序。** 当前主线的输入闭包没有优先展示，模型需从 flat map
   自己恢复工作图。
4. **零行是合法结果，不是 progress signal。** 模型可连续创建许多零行分支，直到 max_steps。
5. **只保留 latest error。** 对即时恢复足够，但没有结构化记录“同一根因已尝试多少次”；
   另一方面，保留更多失败 reasoning 可能造成假设锚定。
6. **固定 system/tool prompt 很大。** version24 为 16,400 字符，当前 version39 external-
   teacher provider prompt 为 21,170 字符。API cache 降低计费/延迟但不消除注意力竞争。
   version40 已证明不能无差别删掉具体输出槽和复杂参数边界。

## 推荐的优化方向

### P0：dependency-aware active/archive renderer

Canonical state 和所有 handle 都不改，只改变 model-visible 排列：

- `ACTIVE RELATIONS`：最近产生、recent-4 中引用、plan/error 引用的 handles，并递归展示其
  dependency closure、完整 derivation 和 reads；
- `ARCHIVED RELATIONS`：较老且近期未引用的 branch，只保留 handle、creator step、operator、
  row_count、columns 和一行 dependency 摘要；
- archived handle 仍可直接作为下一步工具参数；一旦被引用，下轮自动提升为 active；
- source schemas、values 和 terminal grounding 不删除。

这一方向不解释问题语义、不猜 gold、不改执行，只解决 flat state 的注意力层级。

### P0：先做 offline next-action visibility gate

在付费模型实验前，对现有 200 条完整 prefixes 重渲染：

1. 记录每个真实下一步 action 引用的 table、column、step/value；
2. 验证这些引用在新 renderer 的前一轮仍完整可见；
3. 100% 保证 canonical action 可原样 replay；
4. 统计 p50/p95 state chars、row payload、derivation payload；
5. 目标：p95 state chars 至少降低 25%，任何 prefix 不增长，next-action visibility 200/200。

这里使用“真实下一步”只做离线无损审计，运行时 renderer 不能窥视未来 action。

### P1：read subsumption，而不只是 exact dedup

当同一表、同一 predicate/order/offset 的新 read：

- columns 是旧 read 的超集；
- limit 不小于旧 read；
- 返回 rows 完整覆盖旧 read；

则旧 read 的 rows 可折叠，只保留 `equivalent/subsumed_from_steps`。必须先验证 observation
binding 和后续 literal grounding 仍能映射原 step，不能只按文本删除。

当前 fixed-200 没有 exact duplicate resident reads，这一方向主要面向未来长轨迹，不是
解释当前 55 题的主要收益来源。

### P1：扩展 fact-only empty-branch summary

沿用 version39 的安全边界：

- 同一输入/输出形状的多个零行 filters 合并；
- 保留每个 handle、creator step 和 exact predicate；
- 不加入“应该换表/改条件”等 policy advice；
- 可增加事实字段：attempt count、是否存在完全相同 predicate、是否所有分支均为零行。

该方向直接针对 `06299` 和 `04822`，比增加通用 prompt 更可归因。

### P2：错误根因计数，而不是保留全部 rejected reasoning

可以在 `LAST TOOL ERROR` 中增加 Harness-owned、非语义字段：

- same error root occurrence count；
- same canonical attempted action count；
- state unchanged；
- affected argument path。

它不能建议修复方法，但能让模型知道自己正在重复同一无进展动作。先在 `03636`、`02438`
和一组已恢复错误控制上做 micro-gate。不要同时保留所有 rejected reasoning，否则无法区分
计数反馈和模型假设锚定。

### 暂不优先

- recent-10 / full transcript；
- 全部 rejected reasoning；
- 一次性完整 schema/value dump；
- 再次大幅缩短 prompt；
- Harness 自动选择“正确”的 active branch；
- 模型自写 factual memory 或总结后替换原始 state。

这些方向已有负面或不显著证据，或会破坏 Harness-owned factual authority。

## 建议的最小因果实验

新版本只改变 model-visible state rendering，保持：

- public tools、参数、执行与 join 规则不变；
- recent-4 不变；
- system/provider prompt 不变；
- error policy、max_steps、`bird-set` 不变；
- canonical resident state、replay、grounding 不变。

冻结 Gate16：

- 8 个上下文压力 target：优先包括 `06299`、`04822`、`06246`、`06165`、`03636`，再选
  三个 state 大、分支多的失败；
- 8 个动作数/state 大小匹配的正确控制；
- 当前远端 DeepSeek v4 Flash 已更新，必须在同一时间窗重跑旧 renderer baseline，不能用
  2026-07-24 的历史响应直接做模型能力因果对照。

建议扩量门：

1. target 至少恢复 3/8；
2. controls 至少保留 7/8；
3. legal termination 不下降；
4. process errors 不增加；
5. p95 prompt tokens 至少下降 15%；
6. 所有新成功轨迹通过 fresh replay、结构/no-leak gate；
7. 若 target 只减少 token、没有恢复正确率，则将其定位为效率优化，不宣称能力提升。

## 产物

- 定量审计脚本：`src/eval/audit_version24_context_management.py`
- 汇总：
  `data/results/version24_fixed200_failure_reasoning_only_20260730/version24_context_management_summary.json`
- 全 200 题上下文指标：
  `data/results/version24_fixed200_failure_reasoning_only_20260730/version24_all_context_metrics.jsonl`
- 55 条失败指标：
  `data/results/version24_fixed200_failure_reasoning_only_20260730/version24_failure_context_metrics.jsonl`

本审计不执行数据库、不读取或引用 gold SQL，也不把模型 reasoning 当作事实来源。
