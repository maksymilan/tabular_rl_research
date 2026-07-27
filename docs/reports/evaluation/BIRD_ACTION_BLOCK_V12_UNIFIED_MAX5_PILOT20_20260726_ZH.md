# BIRD action-block-v12 统一接口固定 20 题轨迹审计（2026-07-26）

## 结论

新的 `action-block-v12` 没有在这组固定 20 题上提升能力。DeepSeek v4 Flash 在
`bird-set` 下得到 **13/20**，合法终止 **17/20**；配对的 version24 原子工具
baseline 和 action-v4 都是 **16/20**。

这次结果同时支持两个更窄的判断：

1. **环境职责解耦是成立的。** 模型没有调用 `plan` / `update_plan`，没有维护 DAG、
   step 状态或依赖字段；harness 自动调度、保留成功分支、区分 root error 与 blocked
   descendant。20 题只产生 2 个未执行的 blocked 后继。
2. **当前 prompt 和终止接口把模型推向了过度原子化，并引入了新的 carrier/protocol
   摩擦。** 20 题有 20 次空 visible-content carrier retry，最终 3 题未合法终止；
   另外 8 个 protocol error 中有 6 个来自终止块形状或终止证据引用不一致。

因此，13/20 不能解释成“最多 5 个调用压低了能力”：本次没有任何 block 达到 5 个
调用，实际最大宽度只有 4。真正退化点是如何决定 block 边界、如何结束，以及
provider carrier，而不是宽度上限。

## 冻结条件

- cohort：`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` 的前 20 题
- model：DeepSeek v4 Flash
- decoding：temperature 0，thinking enabled，reasoning effort high
- context：4 个成功 assistant/observation pair 的滚动历史
- action budget：30 primitive actions / 30 model turns
- action block：每块 1--5 个调用
- metric：`bird-set`
- gold SQL：不进入模型输入，只用于 harness 终止评分
- protocol：`action-block-v12`
- protocol hash：`65ee2bf989f9b48a`
- system prompt SHA256：
  `80e0de316b39775e5504ba929dd0c1efb2f6480ec96a4a8b72963ed3efd1121b`

第一次 sandbox 内运行的 20 个请求均在网络传输前被系统拒绝，不计为策略结果。
这里报告的是获准联网后完整运行的 `r2`。

## 总体结果

| 方案 | 正确 | 合法 | 平均模型轮数 | 平均 primitive actions | Process errors | Blocked | 总 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| version24 原子 baseline | 16/20 | 20/20 | 6.35 | 6.35 | 1 | n/a | 690,165 |
| action-v4 | 16/20 | 20/20 | 4.75 | 8.15 | 9 | 10 | 450,896 |
| action-v10 low-friction | 15/20 | 20/20 | 4.05 | 7.90 | 6 | 4 | 379,201 |
| **action-block-v12 unified max-5** | **13/20** | **17/20** | **5.75** | **8.10** | **13** | **2** | **527,502** |

与 version24 baseline 配对：

- 13 题两者都对；
- 4 题两者都错；
- v12-only recovery 为 0；
- baseline-only 为 3 题：`00541`、`04189`、`06489`；
- 配对净变化 -3，精确双侧二项检验 `p=0.25`。

与 action-v4 配对也是 13 题都对、4 题都错、v12-only 为 0，action-v4-only 为
`00541`、`00593`、`04189`。

样本只有 20 题，-3 不构成统计显著的能力结论；但 3 个 carrier 终止失败和重复出现的
终止协议错误足以构成工程退化证据。

## Block 行为

v12 共解析出 106 个合法 action block，其中 19 个是终止块，87 个是非终止块：

| 非终止 block 类型 | 数量 | 占比 |
|---|---:|---:|
| 只观察（describe / inspect / read） | 47 | 54.0% |
| 只执行 relational operation | 33 | 37.9% |
| 观察与执行混合 | 7 | 8.0% |

全部合法 block 的宽度分布是：

| 宽度 | block 数 |
|---:|---:|
| 1 | 68 |
| 2 | 27 |
| 3 | 8 |
| 4 | 3 |
| 5 | 0 |

剔除 19 个单调用终止块后，非终止 block 平均宽度为 **1.60**。action-v4 的
非终止 block 平均宽度是 **2.04**，观察/执行混合 block 为 24/75（32.0%）；
action-v10 平均宽度是 **2.33**。

当前 prompt 的“一旦下一决策需要反馈，单调用 block 很正常”被模型学成了偏强的串行
策略。它确实避免了大量 speculative descendants，但也损失了 action block 原本的
主要收益：

- v12 只有 13 个 block 包含同块本地依赖，合计 21 个本地引用；
- action-v4 是 26 个 block、47 个本地引用；
- action-v10 是 27 个 block、49 个本地引用；
- 三个方案都没有观察到前向引用，v12 新增的前向拓扑调度在该 cohort 上没有被使用。

因此，本轮不是“模型一次思考后并行解决更多子问题”，而更接近“把原子的多轮调用套进
统一 action_block 外壳”。

## 推理质量

### 有改善的部分

- 模型推理基本集中在问题、schema、值和结果上，没有重新承担 DAG、状态迁移或
  dependency status 的维护。
- `02935`、`06151`、`06171` 展示了理想路径：先取得必要 schema，再在一个窄 block
  中完成过滤/连接/投影，最后用 grounded evidence 终止。
- `06171` 用三调用依赖链正确保留了四位作者，没有像 action-v10 那样错误去重。
- `04377` 和 `02437` 的 root failure 只阻塞依赖后继，下一轮能够利用 resident success
  修正，说明 harness 的错误边界按预期工作。
- 低摩擦解析进行了 24 次确定性修正：17 次把跨 block 的 `$local_id` 解析为之前的
  resident handle，7 次解析唯一列后缀。它避免了更多接口错误。

### 没有改善的部分

合法解析轮次的 reasoning 平均长度从 action-v4 的约 1,136 字符上升到 v12 的约
1,229 字符；超过 2,000 字符的轮次从 13 增至 18，超过 4,000 字符的轮次从 5 增至
7。reasoning tokens 从 27,592 增至 34,336。

更长的推理没有转化为更好的答案承诺：

- `00541` 对模型输入中明确的 external knowledge 产生了不存在的字符串差异，并反复
  讨论 `person_id`、`credited` 和 `role` 的含义；它实际已经得到正确出生地，最后却
  因终止证据仍使用过期 `$person_info` 而失败，随后 carrier 耗尽。
- `00593` 在第 5 轮已经看到两条满足 explicit medication/reason 条件的记录，却继续
  用 14 轮寻找额外的 “admission” 解释。整题用 20 轮、135,839 tokens，占本次全部
  tokens 的 25.8%，最终仍未执行两个日期差并因 carrier 失败。
- `01152` 的关系路径和“最晚出生日期代表最年轻”都正确，但在 schema 已显示
  `middleName` 的情况下只选择 `firstName,lastName`，得到两列而不是三列。
- `04189` 使用 left join，保留了没有 review 的 free sports app 行。前五行样例与 gold
  相同，但完整 denotation 更大，说明模型没有守住人口集合。
- `05440` 先 left join 全部 Paper 与 Journal 再按年份取 top-1，选中了 journal 为空的
  最新行；它没有先建立“有 journal homepage 的论文”人口。
- `06026` 的跨区域求和推理内部一致，但把重复 West/South 行全部求和为 48.392；
  gold 要求 33.8744。它仍属于既有问题/数据/金标歧义，不是 block 调度错误。

整体上，v12 消除了模型维护环境状态的认知负担，却没有提高语义理解、answer-slot
承诺、join population 或 distinct/grain 判断。当前推理质量更像是“环境推理更干净，
任务推理没有越过模型原有上限”，并且在个别歧义题上变得更冗长。

## 接口与 carrier 错误

13 个 process errors 分为：

| 类型 | 数量 |
|---|---:|
| protocol error | 8 |
| argument validation error | 3 |
| execution error | 2 |

8 个 protocol error 中：

- 4 次把 terminal 与其他调用放进同一 block，违反“terminal 必须是唯一调用”；
- 2 次 terminal evidence 使用了已跨 block 的 `$local_id`，而普通工具位置却允许
  low-friction resolver 自动恢复同类引用；
- 1 次 visible JSON 含非法控制字符；
- 1 次顶层对象不是严格的 `tool + arguments`。

前六次暴露的是终止接口与普通 action 调用不一致，而不是数据库能力问题。

API 层共 136 次请求，等于 115 个模型轮次、20 次 carrier retry 和 1 次 transport
retry。14 个成功返回的 retry 事件都属于：provider 给出非空 reasoning、`finish_reason`
为 `stop`，但 visible content 为空。另有 `00541`、`00593`、`06489` 各自耗尽三次
请求尝试并最终非法终止。

这不是 token length truncation。它更像 DeepSeek split carrier 在当前 API-facing
prompt 下没有稳定地把“思考完成”落实为 visible JSON action。旧 action-v4 在同一
cohort 没有 carrier retry。

## 逐题审阅

| 题目 | action-v4 | v12 | 轨迹判断 |
|---|:---:|:---:|---|
| `00541` | ✓ | carrier | 已找到 York；跨块 `$id` 终止失败后 carrier 耗尽，且 reasoning 幻觉了 external knowledge 差异。 |
| `00593` | ✓ | carrier | 两条药物记录已在第 5 轮可见，仍持续探索至第 20 轮，没有提交日期差。 |
| `01152` | ✗ | wrong | 关系链更短且无错误，但漏掉 `middleName`，输出两列而非三列。 |
| `02437` | ✓ | ✓ | 先猜错取消状态字面量并产生 blocked 后继，观察值后恢复；比 v4 多 4 轮。 |
| `02605` | ✓ | ✓ | 关系结果正确；曾把 terminal 与其他调用混在一块，重试后完成。 |
| `02935` | ✓ | ✓ | 三轮、零错误，属于理想窄 block 轨迹。 |
| `03275` | ✓ | ✓ | 最终三列正确，但用了 10 轮；schema、采样、全量读取和 join 高度串行。 |
| `03390` | ✓ | ✓ | 最终正确；经历 malformed JSON 和过期 `$join_002`，过程质量偏弱。 |
| `04189` | ✓ | wrong | left join 扩大人口集合；前五行看似一致但完整 denotation 错误。 |
| `04377` | ✓ | ✓ | 在 schema 返回前同块猜测 unsupported `starts`，root 失败后恢复。 |
| `04619` | ✓ | ✓ | 正确聚合；两次 domain inspection 略多但证据充分。 |
| `04636` | ✓ | ✓ | 三个单调用 block，答案正确但没有 batching 收益。 |
| `05149` | ✓ | ✓ | 用 row read 消除同名 recipe 歧义后取 calories，探索是有效的。 |
| `05440` | ✗ | wrong | 延续既有 left-join-before-rank 人口错误。 |
| `05952` | ✓ | ✓ | 正确但由 4 轮增至 8 轮，并出现一次跨块 `$id` 终止错误。 |
| `06026` | ✗ | wrong | 连贯地求出 48.392，但仍未解决已知问题/数据/金标歧义。 |
| `06151` | ✓ | ✓ | 4 轮、零错误；low-friction 跨块引用解析成功。 |
| `06171` | ✓ | ✓ | 三调用依赖链正确，5 个 primitive actions；是本轮最符合预期的轨迹。 |
| `06336` | ✓ | ✓ | 最终 word/id 正确；先发生一次 terminal 混块协议错误。 |
| `06489` | ✗ | carrier | v12 已按 external knowledge 正确过滤出 `OBJ_SAMPLE_ID`，但终止前 carrier 耗尽；属于潜在语义恢复、实际得分失败。 |

## 下一步建议

建议保留：

- action block 最多 5 个调用；
- 不向模型暴露 `plan` / `update_plan`；
- resident factual context、依赖发现、拓扑调度、blocked propagation 和错误计数全部由
  harness 负责；
- 当前确定性的 namespace / prior-local-id low-friction resolution。

优先修复三个通用接口问题，而不是继续增加“充分思考/多探索”提示：

1. **把 provider transport 与学生语义 prompt 分开。** DeepSeek API-facing adapter
   应追加已经验证过的简短 carrier 约束：reasoning 限长、必须为 visible action 预留
   token、没有发出 JSON action 就不算思考完成。学生仍只学习统一的
   `<think> + raw JSON` 协议。
2. **让 terminal 使用与普通调用一致的引用解析。** `evidence.table` 应能解析已知的
   prior-block logical id；更进一步，可以允许 terminal 作为 action block 的唯一 sink，
   引用同块结果并由 harness 最后调度，而不是强制整个 block 只能有 terminal。这是
   统一接口语义，不是题目补丁。
3. **纠正 block 边界措辞。** 不再强调“单调用 block 很正常”，改为：一次并行提交所有
   参数已经 grounded 的独立观察；只有当后续参数需要语义解释新反馈时才结束 block。
   这能恢复并行观察和已 grounded 依赖链，而不要求模型维护 DAG。

`01152`、`04189`、`05440` 这类 answer-slot / population / join-type 问题不宜继续用全局
prompt prose 修补。它们更适合由真实 model↔harness 因果教师轨迹提供监督，并在独立
gate 中验证。

## 产物

- 完整轨迹：
  `data/trajectories/batch_plan_20260726/action_block_v12_unified_max5_pilot20_r2.all.jsonl`
- manifest：
  `data/trajectories/batch_plan_20260726/action_block_v12_unified_max5_pilot20_r2.all.manifest.json`
- 逐题审计：
  `data/trajectories/batch_plan_20260726/action_block_v12_unified_max5_pilot20_r2.audit.jsonl`
- 对 version24 baseline 配对：
  `data/trajectories/batch_plan_20260726/action_block_v12_unified_max5_vs_version24_pilot20.paired.json`
- 对 action-v4 配对：
  `data/trajectories/batch_plan_20260726/action_block_v12_unified_max5_vs_v4_pilot20.paired.json`
- 对 action-v10 配对：
  `data/trajectories/batch_plan_20260726/action_block_v12_unified_max5_vs_v10_pilot20.paired.json`
- 未计分的 sandbox 网络失败审计：
  `data/trajectories/batch_plan_20260726/action_block_v12_unified_max5_pilot20_r1.all.jsonl`
