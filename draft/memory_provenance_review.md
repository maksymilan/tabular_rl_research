# Memory 机制再设计 + 溯源完整性审查（待 codex double check）

> 状态：**设计讨论 / 待核对**，尚未动代码。由 maksymilan 提出质疑、Claude 给出结论与改造建议。
> 请 codex 重点核对第 6 节 checklist 中标了代码锚点的**事实断言**，以及第 3 节"改动清单"的**完整性**（有无遗漏的文件 / 调用点 / 反例）。
> 本文所有行号基于审查时的工作区；codex 核对时若行号漂移，以函数名为准。

---

## 0. 三个被提出的问题（原始诉求）

1. **轨迹与上下文的关系**：observation 动作拿到的信息、以及每一步操作产生的表，是否都应进入模型上下文。
2. **memory 的定位**：当前轨迹里的 `add_to_memory` 实际上只是"把信息加进上下文 / 供后续引用"的一步，而 memory 本应是**长期、跨任务**的模块 —— 所以现在的 memory 机制定位有问题。
3. **add_to_memory 应否由 harness 隐式完成**（模型不感知），以及**溯源是否完整** —— 是不是只对聚合类操作做了 memory 链接，而中间过程产生的表、读取过程产生的数据都没有溯源？

---

## 1. 背景：当前"上下文模型"现状（已核实）

模型每一步的 observation 进上下文的形态，见 [`src/eval/rollout.py:84-123`](../src/eval/rollout.py)：

| 步骤类型 | 进上下文的内容 | 锚点 |
|---|---|---|
| `describe_table` / `inspect_column` / `read_subtable` | **完整结果**（列 / 取值 / 行） | rollout.py:101-105 |
| 关系操作（filter/join/group/…） | **metadata-only 句柄**：表名 + 列 + `row_count`，不含行内容 | rollout.py:111-117 |
| 操作结果恰好 1×1 标量 | 句柄 + 保留那个 cell | rollout.py:114-115 |
| 非表结果（如 aggregate 标量列表） | `result_sample` 前 5 行 | rollout.py:118-120 |
| `add_to_memory` | harness 算出的 `{memory_id, value, key, derivation,…}` | rollout.py:93-99 |

**结论**：诉求 1 中"observation 信息进上下文"**已成立**；唯一的设计选择是"关系操作产出的表默认只进句柄、行内容靠 `read_subtable` 按需拉"，这正是既定的"训练主动查表、避免大表撑爆上下文"策略。诉求 1 不需要新增工作（除非要把行内容也默认进上下文，那与"主动查表"冲突，**不建议**）。

---

## 2. 三层概念（把混在 "memory" 一个词下的东西拆开）

| 层 | 是什么 | 当前载体 | 作用域 |
|---|---|---|---|
| **context** | 模型每步看到的信息流 | observation 完整 + 操作表句柄 | 任务内逐 turn 累积 |
| **派生值绑定 derived-value binding**（现 `add_to_memory`） | 把中间标量具名 → 供后续可验证引用 | `memory_id` + `value_ref`，harness-grounded | **任务内** |
| **long-term memory** | 跨任务沉淀的知识 / 事实 / 经验 | **当前不存在** | 跨任务 / 会话 |

maksymilan 的洞察 = 现在第 2 层被错叫成第 3 层。

### Q1 结论：长期 memory 本轮不做

- 当前是**单任务**设定（每题独立、catalog 开局、答完即止），没有跨任务沉淀的承载场景。
- 要它有意义需要切到**多任务 / 多轮会话**设定，属于新研究维度，会大幅增加复杂度、偏离"工具设计 + 稠密过程奖励"主线。
- AGENTS.md 里设计过的 `plan` / `hypothesis` / `evidence_pointer` 也都是 task-level working state，同样不是跨任务 memory。
- **决定**：本轮不做；把 long-term memory 留作"多任务 setting"的未来方向。

---

## 3. Q2：取消 `add_to_memory`，让 harness 隐式完成（核心改造）

### 3.1 结论：应该改，而且它**自证是冗余包装**（已核实）

1. **`memory_id` 恒等于 `mem_<source_step_id>`** —— 见 [`memory_semantics.py:203`](../src/harness/memory_semantics.py) `"memory_id": f"mem_{source_step_id}"`。memory_id 不携带任何 source_step_id 之外的身份信息。
2. **`value_ref` 兜兜转转最终指向产标量的步骤** —— 见 [`emitter.py:108`](../src/harness/emitter.py) 注释 `value_ref -> the add_to_memory step`，而 add_to_memory 的 `source` 又指回 aggregate 步。
3. **最硬的证据：aggregate 步本身已把标量存进 `values[step.id]`**，`add_to_memory` 只是把它"搬"到另一个 id 下：
   - aggregate：[`plan.py:112-113`](../src/harness/plan.py) `if step.tool == "aggregate": values[step.id] = result`
   - add_to_memory：[`plan.py:91-101`](../src/harness/plan.py) `val = values.get(src)`（src=aggregate 步）→ `values[step.id] = val`
   - 执行期解析：[`plan.py:68-69`](../src/harness/plan.py) `out["value"] = values[out.pop("value_ref")]` —— 只是按 id 查 `values`，**与 id 是否来自 add_to_memory 无关**。

   也就是说：如果让 `value_ref` 直接指向 aggregate 步的 id，`resolve_cond` 在执行期一样能 `values[aggregate_id]` 拿到值，**中间那次搬运纯属多余**。

### 3.2 方案

取消 `add_to_memory` 工具；`value_ref` **直接指向产出该标量的 `step_id`**。harness 在 resolve 条件时看到 `value_ref: step_k`：
- 去 step_k 取标量（复用 `memory_semantics.extract_scalar` 的 1×1 非空校验）
- 注入 filter 的 `value`
- 记一条溯源边 `{step: step_k, via: value_ref}`（溯源逻辑不变）

模型侧：只在写 filter 时表达 `value_ref: step_k`，**不再发任何"记忆"动作**。

**信任边界（保持 V2a 精神）**：模型只表达"用第 k 步的标量当阈值"这个**引用意图**；取值、校验、derivation、溯源全归 harness。模型仍**不能伪造**阈值（它只能引用某步，值由 harness 从该步取）。

> ⚠️ 注意一个分支：`add_to_memory` 当前还处理"单行子查询"来源（`source` 是个**表 step** 而非 aggregate），取该表的 `rows[0][0]` —— 见 [`plan.py:95-97`](../src/harness/plan.py) 和 [`compiler.py:265-279`](../src/harness/compiler.py) 的 `_in_subquery` 标量退化。重构后 `value_ref` 指向表 step 时，harness 需对"表 step → 取唯一 cell"和"标量 step → 取 values[id]"两种来源都做 grounding+校验。**这是最容易遗漏的点，请 codex 特别核对。**

### 3.3 改动清单（我的推断 —— 请 codex 核对完整性）

| 文件 | 改动 | 锚点 |
|---|---|---|
| `compiler.py` | `_scalar_subquery` / `_in_subquery` 不再 emit `add_to_memory` step；`value_ref` 直接 = 产标量步的 id（`last.id`） | compiler.py:278-279、282+ |
| `plan.py` | `run_plan` 删除 `add_to_memory` 分支；执行期对 `value_ref` 指向的 step 做"标量 step→values[id] / 表 step→rows[0][0]"取值 | plan.py:91-101、68-69 |
| `memory_semantics.py` | `ground_derived_value` 的调用点从"add_to_memory 执行时"挪到"value_ref 解析 / 步骤标注时"；保留 `extract_scalar` 的 NULL/0 行/多行/多列拒绝 | memory_semantics.py:85、198 |
| `emitter.py` | 删除 `add_to_memory` 的 emit；`_references` 里 value_ref 边改指 aggregate 步（`val` 现在就是 aggregate plan id）；产标量步的 `produces` 标注派生值语义 | emitter.py:106-114、218-219 |
| `rollout.py` | `execute_tool` 删除 `add_to_memory` 分支；`condition_filter` resolve 时做隐式 grounding；`_online_references` 的 `value_ref` 边改指产标量步 | rollout.py:93-99、73-76 |
| `protocol.py` | `TOOL_SPECS` / `_ARG_SCHEMA` 移除 `add_to_memory`；`SYSTEM_PROMPT` 规则 3 改写（不再"add_to_memory then value_ref"，而是"value_ref 指向算出该标量的步骤"）；bump `PROTOCOL_VERSION`，`protocol_hash` 随之变 | protocol.py:103、132-148 |
| `build_sft_data.py` / `fill_think.py` / `splice_think.py` | 渲染与 think 模板里凡处理 `add_to_memory` 的分支 | 待 codex grep 核 |

**验证门（必须全过，复用 V2a gates）**：Spider dev replay 998/998；strict 每工具 schema 0 fail；`backward_slice` 不变式；exec-verified ≈98.4%；unit。

**数据影响**：`subset_180` 中 11/180 条含 `add_to_memory`；全量 v2 train 约 268 条（沿用 v1 统计，待重新生成时确认）。

---

## 4. Q3：溯源完整性

### 4.1 一半要纠正：中间过程产生的表**有**溯源（已核实）

`references` 由 harness 自动写，覆盖**表 dataflow 消费**：见 [`rollout.py:67-72`](../src/eval/rollout.py) / [`emitter.py:100-105`](../src/harness/emitter.py)。一个关系操作引用的输入表，若是某步产出的中间表，references 里就有 `{step:<产表那步>, as:table}`。所以 `filter→join→group` 这条表依赖链是**完整溯源**的，`backward_slice` 能从答案走回源表。**"中间表没溯源"这一点是误解。**

### 4.2 一半确认：读取 / 感知产生的数据**没有**溯源（已核实，且是 V2-ctx 病根）

`_online_references` 全函数（[rollout.py:64-81](../src/eval/rollout.py)）只链接三类：①表句柄消费 ②`value_ref`/`in_table` ③`add_to_memory.source`。**`describe_table` / `inspect_column` / `read_subtable` 从不作为"被引用的目标"出现**：

- `filter(table=singer)` 的 reference 是 `{source: singer}`（指向**基表名**），不指向 describe singer 那一步 → "列知识来自哪次 describe"**未记录**。
- `filter(country='France')` 里 `'France'` 是普通字面量 → "France 来自哪次 inspect 观测"**未记录**。
- `answer_from_context` 引用 evidence 表句柄，不引用 `read_subtable` 那步 → 读证据这一步在溯源图里**悬空**。

> **这正是"6361 次 read_subtable 全是倒数第二步仪式"的结构性病因**：perception 的产出在溯源图里没有任何下游消费边指向它，结构上就是"可删的装饰"，`backward_slice` 根本不经过它 —— 模型自然把它学成仪式而非证据。

### 4.3 统一 references 设计（Q2 + Q3 一次性落地）

决定：既然已经要 bump 协议并重生成数据，就不要只做 Q2（删除 `add_to_memory`）而把 Q3
（感知溯源）留到后面。否则新生成的 perception 数据仍然缺过程奖励依据，后续做 RL 时大概率
还要再次重构、重训、重评。**本轮目标改为：取消 `add_to_memory` + 补齐 perception grounding
provenance，一次性完成。**

为避免术语发散，后续不要再混用 `dataflow references` / `grounding references` /
`grounding_slice` 等多个名字。统一规则：

> 所有 provenance 都写在每个 step 的 `references` 数组里；每条 reference 必须带 `type`。

只保留三类 `type`：

| type | 含义 | 回答的问题 |
|---|---|---|
| `data` | 当前工具的数据输入来自哪里 | 这张表 / 中间表从哪里来？ |
| `value` | 当前工具使用的标量值来自哪里 | 这个阈值 / 标量引用由哪一步算出？ |
| `grounding` | 当前工具使用某列、字面量或证据行的认知依据来自哪里 | 模型之前在哪里看见过这个列 / 值 / 证据？ |

推荐字段：

```json
{
  "type": "data|value|grounding",
  "step": "step_3",
  "source": "program",
  "role": "table|value_ref|schema|literal|evidence",
  "target": "program.Name"
}
```

字段约束：

- `step` 与 `source` 二选一：中间结果引用用 `step`，原始表引用用 `source`。
- `role` 描述引用关系语义。
- `target` 是人和 reward 都能读懂的被支撑对象，例如 `broadcast.Time_of_day`、
  `broadcast.Time_of_day = Morning`、`answer`、`join_002`。
- 旧字段 `as` / `via` 不再作为主语义字段使用；迁移期可以保留兼容，但新 schema 以
  `type` + `role` 为准。

#### 4.3.1 `type: "data"`：数据来源

对应旧的 `{step, as}` / `{source, as}` 表输入边。例子：

```json
{
  "type": "data",
  "step": "step_3",
  "role": "table",
  "target": "join_002"
}
```

表示当前步骤使用的输入表来自 `step_3` 产生的 `join_002`。

如果引用原始表：

```json
{
  "type": "data",
  "source": "program",
  "role": "table",
  "target": "program"
}
```

#### 4.3.2 `type: "value"`：标量值来源

取消 `add_to_memory` 后，`condition_filter.value_ref` 直接指向产出标量的 `step_id`。对应
reference 写成：

```json
{
  "type": "value",
  "step": "step_5",
  "role": "value_ref",
  "target": "T2__Hours = step_5"
}
```

含义：当前 filter 的阈值由 `step_5` 真实产出的标量提供。模型只能引用 step，不能写值；
harness 必须从 history 中抽取并校验标量。支持两种 source：

- 标量工具结果，如 `aggregate` 的 `result_sample == [[v]]`。
- 1 行 1 列表结果，如 scalar subquery 被编译成 table-producing step。

必须沿用 `extract_scalar` 的严格校验：缺失、0 行、多行、多列、NULL 全部 reject，不允许猜值。

#### 4.3.3 `type: "grounding"`：观察依据

`grounding` 不表示数据流，而表示模型某个决策的**认知依据**来自哪次 observation。

列 schema 来自 `describe_table`：

```json
{
  "type": "grounding",
  "step": "step_1",
  "role": "schema",
  "target": "program.Name"
}
```

字符串 literal 来自 `inspect_column`：

```json
{
  "type": "grounding",
  "step": "step_4",
  "role": "literal",
  "target": "broadcast.Time_of_day = Morning"
}
```

最终答案行来自 `read_subtable`：

```json
{
  "type": "grounding",
  "step": "step_8",
  "role": "evidence",
  "target": "answer"
}
```

作用：把 observation 工具从“上下文里出现过的文本”变成“后续步骤显式引用过的依据”，从而让
过程奖励能判断：

- 用了某列但没有 `role=schema` grounding：扣分或拒绝。
- 用了字符串 literal 但没有 `role=literal` grounding：扣分或拒绝。
- 回答前没有 `role=evidence` grounding：说明没有读最终证据。
- `read_subtable` 没有任何后续 grounding 引用：说明是装饰性/仪式性 read。

#### 4.3.4 完整例子

问题：`Find the names of programs that are never broadcasted in the morning.`

可能步骤：

```text
step_1 describe_table(program, broadcast)
step_2 project(program.Name)
step_3 join(program, broadcast)
step_4 inspect_column(broadcast.Time_of_day)
step_5 condition_filter(join_002, Time_of_day = Morning)
step_6 project(filter_003.Name)
step_7 set_op(all_program_names except morning_program_names)
step_8 read_subtable(setop_007)
step_9 answer_from_context(...)
```

`step_5 condition_filter` 的 references：

```json
[
  {
    "type": "data",
    "step": "step_3",
    "role": "table",
    "target": "join_002"
  },
  {
    "type": "grounding",
    "step": "step_1",
    "role": "schema",
    "target": "broadcast.Time_of_day"
  },
  {
    "type": "grounding",
    "step": "step_4",
    "role": "literal",
    "target": "broadcast.Time_of_day = Morning"
  }
]
```

`step_9 answer_from_context` 的 references：

```json
[
  {
    "type": "data",
    "step": "step_7",
    "role": "table",
    "target": "setop_007"
  },
  {
    "type": "grounding",
    "step": "step_8",
    "role": "evidence",
    "target": "answer"
  }
]
```

#### 4.3.5 backward_slice 语义

`backward_slice` 需要明确参数化或拆分：

- `backward_slice(reference_type="data")`：只追踪答案计算链，保持旧语义。
- `backward_slice(reference_type="all")` 或新增 `grounding_slice()`：同时纳入 observation
  grounding，用于过程奖励与 perception load-bearing 检查。

不要无条件把 `grounding` 混进旧的 data slice，否则会改变既有“答案由哪些计算步骤得出”的含义。
但在 reward 里应该显式检查 grounding 链，因为这正是“先观察再行动”的训练目标。

---

## 5. 落地次序建议（更新：一次性完成）

1. **协议重构**：删除 `add_to_memory`，`value_ref` 直接引用产标量 step；`references`
   schema 升级为 `type=data|value|grounding`。
2. **执行与溯源统一**：`plan.py` / `emitter.py` / `rollout.py` 都使用同一套 scalar
   extraction 与 reference 生成逻辑，避免 SFT 与 eval drift。
3. **感知 grounding**：harness 在线维护“已观察 schema / literal / evidence”的索引，后续动作
   自动生成 grounding refs。
4. **验证门**：Spider dev replay、strict schema、`backward_slice(data)` 不变式、grounding
   load-bearing 检查、exec-verified、unit 全过。
5. **重新生成 perception-only 数据**：在新协议上生成感知增强数据，再做 SFT。
6. **后续 RL**：用 grounding refs 设计过程奖励，奖励查证后行动，惩罚无依据列/字面量和装饰性 read。

> 之前“先 Q2、Q3 后做”的方案不再推荐：既然已经要协议 bump 和数据重生成，就应一次性补齐
> grounding provenance，避免感知数据建在一个仍缺过程奖励依据的 schema 上。

---

## 6. 给 codex 的核对清单（请逐条 double check）

**A. 事实断言（带锚点，机械可核）**

- [ ] **A1** `memory_id` 恒为 `mem_<source_step_id>`，无其它身份信息 — `memory_semantics.py:203`。
- [ ] **A2** aggregate 步已把标量存于 `values[step.id]`；add_to_memory 对 aggregate 来源只是 `values[mem_id]=values[agg_id]` 的搬运 — `plan.py:112-113` vs `plan.py:91-101`。
- [ ] **A3** 执行期 `value_ref` 解析只按 id 查 `values`，与该 id 是否来自 add_to_memory 无关 — `plan.py:68-69`。
- [ ] **A4** `_online_references` / `_references` 不把 `describe_table`/`inspect_column`/`read_subtable` 作为被引用目标 — `rollout.py:64-81`、`emitter.py:97-114`。
- [ ] **A5** 中间产出表的消费**有**溯源边（`{step,as}`） — `rollout.py:67-72`。
- [ ] **A6** add_to_memory 存在"单行子查询表来源"分支（非 aggregate），取 `rows[0][0]` — `plan.py:95-97`、`compiler.py:265-279`。

**B. 判断 / 建议（请挑战，找反例）**

- [ ] **B1** 取消 add_to_memory、`value_ref` 指向 step_id 后，**信任边界是否仍成立**（模型能否借此伪造阈值？是否存在模型指向一个非标量 / 多行 step 而绕过校验的路径？）。
- [ ] **B2** 第 3.3 节"改动清单"是否**完整** —— grep 是否还有别处依赖 `add_to_memory` 字符串、`mem_` 前缀、或 `memory_id` 字段（如 SFT 渲染、dashboard、测试、`refine_memory`/`evidence_pointer` 占位）。
- [ ] **B3** 取消后是否影响任何**未实现但已埋点**的类型（`refine_memory`、`evidence_pointer`、`plan`、`hypothesis`）。
- [ ] **B4** Q3 的"信息 grounding 溯源"补法是否会与现有 `references`/`backward_slice` 语义冲突（grounding 边要不要进 backward_slice？会不会把"读了但没用"的 perception 也判为 load-bearing？）。
- [ ] **B5** 次序建议（先 Q2 再生成感知数据）是否合理，有无更省事的路径。

**C. 开放问题**

- [ ] **C1** `value_ref` 指向"表 step"时（单行子查询），重构后 grounding/校验如何与"标量 step"统一，是否需要在 protocol 层区分两种 source 形态。
- [ ] **C2** 是否保留 `memory_id` 作为对外稳定标识（仅改语义为 step_id 的别名），还是彻底删除、SFT 数据里只剩 `value_ref: step_id`。

---

## 7. Codex double-check 结论（2026-06-23）

> 给 Claude 的实现前核对：下面是 Codex 对第 6 节 checklist 的机械核验与设计判断。目标是
> 重构后进入一个**干净协议状态**，不要留下半旧半新的 `add_to_memory` / `memory_id` /
> `mem_` 残留。

### 7.1 事实断言核对

- **A1 成立**：`memory_id` 当前恒为 `mem_<source_step_id>`，见
  `src/harness/memory_semantics.py::ground_derived_value`：
  `"memory_id": f"mem_{source_step_id}`。它本身不携带 source_step_id 之外的新身份。
- **A2 成立**：`aggregate` 步当前已在 `plan.run_plan` 中写入 `values[step.id]`；
  `add_to_memory` 对 aggregate 来源只是 `values[add_to_memory_step_id] = values[source_step_id]`
  的搬运。表 step 来源则取 `rows[0][0]`。
- **A3 成立**：`plan.resolve_cond` 解析 `value_ref` 时只做
  `out["value"] = values[out.pop("value_ref")]`，不关心这个 id 是否来自 `add_to_memory`。
- **A4 成立**：`rollout._online_references` 与 `emitter._references` 当前不会让
  `describe_table` / `inspect_column` / `read_subtable` 成为后续步骤的被引用目标。它们只覆盖
  表句柄消费、`value_ref` / `in_table`、以及 `add_to_memory.source`。
- **A5 成立**：中间表消费已有溯源边。若当前工具消费的是前序 step 产生的表句柄，
  `references` 中会有 step 边；若消费原始表，则是 source 边。
- **A6 成立**：`add_to_memory` 不只处理 aggregate 标量，也处理 1x1 表来源。取消该工具时必须
  继续支持 `value_ref` 指向 table-producing step，并用严格 1x1 非 NULL 校验。

### 7.2 设计判断

- **B1 信任边界仍成立，但前提是 value_ref resolver 必须统一严格校验。**
  模型只能写 `value_ref: "step_k"`，不能写实际值；harness 从 history 中抽取 step_k 的真实输出。
  如果 step_k 不是唯一非 NULL 标量，必须 reject，而不是静默取第一格。
- **B2 第 3.3 节改动清单基本正确，但不完整。** 需要额外清理/更新：
  - `src/harness/executor.py`：旧 `add_to_memory()` 方法。
  - `src/harness/tests/test_compiler.py`：显式期待 `aggregate + add_to_memory + value_ref`。
  - `src/harness/tests/test_verify.py`：注释/期望里有 scalar subquery memory 链。
  - `src/sft/enrich_traj.py`：当前有 `_remap_mem()`、`mem_` 前缀重映射、
    `add_to_memory.source_step_id` 重映射、`supporting_memory_ids` 重映射。
  - `src/tool_design/validate_trajectories.py`：有 `add_to_memory` 校验分支。
  - `src/sft/splice_think.py`：有旧 memory key 替换逻辑。
  - `src/sft/EXPERIMENTS.md`、`AGENTS.md`、`src/harness/README.md` / `DESIGN.md`
    等文档会出现旧术语；实现后应同步。
  - dashboard/attribution 主要是展示层，但若 `supporting_memory_ids` / `memory_id` 字段消失，也要
    确认不会假定旧字段存在。
- **B3 暂未发现已实现的 `refine_memory` / `evidence_pointer` / `plan` / `hypothesis` 会被直接破坏。**
  但 `emitter.TOOLS` 中仍有 `refine_memory` 占位；如果新协议要干净，建议删除未实现工具占位或明确
  标注为非协议工具，避免模型/校验误以为可用。
- **B4 grounding 不应无条件混入旧 backward_slice。**
  旧 `backward_slice` 的语义是答案计算链，建议迁移为 `backward_slice(reference_type="data")`
  或只追踪 `type in {"data", "value"}`。过程奖励另行检查 `type="grounding"` 链。
- **B5 同意一次性做完。** 只做 Q2 会让即将生成的 perception 数据仍缺 grounding provenance，
  后续 RL 前大概率重做数据。现在 bump 协议，就应该同时补齐 Q3。

### 7.3 干净重构目标

实现后，代码与新生成数据中应尽量满足：

- **协议层没有 `add_to_memory` 工具**：
  - `protocol.TOOL_SPECS` 不包含 `add_to_memory`。
  - `_ARG_SCHEMA` 不包含 `add_to_memory`。
  - `SYSTEM_PROMPT` 不再教模型调用 `add_to_memory`。
  - `PROTOCOL_VERSION` 必须 bump。
- **训练/评测对外不再出现 `memory_id` / `mem_` 作为必要字段**：
  - `condition_filter.value_ref` 直接是产标量的 `step_id`。
  - `answer_from_context` 不再需要 `supporting_memory_ids`；若仍保留，必须重命名或明确为
    `supporting_value_step_ids`，避免旧 memory 语义残留。更干净的方案是删除该字段，依赖
    `references`。
  - 新 trajectory 中不应有 `tool_call.tool == "add_to_memory"`。
  - 新 trajectory 中不应有 `produces.kind == "memory"`。
- **内部实现仍可保留“scalar grounding”概念，但不要叫 memory**：
  - 可把 `memory_semantics.py` 改名/拆分为 `scalar_grounding.py` 或至少新增中性 API：
    `ground_scalar_reference(history, source_step_id)`。
  - 如果暂不改文件名，也应避免新代码继续暴露 `memory_id` / `key` / `content` 给模型协议。
- **references 统一新 schema**：
  - 每条 reference 必须有 `type`。
  - 旧 `{step, as}` / `{source, as}` 迁移为 `{type:"data", step/source, role:"table", target}`。
  - 旧 `{step, via:"value_ref"}` 迁移为 `{type:"value", step, role:"value_ref", target}`。
  - 新增 `{type:"grounding", step, role:"schema|literal|evidence", target}`。
- **感知工具必须能被后续 grounding 引用**：
  - `describe_table` 输出的 schema 支撑后续列使用。
  - `inspect_column` 输出的 value domain 支撑后续字符串 literal。
  - `read_subtable` 输出的 evidence rows 支撑最终 answer 或中间关键决策。
- **兼容旧数据要隔离**：
  - 旧 v1/v2/v2ctx 数据可保留，但新协议生成的数据应使用新 `schema_version`。
  - build/eval 脚本如果支持旧协议，应显式按 schema_version 分支，不要让旧 memory 逻辑混入新协议。

### 7.4 实现建议顺序

1. 先改/新增统一 helper：`extract_scalar` / `ground_scalar_reference`，同时支持 scalar result 与
   1x1 table result，严格 reject 非标量。
2. 改 `compiler.py`：scalar subquery 不再 emit `add_to_memory`，`value_ref` 直接指向 last step id。
3. 改 `plan.py`：删除 `add_to_memory` 分支，`resolve_cond` 能用 step id 从 `values` 或 1x1 table
   中解析 value。
4. 改 `emitter.py`：输出新 references schema；产标量 step 标注 `produces.kind="scalar"`；
   `condition_filter` 的 value reference 直接指向产标量 step。
5. 改 `rollout.py`：在线状态从 `memory/memid_to_step` 改成 history-based scalar resolver；
   在线生成 `data/value/grounding` references。
6. 改 `protocol.py`：删除 `add_to_memory` 工具，改 `value_ref` 说明，bump version。
7. 改 `build_sft_data.py` / `enrich_traj.py` / `splice_think.py` / trajectory validator / tests / dashboard。
8. 重新生成 sample trajectories，跑：
   - compiler unit tests；
   - Spider replay；
   - strict per-tool schema；
   - `backward_slice(data)` 不变式；
   - grounding load-bearing 检查；
   - SFT build smoke。

### 7.5 需要 Claude 继续核对的问题

- `answer_from_context` 是否彻底删除 `supporting_memory_ids`，还是迁移为更中性的字段。
  Codex 倾向删除，统一依赖 `references`。
- `grounding role=evidence` 是否只允许引用最近一次 `read_subtable(evidence.table)`，还是允许引用
  中间 read。建议先只强约束最终 answer 的 evidence grounding，避免规则过复杂。
- `describe_table` 一次描述多张表时，schema grounding 的 target 应精确到 `table.column`；
  若后续 join 产生 `T1__Name`，target 应回溯到源列（如 `program.Name`），不要记录为
  `join_002.T1__Name` 后就丢失源表语义。

---

## 8. 统一设计与状态（→ SSOT）

本文是**讨论与逐条核对的存档**。最终冻结的决策、结构化 `target` schema、以及"哪些做了 /
哪些没做"的实现状态清单已收进唯一事实源
[`provenance_redesign.md`](provenance_redesign.md)。后续实现以该文件为准；本文不再增长。

SSOT 相对本文的关键更新：
- `target` 由字符串改为**结构化对象**（reward 直接读字段，join 后回溯源列）。
- 分层时序明确：**A(删 memory) + B(references schema) 本轮做；C(grounding) 与 F 推后但已登记
  清单**，依据是 grounding 属 sidecar、不进 SFT 训练文本、必要时点在 RL。
- 新增 **Feature F：subtable 按句柄归并**（context management，maksymilan 2026-06-23 建议）。
- 已决：删除 `supporting_memory_ids`（靠 `type=value` references 反推）；`role=evidence` grounding
  先只硬约束最终 answer。
