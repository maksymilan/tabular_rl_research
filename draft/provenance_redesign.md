# Provenance + Context 重构 —— 统一设计与实现状态（SSOT）

> **这是本次重构的唯一事实源(single source of truth)。** 讨论过程与逐条代码核对见
> [`memory_provenance_review.md`](memory_provenance_review.md)（Claude 提问 + Codex double-check）。
> 本文只记录**冻结的决策**和**实现状态清单（哪些做了 / 哪些没做）**。
>
> **当前总状态（2026-06-23）：阶段 A + B + 验证门完成，协议升级 v2b、schema_version v3。**
> 剩余 `- [ ]` = 阶段 C(grounding)、F(subtable 归并)、D-3/D-4(感知数据生成 + SFT)，均推后。
> 验证全过：run_all 98/98 + compile 99.9%；emit 296/300 verified；online replay 296/296=100%；
> SFT build 296/296、0 memory 残留。详见 §5 实现记录。
> 关键纪律：本轮只解锁"干净协议 + 感知微调"；grounding(阶段 C) 与 context feature(阶段 F)
> 允许推后，但**必须留在本清单里**，不得遗漏。

---

## 0. 范围

| 编号 | 议题 | 处置 |
|---|---|---|
| Q1 | 长期 / 跨任务 memory | **本轮不做**（单任务无承载场景；未来需多任务 setting） |
| Q2 | 取消 `add_to_memory`，`value_ref` 直指 step_id | **做**（阶段 A） |
| RS | references 统一 schema（`type` + 结构化 `target`） | **做**（阶段 B） |
| Q3 | perception grounding provenance | **设计冻结，实现可推后**（阶段 C，不阻塞感知微调） |
| F | subtable 按句柄归并（context management） | **实证触发**（待首版 SFT 失败模式；见 §5） |

---

## 1. 冻结的设计决策

### 1.1 Q1 长期 memory —— 不做
单任务设定（每题独立、catalog 开局、答完即止）没有跨任务沉淀场景。要它有意义需切到多任务 /
多轮会话 setting，属新研究维度。`plan` / `hypothesis` / `evidence_pointer` 同为 task-level，亦非
跨任务记忆。**结论：本轮不实现；留作多任务 setting 的未来方向。**

### 1.2 Q2 取消 `add_to_memory`
`add_to_memory` 是冗余包装：`memory_id == mem_<source_step_id>`，且产标量步已把值存进
`values[step.id]`，add_to_memory 只是把它搬到另一个 id（依据见 review §3.1，A1–A3/A6 已核验成立）。

**新形态**：取消该工具；`condition_filter.value_ref` **直接指向产出该标量的 `step_id`**。harness 在
resolve 时去该 step 取标量、严格校验、注入、记溯源边。

**信任边界（保持 V2a）**：模型只表达"用第 k 步的标量当阈值"这一引用意图；取值 / 校验 /
derivation / 溯源全归 harness，模型不能写值、不能伪造。

**两种合法 source 必须都支持**（review ⚠️ + Codex A6）：
- 标量工具结果（`aggregate` 等，值在 `values[step.id]`）；
- 1×1 表结果（scalar subquery 编译成 table-producing step，取唯一 cell）。
两者都走同一个严格 `extract_scalar`：缺失 / 0 行 / 多行 / 多列 / NULL 一律 reject。

**内部命名**：保留"scalar grounding"概念但**不再叫 memory**。`memory_semantics.py` 改名/拆为
`scalar_grounding.py`，对外中性 API `ground_scalar_reference(history, source_step_id)`。协议层、
训练 / 评测数据**不再出现** `add_to_memory` / `memory_id` / `mem_` / `produces.kind=="memory"`。

### 1.3 references 统一 schema（含结构化 target）
所有 provenance 写进每个 step 的 `references` 数组，每条带 `type`。三类：

| `type` | 含义 | 回答 |
|---|---|---|
| `data` | 数据输入来自哪里 | 这张（中间）表从哪来？ |
| `value` | 标量阈值来自哪里 | 这个 `value_ref` 由哪步算出？ |
| `grounding` | 某列 / 字面量 / 证据行的**认知依据**来自哪里 | 之前在哪次 observation 见过它？ |

**`target` 改为结构化对象（采纳，不用字符串拼接）**。原因：Codex 的字符串 target
（`"broadcast.Time_of_day = Morning"` / `"T2__Hours = step_5"`）混格式、需 reward 二次解析，且
join 后要回溯源列时字符串扛不住。统一字段：

```json
{
  "type": "data | value | grounding",
  "step": "step_3",          // 与 source 二选一：引用中间结果用 step
  "source": "program",       // 引用原始表用 source
  "role": "table | value_ref | schema | literal | evidence",
  "target": {                // 结构化；reward 直接读字段，不解析字符串
    "handle": "join_002",    //  role=table
    "table":  "broadcast",   //  源表（schema/literal/value_ref 回溯后的源表）
    "column": "Time_of_day", //  role=schema/literal/value_ref
    "value":  "Morning",     //  role=literal（可选）
    "slot":   "answer"       //  role=evidence（"answer" 或中间决策标识）
  }
}
```

- reward / 校验只看 `type` + `role` + `step|source` + 结构化 `target`。
- 需要人读标签时，由结构化字段**确定性渲染**（不手写），沿用项目"description 是 definition 的
  确定性渲染"纪律。
- 旧字段 `as` / `via` 退役（迁移期可兼容，但新 schema 以 `type` + `role` 为准）。
- join 前缀列 `T1__Name` 的 `target` 必须**回溯源列** `{table:"program", column:"Name"}`，不得记成
  `{handle:"join_002", column:"T1__Name"}` 而丢失源表语义（Codex 7.5-Q3）。

### 1.3.1 `target` 如何被确定性构造（harness-owned，非模型）

整条 reference（含 `target`）由 harness 从 **工具调用参数 + tool history + 列血缘** 确定性生成，
模型不写。**offline(`emitter`) 与 online(`rollout`) 必须调用同一个共享函数**（新增
`src/harness/provenance.py::build_references`），否则 SFT 与 eval 漂移 —— 与 `scalar_grounding`
共享同理。

需要的在线状态：
- `handle_to_step`：表句柄 → 产它的 step（已有）。
- `col_lineage[handle][col] -> (源表, 源列)`：**列级血缘，是 grounding `target` 能回溯源列的基础
  设施**。`join_tables` 加 `left/right_prefix` 时建立 `T2__Time_of_day -> col_lineage[broadcast]
  ["Time_of_day"]`（递归回溯）；源表自身 `col -> (源表, col)`；`condition_filter`/`project` 透传；
  `derive_column`/`project as` 新列指向其表达式列。
- （C 阶段）`described{源表->step}` / `inspected{(源表,源列)->step}` / `read{句柄->step}`：已观察索引。

按 role 的构造规则：

| type/role | `step`/`source` 来自 | `target` 字段 | 阶段 |
|---|---|---|---|
| `data`/`table` | table-ref 参数查 `handle_to_step`（中间表）或源表名 | `{handle}` 或 `{table}` | A+B |
| `value`/`value_ref` | 谓词里 `value_ref` 指向的 step | `{table,column}` = 被注入阈值的谓词列 | A+B |
| `grounding`/`schema` | 动作每个源列 → `described[源表]` | `{table,column}` = 源表源列 | C |
| `grounding`/`literal` | string 谓词 → `inspected[(源表,源列)]` | `{table,column,value}` | C |
| `grounding`/`evidence` | `answer.evidence.table` → `read[句柄]` | `{slot:"answer"}` | C |

**本轮 / C 边界**：A+B 的 `data`/`value` 边 `target` 简单可得 —— value 边的列回溯**单步剥前缀**
（`T2__Hours`→`Hours`）即够，不需跨表全血缘，故不被 C 阻塞；`grounding` 三类边 + `col_lineage`
的完整跨表回溯属 C。

### 1.4 Q3 perception grounding（设计冻结，实现见阶段 C）
为每个动作的列 / 字面量 / 证据补 `type=grounding` 边，使 perception 在溯源图里**有下游消费**，
根治"6361 次 read_subtable 全是倒数第二步仪式"（review §4.2）：

- `role=schema`：动作用到的源列 → 最近一次暴露它的 `describe_table` 步。
- `role=literal`：filter 的字符串字面量 → 最近一次确认它的 `inspect_column` 步。
- `role=evidence`：`answer_from_context` → 提供证据行的 `read_subtable` 步。

**硬约束（必须进 `emitter.validate()`，与 backward_slice 不变式同级）**：每个字符串字面量必有
`literal` 边、每个源列必有 `schema` 边、每个 answer 必有 `evidence` 边。否则 grounding 自身又
沦为可选装饰 —— 这正是 Q3 要根治的病。

**backward_slice 语义**（Codex B4，同意）：旧 data-slice 保持纯洁。
`backward_slice(reference_type="data")`（或只走 `type in {data, value}`）= 答案计算链；grounding
**不混入** data-slice，由过程奖励单独检查 `type=grounding` 链。

**为何可推后而不阻塞感知微调**：`references`（含 grounding）是 harness sidecar，**不进 SFT 训练
文本**（SFT 文本只有 `think + tool_call` 和 observation envelope）。grounding 的必要时点是 **RL 过程
奖励**，不是 SFT。补 C 时只需**重跑 trajectory 的 sidecar 生成**（本地分钟级），**不需重训**已训好的
感知 SFT 模型。

**待 Claude 续核（Codex 7.5）**：
- `answer_from_context.supporting_memory_ids` —— **决定：删除**，统一靠 `type=value` references
  反推（不改名，彻底去 memory 语义）。
- `role=evidence` 范围 —— **决定：先只硬约束最终 answer 的 evidence grounding**，中间 read 的
  grounding 按需再加，避免规则一上来爆炸。

### 1.5 分层与时序（关键）
| 层 | 内容 | 改 SFT 文本? | 时序 |
|---|---|---|---|
| **A** | 删 `add_to_memory`，`value_ref`→step_id | 是 | **必须先做** |
| **B** | references 加 `type` + 结构化 `target`，data/value 迁移 | 否（sidecar） | 随 A 一起 |
| **C** | grounding 边在线生成 + validate 不变式 + reward | 否（sidecar） | **RL 前做，不阻塞感知微调** |
| **F** | subtable 按句柄归并（context management） | 否（渲染层） | 推后，见 §3 |

**本轮目标 = A + B**：解锁干净协议 → 重新生成感知数据 → 感知 SFT 拿第一个增量信号。
**C 与 F 推后但已登记在 §2 清单，不得遗忘。**

### 1.6 推理路径（rollout / RL env）改造 —— 与数据生成对称，单独列清

数据生成(`emitter`)与推理(`rollout`)**共用** `protocol` / `scalar_grounding` / `build_references`，
所以下列推理侧改动多数是 A/B/C 的对称落地；单独列出以免被"数据生成"标题掩盖。

- **references 不进模型上下文（关键认知）**：推理时模型每步只看 observation envelope
  `{step_id, status, output}`（`rollout.py:275` `tool_output_message`）；`references`（data/value/
  grounding）只写进 `ctx["history"][step]`，是**纯 harness 旁路**，唯一消费者是
  `backward_slice` / `validate` / reward。**故 B/C 的 references 改造不改变模型可见上下文，只改旁路。**
- **ctx 在线状态重构**（`new_ctx` 现为 `{memory, history, handle_to_step, memid_to_step}`）：
  - A：删 `memory` / `memid_to_step`；scalar resolver 改 history-based（按 `value_ref` 的 step 取标量）。
  - B：加 `col_lineage`（在线随每步维护）。
  - C：加 `described` / `inspected` / `read` 已观察索引。
- **在线 `value_ref` 严格校验（信任边界在 RL 真正落地，推理特有）**：离线 emitter 从 verified Plan
  生成不会非法；在线模型可能给非法 `value_ref`（不存在的 step / 多行多列的表 step）。resolve 时必须
  严格校验，非法抛错 → 主循环（`rollout.py:249`）捕获成 `execution_error` 反馈，**不得静默取值或崩溃**
  （等价于现 `ground_derived_value` 抛 `MemoryGroundingError` 的鲁棒性，迁移后保持）。
- **observation envelope 不变**：A 后 `tool_output_message` 格式不变，只是 turn 序列少了
  `add_to_memory` 这一步；`condition_filter` 的 observation 仍是表句柄。
- **本轮模型可见上下文形态几乎不变**：references 本就不进上下文；A 只让上下文少一个 add_to_memory
  turn。真正改变"上下文组织"的是 **Feature F（subtable 归并）**，已排但推后 —— 本轮不动模型看到的
  上下文布局。

---

## 2. 实现状态清单（哪些做了 / 哪些没做）

> 全部未开始。动工后把 `[ ]` 改为 `[x]`，并在行尾注日期 / commit。

### 阶段 A —— 取消 add_to_memory（改 SFT 文本，先做）
- [x] **A-1** `compiler.py`：`_scalar_subquery` / `_in_subquery` 不再 emit `add_to_memory`；`value_ref` = 产标量步 id（`last.id`）
- [x] **A-2** `plan.py`：删 `run_plan` 的 `add_to_memory` 分支；`resolve_cond` 按 step id 从 `values`（标量步）或 1×1 表取值
- [x] **A-3** `memory_semantics.py` → `scalar_grounding.py`：中性 API `ground_scalar_reference`；保留严格 `extract_scalar`
- [x] **A-4** `emitter.py`：删 `add_to_memory` emit；value reference 指向产标量步；`produces.kind="scalar"`
- [x] **A-5** `rollout.py`：删 `execute_tool` 的 `add_to_memory` 分支；在线 scalar resolver 改为 history-based（去掉 `memory` / `memid_to_step`）
- [x] **A-6** `protocol.py`：`TOOL_SPECS` / `_ARG_SCHEMA` 删 `add_to_memory`；改 `value_ref` 说明；**删 `supporting_memory_ids`**；bump `PROTOCOL_VERSION`
- [x] **A-7** `executor.py`：删旧 `add_to_memory()` 方法（Codex B2 补充）
- [x] **A-8** `tests/test_compiler.py`、`tests/test_verify.py`：更新期望（去掉 aggregate+add_to_memory+value_ref 链）（Codex B2）
- [x] **A-9** `tool_design/validate_trajectories.py`：删 / 改 `add_to_memory` 校验分支（Codex B2）
- [x] **A-10** `sft/enrich_traj.py`：删 `_remap_mem` / `mem_` 前缀重映射 / `supporting_memory_ids` 重映射（Codex B2）
- [x] **A-11** `sft/splice_think.py`：删旧 memory key 替换逻辑（Codex B2）
- [x] **A-12** `sft/build_sft_data.py`：渲染分支去 `add_to_memory`
- [x] **A-13** `rollout.py`：在线 `value_ref` 严格校验 —— 指向不存在 step / 非 1×1 源时抛错，经主循环（`:249`）成 `execution_error` 反馈（信任边界在 RL 落地，不静默、不崩溃）

### 阶段 B —— references 统一 schema（sidecar，随 A）
- [x] **B-1** 每条 reference 带 `type` + 结构化 `target`（§1.3）
- [x] **B-2** `data` 边迁移：旧 `{step/source, as}` → `{type:data, role:table, target}`
- [x] **B-3** `value` 边：`{type:value, step, role:value_ref, target:{table,column}}`
- [x] **B-4** `backward_slice` 参数化：默认 `reference_type="data"`（只走 data+value），grounding 不混入
- [x] **B-5** `emitter.validate()`：保持 data/value reference-integrity gate
- [x] **B-6** 抽共享 `src/harness/provenance.py::build_references`，`emitter` + `rollout` 共用（防 drift）；建立 `col_lineage`（join 前缀回溯基础设施，value 边单步剥前缀即用，grounding 跨表回溯随 C 复用）

### 阶段 C —— perception grounding（sidecar，不进 SFT 文本，RL 前做）
- [ ] **C-1** harness 在线维护"已观察 schema / literal / evidence"索引
- [ ] **C-2** 自动挂 grounding 边：schema（列→describe）、literal（字面量→inspect）、evidence（answer→read_subtable）
- [ ] **C-3** join 前缀列回溯源列，`target` 精确到源 `table.column`
- [ ] **C-4** grounding load-bearing 进 `emitter.validate()` 硬不变式（每字面量有 literal 边 / 每源列有 schema 边 / 每 answer 有 evidence 边）
- [ ] **C-5** `grounding_slice()` 或 `backward_slice(reference_type="all")` 供 reward
- [ ] **C-6** 过程奖励：奖励查证后行动；惩罚无依据列 / 字面量、装饰性 read

### 阶段 F —— subtable 按句柄归并（context management，推后，详见 §3）
- [ ] **F-1** 上下文渲染：同一句柄的多次 `read_subtable` 内容归并到一处，避免散落
- [ ] **F-2** 定义更新策略（替换 / 合并 / 保留最新）与 limit/columns 叠加规则
- [ ] **F-3** offline(SFT 渲染) 与 online(rollout) 归并逻辑一致，防 train/eval drift
- [ ] **F-4** 与 grounding 一致：evidence 边仍指向产生该内容的 `read_subtable` step（归并不改 step 语义）

### 阶段 D —— 验证与数据重生成
- [x] **D-1** 新 `schema_version`；旧 v1/v2/v2ctx 数据按 `schema_version` 分支隔离，旧 memory 逻辑不混入新协议
- [x] **D-2** 验证门全过：Spider dev replay 998/998；strict 每工具 schema 0 fail；`backward_slice(data)` 不变式；（C 完成后）grounding load-bearing；exec-verified ≈98.4%；unit；SFT build smoke；**rollout 在线 replay / live smoke（在线 `value_ref` 解析 + references 生成与离线一致、非法引用优雅 reject）**
- [ ] **D-3** 重新生成感知数据（A+B 完成即可）→ 感知 SFT
- [ ] **D-4** RL 前补 C：重跑 trajectory sidecar 生成（**不重训** SFT 模型）

---

## 3. 新 Feature 详述：subtable 按句柄归并

**来源**：maksymilan 给 Codex 的新建议（2026-06-23），登记于此以免遗忘。

**现状**：三个 observation（`describe_table` / `inspect_column` / `read_subtable`）都进上下文；
table-producing 步的 metadata-only 句柄进上下文；模型 `read_subtable` 时把该句柄的**部分行内容**
追加进上下文。由于 observation 按时间线性追加，**同一句柄的内容会散落在对话历史各处**（多次读同
一句柄、或不同句柄的 subtable 交错），上下文里"subtable 到处都是",分散、冗余、难利用。

**目标**：上下文管理时，把**同一句柄下读到的 subtable 归并到一起**（per-handle 视图区），模型看到
的是"按表组织的当前已读内容"，而非时间散落的碎片。

**待定的设计权衡（实现前要定）**：
- **线性历史 vs 可重组状态视图**：标准 messages 是线性追加；"归并"意味着 harness 主动重组上下文
  （把同句柄 subtable 收拢），改变 observation 渲染方式。需确认这与 chat 模板、loss mask 兼容。
- **更新策略**：同句柄被多次 `read_subtable`（不同 `limit` / `columns`）时，替换？并列合并？保留
  最新？列叠加？（F-2）
- **一致性纪律**：归并逻辑必须 offline（`build_sft_data` 渲染）与 online（`rollout`）**完全一致**，
  否则 train/eval drift —— 这是项目红线（F-3）。
- **与 grounding 的接口**：subtable 内容被归并后，`role=evidence` 的 grounding 边仍须指向**产生该
  内容的 `read_subtable` step**，归并只改"显示位置"不改"step 溯源语义"（F-4）。
- **与 V2-ctx 目标的关系**：这是上下文体量管理（避免大表碎片堆积），与"训练主动查表"不冲突 ——
  它优化的是"读到的东西如何在上下文里组织"，不是"是否要读"。

---

## 4. 参考
- 讨论与逐条代码核对：[`memory_provenance_review.md`](memory_provenance_review.md)
- V2a 现状：`AGENTS.md` Current-state、`draft/v2a_memory_report.md`
- 工具/协议：`src/sft/protocol.py`、`src/harness/{compiler,plan,emitter,rollout,scalar_grounding,provenance}.py`

---

## 5. 实现记录

### 阶段 A + B 完成（2026-06-23；协议 v2a→v2b，schema_version v2-ctx→v3）

**改动文件**：新增 `scalar_grounding.py`（替代删除的 `memory_semantics.py`）、`provenance.py`
（共享 `build_references` + 参数化 `backward_slice`）；改 `compiler.py`（标量子查询 value_ref 直指
产标量步）、`plan.py`（删 add_to_memory 分支，`resolve_cond` 双源取标量）、`emitter.py`（去 memory、
新 references schema、`produces.kind=scalar`、`handle_to_stepid`+`resolve_step`）、`rollout.py`
（去 memory ctx、在线 value_ref 严格校验、共享 `build_references`）、`protocol.py`（删 add_to_memory/
refine_memory、删 supporting_memory_ids、value_ref 说明改 step_id、bump v2b）；周边 `executor.py`/
`gen_trajectories.py`/`enrich_traj.py`/`splice_think.py`/`validate_trajectories.py`/`tests` 去 memory。

**验证门全过**：
- `run_all`：单元 98/98（test_compiler 15、test_emitter 18），compile 1998/2000 (99.9%)。
- emit 端到端：train 300 → 296 verified+legal (98.7%)，失败全是已知 ties/exotic。
- run_spider 1000：round_trip 982/1000 (98.2%)，无新增失败。
- **online/offline 一致**：`run_replay` 重放 296 条 v3 → 296 executed + 296 scored correct (100%)，
  含 5 条标量子查询(value 边)的在线 value_ref resolve。
- SFT build：296/296 渲染，manifest `protocol_hash` = v2b（95c58d18ea4cca28），训练文本 **0 memory 残留**。
- backward_slice：穿过 value 边正确（标量子查询 slice = {aggregate, filter, project}）。

**未做（推后，已在 §2 清单）**：阶段 C(grounding 在线生成 + validate 不变式 + reward)、
F(subtable 归并)、D-3/D-4(重新生成全量感知数据 + 感知 SFT)。

### Feature F 定位修正：实证触发（2026-06-23）
F(subtable 归并)与推理性能/KV cache **不作为研究层面的否决理由**。F 是否做、做 lite 还是 full，
由**第一版感知 SFT 的失败模式**经验决定（是否出现"上下文散落 → 证据混淆/重复读"）；当前线性形态
train/inference 已一致，本轮不带 F。
