# 带文本的表格：文本作为"子表的孪生"（draft）

> 草稿 / 讨论记录，非定稿。日期 2026-06-06。
> 决定：覆盖 table+text 多跳题（FinQA / TAT-QA / MultiHiertt）。带实际信息的文本作为"子表的孪生"放进感知层、可被引用；长文本（开放域）不考虑。
> 约定：纯文本，不用 LaTeX。

---

## 0. 决定一句话

- 带实际信息的随附文本 = 子表的孪生（perception 层，ground truth，可被答案/ memory 引用）。
- 只接"表 + 短的、随表给定的、有界文本"。开放域链接段落（HybridQA / OTT-QA）不做。

---

## 1. 范围

- 接（in）：FinQA / TAT-QA / MultiHiertt —— 表 + 几段随表给定的财报上下文，文本有界。
- 不接（out）：HybridQA / OTT-QA —— 表 + 大量/长的链接维基段落，本质是开放域检索，会冲淡"表格推理"焦点。

判定：文本是"随表一起给定、有界"的就接；需要从语料库里检索的不接。

---

## 2. 为什么进感知层而不是 memory

原则（见 subtable_vs_memory.md）：
```
子表 = 逐字 ground-truth 单元格（perception 层）
memory = 模型写的、必须指回证据的结论（cognition 层）
```
随附文本是逐字 ground truth（和单元格一样是源数据），不是模型解释。若灌进 memory：
1. 重新打破 perception/cognition 分层（memory 又混入原始源数据）；
2. 引用变递归（"从文本得出的结论"要引用同在 memory 里的文本条目，别扭）。

对称的干净设计：
```
表格 ground truth → dynamic_table_context（感知）
文本 ground truth → document_context（感知）        ← 孪生，新增
对两者的结论       → memory（认知），各自引用对应的感知 store
```

---

## 3. 一条分界线：内容证据 vs 元数据

不是所有文本都进感知层。类比 table_meta（是"关于表的元数据"，不是表的数据）：

| 文本类型 | 例子 | 归属 |
|---|---|---|
| 一句话标题/标注/单位 | FeTaQA page/section title、"单位：千元" | 可进 memory 作 harness 元数据（像 table_meta） |
| 承载答案证据的内容段落 | FinQA 里含具体数字的报告文字 | 必须进 document_context 作可引用 ground truth |

判定：这段文本是"答案要引用的证据"吗？是 → 感知层可引用；只是标签/标题 → memory 元数据可以。

---

## 4. 实现时要做的 spec 改动（实现前 edit-spec-first）

在 tool_io_spec.json：
- state 加 `document_context`（文本版子表）：字段建议 `{ passages }`，每个 passage = `{ passage_id, text }`（可选 `source`）。短文本 init 时全给；不需要检索。
- `initial_state` 把 `document_context` 设为**可选**字段（无文本的数据集不带它；现有 7 条轨迹不受影响）。
- `memory_support` 增加 `passages`（必要时再加 `text_spans` = passage_id + 字符区间）。
- `answer_evidence` 增加 `passages`，让最终答案能引用文本片段。
- （可选）若文本稍长但仍有界，加工具 `retrieve_text_context(semantic_match / keyword)`，把相关段落拉进文本缓冲（与 retrieve_row_context 对称）；纯短文本则"全展示 + 引 span"，无需该工具。

在 validator：
- 允许 initial_state 含可选 document_context，并校验其结构；
- 文本引用完整性：被引用的 passage_id 必须存在（与 validate_qualified_column 对称）；
- 允许 memory_support / answer_evidence 出现 passages。

在 tool_io_spec.md（字段字典）：
- 标注 document_context = HARNESS 提供的 ground truth（和表格一样），passages 引用 = 证据引用的一种。

注：现有轨迹都不带文本，以上改动是加法、安全；但要保持 spec + validator + 字段字典三处同步（单一来源纪律）。

---

## 5. 对核心论点的好处

文本做成可引用的 ground-truth 证据后：
- provenance 后向切片可纳入文本 span；
- 忠实度奖励扩展为"引用的文本片段是否真支撑答案"；
- 得到一个干净卖点：**忠实的 table+text 多跳推理**。
文本若埋进 memory，这条扩展就没了。

---

## 6. 待办

- 实现时机：建第一条 FinQA/TAT-QA 轨迹之前，按 §4 先改 spec（spec-first）。
- 决定 text_spans 粒度（整段 passage 引用 vs 字符级 span）——粒度即旋钮，与 process_reward_density.md §5.4 一致。
