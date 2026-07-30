# BIRD version24 + bounded search：精确输出字段 prompt-only Gate50

日期：2026-07-30

> 后续已按用户要求扩展为完整 200 题。最终结果与配对分析见
> `BIRD_VERSION24_SEARCH_OUTPUT_SLOTS_FIXED200_20260730_ZH.md`；完整 200 题结论
> （141/200）取代本报告仅基于前 50 题的判断。

## 结论

在 `version24 + bounded search_values` 上，仅增加“答案字段 / 辅助字段”判别规则和一对
top-1 输出对照示例，固定工具、执行器、状态、评分器与采样配置不变。

完整前 50 题从 **39/50（78%）提高到 41/50（82%）**：

| 配置 | bird-set | Legal | Process errors |
|---|---:|---:|---:|
| 新鲜原始 version24 | 41/50 | 50/50 | 1 |
| version24 + bounded search | 39/50 | 50/50 | 4 |
| version24 + search + 输出字段约束 | **41/50** | **50/50** | **1** |

相对 search-only，逐题为 38 题都对、8 题都错、3 gains、1 regression，净 +2，
双侧 exact McNemar `p=0.625`。因此这条约束确实修复了目标中的典型“保留 ranking
helper”错误，但样本太小、差异不显著，且只恢复到无 search 的新鲜 version24
baseline，没有形成新的总体增益。

相对新鲜原始 version24，逐题为 39 题都对、7 题都错、2 gains、2 regressions，
净变化 0，exact McNemar `p=1.0`。历史 version24 artifact 仍为 42/50。

## 唯一改动

代码分支：`codex/version24-search-output-slots`

实现提交：`cb159cc`

协议版本：`version24-search-values-output-slots-v1`

实际评测 manifest 的 protocol hash：`761946c155142c5d`

在 search-only prompt 上只增加：

1. 终答前将每个候选输出列判为“问题明确要求的答案字段”或“只用于 filter/join/
   group/rank/calculate 的辅助字段”，并移除所有辅助字段；
2. 一对对照例：
   - “Which X has the highest Y?” 只返回 X；
   - “Which X has the highest Y, and what is Y?” 返回 X 和 Y。

没有修改公开工具、参数、search executor、resident state、terminal lowering 或
`bird-set` verifier。provider system prompt 从 17,477 增至 18,358 字符，增加 881
字符。

## 预声明 Gate16

复用冻结的输出形状 Gate16：

`data/eval_inputs/bird_train_version41_output_correction_gate16_20260729.jsonl`

在当前 search-only 轨迹上，该集合包含：

- 5 个当前错误目标；
- 11 个当前正确控制；
- 16/16 合法；
- 0 process error。

运行前固定扩展门槛：

- 至少修复 2/5 当前错误；
- 至多回退 1/11 当前正确；
- 16/16 合法。

结果为 12/16，对 search-only 的 11/16 有 2 gains、1 regression，16/16 合法、
0 process error，刚好通过门槛，因此扩展到冻结前 50。

## Gate50 配对变化

### Gains

1. `bird_train_05316`
   - 问题：`Which chicken restaurant has the highest review?`
   - search-only：最终 top-1 保留 `label, review`，预测 `["petalumas", 3.3]`；
   - 新 prompt：`return_columns=["label"]`，预测 `["petalumas"]`；
   - 这是规则直接针对的修复，模型 reason 也明确说明问题只问 restaurant name，
     没有询问 review value。

2. `bird_train_02901`
   - 问题要求 10 个姓名的 full names，BIRD gold 保留
     `FirstName, MiddleName, LastName` 三个独立字段；
   - search-only 将三列拼为一个字符串；
   - 新 prompt 保留三个独立输出列；
   - 与“精确输出 slot、不要拼接字段”的目标一致。

3. `bird_train_03390`
   - search-only 返回国家 code；
   - 新 prompt 额外 join `country` 并返回国家 name；
   - 虽然结果符合精确输出语义，但它是额外关系选择变化，不是“删除辅助列”的直接
     机械结果，因此不能全部归因于新增规则。

### Regression

`bird_train_06489`

- 问题：`Find the object in image 5 where the object with the coordinate of (634, 468).`
- BIRD gold 返回 `OBJ_SAMPLE_ID = 18`；
- search-only 正确返回 18；
- 新 prompt 将“object”解释为对象类别，返回 `"paper"`；
- 该回退在 Gate16 和独立 Gate50 调用中都重复出现，说明新 prompt 会让模型在这类
  输出槽本身含糊的问题上更主动地选择“语义名称”，而 BIRD gold 可能要求内部 ID。

这说明 prompt 能强化输出选择，却不能确定性解决“自然语言答案槽究竟对应哪个数据库
字段”。后者仍需要外部知识映射、训练覆盖或更明确且不使用 gold 的任务侧证据。

## 成本

相对 search-only：

| 指标 | search-only | + 输出字段约束 | 变化 |
|---|---:|---:|---:|
| 总动作数 | 313 | 304 | -2.9% |
| 总 tokens | 1,727,622 | 1,749,785 | +1.3% |
| Process errors | 4 | 1 | -3 |
| 墙钟时间 | 283.38s | 300.63s | +6.1% |
| search 调用 | 16 | 13 | -3 |
| 使用 search 的题 | 13 | 12 | -1 |

prompt 变长使 token 总量略增；动作数和 process error 下降，但 K=1 下不能把这些变化
解释为确定性的 prompt 效应。

## 判断

1. **对目标错误有效。** `05316` 的多余 ranking 字段被精确修复，`02901` 的字段拼接
   也被纠正。
2. **没有证明总体涨点。** 41/50 只恢复到无 search 的新鲜 baseline，且相对
   search-only 的 `p=0.625`。
3. **存在明确副作用。** 对答案槽含糊的任务，模型可能从 benchmark 所需 ID 转向更自然
   的语义标签。
4. **不 promotion、不扩 200。** 保留为 prompt-only diagnostic；不能据此宣称
   `search_values + prompt` 优于 version24。
5. 如果继续处理输出形状，更可靠的方向是训练中覆盖“答案字段 vs helper 字段”成对
   轨迹，或让 terminal evidence 显式声明精确列并由 harness 校验；不要继续无上限地
   增长全局 prompt。

## Artifacts

- Gate16：
  `data/trajectories/version24_search_single_variable_20260730/version24_search_output_prompt_gate16/verified.all.jsonl`
- Gate16 manifest：
  `data/trajectories/version24_search_single_variable_20260730/version24_search_output_prompt_gate16/verified.manifest.json`
- Gate50：
  `data/trajectories/version24_search_single_variable_20260730/version24_search_output_prompt_gate50/verified.all.jsonl`
- Gate50 manifest：
  `data/trajectories/version24_search_single_variable_20260730/version24_search_output_prompt_gate50/verified.manifest.json`
- 对照 search-only：
  `data/trajectories/version24_search_single_variable_20260730/version24_plus_bounded_search/verified.all.jsonl`
- 对照新鲜 version24：
  `data/trajectories/version24_search_single_variable_20260730/version24_repro_network_retry/verified.all.jsonl`
