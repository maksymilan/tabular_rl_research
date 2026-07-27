# BIRD action-block v23--v26 终止检查与 observation 引用消融

日期：2026-07-26  
模型：DeepSeek v4 Flash  
指标：`bird-set`  
固定诊断集：12 题

## 结论

终止前输出形态检查不适合继续增加到学生 prompt。长版 v23 和精简版 v24 都恢复了一个
空关系应聚合为标量 0 的案例，但在原本正确的输出题上造成更多回归；只保留
observation-only 分类提示的 v25 仍未恢复到 v22 水平。

最终 `action-block-v26`：

- 恢复与 v22 字节完全一致的模型 prompt；
- 保留 action-block adapter 的精确 observation-only 引用错误反馈；
- 不静默把 `read_subtable` 引用改写成其输入表；
- 不修改 atomic prompt、原子工具、合法调用执行、DAG 调度或终止 grounding；
- 不获得 fixed-200 扩展资格，也不能作为 SFT 来源。

## 冻结 12 题

诊断集包含：

- observation-only 误引用：`06151, 04377`；
- v22 仍失败的终止形态/答案槽：`02918, 06489`；
- v22 已正确的输出对照：
  `00454, 00833, 01148, 01290, 02734, 03373, 04290, 05053`。

v22 的 12 题参考结果来自既有固定 40 和条件化 29 题轨迹，不是本轮重新采样。不同运行
仍受 provider 非确定性影响，因此这里只用于否决明显有害的 prompt 变体。

| 版本 | 变化 | 正确 | 合法 | 过程错误 | 模型轮次 | 原子动作 |
|---|---|---:|---:|---:|---:|---:|
| v22 reference | 语义决策边界 | **10/12** | 12/12 | 7 | **43** | 91 |
| v23 | 长版 terminal preflight + observation 提示 | 7/12 | 12/12 | 5 | 59 | 107 |
| v24 | 精简 terminal evidence check + observation 提示 | 8/12 | 12/12 | **4** | 53 | 92 |
| v25 | 只保留 observation 分类提示 | 8/12 | 12/12 | 6 | 51 | 95 |

### 长版 v23

v23 恢复 `02918`：模型不再把空实体关系当最终答案，而是构造一行 count=0。
但它回归：

`00454, 00833, 03373, 05053`。

`06489` 仍选择 object class `"paper"`，没有遵循外部知识要求的
`OBJ_SAMPLE_ID`。因此长清单增加了推理和重规划，却没有稳定修复答案槽映射。

### 精简 v24

v24 仍恢复 `02918`，并保留 `00833`，但继续回归：

`00454, 03373, 05053`。

相同三题在 v23/v24 的连续回归说明，终止检查提示虽然能改变轨迹，却没有形成可靠的
正向选择压力。

### observation-only v25

移除所有新增终止检查后，v25 为 8/12。它恢复 `03373`，但 `00454` 和 `05053`
继续回归，`02918` 也回到错误。新增 observation 分类提示只使两条目标轨迹
`06151/04377` 正确且无错误，整体过程错误仅从参考的 7 降到 6，不足以抵偿准确率变化。

## v26 最终边界

v26 删除新增的 observation 分类 prompt，模型可见 prompt 恢复到 v22：

- 字符数：9,403；
- SHA-256：
  `401a2de1f4573f16f6d214ad5418a6ef414e012b612b3823ae9424c9b6490ca1`；
- 与 v22 manifest 中的 system prompt 逐字相等。

adapter 在本地绑定中记录 observation tool 和其输入引用。当模型把 `$read` 用作 table
或 terminal evidence 时，错误明确报告：

```text
local reference '$read' targets observation-only read_subtable;
observation-only calls do not produce reusable tables or scalar values.
The observed input was '$projected'; if that relation is the intended result,
cite the producer relation instead of the observation call
```

该反馈只在模型已经提交非法引用时出现。它不改变合法轨迹，也不自动把 `$read` 解析成
`$projected`。自动改写不安全，因为 `read_subtable` 可能带 columns/limit，模型可能错误
地把观察窗口理解为派生关系；静默替换会改变其声明语义。

## 决策

1. 保留 v22 的语义决策边界 prompt。
2. 保留 v26 的事实型 observation 引用反馈。
3. 拒绝 v23/v24 的终止前检查 prompt。
4. 拒绝 v25 的 observation 分类 prompt。
5. 输出形态问题留给因果 SFT 轨迹学习；不继续增加学生运行时 prompt。
6. 不启动 v26 fixed-200。

## 验证与产物

- action-block/SFT/tool-scheme tests：51/51；
- eval tests：28/28；
- v23：
  `data/trajectories/batch_plan_20260726/action_block_v23_terminal_preflight_gate12_r1.all.jsonl`
- v24：
  `data/trajectories/batch_plan_20260726/action_block_v24_concise_evidence_gate12_r1.all.jsonl`
- v25：
  `data/trajectories/batch_plan_20260726/action_block_v25_observation_handle_gate12_r1.all.jsonl`
