# BIRD relational-program v6 base-role lowering 4 题门控

日期：2026-07-27

模型：DeepSeek v4 Flash，thinking enabled，reasoning effort high，temperature 0

指标：`bird-set`

数据准入：`diagnostic_only_pending_protocol_scale_gate`

## 唯一改动与门控

v6 不改变公开 action shape。它只修复一个确定性 compiler defect：

- join 的 base 为当前 program node；
- 模型同时声明 `base_role`；
- `on.left` 通过 typed `node+column` 指向该 base 的 bare column；
- compiler 必须把该列降到 `base_role.column`，而不是运行时生成的
  `handle.column`。

这一 lowering 记录在 `work_graph.compiler_lowerings`。真实 SQLite 回归测试覆盖了
rank→join 路径并确认两个原子节点均执行成功。

冻结门：

- 目标：`05440`；
- v5 保留控制：`03275`、`02437`、`04619`；
- 通过要求：目标恢复且控制 3/3 保留。

## 结果

| 指标 | v6 gate4 |
|---|---:|
| 正确 | 3/4 |
| 合法结束 | 4/4 |
| 目标恢复 | 0/1 |
| 控制保留 | 3/3 |
| 模型轮次 | 24 |
| 原子动作 | 32 |
| process error | 2 |
| blocked | 0 |
| 总 token | 110,579 |
| 耗时 | 77.3s |

门控失败，不扩到 20/50/200。

## 目标轨迹审计

目标最终有 11 个成功原子动作、0 个执行失败、0 blocked。模型先用
aggregate→typed `value_from` filter→join→select 得到最大年份的候选集合，随后观察并
改为 left join，再提交合法终端证据。最终仍为 `wrong_answer`。

该实际轨迹没有声明 `base_role`，因此没有触发 v6 的新 lowering；它不能提供模型级
修复收益证据。v6 的 compiler bug 仅由确定性单元/集成测试确认。该题的剩余失败是：

- 最大年份对应多行，模型没有稳定处理题目中的单数/并列语义；
- 终端 reasoning 从多行证据中口头选一行，但 harness 正确地只评分完整证据表；
- 所有执行节点成功，继续改参数 schema 不会自动修复这种选择。

两个 protocol error 均在后续恢复前发生：一次把 named column 加到 predicate
`value_from`，一次发出不存在的 observe operation。它们没有成为训练目标，也没有
造成基础设施失败。

## 结论

v6 是合理的工程修复，但没有通过能力门。连续结果为：

- v4：过程错误/成本改善，但 10/20 低于 v3 的 12/20；
- v5：scalar 微型门 0/2 目标恢复、1 个控制回退；
- v6：确定性 join lowering 修复，目标仍未恢复。

应停止继续堆叠 relational-program prompt/schema。当前工具方向的有效结论限于：

1. 模型声明 typed parameter dependencies、harness 自动建 DAG 可行；
2. primitive node 仍可作为执行、provenance 和未来 process reward 单位；
3. typed refs 能降低一部分协议错误和成本；
4. 终局差距主要是 population、grain、tie、标签选择和最终列语义，需要因果监督或
   reward 学习，而不是更多接口别名。

v6 不进入 SFT/RL，也不启动更大评测。

## 产物

- 轨迹：
  `data/results/relational_program_v6_base_role_gate4_deepseek_v4_flash_20260727/all.jsonl`
- manifest：
  `data/results/relational_program_v6_base_role_gate4_deepseek_v4_flash_20260727/all.manifest.json`

所有产物均为 diagnostic-only，未进入训练数据。
