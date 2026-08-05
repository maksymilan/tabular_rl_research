# 当前 grounding extractor 变更审计（Codex，2026-07-30）

## 结论

用户指定由 Codex 直接复核，不再把轨迹包发送给外部 DeepSeek API。本审计与
`grounding-review-v4-visible-literal-copy` 使用同一判断边界，但明确记录 reviewer
为 `codex-primary-structured-audit`，不伪称 Flash/Pro 双审。

- 当前 provenance 代码重建了全部 703 个 review package。
- 去掉只负责精确定位的 `argument_path`、cell index 和 `replay_binding` 元数据后，
  只有 8/703 个 package 存在语义变化。
- 冻结审计集包含全部 8 个变化 package、6 个未变化 row-edge 控制和 6 个未变化
  schema-edge 控制。
- 20 条轨迹共 63 条 grounding edge：63 valid、0 invalid、0 missing。
- 20 条 terminal dependency：19 valid、1 invalid。
- 当前 extractor 的变化 edge-precision 门禁通过；Process-RL 总门禁仍未通过。

## 冻结输入

- 全 703 包：
  `data/results/process_gate_v2_20260730/current_provenance/packages703_current_provenance.jsonl`
- 全 703 包 SHA-256：
  `e639da0f4b52e9b615790bdeb905c712a05e6724ed604e6d848f21957f715719`
- 20 条选择清单 SHA-256：
  `d2c0441727f3574fd7c14cbac59c456df8c8055d35c732cc7d5e15e4032314c3`
- 实际审计 package SHA-256：
  `2bb279633c3919a6ba8272cb9eacfbe8f3c238dee5cb86220c31b56caf635f2a`
- Codex 逐 edge 结果 SHA-256：
  `7950903d34b30c9e4b9648a2ed0ef7a7c9858e9985b0ae04f9c3b383d92f2a17`

## 审计方法

每条 schema edge 必须满足：

1. source 是实际 `describe_table` 输出；
2. target table 在该输出中；
3. target columns 是实际观察列的子集；
4. source step、target step 和 target action 完整保留。

每条 row edge 必须满足：

1. 被复制值确实出现在 source harness output；
2. `argument_path` 在 target arguments 中精确解析为同一值；
3. 若存在 `replay_binding`，其 row/column index 必须回指同一可见 cell；
4. source/target column 映射及匹配类型完整保留。

上述条件由
`src/rl/materialize_codex_grounding_change_audit.py`
逐边断言；最终 relation 是否完整执行问题约束则由 Codex 逐轨迹复核。模型 authored
reason 不作为事实证据。

## 变化 package

变化的 8 条为：

- `00739`
- `03877`
- `04109`
- `04696`
- `05741`
- `05873`
- `05875`
- `06577`

其中新增的 visible-cell copy 均能从 source output 的精确 cell 定位到 target literal；
`04696` 的变化只是同一个 `wid=1` 被两个 argument path 使用，新增的第二个 binding
同样有效。

## 新确认的 terminal shortcut

`bird_train_04696` 数值上回答正确，但最终步骤为：

```text
project(top_003, ["'àbac-xinès' AS pair"])
```

`top_003` 确实按 `occurrences DESC` 选出了一个 `(w1st, w2nd)`，但 terminal branch
没有把选中的两个 ID 映射回词文本，而是始终投影常量 `àbac-xinès`。若在保持 schema
不变的反事实数据库中交换两组 occurrences，gold 会变成 `àbac-grec`，该轨迹仍会输出
`àbac-xinès`。

因此：

- 三条 grounding edge 均 valid；
- terminal dependency invalid；
- 它属于新的“terminal constant-output shortcut”类别。

历史 v4 Pro review 对该点也给出 invalid；历史 Flash review 给出 valid。本次 Codex
审计独立重读结构化调用后采纳 invalid 结论。

## 门禁决定

`changed_extractor_edge_precision_passed = true`。

`process_reward_ready = false`，原因有二：

1. 当前 23 个正式 RL task 尚无完整、content-hashed、quality-gate-passed 的
   `process-counterfactual-suite-v2`；
2. shortcut 回归套件必须从原来的三类扩为四类，新增 `04696` 的 terminal
   constant-output 类。

不得据此启动 Exp3–Exp6，也不得退回 `denotation-nonempty` 来绕过门禁。
