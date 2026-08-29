# Qwen3-8B v26 `describe_table` 定向超集匹配审计

日期：2026-08-28

## 规则

在同一前置状态下，正确轨迹的 `describe_table.tables` 为 (T_c)，错误轨迹对应动作为
(T_w)。若

\[
T_w \supseteq T_c,
\]

则把这两个 describe 轮次视为匹配，并纳入 state-action 歧义屏蔽。其它工具仍要求 parsed
`tool + complete arguments` 严格相同。

该规则是有方向的包含关系，不是等价关系：

- 错误轨迹多 describe 表：接受；
- 错误轨迹少 describe 表：拒绝；
- 两侧各有独占表：拒绝。

Gold SQL、Gold answer 和模型 reasoning 均未参与匹配。

## Boundary193/K8 主结果

输入是既有 BIRD-train policy-boundary clean-mixed K8 池：193 题、1,544 条原始轨迹；严格重建后
1,516 条可用轨迹、11,645 个决策事件。没有使用 BIRD-dev 1534 训练。

### Describe 包含关系

| 正确动作与错误动作的关系 | 唯一 state-action 对 | event pair | 涉及题 |
|---|---:|---:|---:|
| tables 完全相等 | 198 | 1,119 | 174 |
| 错误动作是严格超集：接受 | 141 | 370 | 89 |
| 错误动作是严格子集：拒绝 | 140 | 328 | 89 |
| 两者不可比较：拒绝 | 73 | 121 | 39 |

141 个严格超集动作对增加的表数：

| 错误动作额外 describe 表数 | 动作对 |
|---:|---:|
| +1 | 106 |
| +2 | 23 |
| +3 | 10 |
| +4 | 2 |

严格超集和严格子集数量为 141 对 140，几乎对称。因此当前数据不能证明“多 describe 表”天然更好或
天然无害；该规则表达的是信息覆盖充分假设，而不是从 outcome 数据学出的因果结论。

### 对 SAAM 屏蔽覆盖的影响

| 指标 | 完整动作严格匹配 | describe 超集放宽 | 增量 |
|---|---:|---:|---:|
| 被屏蔽事件 | 1,525 | 1,747 | +222 |
| 占全部决策事件 | 13.10% | 15.00% | +1.91 pp |
| 初始事件 | 1,059 | 1,274 | +215 |
| 非初始事件 | 466 | 473 | **+7** |
| 涉及题 | 174 | 183 | +9；新增事件分布于 79 题 |
| response tokens | 295,440 | 342,840 | +47,400 |
| response-token 占比 | 11.83% | 13.72% | +1.90 pp |
| 绝对 policy coefficient mass | 17.27% | 19.34% | +2.07 pp |

“新增事件分布于 79 题”与“总涉及题净增 9”并不冲突：多数新增超集匹配发生在本来已经有其它严格
歧义动作的题中。

主要变化集中在第一轮。222 个新增事件中 215 个在 depth 0，只有 7 个是非初始状态。因此该规则
明显改善 opening describe 的对齐覆盖，但几乎没有解决深层工具轮次的 credit 稀疏问题。

### 轨迹是否仍会训练

|  | 严格匹配下至少屏蔽一步 | 放宽后至少屏蔽一步 | 总轨迹 |
|---|---:|---:|---:|
| 正确轨迹 | 692 | 851 | 1,015 |
| 错误轨迹 | 367 | 423 | 494 |

放宽后所有 1,015 条正确轨迹和 494 条错误轨迹仍至少保留一个未屏蔽轮次，没有整条轨迹被删除。

## 两个 representative 初始 batch

为了检查 boundary outcome selection 是否放大增量，又在两个既有 initial-policy、30题×K8 独立引擎
运行上应用相同规则：

|  | Run A | Run B |
|---|---:|---:|
| 严格匹配屏蔽事件 | 129 / 1,596 = 8.08% | 80 / 1,554 = 5.15% |
| 放宽后屏蔽事件 | 131 / 1,596 = 8.21% | 85 / 1,554 = 5.47% |
| 新增事件 | +2 | +5 |
| 新增题 | 2 | 3 |
| 新增非初始事件 | 0 | 0 |

两个独立 batch 都证明该关系真实存在，但增量很小，并且全部发生在初始 describe。这与
Boundary193 的主结论一致：放宽规则主要是第一步对齐规则，不是深层 credit assignment 的解决方案。

## 研究判断

建议把该规则保留为 SAAM-GRPO 的一个可控 ablation，但不要直接当作主定义：

1. **合理处**：describe 更多表确实包含正确轨迹已经获得的 schema 信息，不应仅因多一个无关表就
   必然视为完全不同的信息动作。
2. **风险处**：额外 schema 会改变 observation 长度、模型上下文和后续注意力，也可能带来错误表、
   成本和干扰；超集不是严格语义等价。
3. **实证限制**：它只新增 7 个非初始事件，不能提升深层局部 Q 的统计支持。
4. **推荐比较**：如果进入训练，至少区分 `strict-action SAAM` 与
   `describe-superset SAAM`；不能看到结果后再选择匹配规则。

## 产物

- 审计：`src/rl/diagnostics/audit_describe_superset_matching.py`
- 测试：`src/rl/diagnostics/test_audit_describe_superset_matching.py`
- Boundary JSON：
  `data/results/qwen3_v26_boundary193_k8_state_prefix_20260828/describe_superset_audit.json`
- 两个初始 batch JSON：
  `data/results/qwen3_v26_initial30_rerun_state_prefix_20260828/*.describe_superset_audit.json`

相关诊断测试合计 8/8 通过。
