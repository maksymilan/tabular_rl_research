# Process Reward v1：实现、测试与实际问题记录

> Historical report. Its model-declared final-evidence dependency was replaced by the
> harness-inferred v2 design in `PROCESS_REWARD_V2_REPORT.md`. Keep this file for comparison; do not
> use its readiness conclusion as the current state.

日期：2026-07-15  
状态：奖励计算与策略损失接口已实现；离线审计通过数学不变量，但
`process_reward_ready=false`，当前不得启动 process-shaped RL。

## 1. 目标与范围

本轮实现对应以下逐步奖励：

\[
g_{i,t}=w_BB_{i,t}+w_EE_{i,t}+w_SS_{i,t}+w_FF_{i,t}
\]

\[
c^+_{i,t}=\frac{g_{i,t}}{\sum_k g_{i,k}},\qquad
P_i=\min\left(P_{\max},\sum_t p_{i,t}\right),\qquad
c^-_{i,t}=\frac{p_{i,t}}{\sum_kp_{i,k}}
\]

\[
r_{i,t}=C_i c^+_{i,t}-P_i c^-_{i,t},\qquad
\sum_t r_{i,t}=C_i-P_i
\]

实现目标不是立即训练，而是先回答两个问题：

1. 这些特征能否只依赖 harness 事实稳定计算；
2. 在当前 BIRD SFT-1 教师轨迹上，奖励是否真的形成合理的逐步信用，而不是换一种形式的终局奖励。

本轮只用执行正确的 BIRD train 教师轨迹做离线奖励审计。失败轨迹的负奖励代数已有单元测试，
但尚未用完整真实失败轨迹做同规模经验审计。

## 2. 实现文件

- `src/rl/process_reward.py`
  - 轨迹 harness 重放；
  - `B/E/S/F` 特征提取；
  - 惩罚事件提取；
  - 正负信用归一化；
  - 奖励守恒和正确轨迹正总量断言。
- `src/rl/process_objective.py`
  - 实现逐步加权策略目标；
  - `beta > 0` 时强制要求调用方提供来自冻结 SFT-2 reference 的逐步 KL；
  - 不允许把当前 rollout policy 或 base model 静默当作 `pi_SFT-2`。
- `src/rl/build_process_reward_report.py`
  - 对 JSONL 轨迹逐条重放；
  - 输出完整逐步奖励和聚合报告；
  - 计算 process-RL readiness gate。
- `src/rl/configs/process_reward_v1.json`
  - 当前 pilot 权重；
  - 这些数值是显式审计默认值，不是调优结论。
- `src/rl/tests/test_process_reward.py`
  - 奖励代数、封顶、反馈和搜索缩减测试。

## 3. Harness-grounded 特征口径

### 3.1 为什么必须重放

当前外部教师成功轨迹保存了 `tool_call`、`tool_output` 和前后环境状态，但没有直接保存
`references` / `produces`。因此不能从模型的 `<think>` 文本推断依赖，也不能直接假设相邻步骤存在
因果关系。

离线计算会重新打开对应 SQLite 数据库，通过与在线环境相同的 `execute_tool` 顺序执行每个合法动作。
重放过程中从 `ctx.history` 获取 harness 生成的 data/value references，并重新计算：

- 表 handle 与 producing step 的关系；
- 输入、输出和根表行数；
- 语义环境状态是否变化；
- 空结果；
- 错误事件与下一合法动作的时序；
- 最终答案实际采用的评分分支。

错误动作不重新执行。它们根据轨迹中的 `error_events` 插回原始 action index，并保留错误类型和
状态 hash。这样逐步奖励仍覆盖完整 action timeline，而不是只覆盖 SFT target。

### 3.2 BackSlice：`B`

最终答案只有在以下情况之一成立时才建立终局数据边：

1. `answer_from_context.evidence` 指向的表与 gold denotation 严格相同；
2. 显式答案正确，并且存在一个已生成 handle，其完整 denotation 与 gold 严格相同；
3. 声明的 evidence 仅有列顺序差异，但其允许的完整列排列与 gold 相同。

终局边建立后，调用共享 `provenance.backward_slice` 沿 data/value edge 反向遍历。
位于该依赖链中的 producing step 令 `B=1`。

以下情况不猜测依赖：

- 显式标量答案只是出现在某次 read 的文本里；
- 模型在 `<think>` 中说“根据 step_x”；
- 某个中间表的 `row_count` 偶然等于答案；
- evidence 表比 gold 更宽，而在线评分实际采用了显式答案分支。

### 3.3 新证据：`E`

当前 v1 采用保守定义：最终实际使用的 evidence handle 是 `U_i` 中的证据单元；该 handle 第一次由
harness 创建时令 `E=1`。失败轨迹强制 `U_i` 为空。

这使 `E` 可验证，但也产生一个已知限制：当前 `E` 通常是 `B` 的子集，因而会额外强调最终 evidence
producer，却尚不能独立奖励真正改变后续决策的 `inspect_column` / `read_subtable` 感知步骤。要解决该
问题，需要后续加入 harness-authored perception grounding edge。

### 3.4 搜索空间缩减：`S`

当前只对具有单一可比较行集来源的以下工具启用：

- `condition_filter`
- `extreme_value_select`

lineage-preserving 的 filter、top-k、project、derive 和 window handle 会继承同一个固定根表及
`n_root`。只有同时满足以下条件时才计算正值：

- 当前步骤在 BackSlice 中；
- `n_root > 0`；
- `0 < n_out <= n_in`；
- 输入具有唯一可比较根表。

空结果、扩张结果、join/group 等不可比较操作均为零。实现调用唯一函数
`normalized_search_reduction`，单元测试直接验证：

\[
S(100\rightarrow10)+S(10\rightarrow2)=S(100\rightarrow2)
\]

因此连续拆分过滤不能增加总搜索缩减分数。

### 3.5 反馈响应：`F`

完整 action timeline 中，当前调用合法且执行成功时：

- 上一步是 protocol / argument-validation / execution error：`F=1`；
- 上一步是空结果：仅当规范化动作改变且语义环境状态也改变时 `F=1`；
- 其他情况：`F=0`。

动作变化使用规范化 `{tool, arguments}` hash；状态变化使用 harness resident environment 的语义
snapshot digest。错误动作缺失原始规范化参数时不会猜测其 action signature。

### 3.6 惩罚

当前默认配置：

| 事件 | 默认 lambda |
|---|---:|
| 终局失败 | 1.00 |
| 工具/协议/参数错误 | 0.25 |
| 无反馈的重复调用 | 0.15 |
| 合法但无语义状态变化 | 0.05 |
| 忽略错误或空结果反馈 | 0.20 |

`P_max=0.8`。终局 `answer_from_context` 不因“环境状态不变”被判为 no-state-change。

连续错误会同时产生工具错误惩罚；若第二次错误发生在反馈之后，还会产生 ignored-feedback 惩罚。
空结果后原样重复调用不获得 `F`，并进入 ignored-feedback。

### 3.7 奖励归一化和策略损失

正向信用采用线性归一化，因此 `g=0` 的步骤保持 `c+=0`。若整条轨迹 `sum(g)=0`，仅终步获得
正向兜底信用。惩罚按未封顶的 `sum(p)` 分配位置，再用封顶后的 `P_i` 缩放总量。

代码对每条轨迹强制检查：

- `sum(r) == C - P`；
- 正确轨迹 `sum(r) > 0`；
- 所有权重非负；
- `0 < P_max < 1`。

策略目标接口为：

\[
-\frac{1}{M}\sum_{i,t}r_{i,t}\log\pi_\theta(a_{i,t}|s_{i,t})
+\frac{\beta}{M}\sum_{i,t}\mathrm{KL}_{i,t}
\]

KL 必须由冻结的 SFT-2 reference 计算。当前尚未把 process objective 接入正在使用的训练 runner，
因为 readiness gate 未通过。

## 4. 测试方法

### 4.1 单元测试

运行：

```bash
.venv/bin/python -m unittest src.rl.tests.test_process_reward
```

当前 9 个测试覆盖：

1. 线性正信用归一化，零贡献步骤保持零；
2. 无正向特征时只有终步获得兜底；
3. `P_max` 封顶后正确轨迹仍为正；
4. 失败轨迹总奖励为负；
5. 固定根表搜索缩减的望远镜性质；
6. 空结果和扩张结果的 `S=0`；
7. 错误惩罚与恢复信用同时存在且位置独立；
8. 逐步策略损失按 `r_t` 加权；
9. `beta>0` 时缺失冻结 reference KL 会被拒绝。

结果：`9/9 passed`。

### 4.2 40-episode 小训练切片

输入：

```text
data/trajectories/bird_ds_flash_v4_scale500_rolling4_full_train40.jsonl
```

运行：

```bash
.venv/bin/python src/rl/build_process_reward_report.py \
  --input data/trajectories/bird_ds_flash_v4_scale500_rolling4_full_train40.jsonl \
  --output-dir data/rl/bird_scale500_train40_process_reward_v1 \
  --config-json src/rl/configs/process_reward_v1.json \
  --quiet
```

主要结果：

| 指标 | 结果 |
|---|---:|
| 轨迹数 | 40 |
| replay correct | 40/40 |
| 严格 grounding | 18/40 = 45.0% |
| 显式答案 ungrounded | 22 |
| terminal fallback | 14 |
| ungrounded non-fallback | 8 |
| reward min / mean / max | 0.30 / 0.89625 / 1.00 |
| process_reward_ready | false |

### 4.3 全部 183 条执行正确轨迹

输入：

```text
data/trajectories/bird_ds_flash_v4_scale500_rolling4_full_success.jsonl
```

输出：

```text
data/rl/bird_scale500_success183_process_reward_v1/scored_trajectories.jsonl
data/rl/bird_scale500_success183_process_reward_v1/summary.json
```

主要结果：

| 指标 | 结果 |
|---|---:|
| 轨迹数 | 183 |
| replay correct | 183/183 |
| declared evidence | 50 |
| explicit answer matched handle | 29 |
| explicit answer ungrounded | 103 |
| too large to ground | 1 |
| 严格 grounding rate | 79/183 = 43.2% |
| terminal fallback | 69 |
| ungrounded non-fallback | 35 |
| reward min / mean / p50 / max | 0.30 / 0.8828 / 1.00 / 1.00 |
| process_reward_ready | false |

特征命中：

| 特征 | 命中步骤数 |
|---|---:|
| BackSlice | 216 |
| new used evidence | 79 |
| search reduction | 83 |
| feedback response | 84 |
| error before current action | 81 |
| empty result | 10 |
| changed action after empty | 9 |
| tool error | 81 |
| ignored feedback | 6 |
| no-feedback exact repeat | 0 |
| legal no-state-change | 0 |

84 个 feedback credit 由 76 个错误恢复和 8 个空结果恢复组成。

## 5. 典型逐步奖励

### 5.1 Grounded clean success：`bird_train_00584`

最终 evidence 指向 project handle，BackSlice 为 filter -> join -> project：

| 步骤 | 主要特征 | 最终奖励 |
|---|---|---:|
| condition_filter | `B=1`, `S=0.5364` | 0.3387 |
| join_tables | `B=1` | 0.2204 |
| project | `B=1`, `E=1` | 0.4409 |

总奖励为 1.0。该分配符合预期：有效过滤得到额外搜索缩减信用，最终 evidence producer 因新证据得到
额外权重。

### 5.2 Ungrounded clean success：`bird_train_01915`

显式答案正确，但 harness 无法将答案严格绑定到某个中间 handle。所有 `B/E/S/F` 为零，终步 fallback
得到 1.0。这条轨迹实际上退化为 result-only reward。

### 5.3 Ungrounded recovered success：`bird_train_05475`

| 步骤 | 事件 | 奖励 |
|---|---|---:|
| step_4 | protocol error | -0.25 |
| step_5 | 合法 answer，`F=1` | +1.00 |

总奖励为 0.75。代数正确，但由于终局未 grounding，全部正信用被一个反馈响应步骤占据。这是当前
35 条 ungrounded non-fallback 问题的代表。

### 5.4 多错误后的 grounded success：`bird_train_00395`

两次错误合计惩罚 0.70；后续 group/filter/group 依赖链分享正信用，过滤步骤同时获得 `S`，最终 group
获得 `E`。最终总奖励为 0.30，满足正确轨迹保持正值的约束。

## 6. 实际遇到的问题

### 6.1 成功轨迹缺失持久化 provenance

外部教师 JSONL 没有直接保存 `references/produces`，必须离线重放才能计算 BackSlice。这不会造成
语义错误，但会增加 I/O 和计算成本，也说明在线 RL 应直接从环境 transition 保存这些字段。

### 6.2 最终答案 grounding 覆盖不足

183 条中只有 79 条能严格定位最终 evidence producer。大量教师轨迹通过显式 `answer` 返回标量或短
列表，而没有引用一个可验证的最终 handle。即使答案来自先前的 read，harness 也无法证明模型实际
使用了哪一个 observation。

不能用 `<think>` 文本、值字符串匹配或“最后一次 read”规则补边，否则会把模型自述变成奖励事实。

### 6.3 正信用过度集中

每条轨迹最大单步正信用占比的 p50 和 p90 都为 1.0。原因包括：

- 69 条退化为终步 fallback；
- 35 条 ungrounded non-fallback 的正信用主要由单个反馈恢复动作获得；
- perception step 尚无 grounding edge。

因此当前数据上的奖励还不是真正的 dense process reward。

### 6.4 `E` 与 `B` 高度相关

当前 `E` 是最终 evidence handle 第一次创建，通常对应 BackSlice 中的最后一个 producing step。因此
`E` 主要表现为给该步骤第二份权重，而不是发现一类独立的感知贡献。后续 perception grounding 才能
让 `E` 覆盖 inspect/read 等新证据获取。

### 6.5 大型 denotation 的内存问题

早期实现为了寻找与 gold 一致的中间 handle，直接物化所有候选表。BIRD 大表使全量报告进程内存持续
增长。修复后先比较行数，并把超过 10,000 行的 denotation 保守标记为
`denotation_too_large_to_ground`，不给猜测性信用。当前有 1 条进入该桶。

### 6.6 全量报告并发写入事故

桌面命令会话返回后，两个长重放进程曾继续在后台运行并同时写同一个输出文件。该中间报告已废弃，
随后停止重复进程，并在单一 tmux session 中从头生成最终 183 行报告。长报告必须使用唯一 session
和独占输出目录，完成后同时检查 tmux、JSONL 行数和 summary。

### 6.7 当前权重没有调优

`w_B=w_E=w_S=w_F=1` 和当前 lambda 只是可解释的 pilot 起点。现阶段 grounding 缺陷对奖励形态的
影响远大于权重差异，因此不应先做权重搜索来掩盖数据契约问题。

### 6.8 失败轨迹尚缺真实全量审计

当前 SFT 输入只包含 verified success。失败轨迹的 `sum(r)=-P`、终局失败和错误惩罚已由合成单元
测试覆盖，但在启动 RL 前仍应为 raw rollout failure 增加统一 adapter，并在人类抽查样本上验证负
奖励位置。

## 7. Readiness gate 与当前结论

报告当前采用以下保守 gate：

1. 所有轨迹重新执行正确；
2. 所有奖励满足 `sum(r)=C-P`；
3. 所有正确轨迹总奖励大于零；
4. 严格最终 grounding rate 至少 90%；
5. ungrounded non-fallback 必须为 0。

前三项通过，后两项失败，因此：

```text
process_reward_ready = false
```

这表示代码和数学不变量可用，但当前数据契约下的信用分配不够可靠。不得因为总奖励守恒就直接启动
process-shaped RL。

## 8. 下一步

按优先级：

1. 修改 `answer_from_context` 证据契约：标量答案也必须引用 harness 验证过的 1x1 evidence handle；
2. 为 inspect/read/value observation 增加 harness-authored grounding edge，并接入最终依赖图；
3. 在线轨迹直接持久化 `references`、`produces`、行数 lineage 和状态变化，避免离线二次重建；
4. 为 raw failure rollout 增加奖励 adapter，抽查终局失败、连续错误、空结果重复和忽略反馈；
5. grounding 达标后再做 `w_B/w_E/w_S/w_F` 与 lambda/Pmax 敏感性实验；
6. 从同一 SFT-2 checkpoint 分支 result-only RL 与 process-reward RL，保持任务、K、采样参数和优化预算一致；
7. 最终只在 BIRD dev 上报告泛化结果，BIRD train gold 仅用于训练侧 harness 验证。
