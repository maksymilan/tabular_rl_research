# Atomic 上下文管理 Handle Card 消融 Gate16

## 结果

| 版本 | 改动 | 正确 | Target | Control | Legal | Errors | Mean actions | Reads | Exact rereads | Prompt chars/request | Tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| version39 | 同时间 baseline | 8/15 | 0/7 | 8/8 | 15/15 | 6 | 9.73 | 27 | 0 | 28881 | 1201127 |
| version46 | 仅 handle cards | 8/15 | 0/7 | 8/8 | 14/15 | 11 | 10.73 | 42 | 0 | 29260 | 1664931 |
| version47 | + old rows archived / latest full | 9/15 | 2/7 | 7/8 | 13/15 | 7 | 12.00 | 71 | 24 | 29195 | 1585804 |
| version48 | + interpret-before-act | 9/15 | 2/7 | 7/8 | 13/15 | 10 | 14.47 | 101 | 35 | 29696 | 1943704 |
| version49 | active dependency rows / inactive archive | 8/15 | 0/7 | 8/8 | 14/15 | 8 | 9.20 | 26 | 0 | 28648 | 1169008 |

version49 原始 16 题 artifact 为 9/16、15/16 legal；其中全局配对排除项 `bird_train_06299` 在 version49 中正确（21 步、2 次错误）。为避免利用 version46 的缺失结果选择性计分，上表所有版本仍统一排除该题。该题涉及超出 recent-4 的 dormant branch，实际用了 10 次读取、2 次精确重读，但没有再耗尽 30 步。

## 离线重渲染

对历史 fixed-200 的每个真实 prefix 做纯渲染替换，canonical state 和动作不变：

| 版本 | 末轮平均字符变化 | 解释 |
|---|---:|---|
| version46 | -478 | handle card 净压缩 |
| version47 | -233 | latest full output 抵消大部分 rows 归档收益 |
| version48 | +145 | 总结提示使 prompt 净增长 |
| version49 | -270 | 仅归档 active closure 外的 rows |

version47 全局归档后，历史真实下一动作中有 12 轮、20 个 literal 不再可见；version49 降至 1 轮、4 个 literal，均需重新读取。

## 配对变化

- version39 → version46：0 gains、0 regressions，exact McNemar `p=1.0000`。
- version46 → version47：2 gains、1 regressions，exact McNemar `p=1.0000`。
  - gains: bird_train_02868, bird_train_06246
  - regressions: bird_train_05873
- version47 → version48：1 gains、1 regressions，exact McNemar `p=1.0000`。
  - gains: bird_train_06165
  - regressions: bird_train_02868
- version39 → version48：2 gains、1 regressions，exact McNemar `p=1.0000`。
  - gains: bird_train_06165, bird_train_06246
  - regressions: bird_train_05873
- version46 → version49：0 gains、0 regressions，exact McNemar `p=1.0000`。
- version48 → version49：1 gains、2 regressions，exact McNemar `p=1.0000`。
  - gains: bird_train_05873
  - regressions: bird_train_06165, bird_train_06246
- version39 → version49：0 gains、0 regressions，exact McNemar `p=1.0000`。

## 诊断

1. **Handle card 单独没有能力收益。** version46 与 baseline 在 15 个语义可比任务上完全同 outcome，但 actions 增加 10.3%，tokens 增加 38.6%，process errors 从 6 增至 11。
2. **归档旧 rows 有弱 target 信号，但稳定性不合格。** version47 相对 version46 为 2 gains / 1 regression，p=1.0；Legal 从 14/15 降至 13/15。read calls 从 27 增至 71，其中 24 次是同题内精确重读。
3. **强制先总结没有净增益。** version48 相对 version47 为 1 gain / 1 regression，actions 达 baseline 的 1.49 倍，tokens 达 1.62 倍，精确重读进一步升至 35 次。
4. **主要失败机制是 observation churn。** `bird_train_05873` 在 version47/48 中分别产生 17 / 18 次精确重读并达到 max-steps；baseline 与 version46 都正确。`bird_train_02918` 虽保持正确，也从 baseline 10 步增长到version47 21 步、version48 26 步。
5. **Dependency-aware active/archive 消除了配对子集的 churn，但没有带来能力增益。** version49 在配对任务上为 8/15，与 baseline outcome 完全一致；8/8 controls 全保留，但 0/7 targets 恢复。读取降至 26 次、精确重读 0 次，`bird_train_05873` 从 version47/48 的 30 步失败恢复为 8 步正确，`bird_train_02918` 从 21/26 步回落到 10 步。相对 baseline，actions 仅变化 -5.5%，tokens 变化 -2.7%。

## version49 失败归因

- **最终列/表示不精确（2）**：`02868` 多返回 MiddleName；`00582` 把 First/Last 拼成一个 name 列。两题实体与行集合都已找对。
- **擅自改变问题语义（2）**：`02438` 把 distinct status 改成每单 latest status 并多带 order_id；`06246` 无依据增加 `type='Post Office'` 限制，得到 137 而非 173。
- **聚合/行粒度错误（1）**：`00074` 没有按题目要求使用全局平均订单数量，最终只保留 1 个标题而非 3 个。
- **排序后单行边界错误（1）**：`01213` 找对最新作品 Henry VIII，却返回其全部 47 个角色；gold 的 `ORDER BY Date DESC LIMIT 1` 只保留 1 行。
- **工具参数恢复失败（1）**：`06165` 两次 join 列命名错误后，又用非法 `read_subtable(limit=50)`，触发同类错误上限。

## 决策

version49 不扩到 Gate50，不进入 SFT/RL，也不替换 version39。Gate16 当时仅显示单次运行中的 churn 消失，不能证明跨运行稳定。后续冻结 Gate8 K=3 复测中，version39/version49 都是 20/24 correct 且 15/15 controls；但 version49 平均步数增加 23.0%、reads 增加 71.9%、tokens 增加 28.7%，并出现 3 次 max-steps（version39 为 0）。因此 active/archive renderer 的“上下文稳定性目标”现已被复测否定，不再保留为候选工程 renderer。下一步应针对合法链路上的语义/最终表错误，或先建立不隐藏精确证据的显式状态压缩充分性检查，而不是继续压缩 resident rows。详见 `BIRD_ATOMIC_VERSION49_CONTEXT_STABILITY_K3_GATE8_20260731_ZH.md`。

## 基础设施处理

- 全局语义配对子集排除：bird_train_06299。
- 只允许用 fresh retry 替换原始 `api_error`；wrong answer、工具错误和 max-steps 均不重试。
- 已替换：version46/bird_train_03768。

## 协议边界

四个实验版本均保持 version39 工具、参数、执行、canonical state、grounding、recent-4、external knowledge 和 exact-table terminal 不变。所有轨迹均为 diagnostic-only，不进入 SFT/RL。

离线重渲染与 provider 结果见同目录 JSON；本报告只在五个 16-task artifact 全部存在且 trajectory id 完全配对后生成。
