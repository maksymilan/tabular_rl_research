# BIRD Version54 去 Plan：baseline300 Prefix200 配对能力保持测试预注册

日期：2026-08-06  
状态：**2026-08-06 在外部 API 调用前被用户的新范围取代，未运行、未发送任何数据。**

> 本计划原拟在 baseline300 Prefix200 上做 v53/v54 配对。用户随后明确要求改为当前
> teacher1500 训练候选集前 200 题的 v54 单臂验收，不比较性能。有效计划见
> `BIRD_VERSION54_NO_PLAN_TEACHER1500_PREFIX200_PLAN_20260806_ZH.md`。本文仅保留为范围变更
> 审计记录，禁止按本文启动请求。

## 研究问题

在当前 `native-tool-bundle` 协议中，从 DeepSeek 可见的原生函数集合移除 `plan` 后，模型在
同一批 BIRD-train 任务上的正确率、合法终止和错误恢复能力是否保留。

这不是 atomic 与 direct SQL 的比较，也不是训练准入实验。v53/v54 均保持
`diagnostic-only`，结果不得直接进入 SFT/RL。

## 严格单变量边界

配对两臂为：

| 项目 | v53 control | v54 no-plan |
|---|---|---|
| provider carrier | `deepseek-native-tool-bundle-v1` | 相同 |
| provider-visible functions | 12 个，含 `plan` | 11 个，仅删除 `plan` |
| student runtime prompt | SHA-256 `96d538627da22cd547fef4ff5684cf2c536a06e7356ae4a7d9d0fbdf6f3d8b7e` | 完全相同 |
| teacher-only prompt | v53 reviewed | 仅删除已失效的 `Plan evidence...` 一句 |
| harness/executor | v53 语义 | 相同；底层 replay 兼容代码未删除 |
| pre-state validation | bundle shared pre-state | 相同 |
| history | recent 4 provider turns | 相同 |
| terminal | `answer_from_context` 必须单独调用 | 相同 |
| scorer | hidden `bird-set` | 相同 |

删除 teacher-only 的 plan-evidence 句是移除公开工具后的必要一致性修正，不添加新的解题建议。
如果 provider 幻觉返回 `plan`，客户端按 `unknown_tool` 返回结构化、状态不变的错误；不会执行
该调用。

静态请求大小：v53 为 15,168 字符，v54 为 14,316 字符，减少 852 字符（5.62%）。其中
学生 prompt 完全一致；变化来自一个函数 schema 和一条教师 plan 说明的删除。

## 冻结任务

- source：`data/eval_inputs/bird_train_baseline300_v1.jsonl`
- source SHA-256：`87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03`
- selection：源文件顺序的前 200 条，`start=0, limit=200`，不重新采样
- unique examples：200
- unique databases：67
- example-id sequence SHA-256：
  `9a39d1328009538a8836cd307116108634e65b90925c9e9ea9880a031fcf6450`
- public task identity SHA-256：
  `cc3644d785069d79696c410c5975378b2ecba31c9c0945df014393b417460717`
- identity manifest：
  `data/eval_inputs/bird_train_baseline300_prefix200_v1.manifest.json`

两臂必须跑完整 200 题；不得因中途结果调整 cohort、prompt、工具 schema、重试预算或停止点。

## 冻结运行配置

- provider：官方 `https://api.deepseek.com/chat/completions`，无代理、无 fallback
- model：`deepseek-v4-flash`
- attempts：K=1 / arm，共 400 个 episode
- context：`rolling-legal-history` / recent 4 / `full`
- database context：`catalog-v1`
- denotation：`bird-set`
- max steps：30
- max completion tokens：2,048
- API retries：10
- request timeout：300 秒
- workers：4
- table output rows：0
- diagnostic-only：true

v53：`--atomic-protocol-version version53 --deepseek-carrier native-tool-bundle`  
v54：`--atomic-protocol-version version54 --deepseek-carrier native-tool-bundle`

## 外部数据边界

获得本实验的明确授权后，外部请求只允许包含：自然语言问题、external knowledge、数据库
catalog/schema、当前 causal prefix 和逐步只读工具反馈。不得发送 gold SQL、gold rows、gold
sample、gold row count、私有空结果标签或隐藏验证器数据。授权前不进行任何外部请求。

## 预注册指标与判定

主要能力指标：

1. paired correct：报告 v54/v53 正确数、共同正确、共同错误、gains、regressions、净变化、
   百分点变化和 exact McNemar 双侧检验；
2. 点估计保持门：`v54_correct >= v53_correct - 4`（最多下降 2 个百分点）；
3. 同时报告 paired difference 的 95% 区间；若区间无法排除有意义下降，只能表述为
   “点估计保持”，不能表述为统计等价。

可靠性门：

1. v54 legal termination 不低于 196/200，且相对 v53 最多下降 2 题；
2. v54 不得出现 provider-visible `plan` schema，成功执行的 primitive 中 `plan=0`；
3. 所有 v54 correct 轨迹通过 fresh replay；两臂通过结构、provider-history、bundle-pre-state、
   call/result 顺序和 no-leak 审计；
4. 结构化 process errors 不得比 v53 增加超过 10 个，并逐类报告。

效率只作次要解释：总/均值/中位数/P95 tokens、model turns、primitive calls、bundle size、
API retry、wall time，以及 v54/v53 比值。prompt 变短不自动等价于 provider billed tokens 下降。

只有主要点估计门和全部可靠性门均通过，才判定“在本 Prefix200 K=1 Flash 测试上能力保留”。
该结论不外推到 Pro、学生模型或训练后能力，也不构成 SFT/RL 准入。
