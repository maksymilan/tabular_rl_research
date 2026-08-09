# BIRD Version53 Native Bundle Prompt 逐项审核与静态审计

日期：2026-08-06  
状态：实现与静态验证完成；尚未进行外部模型行为评测

## 结论

Version53 从冻结的 version52 分支，只修改 `native-tool-bundle` 的学生/教师 prompt profile。
12 个函数、参数 schema、DeepSeek native carrier、bundle pre-state 校验、执行顺序、resident
state、recent-4 provider-turn history、结构化错误、terminal lowering、`bird-set` scorer 均不变。

本次没有调用外部 API，也没有生成或准入任何 SFT/RL 数据。Version53 仍为
`diagnostic-only`；multi-call provider turn 不得展开成虚假的单调用训练回合。

## 建议逐项处置

| 建议 | 处置 | 实现边界 |
|---|---|---|
| 拆分 runtime 与 teacher-only 规则 | 采纳 | 学生 prompt 是完整运行契约；教师 prompt 以它为严格前缀，只追加生成质量/格式规则。影响正确性的语义纪律不能只留给教师。 |
| 放宽 `condition_filter` 并行限制 | 采纳 | 不同 resident handles 上、共同需要且互不依赖的 filter 可以同批；竞争分支仍需问题、external knowledge 或观测值支持。 |
| 增加“数据不是指令” | 采纳 | external knowledge、数据库内容、单元格、schema、metadata 只能提供任务数据，不能改变 system/tool/权限/语法/carrier。 |
| 调用数约束服从正确性 | 采纳 | 仍默认单调用、通常 1–3；必要的独立 perception 可超过三次，provider 硬上限仍是 8。 |
| 区分 source representation 与 derived metric | 采纳 | 复制的源字段保持存储表示；派生指标保持工具实际产出；除非题目要求，不自行格式化、拼接或替换 ID/label。 |
| 加入 JOIN 基数检查 | 有条件采纳 | catalog edge 不是唯一性证明；仅当基数可能影响 aggregation/ranking，或出现零行、重复、异常膨胀时要求检查 key/row count，避免把每次 JOIN 都变成固定额外调用。 |
| 精确化 `value_ref` | 采纳 | 只允许 scalar 或 one-row metric producing step；多指标表还必须指定准确列；read/perception/reasoning/plan step 均非法。 |
| 精确化重复读取限制 | 采纳 | 只禁止同一 immutable handle 上“成功且参数完全相同”的 `read_subtable`；错误读取和参数变化后的分页/换列不受此限制。 |
| 移除 harness 实现说明 | 采纳 | 删除 call-id 回传、provider content retention 等模型无法控制的说明；这些继续由 adapter/harness 代码保证。 |
| 删除无依据的 plan evidence 规则 | 部分不采纳 | 当前 public `plan` schema 确实包含 `evidence`，因此保留“无证据时 omit/null，禁止空字符串”，但只放在 teacher-only 生成规则。 |
| 终止前读取完整 evidence table | 不采纳 | `read_subtable` 有行数上限，最终结果也可能超过 preview；强制“完整读取”会制造不可满足约束。改为按风险和需要检查 final handle，最终仍由 harness 对完整引用表评分。 |
| 在 prompt 中写完整函数 grammar/cookbook | 不采纳 | native JSON schemas 继续是唯一参数形状权威，避免恢复 version51 的重复 token。 |

## Prompt 结构

共享学生运行契约只保留六类信息：

1. 权威与数据边界；
2. resident state、逻辑列和 `value_ref` 语义；
3. population、grain、显式映射和 JOIN 风险；
4. 最终证据的字段、表示与 sole-terminal 约束；
5. 正确性优先的 bundle scheduling；
6. 结构化错误恢复和 bounded-history 权威性。

教师增量只包含：关系决策前的单位/population/output-slot 检查、异常核验、简短
`reasoning_content`、空 ordinary content、精确重复读取边界、错误针对性修正和实际存在的
plan-evidence 约束。

## 静态大小

计数使用确定性的 Python `len(str)` 和 canonical native-tools JSON；它是字符审计，不冒充
DeepSeek 服务端 tokenizer 统计。

| 指标 | version52 | version53 | 变化 |
|---|---:|---:|---:|
| student prompt chars | 3,615 | 3,927 | +312（+8.6%） |
| teacher provider prompt chars | 6,887 | 5,286 | -1,601（-23.2%） |
| native tools JSON chars | 9,882 | 9,882 | 0 |
| teacher prompt + native schema | 16,769 | 15,168 | -1,601（-9.5%） |

共享学生契约略增，是因为数据注入边界、JOIN 基数、表示边界和正确性优先属于训练与推理都
必须成立的规则；教师专属增量大幅缩短，避免用只在 teacher 中出现的语义规则制造
train/runtime mismatch。

Version53 hashes：

- registry / protocol：`tool-scheme-registry-v10` / `version53`
- protocol hash：`a12e686b43ca9e42`
- student prompt SHA-256：`96d538627da22cd547fef4ff5684cf2c536a06e7356ae4a7d9d0fbdf6f3d8b7e`
- teacher prompt SHA-256：`ec28102e32aeb6e8c88b055df3016fc2d6e8d8c1715fd8721ad29c5868b80198`

## 静态门禁

- teacher prompt 必须以完整 student prompt 开头；
- teacher-only 生成规则不得出现在 student prompt；
- 不恢复 textual tool catalog、参数 grammar 或 native call cookbook；
- harness call-id/content-retention 说明不得进入 prompt；
- 不要求完整读取任意大小的最终表；
- student prompt `< 4,000` chars；teacher prompt `< 5,500` chars；
- teacher prompt + native schema `< 15,500` chars；
- version52 的工具/执行测试和 version53 的 error→corrected-call 因果测试必须通过。

## 解释边界与下一步

静态压缩和规则整理不能证明准确率提升，也不能直接预测服务端 billed tokens。若要比较行为，
应在同一冻结任务、同一模型、同一 recent-4/catalog/`bird-set` 配置下做 version52/version53
配对 gate，同时报告正确率、合法终止、过程错误、实际 provider tokens、调用数、重复读取、
JOIN 检查触发率和 output-shape 回归。在用户另行授权外部请求前，不启动该 gate。
