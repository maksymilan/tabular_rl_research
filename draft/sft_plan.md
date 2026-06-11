# 工具覆盖补全 + SFT 初验计划

日期 2026-06-10。回答四个问题：剩余工具如何进数据、SFT 怎么做、SFT 前后如何测、数据是否够。

**总原则：v0 SFT 用现有 6,256 条立即启动，验证训练+评测管线；工具覆盖增强并行做，产出 v1 数据后再训 v1。两条线不互相阻塞。**

## 一、剩余工具进入模型的路径（Phase 0，数据侧）

模型"学会"一个工具 = 该工具以正确的使用情境出现在 SFT 数据中。15 个工具按进入批次：

| 批次 | 工具 | 构造机制 | 依赖 | 预计增量 |
|---|---|---|---|---|
| v0（已有） | filter / project / join / group / aggregate / extreme / set_op / answer | gold SQL 编译 | 无 | 6,256 + 914 条 |
| v1-a | `add_to_memory` | 标量子查询编译：aggregate 算标量 → 记入 memory（引用来源步）→ 作为阈值用于后续 filter。验证：标量==子查询值 且 终结果==gold | 改 compiler | 从 550 条 compile_error 中解锁数百条，且轨迹更长 |
| v1-a | `derive_column` | 编译器 un-fold：SELECT/WHERE/ORDER 中的标量表达式拆为独立 derive 步，后续步引用派生列 | 改 compiler | 改造存量轨迹（重新生成） |
| v1-b | `inspect_column` | 注入 grounding 步：对文本/类别列按字面量过滤前，插入 inspect_column 读取值分布（真实执行输出）。随机注入 30–50%，每条 ≤2 个 | 改 emitter（后处理） | 改造 ~4,100 条含 filter 的轨迹的一部分 |
| v1-c | `semantic_match` | 字面量扰动：问题中精确值 V 改写为同义/描述/错拼，filter 步换成 semantic_match，嵌入模型拒绝采样验证解析回同一行集 | 嵌入模型 + LLM API | 目标 1–2k 条增强副本 |
| v2 | `window` | 编译器加 OVER() 分支（executor 已支持） | BIRD/SEDE 接入后才有量 | 少量 |
| v2 | `refine_memory` | 多轮数据（CoSQL/SParC）：后一轮修订前一轮的记忆条目；或留给 RL 阶段自然产生 | 新数据集 adapter | 后置 |
| 待定 | `read_subtable` | 感知内联已覆盖其功能，候选裁撤；若保留则在轨迹开头注入"peek 源表"步 | 设计决策 | — |

v1 完成后：13/15 工具有数据，总量约 8–10k 条。全部机制使用既有工具，不新增工具。

## 二、SFT 数据准备（Phase 1）

1. **格式转换** `scripts/sft/build_sft_data.py`：trajectory JSON → messages（sharegpt 格式）：
   - system：工具规范（紧凑 schema）+ 调用规则（一次一个 tool_call、JSON 格式、必须以 answer_from_context 终止、引用证据表）。
   - user：dataset_overview（紧凑渲染）+ question。
   - 循环：assistant = think + tool_call JSON；observation = tool_output（含内联表内容）。
   - **loss 只算 assistant tokens**，observation/工具输出不参与损失。
2. **think 字段**：v0 直接用现有模板（目的是验证管线）；v1 用 LLM 以可见状态为条件重写，校验不引入可见状态之外的事实（数字/实体 ⊆ 已展示内容）。
3. **统计与配比**：
   - 实测 token 长度分布定 cutoff_len（含内联内容，预估 95 分位 4–8k，需脚本实测）。
   - 长度配比：len-2/3 占 48%，对短轨迹降采样或对 ≥6 步轨迹上采样，避免模型偏向两步答题。
   - spider_dev.jsonl 的 914 条不进训练（评测保留）。
4. 产出：`data/sft/spider_v0.jsonl` + 统计 manifest（token 分布、长度配比、工具直方图、生成 commit）。

## 三、SFT 训练（Phase 2）

| 项 | 选择 | 理由 |
|---|---|---|
| 基座 | Qwen2.5-7B-Instruct；另用 3B 做快速迭代 | 工具调用能力底子好，单卡可训 |
| 框架 | LLaMA-Factory（备选 ms-swift / trl） | sharegpt 格式 + observation 掩码开箱即用 |
| 方法 | LoRA r=16–32, lr 1e-4, 2–3 epochs, bf16, cosine | 6k 量级数据全参易过拟合，LoRA 足够验证 |
| 上下文 | cutoff_len 按实测，8192 起步 | 内联表内容占大头 |
| 硬件 | 7B LoRA + 8k ctx：单卡 ≥40G；3B 可 24G；QLoRA 兜底 | |
| 流程 | 先 3B + 500 条子集冒烟（确认 loss 下降、输出可解析）→ 7B 全量 | 把格式错误暴露在小实验里 |
| 选 ckpt | 用 dev 子集（50–100 题）rollout 的执行准确率选，不只看 eval loss | loss 低 ≠ 工具链正确 |

## 四、评测协议（Phase 3）—— SFT 前后同一协议

**缺口组件（最优先做）**：rollout runner `scripts/eval/rollout.py`。模型生成 think+tool_call → 解析 → harness 执行 → 返回 tool_output（含 preview）→ 循环，直到 answer_from_context 或 max_steps=20；非法调用返回错误信息、允许重试 ≤2 次。模型经 vLLM OpenAI 兼容端点服务，temperature=0。此组件后续直接复用为 RL 环境交互循环。

**评测集**：Spider dev 全部 1,034 题。gold = gold SQL 在真库执行的行集（与我的编译器是否覆盖无关，覆盖率 100%）。另报 914 条 in-coverage 子集切片。

**指标**：
1. 执行准确率（主指标）：答案行集 == gold 行集（排序不敏感，数值容差）。
2. 格式合法率：tool_call 可解析 / 工具名合法 / 参数合法 / 表引用存在（步级 + 轨迹级）。
3. 过程指标：平均步数、重试次数、引用完整性、工具分布 vs 参考轨迹。
4. 按 Spider hardness（easy/medium/hard/extra）分桶。

**对照**：
| 组 | 设置 | 含义 |
|---|---|---|
| A | 基座 + 工具规范 + 2 条示例轨迹（few-shot） | SFT 前能力 |
| B | SFT 后，零样本，同 system prompt | SFT 后能力 |
| 上界 | 编译轨迹的 execution-verified 覆盖（dev 88.4%） | 数据管线上限 |
| C（后置） | 基座直接写 SQL | 研究主对照，v1 之后做 |

预期结果形态：A 合法率低（JSON/参数错误为主）、B 合法率 >95% 且执行准确率显著提升。这是第一个可汇报的 SFT 结论。

## 五、数据是否足够

- **数量**：6.2k 条对 7B LoRA 的格式学习足够（工具调用 SFT 常见 1k–10k 量级）。v0 结论：够。
- **真正的不足是多样性，不是数量**：工具 8/15、think 模板化、单数据集、长度偏短（len-2/3 占 48%）。这些影响"模型只会 SQL 形状的轨迹"，不影响 v0 管线验证。
- **决策门**：v0 SFT 后若合法率 >95% 且执行准确率明显高于组 A → 管线成立，投入 v1 数据（工具覆盖 + think 重写 + 配比调整）；若不达标，先修格式/数据问题再扩数据。

## 六、执行顺序

| 周 | 事项 |
|---|---|
| W1 | rollout runner + 组 A 基线跑通（最大未知，先做）；build_sft_data v0；token 长度实测 |
| W2 | 3B 冒烟 → 7B 全量 SFT → 组 B 评测 → before/after 对照表 |
| W3+ | Phase 0 增强（子查询→memory、derive un-fold、inspect 注入）→ v1 数据 → v1 SFT；semantic_match 扰动管线 |
