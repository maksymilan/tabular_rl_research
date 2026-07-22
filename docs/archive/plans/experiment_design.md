# 实验设计汇总（draft）

> 草稿 / 讨论记录，非定稿。日期 2026-06-06。
> 把讨论中提到的实验设计集中到一处，并标注每个实验"支撑哪个论点"。
> 详细推导见同目录 process_reward_density.md、subtable_vs_memory.md。约定：纯文本，不用 LaTeX。

---

## 0. 总览：每个实验对应一个论点

| 实验 | 支撑的论点 |
|---|---|
| E1 核心 A/B：工具+显式状态 RL vs 代码执行 RL | 我们的环境让"高保真过程奖励"可定义可计算（核心 novelty） |
| E2 子表必要性消融 | 子表不冗余，memory-only 会崩（架构辩护） |
| E3 adapter 强度消融 | 结果不是"强检索器替我们干活"（消除 adapter 混淆） |
| E4 奖励成分消融 | 细粒度有据过程奖励确实有用（奖励设计贡献） |
| E5 baseline 套件 | 标准严谨性 / 各组件贡献拆解 |

通用目标定位：不在纯准确率上硬拼代码 agent，而是"准确率持平 + 忠实度/效率/可解释性更优"。

---

## E1. 核心 A/B：工具+显式状态 RL vs 代码执行 RL

(详见 process_reward_density.md §7)

- 同一 base 模型（如 Qwen3-8B）。
- A 臂 = 代码执行 RL（复刻 TableGPT-R1 式 Python agent）。
- B 臂 = 抽象工具 + 显式状态 RL（本方案）。
- 测量（不只准确率）：
  - rho_fid：逐步奖励 r_t 与反事实贡献 c_t 的相关性（c_t 由"抹掉/中性化第 t 步后重测答对率"估计）。
  - 证据引用忠实度（见 §6 指标）。
  - 样本效率（学习曲线：准确率 vs 训练步/样本；到达 X% 所需步数）。
  - reward hacking 频率。
- 预期：B 的 rho_fid 显著高于 A（A 的过程奖励是 local validity，与 c_t 近乎不相关）；B 在忠实度/效率上更优。

---

## E2. 子表必要性消融（memory-only vs 子表+memory）

(详见 subtable_vs_memory.md §5)

- A = memory-only（不给子表，只能读写 memory 作答）。
- B = 子表 + memory（现状）。
- 按问题类型分别测：幻觉率 / 引用忠实度 / 准确率。
- 预期：A 在查找/抽取类和忠实度上崩，B 明显更稳。
- 作用：把"子表是否必要"从设计断言变成实测命题。

---

## E3. adapter 强度消融（匹配后端强度阶梯）

(来自 semantic_match 讨论 + agent.md 弱点 #6)

- 把语义匹配后端做成可换、可记录的组件，跑强度阶梯：
  - 词法（BM25 / RapidFuzz / 子串）→ 小嵌入模型 → 强嵌入模型 →（上界）LLM-judge。
- 更广义：simple adapter vs strong adapter。
- 测量：
  - 各档下的策略准确率；
  - "模型是在推理还是在蹭检索器"——例如：固定策略、只换 adapter，看准确率随 adapter 强度的增益曲线；增益越陡，说明越依赖 adapter（混淆越大）。
- 关键点：嵌入模型是**固定的环境基础设施**（类比 SQLite 引擎），不参与训练、配置记入 tool_history。
- 作用：正面回应"adapter 太强、是不是它替你干活"的审稿质疑。

---

## E4. 奖励成分消融（过程奖励的价值）

- 阶梯：
  - R0 仅终端奖励（最稀疏）；
  - R1 +粗过程奖励（TableGPT-R1 式 ±0.1，local validity）；
  - R2 +细粒度有据过程奖励（证据获取 / 剪枝正确性 / 引用忠实度，基于 provenance 切片）。
- 测量：准确率、rho_fid、样本效率、reward hacking 频率。
- 另做奖励尺度消融：过程奖励 vs 终端奖励权重比（防止其一压过另一）。
- 预期：R2 > R1 > R0，尤其在样本效率和忠实度上；R2 的 rho_fid 最高。
- 作用：证明"细粒度有据过程奖励"本身是贡献，而非可有可无的塑形。

---

## E5. baseline 套件（组件贡献拆解）

(来自 agent.md 设计建议)

至少包含：
- 直接作答（无工具）；
- 全表塞进上下文；
- SQL / Python / code-agent baseline（对照表达力上界）；
- 仅检索、不剪枝；
- 剪枝、无 memory；
- 完整 harness（本方案）。
作用：拆出"检索""剪枝""memory""过程奖励"各自的边际贡献。

---

## 6. 指标定义（统一口径）

- 准确率：按答案类型（Boolean/Category/Number/List）精确匹配，并按问题类型（查找/聚合/多跳/多表）分层。
- rho_fid（过程奖励保真度）：corr_t(r_t, c_t)，c_t = Pr(答对|保留 t 步) - Pr(答对|抹掉 t 步)（消融估计；金标准更严格用 Shapley）。
- 引用忠实度：answer_from_context 引用的证据中，真正支撑答案的比例。可用 (a) 引用是否落在 provenance 切片内，(b) verifier 判"答案能否由所引证据推出" 两种方式测。
- 样本效率：学习曲线 / AUC / 到达 X% 所需训练量。
- 上下文压缩：每条轨迹平均展示给模型的 行/列/cell/token 数（context_budget 指标）；最小充分上下文比例。
- reward hacking 指标：无足够证据就作答、引用无关证据、重复/空转工具调用等退化行为频率（TableGPT-R1 记录过 corrupted tokens / 跳过推理）。
- 剪枝正确性：drop_context 删除的行中，落在 provenance 切片外（正确剪枝）vs 切片内（误删 gold 证据，重罚）的比例。

---

## 7. 评测基准对齐（与对照工作放同一张表）

- 已用 / 主力：TableBench、Spider、（多表多跳）BIRD、WTQ。
- 数值多跳：FinQA / MultiHiertt（注意只取答案落在表里的子集，见 table_text_handling.md）。
- 评测对齐 TableGPT-R1 / SemEval：RealHitBench、InfiAgent-DABench、DataBench（full + Lite，天然是上下文规模消融）。

---

## 8. 待办 / 需先定的前提

- E1/E4 需要可运行 harness + reward + 反事实消融设施（与 trajectory 构造共用 harness）。
- rho_fid 的 c_t 估计成本高（每步消融重测）；先在带证据标注的小样本上测，别全量。
- 证据标注来源：数据集自带（FeTaQA 高亮 cell 等）或正确轨迹切片当伪标签（见 process_reward_density.md §6）。
- E3 需要 semantic_match adapter 接口落地（多语言小嵌入模型，版本钉死）。
