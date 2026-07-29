# 八篇 Text-to-SQL 相关工作 Introduction 中文翻译与 SFT 动机分析

日期：2026-07-29

## 阅读范围与说明

本文基于以下本地论文逐页核对 Introduction，翻译采用忠实意译：保留作者的问题定义、
论证链、方法主张、贡献和正文中的结果数字，但省略文献编号。结果数字均是论文作者的
报告值，不代表本项目已经独立复现。

- `OmniSQL: Synthesizing High-quality Text-to-SQL Data at Scale`
- `SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning`
- `Reasoning-SQL: Reinforcement Learning with SQL Tailored Partial Rewards for
  Reasoning-Enhanced Text-to-SQL`
- `Arctic-Text2SQL-R1: Simple Rewards, Strong Reasoning in Text-to-SQL`
- `SQL-Trail: Multi-Turn Reinforcement Learning with Interleaved Feedback for Text-to-SQL`
- `Reward-SQL: Boosting Text-to-SQL via Stepwise Execution-Aware Reasoning and
  Process-Supervised Rewards`
- `Progress-SQL: Improving Reinforcement Learning for Text-to-SQL via Progressive Rewards`
- `SQL-ASTRA: Alleviating Sparse Feedback in Agentic SQL via Column-Set Matching and
  Trajectory Aggregation`

为了更清楚地看到研究路线的演化，下面没有按文件名排序，而是按“扩大数据覆盖 →
结果奖励 → 局部奖励 → 简单结果奖励与强初始化 → 多轮环境交互 → 过程奖励 →
轨迹进步奖励”的逻辑排列。

## 一句话结论

这些工作并没有共同证明“多加一些正确 SFT 轨迹就能超过强 Direct-SQL baseline”。
它们实际上分别依赖了五种不同的增益来源：

1. **大规模覆盖**：OmniSQL 用 250 万合成样本建立强 SQL 先验。
2. **强初始化后的策略重排**：Arctic 从 OmniSQL 等强 SFT checkpoint 出发，再用在线
   RL 把正确候选提到更高概率。
3. **多候选探索**：SQL-R1、Reasoning-SQL 和 Arctic 都依赖同题多次 rollout，使奖励
   能在候选之间产生学习信号。
4. **数据库反馈进入决策过程**：SQL-Trail 和 Reward-SQL 不满足于“先想完、最后执行”，
   而是让中间 SQL/CTE 实际执行，再根据观察修正。
5. **从终局奖励转向过程信用**：Reward-SQL 和 Progress-SQL 直接把研究问题定义成
   “哪一步有用、轨迹是否在进步”，而不只是最终 SQL 是否正确。

你当前的工具 checkpoint 在 greedy 上与 Direct SQL 基本持平、但 Pass@4 明显领先，
更像是“正确轨迹已经进入策略支持集，但没有稳定成为最高概率轨迹”，而不是模型完全
没有学到工具能力。因此，下一步更应解决**状态覆盖、终止/恢复监督、同状态动作区分和
在线策略重排**，而不是继续无差别堆叠容易的成功轨迹。

---

## 1. OmniSQL：把问题归因于训练分布覆盖不足

原论文：`/Users/hudou/papers/reference/OminiSQL.pdf`

### Introduction 中文翻译

Text-to-SQL 将自然语言问题转换为可执行 SQL，使非专家也能有效地与数据库交互。
这一能力支撑了大量以数据为中心的应用，也受到自然语言处理和数据库两个研究社区的
广泛关注。

**现有最优方法的优势与局限。** 大语言模型推动了 Text-to-SQL 的显著进展。当前最优
系统往往采用多智能体协作框架，让不同智能体分别完成 schema linking、SQL 生成、
SQL 修正和候选选择等子任务。其中，SQL 生成仍是核心。围绕这一核心组件，现有方法
大致分为提示式和微调式两类。

提示式方法通过精心设计的 prompt 使用强大 LLM，通常依赖通过 API 调用的闭源模型；
微调式方法则在 Spider、BIRD 等既有数据集上训练 LLM。两者虽然都取得了很好的
benchmark 表现，但在真实应用中存在明显问题。提示式方法成本高、存在数据隐私风险，
而且由于依赖 API，用户很难控制和定制模型。微调式方法则常常难以泛化到复杂问题或
特定领域数据库，因为公开数据对真实场景的覆盖有限。

例如，把 Qwen2.5-Coder-7B-Instruct 在 Spider 和 BIRD 训练集上微调后，它在同分布
开发集上表现良好，但在分布外的 ScienceBenchmark 和 EHRSQL 上，执行准确率分别
只有 43.8% 和 31.4%。相比之下，零样本提示 GPT-4-Turbo 分别达到 59.2% 和
43.1%。这一差距表明，现有微调方法的泛化能力有限。

因此，一个有前景的方向是用**大规模、多样且高质量的训练数据**增强开源模型的
Text-to-SQL 能力。这既可能改善模型性能和泛化，也允许模型在本地部署，从而降低成本、
保护数据并支持定制。不过，大规模人工标注通常不可行。早期数据增强方法虽然能扩充
已有数据集，但多数只是继续生成符合原数据分布的样本，因而在多样性、质量和扩展性上
仍然有限；不少方法还需要专家手写复杂模板或语法，进一步限制了实用性。

**本文方案。** 作者提出一种新的 Text-to-SQL 数据合成框架，相比已有增强方法有三个
特点：

1. 自动化：整个过程几乎不需要人工参与；
2. 可扩展：能够跨大量领域生成大规模、多样且高质量的数据；
3. 真实：合成数据尽量对应真实用户需求和使用场景。

要同时实现自动、可扩展和真实并不容易。第一个挑战是：在自动化和规模扩张的同时，
如何维持数据质量和多样性。作者把复杂的合成过程拆成若干更容易控制的阶段，每一步
都由 LLM 自动执行：

1. 从 Web 表格出发，让 LLM 推断合理的业务背景并合成真实感较强的数据库。数据库
   包含多张有主外键关系的表，以及表名、表描述、列名、类型、列描述和示例行等元数据。
   Web 上数量庞大的表格为跨领域扩展提供了来源。
2. 根据合成数据库生成有意义的 SQL。
3. 使用回译把 SQL 转换成语义等价的自然语言问题。作者认为 SQL-to-question 比
   question-to-SQL 更容易保证正确性，因为从 SQL 解释成自然语言比从模糊问题反推
   唯一 SQL 更准确、歧义更少。
4. 为每条样本生成逐步的 CoT 解答，描述从问题和数据库构造 SQL 的中间推理过程，
   既提高可解释性，也提供更丰富的训练信号。

第二个挑战是让合成数据贴近真实需求。SQL 应覆盖从简单检索到高级分析的不同难度，
因此作者定义了 simple、moderate、complex、highly complex 四级 SQL 复杂度，并在
生成时显式控制难度。用户表达也很多样，因此作者定义了九种语言风格：正式、口语、
祈使、疑问、描述、简洁、模糊、隐喻和对话式，并让 SQL-to-question 阶段按指定风格
生成问题。

为了验证框架，作者构建 SynSQL-2.5M，包含 2,544,390 个
`<database, question, SQL, CoT>` 四元组，覆盖 16,583 个合成数据库。作者从数据库、
问题、SQL 和完整样本四个维度评估质量，并声称它在几乎所有指标上优于人工标注的
BIRD。

在 SynSQL-2.5M 上，作者训练了 7B、14B 和 32B 三种规模的 OmniSQL，并在九个数据集
上评估：Spider 开发/测试集和 BIRD 开发集；Spider2.0-SQLite、ScienceBenchmark、
EHRSQL 三个领域数据集；以及 Spider-DK、Spider-Syn、Spider-Realistic 三个鲁棒性
数据集。作者报告 OmniSQL 以更小参数量达到新的平均最优水平，匹配或超过
DeepSeek-V3、Qwen2.5-72B-Instruct、GPT-4-Turbo 和 GPT-4o。

作者把贡献概括为：

- 提出自动、可扩展的数据合成框架，覆盖真实数据库、难度可控 SQL、多风格问题和
  CoT；
- 发布首个百万规模的 Text-to-SQL 合成集 SynSQL-2.5M，并据此训练 7B/14B/32B
  OmniSQL；
- 在多项 Text-to-SQL benchmark 上以更少参数达到新的最优表现，并开源代码、数据和
  模型。

### Motivation 的本质

OmniSQL 不认为主要瓶颈是优化器或奖励，而认为是**公开训练数据只覆盖了很窄的
数据库、SQL 结构和语言表达分布**。它的解法是先扩张任务分布，再做普通 SFT。

这对当前项目的提醒是：4,119 个 action target 或约 700 条成功 episode，承担的是比
普通 SQL SFT 更复杂的学习目标——既要学协议，又要学何时观察、如何操作关系、如何
恢复和终止。它与 250 万条直接 SQL/CoT 数据并不是同一个数据量级。

---

## 2. SQL-R1：把 SFT 视为固定模仿策略，用结果反馈促进探索

原论文：`/Users/hudou/papers/reference/SQL-R1.pdf`

### Introduction 中文翻译

NL2SQL/Text-to-SQL 把自然语言问题转换为结构化 SQL，让用户无需数据库专业知识也能
与数据库交互。近年的方法显著改善了数据库应用中的人机交互，并服务于多种数据科学
分析任务。现有研究主要优化 workflow 及其组件，例如 schema linking、内容检索、
生成修正等。

尽管如此，在复杂数据库场景中提升推理表现仍然很困难。复杂 schema 容易造成多表
join 和嵌套查询错误，单独训练的模型也很难处理复杂语义。大量工作通过 SFT 训练
开源 LLM，希望以较小模型获得接近 GPT-4/GPT-4o 等闭源模型的准确率。但 SFT 对
数据库 schema 结构和训练数据规模依赖很强，这会导致模型在新数据库上的领域适配和
泛化不稳定。缺少可解释的 NL2SQL 推理过程，也限制了它在金融、医疗等高风险领域的
使用。

强化学习近年来展现出训练 LLM 推理能力的潜力。与 SFT 相比，RL 可以通过和环境
交互动态调整决策策略，因此在复杂推理任务中可能取得更好表现。类似方法已用于金融
推理、搜索和数学推理。

受此启发，作者提出用 RL 训练 NL2SQL 推理模型 SQL-R1，并围绕三个问题展开：

**Q1：能否为 NL2SQL 设计专门的 RL 方法，并成功训练出推理模型？** 与 SFT 不同，
RL 直接优化模型是否生成了准确表达用户意图的 SQL。困难在于怎样设计有效的反馈；
恰当的奖励结构可能显著改善表现。

**Q2：RL 模型是否需要特定的 cold start？** 对已有基座模型做合适的冷启动，可以
增强 instruction following、激活 SQL 生成能力，并让后续 RL 探索产生更高质量的
候选。怎样设计冷启动本身也是一个挑战。

**Q3：能否建立可持续的数据工程来训练稳健、高效的 NL2SQL 推理模型？** RL 依赖
高质量数据，但真实 NL2SQL 训练数据不足。如何用数据工程支撑训练、鲁棒性和泛化，是
必须解决的问题。

作者将贡献概括为：

- 提出显式 NL2SQL 推理模型 SQL-R1，只使用约 5K NL2SQL 数据进行 RL 训练，报告在
  Spider-Test 和 BIRD 上分别达到 88.6% 和 66.6%，并能输出详细推理过程；
- 系统研究 cold-start，提出 SFT + RL 的训练策略，并分析合成数据对性能和鲁棒性的
  作用。

注：该版本摘要写 BIRD 67.1%，Introduction 贡献列表写 66.6%，正文不同设置还报告
其他数字。引用时必须绑定具体模型、推理设置和评测脚本，不能只写“SQL-R1 是 67%”。

### Motivation 的本质

SQL-R1 的核心假设是：SFT 学到的是训练分布中的**固定生成策略**；当 schema 或语义
发生变化时，模型不能根据结果成败主动调整。RL 的价值不是额外展示一条标准答案，
而是从同题多个候选中提高得到正确执行结果的推理方式的概率。

不过，它并不是“完全不用 SFT”。论文自己把 cold-start、合成数据和 RL 数据工程列为
核心问题。也就是说，RL 的收益建立在模型已经会生成 SQL、会遵守输出格式并且 rollout
中存在足够正确候选的前提上。

---

## 3. Reasoning-SQL：认为二值 EX 太稀疏，需要 SQL 特定的局部奖励

原论文：`/Users/hudou/papers/reference/Reasonning-SQL.pdf`

### Introduction 中文翻译

Text-to-SQL 连接人类语言与数据库查询，把自然语言问题转换为可执行 SQL。LLM 带来
了显著进展，当前多数最优系统都依赖 LLM 合成 SQL。但这些方法的上限受制于基座模型
的推理能力，因此需要显式增强 Text-to-SQL 推理。准确 SQL 不仅要求理解自然语言，
还要求在复杂数据库 schema 上进行精细推理。

传统的 LLM 适配主要依靠 SFT 或 few-shot prompting；当问题存在歧义、需要多步推理
或需要较长思考时，这些方法经常不足。o1 和 DeepSeek-R1 等 reasoning model 表明，
在推理时投入更多计算、生成经过权衡的推理轨迹，可以提高答案质量。受此启发，作者
提出 Reasoning-SQL：用 RL 鼓励模型生成详细中间推理，最终得到更准确的 SQL。

RL 训练的核心是奖励函数。Text-to-SQL 最直观的奖励是执行准确率，但它是二值且稀疏
的：当模型只正确理解了部分逻辑或 schema 关系时，最终执行错误会让这些部分也得不到
有效反馈。稀疏奖励会阻碍策略优化。

为此，作者设计了组合奖励，把多种局部信号合在一起：

- LLM-as-a-Judge；
- 语法检查；
- schema linking；
- N-gram 相似度。

作者使用 GRPO 整合这些奖励。对每个输入生成多个候选 SQL，并在组内进行相对评估，
从而同时优化中间推理过程和最终执行准确率。

消融实验显示，这些局部奖励比只使用执行准确率更有效。作者在 BIRD、Spider、
Spider-DK 和 Spider-Syn 上验证，声称较小模型不仅超过传统 SFT，还超过更大的闭源
基座模型；集成进标准 Text-to-SQL pipeline 后达到 72.78% 执行准确率，同时推理成本
降低 93%。作者还声称，模型通过 RL 自然形成的结构化推理优于人工设计的逐步 prompt。

主要贡献是：

1. 提出首个自动优化 Text-to-SQL 推理过程的 RL 框架；
2. 提出一套 SQL 特定的局部奖励并做系统消融；
3. 以较小开源模型取得有竞争力且低成本的结果。

### Motivation 的本质

Reasoning-SQL 接受“最后执行正确”是最终目标，但认为它对学习而言信息密度太低。
它希望用 schema、语法、文本相似度和 AI judge 告诉模型“虽然还没完全正确，但哪些
部分更接近”。

这条路线的风险也很直接：局部指标只是最终语义正确性的代理。模型可能学会提高
schema overlap、N-gram 或 judge 分数，却没有提高真实 denotation。它适合作为
outcome reward 的补充，而不应替代终局执行正确性。对当前项目而言，模型自述
`<think>` 也不能成为事实或奖励依据；只有 harness 可验证的状态变化适合进入过程
信用。

---

## 4. Arctic-Text2SQL-R1：主张强数据、强初始化之后，简单奖励反而更稳

原论文：`/Users/hudou/papers/reference/Arctic-Text2SQL.pdf`

### Introduction 中文翻译

把自然语言问题转换为 SQL 是自然语言理解和人机交互中的核心问题。可靠的
Text-to-SQL 系统可以让非技术用户直接用自然语言查询结构化数据库，从而降低数据
分析门槛。

LLM 已显著改善 SQL 生成的流畅度和表面覆盖，但要生成真正正确且可执行的 SQL，
尤其是涉及多表 join、嵌套逻辑和细致 schema 理解的复杂查询，仍然困难。多数方法
依赖 `(question, SQL)` 对进行 SFT，但这种训练往往不能促进可靠、可泛化 SQL 所需的
中间推理。

作者提出 Arctic-Text2SQL-R1，一个通过 RL 生成高质量可执行 SQL 的框架和模型族。
它使用基于执行正确性的轻量奖励，避免脆弱的局部 reward shaping，并让训练更稳定、
目标与最终任务更一致。作者强调：真正有效的组合是**高质量数据、强 SFT 初始化、
简单执行奖励和合适的在线训练实践**，而不是单独依靠某一种技巧。

主要贡献包括：

- **简单、可扩展的 Text-to-SQL RL。** 使用执行驱动的轻量奖励，在不同模型尺寸和
  benchmark 上稳定扩展，而无需复杂 reward design。
- **多 benchmark 的最优表现。** 论文报告 32B 在 BIRD leaderboard 达到 71.83%，
  14B 超过 70%，7B 超过此前 70B 级系统，并在六个数据集上超过多种 SQL 专用和
  通用模型。
- **数据与训练策略同样关键。** 给出数据过滤、合成生成和模型筛选的实践；在线 RL
  从强监督 checkpoint 出发，配合恰当 prompt 继续提升。
- **广泛评测。** 六个数据集覆盖不同 schema 复杂度和查询难度，以降低只对单一数据
  或评测格式过拟合的风险。
- **推理时可扩展。** value retrieval 和 majority voting 可以在较小系统开销下继续
  提高准确率。
- **同时报告正面和负面经验。** 为后续 RL Text-to-SQL 研究提供实践依据。

Introduction 的表格把 Arctic 与复杂奖励方法对照：Reasoning-SQL 使用执行、语法、
N-gram、LLM、schema 和格式奖励；SQL-R1 使用执行、长度、语法和格式奖励；
Think2SQL 使用 precision、recall、cardinality 和两项格式奖励；Arctic 只列
`EX + syntax`。因此，“execution-only”更准确的理解是：**唯一有语义方向性的正奖励
来自执行结果**，语法信号主要用于处理不可执行输出，而不是用多个代理指标定义
“部分语义正确”。

### Motivation 的本质

Arctic 与 Reasoning-SQL 的判断几乎相反：它认为复杂局部奖励容易脆弱、产生局部
最优或偏离最终任务；只要数据经过过滤、初始化足够强、prompt 对齐、在线 rollout
中有可学习的正负样本，简单 EX 就足够有效。

它也是解释当前“为什么论文 7B 远高于我的 7B”最关键的一篇。论文完整实验表明，
Arctic-Text2SQL-R1-7B 的强版本以 **OmniSQL-7B** 为初始化，而 OmniSQL 已经在
SynSQL-2.5M 等大规模数据上获得很强的 SQL 先验。论文表中的 BIRD-dev 结果是
OmniSQL-7B 63.9%，Arctic-Text2SQL-R1-7B 68.9%。因此正确的增益归因是：

> 250 万级 SQL SFT 建立 63.9% 的强起点，再由在线 GRPO、数据过滤和 prompt 对齐
> 提高约 5 个百分点。

它不是“原始 Qwen2.5-Coder-7B + 少量简单奖励直接达到 68.9%”。

---

## 5. SQL-Trail：认为真正缺失的是数据库闭环，而不是更长的静态 CoT

原论文：`/Users/hudou/papers/reference/SQL-Trail.pdf`

### Introduction 中文翻译

Text-to-SQL 让非专家可以通过自然语言访问结构化数据库。LLM 已取得显著进展，但在
BIRD-SQL 等困难 benchmark 上，最强 AI 系统与人类专家之间仍有明显差距。

作者认为，这个差距主要来自主流的**单次生成范式**：给定问题和 schema，模型直接
输出 SQL，却不把数据库环境的执行结果或错误信息纳入推理过程。Text-to-SQL 本身还
有四个困难：

1. 自然语言问题经常有歧义；
2. schema 可能庞大、复杂、含噪，实体名语义也可能模糊；
3. SQL 语法严格，几乎没有容错空间；
4. 高计算成本和高质量公开训练数据短缺带来实践限制。

很多失败来自错误 schema linking，以及嵌套子查询、多跳 join 和复杂聚合等难题。
人类专家之所以更强，是因为他们会反复与数据库交互：探索 schema，把复杂问题拆成
小查询，检查中间结果，再根据执行反馈修正和调试。

工具增强 RL 在搜索、UI、代码执行等任务上已经表明，多轮环境交互可以让 agent 获取
缺失信息、修正中间假设并根据反馈自我纠错，从而提高准确率和跨领域鲁棒性。基于这一
范式，作者提出 SQL-Trail，一个通过多轮 RL 训练的 Text-to-SQL agent。

与 SQL-R1 等单次 RL 不同，SQL-Trail 在闭环中进行数据库探测、schema 探索和基于
执行的自我修正。它包含两个关键设计：

- 难度感知的 turn budget：简单问题少思考，困难问题分配更多交互；
- 组合 reward panel：同时鼓励执行正确和高效的长程行为，提供稠密逐步指导。

作者将贡献概括为：

1. 提出统一的多轮 RL 训练框架和自适应 turn budget；
2. 系统比较单次与多轮 RL，分析组合奖励，并研究多轮 agent 的推理效率；
3. 报告较强的数据效率和分布外泛化。作者称 OmniSQL 每 1,000 条同分布训练样本只
   带来约 0.005 个百分点的增益，而 SQL-Trail 使用分布外数据时每 1,000 条可带来
   4.6 个百分点；7B 模型在多个 benchmark 上平均超过更大的闭源模型约 5%。

### Motivation 的本质

SQL-Trail 认为静态 CoT 的问题不是“不够长”，而是**没有新观察**。如果模型在生成
完整 SQL 前从未执行任何东西，那么更长 reasoning 仍可能围绕错误的表、join 或值
进行自洽推演。环境反馈改变的是信息集，而不只是计算量。

它与当前项目的研究初心最接近，但有一个决定性接口差异：SQL-Trail 的主要动作是
提交候选 SQL、观察执行结果、再修订 SQL。它最大程度保留了 Qwen2.5-Coder 已有的
SQL 生成先验。当前项目要求模型把问题编译为一串细粒度 typed relational tools，
还要学习 handle、resident state、引用和终止协议。后者有更强的可审计性和过程信用
潜力，但也显著扩大了 SFT 必须学习的新策略空间。

论文的结果也显示阶段作用不同：在其 7B 设置中，未微调多轮 agent、SFT、RL 的
BIRD-dev greedy 分别约为 50.9%、57.8% 和 60.1%。SFT 主要改善格式、短轨迹和基本
工具行为；RL 再改善 schema linking、执行修正和候选一致性。这比期待 SFT 一步完成
所有探索策略更合理。

---

## 6. Reward-SQL：用可执行 CTE 作为过程单元，把思考落到数据库状态

原论文：`/Users/hudou/papers/reference/Reward-SQL.pdf`

### Introduction 中文翻译

Text-to-SQL 把自然语言问题转换为可执行 SQL，使非技术用户能够访问关系数据库。
尽管 LLM 有很大进展，在复杂数据库上生成包含多表 join 和嵌套结构的复杂 SQL 仍然
困难。

**现有 RL 方法及其局限。** 近期工作把 RL 作为后训练策略，通过把复杂 SQL 生成拆成
中间 reasoning step，再用反馈优化模型。它们通常使用 GRPO、DPO 等方法鼓励结构化
逐步推理。但作者认为仍有两个问题：

1. **缺少逐步、执行感知的推理。** 多数方法只是在最终 SQL 前生成一些文字 reasoning；
   生成期间模型并不与数据库交互，只有最终 SQL 执行后才看到错误。缺少中间执行信号，
   限制了复杂 SQL 表现。
2. **缺少过程监督奖励。** 现有方法主要在最终 SQL 完成后提供 outcome reward。它能
   鼓励最终执行正确，却无法精细调整中间轨迹，复杂查询中的错误会逐步累积。

作者用一个预实验支持这一判断：Qwen3-8B 在 BIRD train 上用 outcome-only GRPO
训练后，BIRD dev 只有 63.0%。错误主要是 filter condition 161 例（32.8%）、表选择
115 例（23.4%）、列选择 81 例（16.5%）。共同根因是模型一次性生成整个 SQL，无法
查看中间数据库状态，因此不能检查 join 是否重复行、filter 是否符合真实数据分布。

论文给出一个真实失败：问题要求 season 5 中平均每场 runs 最高的五位球员。模型文字
推理看似合理：按球员汇总 runs、统计比赛数、计算平均值、排序取前五；但 SQL 多 join
了一张 player 表，导致 runs 被重复累计。分母使用
`COUNT(DISTINCT match_id)` 也无法修复被放大的分子。因为中间 join 从未执行检查，
模型只能在最终答案错误后知道“某处有问题”，却不知道哪一步出了错。

**CoCTE：分治且执行感知的推理框架。** 作者借鉴数据库工程师写复杂 SQL 的方式：
不是一次写完，而是逐步建立中间 view，用 Common Table Expression（CTE）组织逻辑、
检查中间结果。每个 CTE 都是可以验证和复用的构件，最终查询由已验证组件组合而成。

CoCTE 把复杂查询拆成一串可执行 CTE，每一步生成后立即执行，把数据库反馈带回后续
推理。这样模型能在过程中发现并修正错误，提高复杂 SQL 的准确率和可解释性。上述
例子会被拆成：筛选 season 5 的比赛；计算每位球员总 runs；统计其比赛数；计算平均
值；最后连接球员名并排序。中间表会让模型看到正确的粒度和分母，从而避免错误 join。

**Reward-SQL 的三个阶段。**

1. **模型初始化。** 通用 LLM 不会自然遵守 CoCTE 轨迹格式，因此先合成 CoCTE 数据做
   SFT，让模型学会这种结构化交互，作为 RL 初始化。
2. **过程奖励设计。** 每个中间步骤都绑定可执行反馈，提供细粒度、execution-aware
   监督，而不是只在完整 SQL 后打分。
3. **过程监督 RL 与推理。** RL 同时使用过程和终局信号；推理时用过程 reward 支持
   高效候选选择和结构化探索。

实现它有两个技术挑战。第一，怎样为 CoCTE 中间轨迹设计可靠过程奖励。作者使用组合
PRM：轨迹评分模型估计每个中间步骤的正确性；inverse-entropy weighting 根据各步骤
对最终查询的相对贡献分配权重，强调信息量更大的步骤。

第二，怎样在稠密 reward 与稳定训练之间取得平衡。组合不当会导致发散或 reward
hacking，即模型只追求高过程分数而不生成正确答案。作者因此把 process reward 和
outcome reward 放入统一目标；推理时还用过程 reward 做 Best-of-N 选择。

作者报告，stepwise execution-aware reasoning 把表选择和列选择错误分别减少 42.6%
和 27.2%，process reward 对 GROUP BY 错误的降幅达到 82.4%。完整 Reward-SQL
使用 8B 模型在 BIRD 达到 70.3%，并在五个分布外 benchmark 上无需重训保持较强表现。

主要贡献是：

1. 提出 CoCTE，把复杂 SQL 拆成一串可执行 CTE；
2. 提出 Reward-SQL，解决过程奖励和过程监督 RL/推理问题；
3. 在 BIRD 和分布外数据上验证效果。

### Motivation 的本质

Reward-SQL 区分了两件常被混在一起的事：

- 写一段看起来合理的 CoT；
- 生成一个可执行、能改变后续信息状态的中间程序。

只有后者能为过程信用提供可验证事实。它与当前项目“模型 reasoning 不作为事实权威、
harness observation 才能控制 factual provenance”的原则高度一致。

但 Reward-SQL 选择 **CTE** 作为动作单位，而不是更细的通用关系原子。这是值得认真
对照的设计：CTE 既可逐步执行、验证，又继续使用 Coder 模型已经熟悉的 SQL 表达。
当前 typed tools 的优势是参数约束、可追踪 provenance 和局部 credit 更清楚；代价是
动作序列更长、协议更重、需要从很少数据中学习一个全新的中间语言。

---

## 7. Progress-SQL：最终正确还不够，奖励应描述轨迹是否真的在改善

原论文：`/Users/hudou/papers/reference/Progress-SQL.pdf`

### Introduction 中文翻译

LLM 已显著推进 Text-to-SQL。近期 RL 方法通常优化一次性 reward：模型单轮生成一个
SQL，reward 只根据它的执行结果计算。这类 reward 很稀疏，对 SQL 生成的指导有限，
因此难以高效探索正确 SQL，尤其是包含 join 和 aggregation 的复杂查询。

SkyRL-SQL 等工作已引入多轮 rollout：模型为同一个问题生成一串 SQL，每轮获得执行
结果并用于下一轮生成。但它仍然只根据最后一轮 SQL 计算 reward，所以没有真正摆脱
one-shot reward 的局限。换句话说，**最后状态的 reward 无法描述多轮轨迹中的动态
修订行为**。

为解决这个问题，作者提出带 progressive reward 的多轮 RL 框架 Progress-SQL。
首先构建 Oracle-guided Diagnostic Tree（ODT），把 SQL 抽象为 clause-level 结构
profile，并为下一轮修订生成诊断反馈。训练时比较预测 ODT 与 gold ODT，让模型根据
结构反馈修正 SQL。

与 one-shot reward 不同，progressive reward 定义在整条 SQL trajectory 上，衡量
最终 SQL 相比初始 SQL 是否在结构和词法对齐上变好。它还加入：

- progression latency reward：鼓励更早达到正确；
- execution status reward：鼓励从无效 SQL 恢复。

因此，目标函数偏好那些有效改善、较早正确且能从执行错误中恢复的轨迹。作者在 BIRD、
Spider 及 Spider 鲁棒性变体上评测；以 7B 为基座时，在 BIRD Dev、Spider Dev 和
Spider Test 上的执行准确率平均提高 8.5%，Spider Dev test-suite accuracy 平均提高
6.3%。

主要贡献是：

1. 提出多轮 Text-to-SQL RL 框架和 clause-level ODT 诊断反馈；
2. 设计显式衡量初始 SQL 到最终 SQL 改善程度的 progressive reward，并加入早正确
   和执行状态 reward；
3. 在多个 benchmark 和不同基座模型上同时改善 EX 与 test-suite accuracy。

### Motivation 的本质

Progress-SQL 指出了 outcome-only 多轮 RL 的一个真实盲点：两条最终都失败的轨迹，
可能一条从严重错误走到了几乎正确，另一条完全没有进展；终局二值 reward 却把它们都
视为 0。类似地，两条最终都正确的轨迹，一条第二步就正确，另一条反复试错到最后一步，
终局 reward 也无法区分效率。

不过，ODT 在训练时使用 gold SQL 的结构 profile。它属于有特权监督的 reward
shaping，不能直接搬到当前“gold SQL 对 actor/teacher 隐藏，只由 harness 做终止校验”
的主线。可借鉴的是“奖励轨迹改善和恢复”的思想，而不是把 gold 结构反馈暴露给模型。

---

## 8. SQL-ASTRA：同时解决多轮信用聚合和单步稀疏反馈

原论文：`/Users/hudou/papers/reference/SQL-ASTRA.pdf`

### Introduction 中文翻译

近年来，Agentic RL 受到广泛关注。它让 LLM 与环境进行多轮交互，从而完成 deep
research、Web search 和代码执行等更复杂的任务。多数场景通过 RLVR 提升能力，即根据
最终结果是否正确提供反馈。

但在需要探索性推理的复杂任务中，Agentic RL 仍有三个核心问题：

1. **范式约束。** Agent 本来面向多轮交互，但 Text-to-SQL 等具体领域多数仍停留在
   单轮静态生成。它不能体现人类数据分析师通过多个试探查询收集上下文、修正策略的
   动态过程，因此限制了模型解决复杂真实任务的能力。
2. **信用分配。** 多轮轨迹往往只根据最后一轮反馈打分。这种 all-or-nothing 方法把
   整条交互视为黑箱，无法判断哪些中间步骤真正帮助了最终结果。
3. **微观奖励稀疏。** 即使每一步都有执行反馈，reward 也经常只是粗糙的 0/1 信号。
   它忽略“部分正确”查询中包含的信息，无法提供足够细粒度的方向，降低 RL 的效率和
   鲁棒性。

为同时解决这些问题，作者提出三项设计：

1. 构建多轮交互框架，让 agent 通过数据库交互反复获取上下文和修订 SQL；
2. 提出 Aggregated Trajectory Reward（ATR）。ATR 使用非对称状态转移矩阵聚合整条
   reasoning path 上的分数，显式鼓励持续、单调改善。作者把推理视为动力系统，并用
   Lyapunov stability 论证 ATR 相当于 energy dissipation operator，可以惩罚循环、
   促进单调收敛；
3. 提出 Column-Set Matching Reward（CSMR），作为即时的稠密 step reward。它执行
   每轮 SQL，对预测结果和 gold 结果的各列 value set 做归一化匹配，把二值 0/1 转成
   `[0,1]` 的部分正确分数。CSMR 既给每一步提供方向，也是 ATR 聚合的输入。

论文声称，在 BIRD 上，完整方法比 binary-reward GRPO 高约 5 个百分点，并在相同
OmniSQL-7B 初始化下超过 Arctic-Text2SQL-R1-7B。

### Motivation 的本质

SQL-ASTRA 把 Reward-SQL 和 Progress-SQL 的问题合到一起：

- CSMR 解决**同一步执行结果只有 0/1、无法看出部分接近程度**；
- ATR 解决**每一步的分数怎样沿整条多轮轨迹聚合，并抑制退步和循环**。

这与当前项目观察到的大量相邻重复/no-progress 很相关。它的重要启发是：过程 credit
不仅要判断某一步局部好不好，还要让“从较好状态退回较差状态再恢复”的循环产生净
负收益，否则 agent 可能通过来回摆动获得或维持高分。

但 CSMR 每一步都把预测执行结果与 **gold SQL 的执行结果**做列级部分匹配。这是
privileged reward shaping，不符合当前主线“gold 只做 terminal denotation，不用它
定义中间唯一方向”的边界。可以借鉴非对称退步惩罚、轨迹聚合和循环抑制，不能直接把
CSMR 当作当前 grounded process credit。

### 论文真正训练的模型不是 Coder-7B agent

SQL-ASTRA 的主体实验有两组：

- Qwen2.5-7B-Instruct 从头做 BIRD RL：8,958 个过滤后的 BIRD train 样本、每题 8 个
  rollout、5 epochs、最多 3 次 SQL tool call、32 张 A800-80G；完整
  `Agentic SQL + CSMR + ATR` 报告 64.2%；
- OmniSQL-7B 先用 Format-6K 学 tool format，再做 BIRD + Spider RL，报告 69.1%。

作者反而明确写道：Qwen2.5-Coder 过度专门化，instruction following 较差，难以学习
agent format；即使先用 SFT 修格式，也没有表现出足够的多轮探索能力。因此表中的
“Qwen2.5-Coder-7B = 58.2”只是外部 baseline 行，不是 SQL-ASTRA 在 Coder-7B 上训练
agent 后的结果。

### 为什么它表中的 58.2% 不能与当前 48.76% 直接比较

1. **SQL-ASTRA 没有复跑这行。** 表中把它写成 `Qwen2.5-Coder-7B (Team, 2024)`，
   RL Data 和 Base Model 都是空值。它是从外部结果表转引的，不属于论文自己的
   “Our Results”。
2. **所引 Qwen primary source 本身不是 58.2。** Qwen2.5-Coder Technical Report
   Figure 12 中，Qwen2.5-Coder-7B-Instruct 的 BIRD 是 **51.1%**；该报告说明其统一
   prompt 含 table representation、table-content examples、optional knowledge 和
   question。
3. **58.2 可追溯到 SQL-R1 的实验体系。** SQL-R1 论文把 base Coder-7B 写成 58.2；
   其官方仓库当前 commit `ccfb511e81fdb3a8e87bd3da7f3d71882cedd532` 的输入构造
   与当前项目明显不同：
   - 用 DDL，而不是当前 compact JSON schema；
   - 包含列 comment、PK/FK；
   - 每一列放最多 6 个从真实 dev database 采样的 representative values；
   - 把 BIRD evidence 直接拼到问题前；
   - 使用明确的 step-by-step/`<think>`/SQL code-block 指令；
   - `max_output_len=2048`，而当前可比 baseline 为 1024。
4. **它的“greedy”并非严格 temperature-0。** SQL-R1 官方 `inference.sh` 默认
   `temperature=0.8, n=8`；`evaluation_bird_post.py --mode greedy_search` 只是取
   `pred_sqls[0]`，即八个采样候选中的第一个，而不是确定性 greedy。SQL-ASTRA 把
   这类外部结果统一放进 “Bird (Greedy)” 列，标签不能证明生成参数一致。
5. **底层 EX 比较器反而基本一致。** SQL-R1 evaluator 和当前 `bird-set` 都比较
   `set(predicted_rows) == set(gold_rows)`。SQL-R1 使用 10 秒 query timeout，当前
   Coder baseline 使用 20 秒且没有 timeout 边界样本；这不是 9.44 点差距的主要来源。

因此，更可靠的参照关系是：

| 结果 | 口径 |
| --- | --- |
| 48.76% | 当前固定 revision、temperature 0、无列样例值、1024 output、严格可复现 |
| 51.1% | Qwen 官方报告；带 table-content examples/optional knowledge 的统一 prompt |
| 58.2% | SQL-R1 风格富 schema/value prompt；公开脚本所谓 greedy 实为第一个 T=0.8 sample |

目前不能把 48.76 → 58.2 的每一点精确分给 prompt、values 或 sampling；需要在同一
checkpoint 上做控制实验。最小的四行 ablation 应为：

1. 当前 prompt，temperature 0；
2. SQL-R1 DDL + column comments，但不加 values，temperature 0；
3. 再加每列 representative values，temperature 0；
4. 完整 SQL-R1 prompt，temperature 0.8，取第一个 sample。

只有第 2/3/4 行与第 1 行的配对差值，才能说明你的真实 baseline 还缺多少 prompt-side
能力，而不是引用不同论文的不可比数字。

---

## 9. 八篇工作的 Motivation 对照

| 工作 | 它认为 SFT/既有方法缺什么 | 核心干预 | 真正新增的学习信号 | 主要风险 |
| --- | --- | --- | --- | --- |
| OmniSQL | 训练数据库、SQL 结构和语言风格覆盖太窄 | 250 万合成样本 SFT | 更广的监督分布 | 合成分布与真实需求仍可能有偏差 |
| SQL-R1 | SFT 只模仿固定策略，复杂 schema 上不主动探索 | cold-start + GRPO | 同题候选的执行成败 | 正确 rollout 太少时无有效优势 |
| Reasoning-SQL | 二值 EX 对部分正确没有反馈 | SQL-specific partial rewards | schema/语法/N-gram/judge | 代理奖励可能被优化而语义没变 |
| Arctic-R1 | 复杂 reward 脆弱，偏离最终目标 | 强 SFT 初始化 + 在线 GRPO + 简单 EX | 终局执行正确性 | 容易误以为强结果来自“RL alone” |
| SQL-Trail | 单次生成看不到数据库反馈 | 多轮 SQL 执行、修订、turn budget | 新的数据库 observation | 交互成本高，reward panel 仍可能偏 |
| Reward-SQL | 文字 CoT 不执行，outcome reward 无法定位步骤 | 可执行 CTE + PRM + GRPO | 中间执行状态与过程分 | PRM 错误、Best-of-N 成本、reward hacking |
| Progress-SQL | 最终状态 reward 不描述多轮是否改善 | ODT feedback + progressive reward | 相邻/首尾状态的进步与恢复 | 依赖 gold SQL 结构，存在特权监督 |
| SQL-ASTRA | 多轮只有终局分；单步 0/1 又太稀疏 | SQL agent + CSMR + ATR | gold-result 部分匹配与非对称轨迹聚合 | 中间步骤依赖 gold denotation；部分匹配可被利用 |

把它们串起来，可以得到一条清晰的研究演化：

> 先让模型见过更多 SQL（OmniSQL） → 再让执行正确性重排候选概率（SQL-R1/Arctic）
> → 为稀疏奖励添加局部方向（Reasoning-SQL） → 让模型真正进入数据库闭环
> （SQL-Trail/Reward-SQL） → 最后研究多轮轨迹中每一步如何分配并聚合信用
> （Reward-SQL/Progress-SQL/SQL-ASTRA）。

当前项目位于这条路线最右侧，但训练数据量仍接近最左侧方法的小规模 pilot。这就是
“研究目标很先进、SFT 表现却追不上 Direct SQL”的结构性原因之一。

## 10. 对当前 SFT 结果的重新诊断

### 10.1 现在并不是“模型 SQL 能力整体严重下降”

当前审计给出的可比结果是：

| 对照 | 结果 |
| --- | ---: |
| Source Qwen2.5-Coder-7B Direct SQL greedy | 748/1534 = 48.76% |
| 外部教师 SFT + carrier repair 后 Direct SQL greedy | 742/1534 = 48.37% |
| checkpoint-560 工具 greedy | 734/1534 = 47.85% |

SFT 模型直接写 SQL 只下降 0.39 个百分点，配对检验不显著。这说明平均意义上的底层 SQL
能力没有整体崩塌。真正明显的问题集中在：

- challenging 子集从 34.48% 降到 24.14%；
- 长 reasoning 挤占输出预算，出现不完整 carrier；
- 工具轨迹重复、缺少终止、到达训练数据未覆盖的 student state 后不会恢复。

### 10.2 Pass@4 说明正确策略已经存在，但 greedy 排序没有学好

按严格 BIRD reference EX：

| 指标 | checkpoint-560 工具 | Direct SQL base | 工具差值 |
| --- | ---: | ---: | ---: |
| sampled Pass@1 | 50.72% | 47.39% | +3.32 pp |
| Pass@2 | 60.23% | 53.46% | +6.78 pp |
| Pass@4 | 67.34% | 58.93% | +8.41 pp |

这表示工具 policy 的支持集中已经包含更多正确轨迹，且 k 越大覆盖优势越明显；但
greedy 仍为 47.85%，没有把这些轨迹稳定放到 mode。最符合这个现象的不是“模型没学会”，
而是：

1. SFT 只提高了成功教师动作的似然，没有看到同一 student state 下“这个动作为什么
   比另一个动作好”；
2. 正确轨迹被分散在多种探索方式中，局部错误和冗余调用也保留较高概率；
3. 训练目标按 token 平均，长 `<think>` 和系统上下文淹没了真正决定成败的短 JSON
   action、carrier 边界和终止动作；
4. 当前没有一个测试时可用的选择器把 Pass@4 的 oracle coverage 转换为 top-1。

这恰好是 SQL-R1/Arctic 的“在线策略重排”和 Reward-SQL 的“过程选择器”要解决的
问题。

### 10.3 当前 SFT 数据的缺陷比总条数更关键

当前外部教师数据有 4,119 个 action records、703 个 contributing episodes，但：

- 153/703（21.8%）episode 没有保留下最终 `answer_from_context` target，形成只教前缀
  不教终止的 orphan prefix；
- feedback recovery 只占 2.0%，而效果更稳的 SFT-1 为 9.0%；
- 数据主要来自强教师访问的状态，而 7B student 会在更早处犯错并进入不同状态，
  多轮 exposure mismatch 会逐步放大；
- 当前 system prompt 字符数约为 SFT-1 的 2.23 倍，典型 record 约长 1.6 倍，
  6,400-token cutoff 已经实质改变了 episode 的保留结构；
- 许多成功轨迹是“教师自然成功”，并没有覆盖 student 的重复调用、错误列、错误
  handle、无法终止等实际失败状态。

因此，“轨迹都正确、教师更强”不等于“更适合 student 学习”。对交互 agent，
**student 实际访问状态上的纠正轨迹**往往比干净但离策略分布很远的专家轨迹更重要。

## 11. 最值得立刻验证的研究灵感

### 11.1 增加一个 SQL-native 多轮 agent baseline，隔离工具抽象成本

建议在同一 Qwen2.5-Coder-7B、同一任务、同一数据库和相同总 token/执行预算下比较：

1. Direct SQL，一次生成；
2. SQL-native agent：`execute_sql → observation → revise_sql`；
3. CTE-native agent：逐步生成可执行 CTE；
4. 当前 typed relational tools。

这个实验能回答一个目前尚未隔离的问题：

> 性能差距来自“模型不会利用数据库反馈”，还是来自“模型需要重新学习过重的 typed
> tool 中间语言”？

如果 SQL-native/CTE-native 多轮显著高于 typed tools，说明信息闭环本身有效，但当前
动作抽象和 SFT 样本效率是瓶颈。如果 typed tools 在 OOD、错误恢复、grounding 或
过程信用上更强，才构成它相对 SQL-Trail/Reward-SQL 的实质研究价值。

### 11.2 把下一批数据从“成功 episode 集”改为“关键决策状态集”

下一轮不应按完整成功轨迹平均扩充，而应优先收集：

- 当前 student greedy 失败、但 Pass@4 中至少有一条成功的题；
- 同一 resident state 下同时存在错误动作和成功动作的 pair；
- 相邻重复、错误列/表、join 粒度错误、提前终止和不终止状态；
- 一次环境反馈后能够明确恢复的状态；
- challenging 和 OOD schema 上的决策点。

每条样本都应 episode-complete，必须包含 grounded terminal target；超过 cutoff 时应
缩短上下文或整条 episode 拒绝，不能保留 orphan prefix。

这会把数据目标从“模仿教师如何做完题”改成“在 student 真正犹豫或犯错的状态，学习
哪个动作改变成功概率”。

### 11.3 将 SFT 的职责收窄，后续能力交给在线 RL

可以把训练职责分成三层：

1. **协议层 SFT**：只保证 carrier、合法调用、handle 使用和可靠终止；对 action JSON、
   边界 token 和 terminal target 使用显式 loss weighting。
2. **student-state correction SFT / preference**：在同一状态上对比 bad action 与
   teacher-repaired action，重点学习恢复和避免 no-progress。
3. **在线 RL**：从当前 policy 实际 rollout，同题多采样，用 `bird-set` terminal reward
   做不可替代的最终目标，再与 harness-grounded process credit 做严格消融。

SFT 不需要独自学会“探索的最优策略”。它只需要把 policy 推到能够稳定交互且具有
足够非零成功率的区域；随后由在线 RL 把 Pass@k 中已有的正确轨迹提升为 top-1。

### 11.4 过程奖励应来自可验证状态，不奖励“像推理”的文本

Reasoning-SQL 的 schema/N-gram/judge reward 可以作为对照，但当前主线更适合坚持：

- terminal `bird-set` 是最终语义目标；
- 非法参数、执行错误、精确重复/no-progress 可获得局部非正信用；
- 后续合法恢复与产生正确终局的因果边可以单独记 credit；
- plan 和 `<think>` 只作为控制文本，不作为事实或正确性证据；
- 任何正过程奖励都必须通过 replay 和 provenance/grounding gate。

可以借鉴 Progress-SQL 的“改善”概念，但使用不依赖 gold SQL 的版本，例如：

- 从不可执行状态恢复到可执行状态；
- 从明确的 harness validation error 恢复；
- 消除相邻精确重复；
- 在同一成功 episode 中，对真正被后续引用的 grounded producing step 记局部贡献；
- 奖励更早的正确终止，但不能单独奖励更短轨迹。

这些信号仍需和 outcome-only control 做对照，防止模型只学会“合法、短、看似有进展”
却不提高 denotation。

### 11.5 把 Pass@k 覆盖转成可部署 top-1

当前最有价值的资产是 Pass@4 达到 67.34%，说明搜索空间中已经有较多正确轨迹。可同时
推进两条线：

- **训练时**：用 GRPO/相对优势把成功 rollout 的动作概率提高，而不是继续均匀 SFT；
- **推理时**：训练独立、不能访问 gold 的 trajectory/result selector，或研究执行
  结果聚类、过程一致性和 grounded trajectory features。

必须区分：

- Pass@k：使用隐藏 gold 判断 k 条里是否至少一条正确，是 oracle coverage；
- Majority/PRM selection：测试时真实可用的选择方法；
- Greedy：最终策略 mode。

论文里的多数投票或 PRM@N 不能与 Pass@k 混报，但它们可以帮助判断 67.34% 的上限中
有多少能被实际回收。

## 12. 建议的最小实验矩阵

| 阶段 | 模型/接口 | 目的 | 必报指标 |
| --- | --- | --- | --- |
| A | base Direct SQL | 原始 SQL 能力 | greedy EX、sample pass@k |
| B | base + 当前 tools，无训练 | 工具 prompt/接口零样本成本 | EX、legal、steps、errors |
| C | SFT checkpoint Direct SQL | 底层 SQL retention | 全量及 challenging EX |
| D | SFT checkpoint + tools | SFT 工具能力 | greedy、pass@k、终止、重复率 |
| E | SFT + SQL-native/CTE-native agent | 动作抽象消融 | 预算匹配 EX、成本、恢复 |
| F | outcome-only online RL | 终局策略重排 control | greedy、pass@k、reward groups |
| G | grounded process RL | 项目主方法 | 相对 F 的 paired gain、edge precision |
| H | selector/majority/PRM | Pass@k 到可部署 top-1 | selected EX，不使用 gold |

所有行应固定 task ids、`bird-set`、external knowledge、数据库快照、解码参数和执行预算。
另外按 simple/moderate/challenging、需要值探索/不需要值探索、seen/OOD schema 分层。

## 13. 最可能形成论文主线的表述

基于这些 Introduction，当前工作最有潜力的 motivation 不是：

> “我们也用 GRPO 提升 Text-to-SQL。”

而是：

> 现有 RL Text-to-SQL 多数把信用分给完整 SQL、文字 reasoning 或由 gold SQL 导出的
> 代理结构；即使引入多轮执行，也缺少对可验证关系操作、错误恢复和信息获取行为的
> 因果信用。我们研究一种 harness-grounded 的数据库 agent，使每个事实状态变化都可
> replay、可追踪 provenance，并在不把模型自述 reasoning 当作事实、不指定唯一 gold
> 工具路径的条件下，为实际工具轨迹分配过程信用。

这个主张要成立，需要先通过两个反证门：

1. 在预算匹配下，typed tools 至少在某类需要探索、OOD 或复杂恢复的任务上超过
   SQL-native 多轮 agent；
2. grounded process credit 相比 outcome-only RL 带来配对提升，而不是只改善合法率、
   轨迹长度或 reward proxy。

如果这两个门能通过，当前 SFT 不涨点并不是项目失败，而是揭示了一个更精确的问题：
**仅靠成功轨迹模仿，无法让小模型学会可泛化的数据库探索策略；真正需要学习的是
在自身访问的状态分布上，哪些可验证操作会提高最终成功概率。**
