# 过程奖励的稠密性：形式化思考（draft）

> 草稿 / 讨论记录，非定稿。日期 2026-06-06。
> 主题：为什么"稠密、有据的过程奖励（dense, high-fidelity process reward）"可能是本方案相对现有表格 RL 工作的核心创新点，以及如何把"稠密性"形式化。
> 约定：所有公式用纯文本/ASCII（当前查看环境不渲染 LaTeX）。

---

## 0. 核心创新点假设（the bet）

不要在"表达力 / 准确率"上和 code/SQL agent 硬碰（会输：我们的工具表达力 ⊆ SQL，而 NL2SQL、TableGPT-R1 已证明 code/SQL 执行在 8B 规模就能接近 SOTA）。

真正可立的论点是关于**学习环境与信用分配**，而非任务求解能力：

> 抽象工具 + 显式状态，构成一个能定义并计算"稠密、高保真过程奖励"的环境；
> 而执行式（SQL/pandas）agent 在结构上做不到这一点。

"过程奖励是否真的能做稠密、有据"是这套方案能否比现有 RL 工作更有创新性的关键。本文件是对"稠密性"形式化的初步思考。

---

## 1. 对照：TableGPT-R1 的奖励方式

三层：
- 终端奖励（任务自适应路由）：确定性任务（SQL/数学/带标签）→ 规则/执行验证；开放任务 → criteria-injected 奖励模型（强 teacher 先生成评分标准，LLM judge 按标准打 0–10）。
- 过程步奖励（中间步 t<T）：选对函数且执行成功但未结束 +0.1；选错函数 -0.2；选对但执行失败 -0.1。
- 聚合：R(tau) = sum_t ( R_base(t) + R_reg(t) )，R_base(T)=终端验证，R_base(t<T)=过程步奖励，R_reg=长度/重复/无效反思惩罚 + need_plot 反作弊。

关键洞察（也是我们的机会）：
它的过程奖励衡量的是"这一步本身合不合法、跑没跑通、有没有推进"（local validity / progress），
**不是**"这一步对最终正确答案有没有贡献"（causal contribution）。
即：覆盖率高，但保真度约等于 0。TableGPT-R1 自己也承认 agentic 轨迹的长程 credit assignment 很脆。

---

## 2. 为什么 SQL/代码执行"不稠密"——结构性理由

不是"语义模糊"，而是结构性的：

> 过程级贡献，只有当中间状态被物化成"可寻址对象 + 带来源边（provenance）"时才有定义。

- 一条 SQL 是一个原子动作：吐一次、拿一个结果，中间没有可奖励的步 → 天然只有终端奖励，最稀疏。
- 多条 SQL：每条产出一张表，但 SQL 优化器把 WHERE/GROUP BY/JOIN 融合成一次执行，子操作的中间关系态不被物化 → "这个 WHERE 对答案贡献多少"无从定义。
- pandas blob 同理：一段代码内部的中间 DataFrame 不是可寻址、可单独奖励的对象。

我们的设计：每个工具 = 一步 = 一次显式状态转移，输入（哪些行/列）和输出（新可见态 / memory）都是可寻址对象 → 逐步 provenance 和逐步贡献都良定义。这是"稠密过程奖励"能成立的结构前提，是显式状态环境独有的。

---

## 3. 形式化"稠密性"：两个正交轴

把"稠密"拆成两维，别当成一个标量：

```
覆盖率  rho_cov = (有信息量的步数) / T
        = (1/T) * #{ t : r_t 非零且与决策相关 }

保真度  rho_fid = corr_t( r_t , c_t )
        即 "每步奖励 r_t" 与 "该步真实贡献 c_t" 的相关性
        （或 rho_fid = 1 - mean_t |r_t - c_t| / 归一化项）
```

- r_t = 给第 t 步的奖励；c_t = 第 t 步的真实贡献；T = 轨迹步数。
- 好的稠密奖励 = 高 rho_cov 且 高 rho_fid。
- TableGPT-R1 = 高覆盖、低保真；我们的目标 = 高覆盖 + 高保真。
- rho_fid 就是我们和它的差距，可用消融实验直接测量（见 §7）。

真实贡献 c_t 用反事实定义：

```
c_t = Pr(答对 | 保留第 t 步) - Pr(答对 | 抹掉/中性化第 t 步)
```

更严格：对"步的联盟"求 Shapley 值（在所有步子集上平均第 t 步的边际贡献）。

---

## 4. provenance 后向切片：核心机制（把"反推最短路径"的直觉做严谨）

文献名字（便于查阅）：
- 从答案引用证据往回追每个数据是哪步产生的 = 数据来源 / 血缘（data provenance / where-provenance，Buneman 等）；
- 反推出"答案依赖的步集" = 程序切片（program slicing，Weiser）的后向切片；
- 整体上是用结构近似反事实贡献；与势函数塑形（potential-based shaping，Ng 1999）和因果/事后信用分配（hindsight credit assignment）相关。

核心可靠性性质：

```
设 Slice(E) = 答案证据 E 在来源 DAG 上的后向闭包（答案依赖的步集）

性质（soundness）:  t ∉ Slice(E)  =>  c_t = 0   (对"所产生的那个答案"而言)
```

含义：切片外的步，可证明对答案零贡献 → 这是一个可靠的"废步检测器"。
反向不成立：t ∈ Slice(E) 不代表 c_t > 0（该步输出可能与默认值无异、可被替代）。
所以：**切片精确界定"废步"，只粗略界定"关键步"。** 关键性还需反事实/Shapley 进一步敲。

provenance 的本质：它不是凭空造的稠密奖励，而是把稀疏的终端正确性信号、沿因果线分发到各步的机制（一个结构化的信用分配 / 势函数塑形实现），因此天然不改变最优策略。

---

## 5. 必须钉死的设计选择 / 开放问题

1. 必要 ≠ 关键（necessity vs pivotality）
   切片给的是"有用候选集"，不是"关键集"。要不要再用反事实/Shapley 区分"关键"和"可替代"，是一个精度 vs 成本的取舍。

2. "最短路径"是欠定义的
   - 最短按什么度量？步数？
   - 最小充分子图通常不唯一（多条等价证据路径）→ 正是 agent.md 弱点 #1 担心的"多条正确路径取交会欠奖励"。
   - "最短"未必"最好"：冗余证据有时带来鲁棒性。
   - 必须先决定目标：是"最小充分"（效率导向）还是"充分且鲁棒"（可靠导向）。两者奖励设计不同。不要默认"最短=最优"。

3. 必须以"答对"为条件（correctness conditioning）
   provenance 解释的是"模型产生的那个答案"，不管对错。答案错时，反推出的"必要步"是"高效走向错误"。所以 provenance 奖励必须与终端正确性（verifier）耦合。

4. 粒度即旋钮（granularity）
   cell / row / column / memory 级 provenance 给不同分辨率的信用；细 → 更稠密但更噪。状态已支持这几级引用，所以"粒度"是一个可形式化、可消融的超参。

5. soundness 易证、completeness 难证
   "切片外=零贡献"（soundness）容易证；"切片内都有用"（completeness）一般不成立。要明确论文里证到哪一步。

---

## 6. 连带好处（同一个 Slice 的复用）

1. 一次切片，喂两个奖励：
   - 证据获取奖励：有没有取到 Slice(E) 内的必要证据；
   - 剪枝奖励：drop_context 删的是不是 Slice 外的废行。
     ```
     r_prune = #(删除的废行) - lambda * #(删除的必要行)
              其中"必要行" = 落在 Slice(E) 内的行
     ```
   代码 agent 没有"从上下文移除"这个显式动作，这个奖励它结构上无法表达。

2. 自监督的 gold 证据标签：
   缺"标准证据"标注时，用正确轨迹的切片当伪标签（或多条正确轨迹切片取交/并）。
   闭合 agent.md 里"无标注下能否得到证据伪标签"这个开放问题。

3. 粒度作为消融维度（见 §5.4）。

---

## 7. 形式化目标（thesis，纯文本版）

```
把"过程奖励稠密性"定义为 (rho_cov, rho_fid)，
其中 rho_fid = 逐步奖励 r_t 与反事实贡献 c_t 的相关性；
用 provenance 后向切片做高保真信用分配的廉价可靠代理（可靠判废步），
用反事实/Shapley 消融做 rho_fid 的评估金标准。
论点 = "我们的环境让高保真过程奖励可定义、可计算，而执行式 agent 做不到"。
```

测量协议（怎么"证"而非"嘴硬"）：
- 同一 base（如 Qwen3-8B），两臂：A = code-execution RL（复刻 TableGPT-R1 式）；B = 抽象工具 + 显式状态 RL。
- 在带证据标注的样本上，对每步做反事实消融估计 c_t，计算两套奖励各自的 rho_fid。
- 预期：B 的 rho_fid 显著高于 A（A 的过程奖励是 local validity，与 c_t 近乎不相关）。
- 同时比较：证据引用忠实度、样本效率/学习曲线、对新操作组合的泛化、reward hacking 频率。
- 目标定位：准确率持平 + 忠实度/效率/可解释性更优（而非单纯刷准确率）。

---

## 8. 下一步要先定的三件事

1. 目标是"最小充分"还是"充分且鲁棒"。
2. 信用分配的粒度（cell / row / column / memory）。
3. provenance 切片与反事实贡献的一致性证明到什么程度（soundness 容易，completeness 难）。

定了之后，可以把奖励函数和 rho_fid 测量协议写成形式化草稿。

---

## 相关概念 / 待读

- Potential-based reward shaping（Ng, Harada, Russell 1999）—— 势函数塑形不改最优策略。
- Data provenance / lineage（Buneman 等；why/how/where-provenance）。
- Program slicing（Weiser）—— 后向切片。
- Shapley value credit assignment / counterfactual credit assignment / hindsight credit assignment。
- 对照工作：TableGPT-R1（arXiv 2512.20312，code-execution RL，过程奖励粗）；NL2SQL agentic（SemEval-2025 Task 8 系统描述，无训练）。
