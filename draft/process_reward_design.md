# 过程奖励函数设计 — draft / SSOT 草案

> 草稿,待完善(2026-06-28)。本文件是"奖励**怎么设计/怎么实现**"的 SSOT;
> "**为什么**稠密+高保真过程奖励是创新点"的形式化在 `draft/process_reward_density.md`。
> 约定:公式纯文本/ASCII(查看环境不渲染 LaTeX)。
>
> 状态标注:
> - 【已有】`provenance.backward_slice` / `build_references`、感知工具
>   (`preview`/`read_subtable`/`describe_table`/`inspect_column`)、`env.py`、`reward.py`(occurrence 型 pilot)。
> - 【待造】`build_preconditions`、`grounds()`、`Φ_ground`、grounding 边(V2c deferred)、self-slice 奖励接线。

---

## 0. 一句话哲学

> **奖励只奖两样:答案对不对(路径无关的锚)+ 每步对那个对/错贡献多少(沿因果链摊开的稠密项);
> 绝不奖"调用了哪个工具""调用了几次"。**

两条总原则贯穿全文:
1. **路径无关**:同一问题的 gold SQL 有多条等价分解。任何项都不对齐**单条** gold 路径——锚只验"答案",过程信用切"模型自己的轨迹"或"模型自己下一步的参数"。
2. **训练用便宜代理,验证用昂贵金标准,两者分开**:训练用 provenance 切片 / 势能;验证(测 ρ_fid)用反事实 / Shapley / OPE。反事实**只当裁判,不进训练**(它是通用度量,不是创新点)。

---

## 1. 三层奖励架构(全部路径无关)

记一题:gold 答案 `A*` = gold SQL 执行 denotation;模型轨迹 τ 每步有 harness 写的
`references` 边(`provenance.build_references`)和 `step_id`(见 `env.py`)。

### Layer 0 — 终端验证器(锚,主导项)
```
R_term = +1     denotation(模型答案) == A*    (执行验证;任何合法分解都给 → 路径无关)
         -ε_w   legal 但答错        (默认 ε_w=0.2,对齐现 reward.py terminal_wrong)
         -ε_f   illegal             (protocol/exec/context_overflow/max_steps)
```
比任何过程项高一个量级。下面所有项都是它的 shaping,**不得改变它定义的最优解**。

### Layer 1 — 执行信用:正确轨迹的 self-slice(离线、终端条件)
```
当 R_term = +1:
   S = backward_slice(τ)              # 模型自己这条轨迹的答案依赖集(provenance.py 已实现)
   每步 a_t:  +β   若 t ∈ S          # on-slice:对正确答案有贡献
              -β'  若 t ∉ S          # off-slice:可证废步(soundness: t∉Slice ⇒ c_t=0)

当答错:  不做 self-slice(错答案的切片 = "高效奔向错误")
         → 失败任务的部分信用走 Layer 2 势能 + critic 的"进展",不走 self-slice
```
self-slice 是 c_t 的 **sound 廉价代理**;多样轨迹自动解决(每条合法路径切自己的步)。
`backward_slice`、verifier 都已就绪 → **这一层今天可接**。

### Layer 2 — 观察信用:grounding 势函数(在线、第二条护城河)
观察的价值 = 它把"被需要、且当前不确定的前提"变成"已知"。势能 = 已 ground 的需要前提占比。
详细实现见 §2。PBRS 保证策略不变 + telescoping 防 farming。**依赖 V2c grounding 边 → reward-v2。**

### 合成与正则
```
R(τ) = R_term                                          # Layer 0 锚(主导)
     + Σ_t exec_credit(a_t)                            # Layer 1 self-slice(仅正确)
     + Σ_t [γ·Φ_ground(s_{t+1}) - Φ_ground(s_t)]       # Layer 2 观察 VoI
     + Σ_t [γ·V(s_{t+1}) - V(s_t)]   (可选)            # learned critic:在线 + 失败任务进展
     - c_step·T - c_rep·#重复调用                       # 正则(保留现 reward.py 的 step_cost/repeat_call)
```

### 必须从现 `reward.py` 删掉的项(farming 源)
```
valid_tool_call: 0.01           → 删(按次计,describe_table 提款机)
read_evidence_before_answer:0.05→ 删(直接奖励了实测的 read_subtable 仪式:V2-ctx 6361 次全是倒数第二步)
保留: terminal_correct / terminal_wrong / step_cost / repeat_call;其余换成 Layer 1/2。
```

---

## 2. Layer 2 观察势能的具体实现(锚定真实代码)

Φ_ground 本质是三件 bookkeeping:① 编译期抽"需要 ground 的前提集 N";
② rollout 期每次观察判定"它 ground 了哪个前提";③ 维护集合算 Φ 与逐步奖励。

### 2.1 condition tree / args 的真实形状(来自 `compiler.py::_condition`)
filter 的 `conditions` 叶子谓词(决定哪些前提需要观察 ground):
```
{column, op, value}            # 字面量谓词        ← 唯一需观察 ground 的(值域不确定)
{column, op, value: str}       # 字符串字面量      ← 同上
{column, op, column_value}     # 列=列            ← 无字面量,不需值观察
{column, op, value_ref: sid}   # 标量子查询        ← 由"计算"ground(value 通道),非观察
{column, op:"in", values:[...]}# IN 字面量         ← 字面量,需 ground
{column, op:"in", in_table:sid}# IN 子查询集合      ← 由计算 ground
{column, op:"between", low,high}# 区间字面量        ← 需 ground
{column, op:"like", value}     # LIKE 字面量       ← 需 ground
{column, op:"is_null"}         # 无值
{and|or:[...]} / {not:...}     # 布尔树,递归
```
join 的 `on`(来自 `_join_on` / `_join_on_internal`):`[{"left": key, "right": key}]` → join 键前提。

### 2.2 `build_preconditions(plan)` — 编译期抽 N 【待造】
遍历 gold Plan(`list[Step]`),对每个 step 的 args:
```python
def build_preconditions(plan):
    N = []
    for step in plan:
        if step.tool == "condition_filter":
            for leaf in walk_conditions(step.args["conditions"]):   # 复用 plan._value_ref_ids 的遍历骨架
                if has_literal(leaf):     # 含 value / values / low/high / like-value,且 column 是基础列
                    N.append(("literal", table_of(step), leaf["column"], literal_of(leaf)))
                # column_value / value_ref / in_table / is_null → 不进 N(非值域不确定)
        if step.tool == "join_tables":
            for pair in step.args["on"]:
                N.append(("join", pair["left"], pair["right"]))
        # 列前提:任何 column 引用 → 多数已被 dataset_overview 解析 → 默认不进 N(见 2.6 门控)
    return N
```
`N` 是该题的小集合,编译时算好,随题塞进 env。纯遍历 args,确定性。

### 2.3 `grounds(p, tool, output)` — 匹配到真实感知工具 【待造】
映射到 `executor.py` 真实工具及其返回结构:
```
inspect_column(table,column) -> {column, distinct_count, has_null, frequent_values, truncated}
   → ground ("literal", table, column, value):  output.column==column 且
     (value ∈ frequent_values  或  not truncated)        # 值域被看见 → 能确认拼写/存在性
describe_table(tables) -> {tables:[{columns:[{name,type,pk}], foreign_keys:[{column,references}]}]}
   → ground ("column", table, column):  column ∈ 该表 columns.name
   → ground ("join", left, right):       该 fk 出现在 foreign_keys
preview(table) -> {columns, rows, ...} / read_subtable -> 行
   → ground ("literal", ...):  若该列在 columns 且 value 出现在 rows 的该列(弱形式,谨慎用)
```
关键:**`describe_table` 只有 schema、没有值 → 不满足 literal 前提**(这就是"刷 describe 拿不到分"的机制)。
ground 字面量的正主是 **`inspect_column`**(它的 `frequent_values` 就是为 ground filter 字面量设计的,见其 docstring)。

### 2.4 Φ_ground + 逐步奖励 + env 钩子 【待造】
```python
# env.reset():  self.N = build_preconditions(self.gold_plan);  self.grounded = set()
# env.apply_model_output() 中,工具执行后:
def phi(self):
    return (len(self.grounded & set(self.N)) / len(self.N)) if self.N else 0.0
if is_observation(tool):                       # preview/read_subtable/describe_table/inspect_column
    before = self.phi()
    for p in [p for p in self.N if p not in self.grounded]:
        if grounds(p, tool, output):
            self.grounded.add(p)
    turn["reward_obs"] = GAMMA * self.phi() - before     # 势能增量
```
就嵌在现 `env.py::apply_model_output` 的工具分支里,多维护一个 `grounded` 集合。

### 2.5 一条 worked trace(问题 "How many singers are from France?")
```
gold Plan:  s1: condition_filter(table=singer, conditions={column:country, op:'=', value:'France'})
            s2: aggregate(table=s1, op=count)
build_preconditions → N = { ("literal","singer","country","France") }   # 列/表在 overview → 不进 N

A 先观察(好): inspect_column(singer,country)→frequent_values含'France' → grounded={p1} Φ:0→1 reward=+1
              filter(country='France') Φ:1→1 reward=0 ; aggregate→答案对
B 直接猜(坏): filter(country='French')→0行  没观察 grounded={} Φ:0→0 reward=0 ;终端通道扣分
C 刷分(被堵): describe_table(singer)→只有schema  grounds(p1)=False Φ:0→0 reward=0   ← 刷不到
              inspect_column 第一次 →Φ:0→1 +1 ; 再来一次 → p1已在集合 ΔΦ=0 reward=0  ← telescoping
```

### 2.6 两个变体(对应 reward-v1/v2 与"实时")
```
(a) gold-前提势能(离线): N 取自 gold 执行步参数(§2.2)。可编译、定义清晰 → reward-v2。
(b) self-前提势能(在线、路径无关): N 取自【模型自己下一步要执行的参数】,
    gold 只用来提供"正确下一步 a*",配合 Direction C(§4)测"观察把策略推向 a* 多少"。
```
**⚠️ 唯一"软"的一环:不确定性门控** —— 一个前提是否"开放/需要观察",
即 `resolved(p, s_t)`(当前可见上下文能否解析它)。
- 列/表前提:多在 `dataset_overview` 里 → resolved → 默认不进 N(否则退化成仪式)。
- 字面量前提:overview 没有值 → open → 进 N。**这是 VoI 最高、可干净实现的一类。**
- 门控的精确判据、以及 column/join 的 `grounds` 规则,是本设计**留待用户继续思考**的部分。
  建议:**reward-v2 先只对 literal 前提实现 Φ_ground**,跑通再扩 column/join。

---

## 3. 性质小结(为什么这套同时满足前面所有约束)

| 担忧 | 解 |
|---|---|
| 反事实太通用、非创新 | 只当**验证**(§5),不进训练;训练用 sound 的 slice/势能代理 |
| gold 只有执行轨迹、学不会观察 | 观察走独立 VoI/grounding 通道(Layer 2),不靠 gold 执行切片 |
| gold 路径刚性、同问多轨迹 | 锚只验**答案**;Layer 1 切**模型自己**轨迹;Layer 2 (b) 锚**模型自己**参数 → 全程路径无关 |
| 奖励太稀疏 | Layer 1/2 把终端 1 比特沿因果链摊成稠密 |
| 失败任务无信号 | Layer 2 势能 + critic 给"进展";Layer 1 只在正确时开(条件化) |
| 简单工具刷分 | 删按次项;PBRS telescoping + VoI 门控 + advantage 自动压平易命中动作 |

farming 与"学不会观察"是同一旋钮两端,**VoI 门控同时解决**:有 VoI(不确定)→ 奖 → 学会;
冗余(已知)→ ΔΦ=0 → 刷不动。

---

## 4. 实时奖励的创新角度(给导师的方向)

"dense 比 sparse 高效"是通用结论,**不能当卖点**。真创新来自你的环境让实时奖励能做到别人做不到的:

- **Direction A —「环境即逐步验证器」**:harness 在工具调用当下确定性地知道合法性/grounding/on-slice,
  给出**执行接地、逐步可证 sound** 的实时过程奖励,而非学出来的 PRM 代理(结构化 vs 统计)。
- **Direction C —「观察的实时价值 = 它当场把策略推向正确动作多少」**(最深、最贴项目):
  ```
  r_obs(t) ∝ log π(a* | s_t^+) - log π(a* | s_t^-)
            = 这次观察把"正确下一步 a*"的概率质量抬高了多少(realized information gain)
  ```
  只有在线才成立(要看观察前后的策略分布);code-agent 没有可分离观察动作,结构上算不出。
- **Direction D —「自验证环境上的 test-time RL」**(回应"推理时更新参数"的字面诉求,最激进):
  harness 在测试时**无需标签**就能给 sound 奖励 → 测试时自跑几条 rollout、几步梯度再给答案。

注意术语:标准 RL 里"实时/online 奖励"**不在生成途中更新参数**(生成与梯度更新是两相);
"边推理边改参数"是 test-time training(Direction D),是另一回事。

---

## 5. 验证协议(与训练分开)
训练用便宜代理(slice/势能);**验证**用反事实/Shapley/OPE 估每步 c_t,测
`ρ_fid = corr_t(r_t, c_t)`,跑 A(code-agent RL)vs B(本方案)的对照。这条是论文裁判,不进 reward。
(详见 `process_reward_density.md` §3/§7。)

---

## 6. 落地顺序(诚实区分今天能做 vs 依赖 V2c)
```
reward-v1(今天可做): Layer0 锚 + Layer1 self-slice + 删 farmable 项 + 正则        ← 全离线
                     依赖件已就绪:backward_slice、verifier、env.turn 结构
reward-v2:           加 Layer2 grounding 势能(先只 literal 前提)
                     依赖:V2c grounding 边 + build_preconditions + grounds() + VoI 门控
reward-v3:           加 learned critic(真在线 + 失败任务进展);多正确 rollout 切片求交做鲁棒证据
```
建议:**先实现 reward-v1**,验证"self-slice 过程奖励 > 现 occurrence 奖励"这个最小命题;v2/v3 待 grounding 边到位。

---

## 7. 待定旋钮(留待继续思考完善)
1. **不确定性门控** `resolved(p, s_t)` 的精确判据(§2.6)—— 最核心、最易退化成仪式的一环。
   现实路径:注入不确定性的训练分布(歧义 schema、'France'/'French' 不匹配)使其可判。
2. **`grounds(p,·)` 对 column / join 前提的规则**(literal 已可干净实现)。
3. **off-slice 惩罚 β'** 大小 → "最小充分" vs "充分且鲁棒"。
4. **slice / grounding 粒度**(cell/row/column/memory)→ 稠密度 vs 噪声。
5. **失败任务部分信用**:critic-only,还是用"多正确 rollout 切片求交"的鲁棒必要集做参照。
6. Layer 1/2 各自的权重 β、势能折扣 γ,以及与终端量级差(保持终端高一个量级)。

---

## 8. 相关代码与文档索引
- 形式化(为什么):`draft/process_reward_density.md`
- provenance 切片 / 边:`src/harness/provenance.py`(`backward_slice`、`build_references`)
- 编译器 / Plan IR:`src/harness/compiler.py`(`_condition`、`_join_on`)、`src/harness/plan.py`(`Step`、`TABLE_REF_ARGS`、条件树遍历 `_value_ref_ids`)
- 感知工具真实返回:`src/harness/executor.py`(`inspect_column`/`describe_table`/`preview`/`read_subtable`)
- RL 环境 / 现 pilot 奖励:`src/rl/env.py`、`src/rl/reward.py`
- 观察作为上下文管理(非 memory)+ 反命名:`draft/context_management_design.md`、`draft/subtable_vs_memory.md`
- 观察数据构造方向:`draft/reflection_trajectory_data_plan.md`;provenance/V2c:`draft/provenance_redesign.md`
