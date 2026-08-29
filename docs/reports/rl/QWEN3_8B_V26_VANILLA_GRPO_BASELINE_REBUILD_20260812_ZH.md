# Qwen3-8B atomic v26 vanilla GRPO baseline 重建

日期：2026-08-12

状态：实现与无更新 rollout 门禁已完成；representative cohort 被门禁拒绝；policy-boundary S1 screening 正在 `table_rl` GPU0/1 与 NewGNN GPU6/7 上按冻结四分片合同运行。尚无可报告的 RL 性能提升。

## 结论先行

旧实验没有证明 GRPO loss 写反或模型完全没有更新。更直接的失败来源是：

1. Qwen3 SFT1 是冻结 atomic `version26`，而旧训练入口可能从当前 checkout 导入 `version39` prompt/runtime；训练与评测合同不一致。
2. Qwen3 thinking 开关曾以错误的 tokenizer 参数层级传入，配置没有真正绑定 chat template 行为。
3. 旧训练只有 60 题、K=8、6 个 optimizer updates，而且这 60 题按 Qwen2.5 策略的 mixed outcome 选择，并非 Qwen3 SFT1 的策略边界题。
4. 旧的 `{1.0, 0.2, 0.0}` reward 会把“合法但答案错误”当正信号；终局 advantage 又广播到多轮轨迹的所有 authored token，容易强化成功恢复前的错误动作。
5. 1024-token rollout 截断、context overflow 和 timeout 曾可能作为 reward=0 的语义失败进入 GRPO。

当前重建的 baseline 只验证最基础的 binary terminal-reward GRPO：不使用 process reward、rank loss、schema reward、自研 credit 或 KL。只有 matched full-dev 结果通过预注册门槛后，才能声称 baseline 有提升。

## 冻结 vanilla 合同

- 初始策略：Qwen3-8B atomic v26 SFT1 checkpoint-560。
- protocol：`version26`，hash `4da19387399bd3a5`。
- student prompt SHA-256：`848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`。
- reward：binary denotation correctness，只有 `{1, 0}`。
- GRPO：K=8，temperature=0.8，top-p=1.0，PPO iteration=1，clip=0.2。
- optimizer：AdamW，FP32 LoRA trainable tensors/optimizer moments，LR `8e-7`，constant schedule，weight decay 0.1。
- rollout：最多 30 agent steps、每 turn 2048 new tokens、16384 context、recent-4 causal history、Qwen3 thinking=true。
- reduction：whole-trajectory token mean；同质 K8 group 的 standardized advantage 严格为零。
- KL：baseline `beta=0`；非零 KL 的 frozen-SFT reference 能力仅保留给后续独立 arm。

## Representative first32 probe 结果

输入是从 BIRD train 合法 universe 按 DB、问题长度与 external knowledge 做代表性分层的 600 题；它排除了 held-out baseline300、SFT1 source tasks 与旧 mixed60。probe 使用其中冻结顺序的 first32、初始 SFT1、K=8、零 optimizer updates。

结果：

- 32 groups / 256 trajectories 完整生成；181 条 correct。
- 249/256（97.27%）semantic-eligible。
- 11/32 mixed groups，低于预注册门槛 20/32。
- correct-count histogram：`0:5, 1:2, 2:1, 3:1, 5:1, 6:1, 7:5, 8:16`。
- 7 条 `generation_length`，来自 3 个 task；全部被排除，没有当作 reward=0 训练。
- context overflow=0，timeout=0。

因此 representative-600 formal train 被 fail-closed 拒绝。16/32 题是 8/8 correct、5/32 是 0/8 correct；这些同质组对 binary GRPO 提供严格零 advantage。这个结果直接支持“旧 cohort 对当前 Qwen3 没有足够训练信号”，而不是“GRPO 数学实现完全不更新”。

远端只读产物：

```text
/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_vanilla_grpo_20260812/
  probe_first32_k8_seed20260812/
    groups/
    manifest.pending.json
    trajectories.jsonl
  logs/probe.log
  logs/probe_generation.log
  status/probe.status
```

## Policy-boundary fallback v1

只有 representative probe 的 identity/eligibility 合同成立、但 mixed 数不足时才允许进入该 fallback；当前条件满足。

1. S1 对冻结 representative600 全量做 initial-SFT1 K8 screening；first32 逐字节复用。旧双卡 run 已原子提交的 37 个后续 group 也只读继承；其余 531 题按冻结 continuation plan 在 `table_rl` GPU0/1 与 NewGNN GPU6/7 四个互斥分片生成。六个仅 `table_rl` 具备数据库副本的 DB，其任务只分配到前两个分片。
2. 每组必须 8/8 trajectories 都 semantic-eligible，且无 generation length/OOM、context overflow、structured timeout、identity/logprob 缺失，才可参与选择。
3. `c=#correct`。`1<=c<=7` 是 mixed boundary；`2<=c<=6` 是 core boundary；排序 uncertainty 为 `c*(8-c)`，即 4 优先于 3/5、2/6、1/7。
4. S1 只有在 mixed>=360 且 core>=240 时直接选择；否则才允许启动预先隔离、与 S1/SFT/dev/旧 mixed60 均不重叠的 extra600 S2。S1+S2 仍达不到两个阈值就停止，不降门槛、不启用第三池。
5. 在相同 uncertainty bucket 内，按公开字段 DB、question-length quintile、external-knowledge 的分布缺口做确定性平衡，从 qualifying groups 一次冻结 332 题；再固定 hash split 为 train300 + validation32。selector 不读取 gold SQL/result。
6. validation32 用初始 SFT1 和独立 seed fresh K8；要求 mixed>=20/32、256/256 trajectories 都 semantic-eligible，且无 generation length/OOM、context overflow、structured timeout 或 identity 污染。失败后不换题、不补抽。
7. formal train 使用 train300，TRL 0.29 `RepeatSampler` 在固定 data seed 下产生两次确定性 shuffled pass；每一遍 300 个 identity 恰好出现一次。30 prompts/update、20 updates、K8，共 4800 fresh on-policy trajectories。step20/final 是唯一 primary；step10 仅作训练诊断。

当前冻结 S1 原始根目录与四分片 continuation 根目录：

```text
/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_boundary_screen_s1_2gpu_20260812
/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_boundary_screen_s1_crosshost_v2_20260812
```

continuation plan 在启动前绑定了旧 69 个完整 K8 group、其余 531 个 task、四份互斥 assignment 及各自 SHA。跨主机 group 只以同目录原子发布后的 `TASK.json` 为完成；`.json.next` 不可晋级。最终还必须在 `table_rl` 的新 staging 目录中逐组校验并恰好合并为 600 groups / 4800 trajectories，旧 S1 目录不会被就地覆盖。

## 预注册的 Arm B（只在 Arm A 训练前失败时启封）

这部分在 full S1/S2 outcome 揭晓、任何新 RL 训练和任何 confirmatory full-dev 之前冻结，不修改 Arm A 的阈值。first32 中只有 4/32 是 `c=2..6` core boundary；在把这 32 题仅当规划先验、而非正式统计结论时，S1+S2 同时通过现有 mixed/core 门槛的后验预测概率约为 10%–16%。因此预先准备一个更宽但仍是基础 binary GRPO 的独立 Arm B：

1. 触发条件只允许是 Arm A 在训练前因 S1+S2 readiness 失败，或 boundary32 fresh validation 失败而终止。若 Arm A 已训练或看过 confirmatory full-dev，Arm B 永不以本合同启封。
2. 从同一 frozen eligible universe 排除 baseline300、SFT1 unique678、旧 mixed60 及所有实际 S1/S2 screen IDs；仅按 DB、问题长度、external knowledge 与固定 SHA 选择 wide3000，不读取任何 rollout outcome、gold 或 dev。
3. F1：初始 SFT1、独立 seed、K8 筛 3000 题；要求至少 640 个 8/8 clean mixed group，否则停止。按 `c*(8-c)` 与公开轴平衡一次冻结 640 题。
4. F2：对 640 题用第二个独立 seed fresh K16；要求至少 384 个 16/16 clean 且 `c=2..14` 的 robust core。选择 384 后固定拆分 train320 + validation64。
5. F3：validation64 用第三个独立 seed fresh K16；要求 1024/1024 clean、至少 56/64 mixed、至少 48/64 robust core。失败不换题、不补抽、不降阈值。
6. 训练仍固定 binary result-only、beta=0、PPO iteration=1、clip=0.2、LR `8e-7` 和 whole-trajectory token mean；train320、K16、20 prompts/update、两遍 deterministic shuffled pass，共 32 updates / 10,240 fresh on-policy trajectories。step32/final 是唯一 primary。
7. Arm A 与 Arm B 最多一个进入 confirmatory full-dev；两者沿用同一个 fresh matched `>=+1pp + McNemar p<0.05 + legal不降` 门禁，避免事后在同一 dev 上挑实验。

Arm B 的 full-dev 前冻结预算为 45,504 trajectories（24,000 + 10,240 + 1,024 screening/validation，另加 10,240 formal training）。它不使用“抽到 mixed 才在线更新”或 reward-conditioned replay；formal rollout 仍由当前策略正常生成，同质组自然产生零 advantage。

## 已修复的实现问题

- tokenizer `enable_thinking` 改为 Hugging Face chat template 的顶层 Jinja kwarg。
- 训练入口在导入 rollout/environment 前 bootstrap 冻结 v26 runtime，并校验 101-file runtime tree SHA 与实际 module paths。
- 新增隔离 `tool_environment_v26.py`，训练 overlay 不得 shadow frozen `eval/sft/harness/tool_modules`。
- result-only 路径不再 eager import process-credit/current-harness 模块。
- binary reward 不再给错误但合法的 terminal answer 0.2。
- `generation_length`、generation OOM、context overflow、终局或 recovered structured timeout 会保留审计证据，但整条 trajectory 退出 GRPO normalization/backward。
- policy logprob、vLLM sampling logprob、group advantage、trajectory-token weighting、LoRA/vLLM sync 与 importance-ratio diagnostics 均有 CPU/GPU smoke 覆盖。
- 可选非零 KL reference 改为冻结的初始 SFT1 adapter，而不是裸 Qwen3 base；本 baseline 仍固定 beta=0。

## 唯一性能判定

训练完成后，必须 fresh 顺序运行 SFT1 与 final 两臂，使用完全相同的 frozen v26 runtime、BIRD dev1534 input、base revision、24 concurrency、vLLM serving、thinking=true、temperature=0、top-p=1、2048/16384 token contract 和 10s SQLite timeout。

final 是唯一 primary；中间 checkpoint 不参与挑选。晋级同时要求：

- full 1534 matched evaluation 身份有效；
- final accuracy 至少比 fresh SFT1 高 1.0pp（净增至少 16 题）；
- exact two-sided McNemar `p<0.05`；
- legal termination 净变化不为负。

未过门槛只能报告“在冻结预算与 cohort 下未检测到可靠提升”，不能挑最好 checkpoint，也不能据此断言 vanilla GRPO 普遍无效。

## 运行边界

- 旧 `table_rl` RL/vLLM 实验已停止。
- 当前 `table_rl` 运行本报告的 boundary screening 分片 0/1；NewGNN 运行分片 2/3。四个 worker 都使用相同的 frozen v26 runtime、SFT1、K8、seed 与 decode 合同。
- NewGNN 上原 checkpoint4 full-dev evaluation 未被停止、未收到信号，并已于 1534/1534 自然完成后自行退出。结果为 830/1534（54.11%）；它与历史 SFT1 不是 fresh same-runtime/concurrency 对照，因此只是旧诊断，不是本 baseline 的 matched primary。
