# RL 性能优化与启动前检查

更新：2026-09-14。本文是项目的性能优化共享记录；每次新建或恢复 RL 实验前必须阅读。
实验策略准入仍由 `final_project_contract.md` 和 `decision_register.md` 决定。

2026-09-14 后续错误半权重4B对照已锁定原4-bit/SDPA/checkpointing/actor/1row-4096/Graph配置，
preflight和credit数值一致性通过；训练已完成4/4 updates并生成checkpoint-4，rollout correct rate为
`0.408333/0.429167/0.370833/0.437500`。checkpoint-4评测的前两次启动分别因导入路径和共享编译
缓存损坏失败，未影响训练；第三次独立根已完成双卡分片和1534题合并，结果为902/1534正确、
1359/1534合法。见decision register与`qwen3_4b_later_error_half_saam60_20260914_r1`。严格
fresh replay gate仍未通过；随后完成`timeout-barrier replay audit v1`，确认唯一cascade为timing-sensitive，
diagnostic gate通过但promotion gate继续关闭。

### 2026-09-14 夜间 first-error-capped 对照（已完成，未准入）

对已完成 correctness-only SAAM 的 960 条 frozen rollout 做了 gold-free 静态 credit 审计：
当前 A 的正确轨迹负向 coefficient mass 为 104.14，首个非 timeout deterministic Harness
error 封顶的 B 为 95.08；正确负向 transition 68→61，封顶 30 个后续 error transition，
其中 7 个来自最终正确轨迹，错误轨迹正向 credit 未增加。静态 gate 通过后，B 已在
`table_rl` 根 `/home/dengyan/tabular_rl_outputs/overnight_first_error_capped_20260914_r3`
启动；同一 checkpoint/cohort、4-bit/SDPA、actor old-policy、KL=0和 CUDA Graph，唯一算法
变化为 `saam-first-error-capped`。训练根已完成4/4 updates并生成 checkpoint-4，adapter SHA-256
为 `5730d15dd91500f096a5343ec15b054d776cde586ea1474f968dae48d99490f2`。匹配评测根
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_first_error_capped_saam60_checkpoint4_graph_actionable_20260914_r2`
的双卡767/767分片已完成，结果为877/1534正确（57.1708%）、1344/1534合法（87.6141%）。
`merged/paired_comparison_provenance.json`确认五臂1534个唯一example index且question/db_id/split零错配，
config/runtime/protocol/feedback/decode一致；`fresh_runtime_replay_admission` gate为true，但底层
fresh runtime audit仍有112个step divergence和4个timeout timing divergence，fresh execution audit
gate为false（9个replay exception、100个derived-legal disagreement，correctness disagreement为0）。
因此 admission通过不等于raw replay diagnostics全部清零；owned cleanup已完成。相对 correctness-only SAAM 为
63/95（净-32，-2.086 pp，p=0.0134012），相对4B SFT epoch4为64/90（净-26，-1.6949 pp，
p=0.0435999），相对three-level SAAM净-20题，相对vanilla净-13题。单次对照显示准确率下降，
不支持推广该 credit 规则，也不改变8B主线；r1/r2为已停止的实现/启动诊断根。

## 已确认的适用范围

| 项目 | 当前代码与证据 | 8B / 改变 RL 策略时的边界 |
|---|---|---|
| vLLM CUDA Graph | 共享 `start_vllm_server.sh` 的 `VLLM_ENFORCE_EAGER` 默认0；4B单组实测生成明显加速 | 可供8B验证；与reward、SAAM、span权重、KL计算独立。8B尚未完成本优化的性能验证；CUDA Graph占额外显存，需重新验证KV cache容量和长轨迹 |
| gradient checkpointing | `--gradient-checkpointing/--no-gradient-checkpointing`；replicated BF16和4-bit默认均开启 | 与reward/credit无关；4B/3090关闭后OOM，不能推广到8B关闭。FSDP仍使用独立activation checkpointing配置 |
| attention backend | `--attn-implementation eager/sdpa/flash_attention_2`；当前CLI默认eager，可由`TABLE_RL_ATTN_IMPLEMENTATION`覆盖 | SDPA在4B单组未见稳定收益；Flash-Attention未验证，不能写成已优化。以实际模型后端和版本为准 |
| old-policy score复用 | `--old-policy-logprob-source actor`为默认；`sampling`为显式诊断 | 会改变old-policy分母以及actor-vLLM重要性校正，不能随策略变更直接沿用；当前实现禁止与非零KL组合 |
| 更大microbatch/token cap | 4B单组1 row/4096比8 rows/8192更快 | 不能仅看显存空闲或microbatch数量；要比较相同有效训练量的完整update时间、padding和峰值显存 |

`VLLM_ENFORCE_EAGER=0`表示允许图捕获/重放，主要降低逐token的CPU调度与kernel launch开销；
它不等于单独安装或启用FlashAttention，也不保证逐token输出完全一致。图捕获会占用额外GPU
内存，见[vLLM官方说明](https://docs.vllm.ai/en/v0.18.2/configuration/conserving_memory/)。
checkpointing通过丢弃部分激活并在反向重算换取显存，见[Transformers官方说明](https://huggingface.co/docs/transformers/grad_checkpointing)。

## 实验启动前必须核对

1. **查实际launcher和隔离runtime，不只看共享源码默认值。** 当前共享 server 和本地
   8B launcher 默认 `VLLM_ENFORCE_EAGER=0`；已在途或冻结runtime不会因本地改动自动更新，
   仍须从该运行的启动日志确认。A100 本轮因被其他用户占用未测，8B证据来自 table_rl 双3090。
2. 核对生效的base storage、checkpointing、attention、old-policy source、microbatch rows/
   token cap、生成并发、context/token上限、模型/adapter/cohort/runtime/implementation identity。
   trainer会记录checkpointing、attention、old-policy source；CUDA Graph的实际生效值仍需
   从server启动参数/日志确认并记入该运行的启动证据，不能只从trainer manifest推断。
3. 8B、更大K/上下文或更换GPU需要代表性长轨迹的完整update验证，包括模型权重同步、前反向、
   optimizer step和峰值显存；评估吞吐必须包含rollout、old-policy、policy和新增算法阶段。
4. 正式策略保留 `actor` old-policy scoring。`sampling`须独立命名和记录为算法ablation；
   不得以节省时间为由静默移除重要性校正，不能宣称它不改变GRPO/PPO更新。

CUDA Graph可继续加速使用同一online vLLM生成链路的vanilla GRPO、SAAM、four-level或
span加权实验。若改成离线固定轨迹池，则没有online rollout可加速；若加入KL、ranking或
critic等额外前反向，生成节省仍存在，但总加速比例会变化。更换训练框架后，当前CLI选项
不会自动迁移，需要检查新框架的生成与训练入口。

## 4B单组性能证据

所有下列结果是table_rl双3090、Qwen3-4B SFT epoch4 `checkpoint-6380`、Atomic v26、新60题
输入内的单题K=8、1 optimizer update诊断，非30题/240轨迹的完整规模update，也非8B结果。
cohort SHA-256：`1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca`。
初始4B adapter SHA-256：`0bc7b8b644ab8aba0c8f0762ff58bba66f5b7e917182bf5ada10e6b3ce522d19`。

| 诊断 | rollout秒 | old-policy秒 | policy前反向秒 | 时间口径与结果 |
|---|---:|---:|---:|---|
| 原1 row/4096 | 203.85 | 22.92 | 74.87 | trace update 303.55秒；37个训练transition |
| 8 rows/8192 packing | 201.01 | 26.60 | 113.27 | trace update 342.73秒；microbatch 37→18，反而慢 |
| checkpointing+SDPA | 206.70 | 22.90 | 75.02 | trace update 306.49秒；37个训练transition |
| CUDA Graph+actor（埋点） | 74.67 | 28.64 | 94.21 | trace update 199.43秒；47个训练transition |
| CUDA Graph+actor（重复） | 74.58 | 28.39 | 93.64 | step_time 198.64秒 / train_runtime 199.95秒 |
| CUDA Graph+sampling | 74.70 | 约0 | 93.94 | step_time 170.49秒 / train_runtime 171.86秒 |
| 关闭checkpointing | 200.17 | 22.89 | 未完成 | 长轨迹阶段约23.25GiB后OOM，不采纳 |

注意：8条是episode数；37/47是进入训练的transition数，不能称为37/47条轨迹。原SDPA
smoke共有48个原始turn而37个进入训练；在线采样内容并不完全相同，也未做完整固定rollout
的梯度等价验证。34%/43%仅为这些smoke的耗时差异，不能线性承诺正式update或8B的加速比例，
不能作为accuracy改善证据。

`sampling`记录的 `rollout/old_policy_sampling_reused=1`，令
`importance_sampling_ratio`（actor-vLLM校正因子）恒为1；PPO中current/old概率比仍会随参数
变化，不能把所有ratio都称为恒1。此实现仅在KL=0的单update smoke验证。

## 2026-09-12 长序列 OOM诊断与SDPA恢复

8B SDPA恢复根已完成至`global_step=4/4`并释放GPU2/3；checkpoint-4 adapter SHA-256为
`207a2b59d25f21122a3dd5d164e4734be7e0f3b8c0933e3a53f2b3909a11d272`。最后240条rollout的离线审计为
209条可训练trajectory、1777个transition，最大`prompt+response`为9771 tokens。其独立
`actionable-error-v1` BIRD-dev1534匹配评测为931/1534，对照checkpoint-6380 SFT的928/1534是
66 gain / 63 regression、净+3题，exact McNemar p=0.8603；这不构成显著能力增益，也不能归因为
SDPA导致变化。table_rl 4B SAAM `r2_sdpa`也已完成`global_step=4/4`、960条rollout和precision审计，训练及
online vLLM资源已释放；其checkpoint-4的独立`actionable-error-v1` BIRD-dev1534评测也已完成。两个恢复根
仍须分别解释，不能合并训练身份。4B SDPA恢复的`actionable-error-v1` BIRD-dev1534为
897/1534、1358合法；相对同题号4B SFT为净-6题（77/83，-0.391 pp，exact McNemar p=0.6928），
相对vanilla RL为净+7题（88/81，+0.456 pp，exact McNemar p=0.6445），均未达到能力准入。

NewGNN 的 8B `c=2–6` recovery 在 checkpoint-3 后的最后 update 再次 OOM。失败批次的
240 条 rollout 中有 215 条可训练 trajectory、1,847 个 transition；离线重建的最大
`prompt+response` 为 15,528 tokens，超过 8,192 的只有 6 个，超过 12,288 的 4 个，且
4 个都属于 `rl_4852_sample_1`。根因不是多行 micro-batch packing 过大：
`build_transition_microbatch_ranges` 为保证进度会单独接纳一条超过预算的 transition。
因此把 token budget 从 8,192 降到4,096不能提供单行上限；失败申请为716MiB时显存仅余约661MiB。

table_rl 的 4B SAAM 三等级诊断在第一批 240 条 rollout 后也复现了同类问题：最大 transition
为13,212 tokens，eager attention softmax 单次申请9.29GiB，尚未提交 optimizer checkpoint。
这条最长 transition属于正确轨迹，不能通过删“错误样本”规避。

两项均另立隔离恢复根，只把 trainer attention backend 切为 `sdpa`；保留 checkpointing、
4-bit、actor old-policy、microbatch=1/4096，以及原 reward/cohort/rollout/context。SDPA
恢复不是 accuracy 结果，也不与 eager 失败根合并；必须等完整 update/checkpoint 和匹配评测
后再判断。若 SDPA 仍 OOM，先保留证据；`max_context_tokens=14336`只作为另立命名根的
备用边界，不自动应用。

## 可追溯证据与实现

远端统一前缀：`table_rl:/home/dengyan/tabular_rl_outputs/`。

## 2026-09-13 4B/旧8B耗时解释与下一轮测速门禁

4B 两条完整四步运行的训练计算几乎相同：vanilla 为 29,842.95 秒（8小时17分23秒），
SAAM-SDPA 为 29,942.40 秒（8小时19分02秒），差异仅 0.33%。因此“有效优势变多导致
SAAM 本身更慢”没有得到支持。SAAM 四步的时间拆分为 rollout 7,307.46 秒（24.4%）、
actor old-policy 前向 5,154.32 秒（17.2%）、policy 前向/反向 17,160.86 秒（57.3%），
其余约 320 秒；训练保留 6,682 条 transition，原始 8,360 条，retokenized 的 prompt+
response padded token 总量约 2,317 万。优势是否为零只决定哪些行进入 policy loss，不能把一次
Transformer 前向变成稀疏计算；只要行保留，仍要做完整序列的前向和反向。SAAM 新增的 mask、
计数和局部系数计算属于秒级 CPU/GPU 张量操作，未形成可观测的端到端开销。

此前文档能核验到的旧 8B 记录是“previous matched training 6h05m”的规划/运行口径，不能
直接当作与当前 4B 四步完全同身份的实测基线；旧单组 8B 诊断也只有一组、有效 transition
不同，不能外推完整 update。因此目前不能把 3–4 小时差额归因于组件优势数量，也不能声称
当前 8 小时已达到硬件理论上限。当前循环是 rollout 与单卡 trainer 的同步流水：一张卡在生成
或 old-policy/反向时，另一张卡可能等待；`nvidia-smi` 瞬时利用率不是 FLOP/带宽 roofline。

后续速度实验只在 table_rl 建立隔离、短时、固定 transition 的门禁，按同一序列长度分桶比较
端到端 update、rollout、old-policy、policy 前反向、峰值显存和 padded tokens：

1. 保持 actor old-policy、KL=0、checkpointing、SDPA 和 microbatch=1；固定 example 的
   CUDA Graph 复测已完成，4B one-group 的 rollout/old-policy/policy 阶段为74.42/28.38/93.73
   秒，完整证据见下文。它仍只代表短诊断，不代表8B完整 update。
2. 对 4B 单独测试 replicated BF16 base（同一 FP32 LoRA/AdamW、同一 transition）与 4-bit
   base；固定 transition 已显示 BF16 可能减少 NF4 反量化开销，但改变数值执行，只有在真实
   batch 的峰值显存和 loss 对齐后才能作为独立速度 arm。不得把该结论外推 8B。
3. 评估“同一 actor 前向同时产出 old/current log-prob”的代码优化；固定 transition 的
   token/gradient 门禁已通过，但仅在 PPO iteration=1、dropout=0、KL=0、无 ranking 且 optimizer
   step 前权重严格相同的条件下注册。下一步必须做真实 batch 集成对齐和一 update benchmark；
   它若成立，理论上最多移除当前 old-policy阶段约17%的工作，不能直接宣称无损。
4. 继续保持 `sampling` 复用为单独算法消融：它在 8B 单组省约 84.7 秒，但把 actor-vLLM
   importance correction 改成 sampling 分母，不能作为正式主线的无损加速。

未完成真实 batch 集成的数值、checkpoint/OOM和实现身份门禁前，不启动新的长 8B/4B 正式 RL；
任何新 arm 都须生成独立 manifest、implementation lock、precision audit 和 `project_records` 记录。

### 2026-09-13 PyTorch profiler 精度门禁

第一份独立 profiler 缺少正式 trainer 的 BF16 autocast，出现FP32/math attention与4096
OOM；此配置失配，不能作为正式训练瓶颈或显存上限。正式路径的
`table_rl:/home/dengyan/tabular_rl_outputs/perf_profiler_4b_20260913_r2_autocast/` 使用
BF16 autocast、4-bit NF4/FP32 LoRA/SDPA/checkpointing，504/504个LoRA参数均有有限梯度。
不启用 profiler 的三次计时中位数为：2048=1.481s、4096=3.016s、8192=6.505s；对应 peak
allocated 为4.92/6.12/8.90GiB，reserved为5.52/6.69/9.96GiB。8192 trace已确认使用
FlashAttention，CUDA self time主要为mm39.91%、flash backward11.53%、flash forward8.81%、
copy12.29%和mul10.48%；Command Buffer Full占CPU self time约45%，但受profiler扰动，不能
单独视为launch瓶颈。该 workload是synthetic last-token loss，不是完整GRPO update；4096/8192
在该路径均未OOM。随后完成同一 adapter/LoRA 参数的 BF16 paired profiler：2048/4096/8192
中位数为1.257/2.591/5.751s，相对4-bit快15.2%/14.1%/11.6%，peak allocated为
8.60/9.55/11.47GiB；12288/14700长度为9.538/12.238s，相对4-bit的10.686/13.516s
快10.7%/9.5%，BF16 peak allocated 13.39/14.52GiB。BF16 仍是改变量化执行的独立 arm，
尚未切换正式训练；长序列 profiler 根分别为
`table_rl:/home/dengyan/tabular_rl_outputs/perf_profiler_4b_20260913_bf16_long/`和
`.../perf_profiler_4b_20260913_4bit_long/`。脚本和说明在`archive/diagnostics/20260913_profiler/`。

### 2026-09-13 固定 transition 的 old/current 融合门禁

在 `table_rl` 对同一条真实 rollout transition 做了 5 次稳定计时：来源为
`row_index=3/example_index=7365/turn_index=0`，prompt 1,815、response 142、总长 1,957
tokens；模型为 Qwen3-4B `checkpoint-6380`，SDPA、gradient checkpointing、BF16 autocast，
不执行 optimizer step。将 actor 的 old-policy 与 current-policy log-prob 合并为同一次
forward 后，4-bit 的 steady median 为 **1.3895s**，两次 forward 的 steady median 为
**1.8167s**，加速 **1.307x**（约减少23.5%时间）；BF16 分别为 **1.1659s** 和 **1.5220s**，
加速 **1.306x**（约减少23.4%时间）。两种精度的 token-level current log-prob 和 detached
old log-prob 最大绝对差均为0；梯度余弦相似度均为1.0，差异仅为 BF16/算子舍入量（4-bit
最大梯度差4.88e-4、L2差4.995e-3；BF16最大梯度差7.32e-4、L2差1.160e-2）。

同一单前向口径下，BF16 约比4-bit快16.1%，但 peak allocated 为
`9,074,672,128 B` 对 `5,398,452,224 B`，约1.68倍；因此 BF16 只能作为独立的显存/数值
对照，不能因为这条短 transition 就切换正式8B或长序列训练。证据根为
`table_rl:/home/dengyan/tabular_rl_outputs/perf_gate_fixed_transition_20260913/run5/4bit_fusion/`
和 `.../run5/bf16_fusion/`。

这只是固定单 transition、无 optimizer 的实现门禁，不是完整 GRPO update 的吞吐或 accuracy
结果。融合只有在 PPO iteration=1、dropout=0、KL=0、无 ranking、且 old/current 使用同一
次 optimizer 前权重时才可等价；在把它放入 trainer 前仍需用真实 transition batch 做 token/
gradient 对齐、checkpoint 恢复和 OOM 回归，并生成新的 implementation lock/manifest。当前
正式 arm 仍保留 actor old-policy scoring，未静默启用融合或 BF16。

### 2026-09-13 CUDA Graph 复测

在同一隔离 source 上以 `example_index=7365` 做了一组 online one-group 复测：4-bit、SDPA、
gradient checkpointing、actor old-policy、KL=0、transition micro-batch 1/4096，vLLM
`VLLM_ENFORCE_EAGER=0`。vLLM 日志确认完成 mixed prefill/decode 与 decode CUDA graph capture。
本次实际收集 8 episodes、47 transitions、11,397 response tokens；rollout collect **74.42s**，
old-policy forward **28.38s**，policy forward/backward **93.73s**，weight sync 1.72s，
`step_time=198.55s`（`train_runtime=199.92s`）。该结果与此前 Graph smoke 的约200秒量级一致，
但采样轨迹和有效 transition 不同，不能与 eager 运行直接做总时长差分；本次诊断状态为
`complete_audited`，owned vLLM/trainer 已释放。证据根为
`table_rl:/home/dengyan/tabular_rl_outputs/perf_gate_cuda_graph_20260913/`。

这次复测只确认 Graph 配置在该短链路上生效，不改变正式 RL 的 actor old-policy、reward 或
precision contract；8B/A100 和完整30题 update仍需独立门禁。

- `perf_audit_qwen3_4b_v26_oneupdate_20260912_retry1`
- `perf_audit_qwen3_4b_v26_oneupdate_20260912_retry2_pack8192`
- `perf_audit_qwen3_4b_v26_fast_mode_20260912_gc0`（OOM）
- `perf_audit_qwen3_4b_v26_fast_mode_20260912_sdpa`
- `perf_audit_qwen3_4b_v26_fast_mode_20260912_vllmgraph`
- `perf_audit_qwen3_4b_v26_fast_mode_20260912_sampling`（名称含sampling但manifest实际为actor；只作actor重复）
- `perf_audit_qwen3_4b_v26_fast_mode_20260912_sampling2`（manifest实际为sampling）

各根内读取`train/run_manifest.json`、`train/implementation_lock.json`、`train/training_precision.json`、
`logs/train.log`、`logs/vllm.log`、`perf_trace.jsonl`（若存在）和`status`，以实际字段识别运行。
早期报告见[性能审计](../reports/rl/PERFORMANCE_AUDIT_TABLE_RL_20260912.md)；本页补充更严格的
计数、时间口径和适用范围。`docs/reports/`被仓库忽略，不能把唯一的启动指导只放在那里。

共享实现位于`src/rl/frameworks/trl/{run_transition_grpo.py,transition_grpo.py,start_vllm_server.sh}`；
日期诊断wrapper和埋点位于`archive/diagnostics/20260912_perf_audit/`，不作为正式入口。
2026-09-12适配核查已修正resolver误将replicated BF16默认checkpointing关闭的问题，恢复
历史BF16/4-bit均开启，3个CPU回归测试通过；未因此重启、改写或迁移任何在途实验。


## 8B table_rl 单组性能证据

以下均为 Qwen3-8B `checkpoint-6380`、Atomic v26、新60题文件、单题 K=8、1 optimizer
update 的隔离诊断，训练卡与 vLLM 卡分别使用 table_rl 的两张 RTX 3090；不是正式 30题/update
RL 结果，也不包含 A100 测量。

| 诊断 | rollout | response tok/s | old-policy | policy前反向 | train_runtime | 备注 |
|---|---:|---:|---:|---:|---:|---|
| eager | 152.67s / 13,903 tok | 91.1 | 1.31s | 2.52s | 161.23s | 46 raw transitions，只有1个进入训练，不能作完整 update 对照 |
| CUDA Graph + actor | 166.88s / 22,749 tok | 136.3 | 84.68s | 276.77s | 534.35s | 57 transitions 全部有效；与 eager 的有效训练量不同 |
| CUDA Graph + sampling | 166.72s / 22,749 tok | 136.4 | 约0s | 277.08s | 449.24s | 节省约84.7s，但改变 old-policy 分母，只能作算法消融 |
| CUDA Graph + sampling + `max_num_batched_tokens=32768` | 166.76s / 22,749 tok | 136.3 | 约0s | 277.45s | 449.69s | 与默认 scheduler 无可见收益，保留为可选参数 |
| CUDA Graph + sampling + microbatch=2 | — | — | — | — | 失败 | 3090 在 backward OOM，不能盲目增大并发 |

因此目前对 8B/3090 可采纳的安全性能配置是 CUDA Graph、actor old-policy、microbatch 1 /
4096 token；sampling 复用只用于明确标注的速度消融。Graph 相对 eager 的 rollout tok/s 约提升
50%，但不能把两次不同随机轨迹的总 update 时间直接相减，也不能据此声称 accuracy 提升。
# 2026-09-13 accelerated continuation

The first post-gate same-identity rerun was stopped before its first update because it duplicated
an already completed result; its root
`/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_saam_threelevel_spanbalanced60_table_rl_20260913_graph_r2`
is retained as performance evidence and is not an accuracy result. The next registered arm is a
new correctness-only `binary + SAAM` 4B experiment in
`/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_correctness_only_saam60_table_rl_20260913_graph_r1`.
It uses the same checkpoint/cohort and the validated 4-bit NF4, SDPA and vLLM CUDA Graph path
(`VLLM_ENFORCE_EAGER=0`). It completed four updates and produced checkpoint-4; the immutable
manifest, implementation lock, precision audit and matched BIRD-dev1534 evaluation are complete.
The remaining promotion gate is a complete fresh replay audit.

The correctness-only binary+SAAM run has now completed `global_step=4/4` with
`train_runtime=26093.7672s` (about 7h14m53s); GPU0/1 were released. This is the first new reward
arm after the performance gate, so its runtime is useful for throughput comparison. Its matched
accuracy is now known, but the run remains non-promoted because the gain is not significant and
the complete fresh replay audit is still outstanding.

The first matched BIRD-dev1534 attempt under
`.../qwen3_4b_correctness_only_saam60_checkpoint4_graph_actionable_20260914_r3` was stopped before
completion because its manifest recorded `error_feedback_version=legacy`; its partial 532/767 and
565/767 shards are not an accuracy result. The canonical r4 attempt is
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_correctness_only_saam60_checkpoint4_graph_actionable_20260914_r4`.
Its immutable config records `actionable-error-v1`; both 767-example shards completed and merged to
909/1534 correct and 1359/1534 legal, with all owned GPU/vLLM processes cleaned up. Compile cache
was disabled only to avoid the prior shared-cache checksum failure; the two earlier startup failures
remain separate and have no GPU residue. Same-item comparison against 4B SFT epoch4 is 77 gains /
71 regressions (net +6, +0.391 pp, exact McNemar p=0.6812) for correctness and 62/75 (net -13,
-0.847 pp, p=0.3052) for legality. Against three-level SAAM it is 74/62 (net +12, p=0.3456), and
against vanilla it is 84/65 (net +19, p=0.1401); none passes the accuracy admission gate. The
canonical paired artifact is `results/merged/paired_comparison.json`.
Its adjacent `paired_comparison_provenance.json` binds all input hashes and records zero mismatches
for `question`, `db_id` and `dataset_split` across all 1,534 example indices.
