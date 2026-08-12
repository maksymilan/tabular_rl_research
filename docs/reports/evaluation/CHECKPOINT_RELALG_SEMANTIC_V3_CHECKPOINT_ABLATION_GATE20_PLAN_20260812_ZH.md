# Semantic-v3 Checkpoint-on vs Disabled fresh Gate20 plan

日期：2026-08-12
状态：预注册；尚未产生 live 结果

## 核心问题

在已通过 Gate100 行为保持且显著降低 token 的 `semantic-v3-v24` 接口上，模型自主 checkpoint
是否比始终只调用 Atomic 工具更能提高长轨迹正确率或可靠性。

## 单变量与 cohort

- fresh disjoint positions：300–319，20 pairs / 40 paid episodes
- 共同：Atomic semantic-v3-v24、A Text-JSON、recent-4、official Flash、同一 typed Harness
- treatment：`model-choice-commit-v6`，max checkpoints 8，restore 0，eligibility `none`
- control：`checkpoint-disabled-v1`，max checkpoints 0，restore 0
- 除 teacher checkpoint guidance 与 max-checkpoint capability 外，其余配置一致
- contiguous disjoint cohort 在看到结果前冻结；不按 gold、question、reasoning 或历史 outcome 选择

## 预算与调度

- endpoint/model：`https://api.deepseek.com/chat/completions` / `deepseek-v4-flash`
- 每 episode：turns20、primitive calls30、attempts50、tokens350k、wall3600秒
- 40 episodes 总硬上限：14,000,000 tokens
- 四个动态 lanes；同题两臂相邻，pair order 交替；最多四个 episode 同时在途

## 判断

先验证 treatment checkpoint exposure >=8/20；不足则 underexposed/inconclusive。若暴露充分，主指标
为 paired `bird-set`、strict、legal；并报告 checkpoint-used symmetric subset、turns、tokens、
errors、max-turn/provider failures。Checkpoint 至少需 correct 不低于 control、legal 不低于 control、
errors不增加超过2，且 checkpoint-used subset 不出现净回退，才值得扩大。所有 record 必须通过
identity、structure、cohort/budget 与 fresh replay。无 restore；无 SFT/RL admission。
