# version26 matched evaluation contract

更新时间：2026-09-06

评测只验证当前 Atomic version26 模型身份；历史 scheme 的结果不能写入当前主线结论。
训练完成后的统一衔接规则也在本页末尾，避免维护平行 handoff 文档。

## 固定评测身份

- runtime：Atomic version26，commit `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de`
- protocol hash：`4da19387399bd3a5`
- carrier：`think-json-v1`
- prompt SHA：`848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`
- model：Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`
- task set：BIRD-dev 1,534 题，必须记录精确 path/hash
- decode：greedy `n=1`、temperature `0`、top-p `1`，episode 内严格因果串行
- limits：max 30 steps；max tokens 至少 2,048
- scorer：`bird-set`，terminal 结果引用 Harness 的 exact artifact/列顺序

冻结 SFT1 checkpoint-560 的小样本可行性验证结果为 838/1534 = 54.63% `bird-set`；匹配
base 为 270/1534 = 17.60%。这是 SFT1/早期 RL 验证锚点，不是正式 checkpoint-6380 RL 的
baseline；checkpoint-6380 必须按同一 matched contract 重新评测。

## 评测门禁

1. candidate 和 baseline 使用相同题号、database snapshot、external knowledge、decode、
   max steps、scorer、runtime/prompt/history 和 evaluator。
2. 两个 run 都必须有完整 `run_manifest.json`、`implementation_lock.json`、precision audit、
   final checkpoint、source snapshot 和 evaluation identity。
3. 结果必须覆盖 1,534/1,534 唯一题号，API/断线污染为 0；否则只能作进度记录。
4. candidate-only 绝对分数不能宣称提升；gain/regression、McNemar 和 promotion 需要同题
   matched baseline。
5. 评测输出与 RL 训练输出隔离，不能覆盖既有目录或混用 protocol identity。

## 资源位置

- RL 主实验：`a100`，一张 A100 做单进程 replicated trainer，另一张做 online vLLM；评测完成前不要在其上启动第二套 evaluator。
- matched evaluation：优先 `table_rl` 或 `NewGNN`，GPU/port 由 handoff 脚本显式绑定并记录。
- evaluator：`reproductions/trust_sql/qwen3_8b_atomic_sft1/`
- handoff：`src/rl/evaluation/run_v26_matched_eval_handoff.sh`
- baseline cohort：`data/eval_inputs/bird_train_baseline300_v1.jsonl` 仅用于 train-side
  diagnostics；BIRD-dev full eval 使用其独立固定题集，二者不可混称。

## 训练后 handoff

训练结束后由 launcher 先完成 manifest、implementation lock、precision audit、checkpoint
和 source/runtime identity 检查，再在 `table_rl` 或 `NewGNN` 选择空闲 GPU 执行 matched greedy
evaluation。candidate 与 baseline 必须使用相同题号、数据库快照、prompt、carrier、decode
和 scorer；任一身份、覆盖率、API/断线污染或 fresh replay 检查失败即停止并保留失败记录。

## 结果解释

`bird-set` 比较行集合并忽略行序；列顺序仍由 terminal exact-artifact contract 固定。所有
报告同时写 accuracy、legal termination、tool/process errors、tokens、steps 和审计状态。
传输失败、模型解析失败、Harness argument error 和语义错误分开统计，不把 transport block
写成模型能力结论。

旧 direct-SQL、iterative-SQL、checkpoint-relalg、Atomic v24/v39/v51/v54 和其他 scheme
仅可用原始 identity 在隔离目录 replay；当前评测入口不再为它们保留平行默认配置。
