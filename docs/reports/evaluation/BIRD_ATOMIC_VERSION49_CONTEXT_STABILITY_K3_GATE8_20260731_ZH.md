# version39 / version49 上下文稳定性 K=3 Gate8

每个版本包含 3 次独立运行，每题每次恰好一个 semantic attempt；不使用 pass@K、成功早停或 verifier 选择。指标为 `bird-set`。
本实验验证的是 version49 active/archive renderer 的重复运行稳定性，不是对模型总体准确率的无偏估计，也不验证由模型自由撰写长期摘要的另一套协议。

| 版本 | Correct | Legal | Control | Mean steps | Errors | Reads | Exact rereads | Max-step | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| version39 | 20/24 | 22/24 | 15/15 | 9.96 | 13 | 64 | 5 | 0 | 1932655 |
| version49 | 20/24 | 21/24 | 15/15 | 12.25 | 10 | 110 | 15 | 3 | 2488137 |

三轮逐轮正确数完全相同，因此 24 个配对 attempt 为 0 gains / 0 regressions。但 version49 的平均步数增加 23.0%，row reads 增加 71.9%，tokens 增加 28.7%。
总 prompt tokens 为 1796337 → 2298044；即使除以语义步骤，仍由 7516 增至 7816（+4.0%）。因此总成本上升同时来自更长轨迹和未降低的单步上下文。

## 逐轮

| Run | v39 correct/legal | v49 correct/legal | v39 steps | v49 steps | v39 reads/rereads | v49 reads/rereads |
|---|---:|---:|---:|---:|---:|---:|
| run1 | 7/8 / 7/8 | 7/8 / 8/8 | 9.50 | 10.25 | 20/0 | 24/1 |
| run2 | 7/8 / 8/8 | 7/8 / 7/8 | 9.50 | 11.75 | 20/1 | 35/7 |
| run3 | 6/8 / 7/8 | 6/8 / 6/8 | 10.88 | 14.75 | 24/4 | 51/7 |

## 逐题

| Task | v39 correct | v49 correct | v39 steps | v49 steps | v39 reads/rereads | v49 reads/rereads |
|---|---:|---:|---:|---:|---:|---:|
| bird_train_05873 | 3/3 | 3/3 | 7.67 | 7.67 | 3/0 | 4/0 |
| bird_train_02918 | 2/3 | 2/3 | 9.00 | 18.00 | 8/0 | 32/6 |
| bird_train_06299 | 0/3 | 0/3 | 14.67 | 24.33 | 29/5 | 48/9 |
| bird_train_02196 | 3/3 | 3/3 | 7.33 | 8.67 | 3/0 | 3/0 |
| bird_train_02179 | 3/3 | 3/3 | 7.33 | 6.67 | 3/0 | 2/0 |
| bird_train_03768 | 3/3 | 3/3 | 11.33 | 11.00 | 6/0 | 6/0 |
| bird_train_00166 | 3/3 | 3/3 | 9.00 | 7.67 | 1/0 | 2/0 |
| bird_train_00808 | 3/3 | 3/3 | 13.33 | 14.00 | 11/0 | 13/0 |

## 预注册条件

- PASS `accuracy_non_regression`：20/24 versus 20/24
- PASS `control_non_regression`：15/15 versus 15/15
- FAIL `legal_at_least_95_percent`：21/24
- PASS `median_exact_rereads_zero`：0.0
- FAIL `mean_steps_within_10_percent`：12.250 versus 9.958
- FAIL `tokens_within_10_percent`：2488137 versus 1932655
- FAIL `churn_tasks_no_max_steps`：05873=0, 02918=1
- PASS `churn_tasks_at_least_two_of_three_correct`：05873=3/3, 02918=2/3
- FAIL `dormant_branch_bounded_reads`：max reads=22, max exact rereads=6

## 轨迹审计

- `bird_train_02918`：version39 三轮 reads 为 2/4/2，version49 为 7/5/20。version49 第三轮有 5 次 exact reread，并在部门历史与人员信用卡映射之间反复读取，没有推进到信用卡到期年份过滤，最终 max-steps。
- `bird_train_06299`：两个版本都是 0/3，说明 active/archive 不是该题失败的唯一原因。但 version39 三轮均未 max-steps；version49 后两轮 reads 为 22/18，均跑满 30 步，反复重建同一菜单过滤/连接分支。归档策略放大了既有求解困难。
- 五个稳定 controls 在两边均为 15/15。当前证据支持“简单任务行为保持”，不支持“困难长程任务上下文更可控”。

## 基础设施说明

version39/run1 的 `bird_train_06299` 首个 worker 请求长期无返回且未写入 terminal record；主进程中断后仅以 `--resume --start 2 --limit 1` 补齐该缺失记录。原请求没有可选择的语义结果，因此未形成 pass@K 或 verifier 选择。

## 决策

version49 未通过预注册稳定性条件；不应替换 version39，也不继续叠加新的语义/终止改动。

全部输出均为 `diagnostic_only_pending_protocol_scale_gate`，不进入 SFT/RL。
