# Qwen3-4B comparison matrix

This directory owns the Qwen3-4B scale controls requested for comparison with TRUST-SQL and the
completed Qwen3-8B atomic SFT1 run. Scores from different interfaces stay in separate columns.

Planned/evaluated arms on the same 1,534-question BIRD-Dev snapshot are:

1. raw one-turn Direct SQL (`canonical-json-v1`);
2. raw SQL-only iterative feedback (`iterative-sql-v6`);
3. raw author four-phase Explore/Propose/Generate/Confirm, unknown schema;
4. frozen historical atomic-version26 base;
5. the same atomic-version26 base after the same external-teacher SFT1 QLoRA targets.

The paper's Qwen3-4B unknown-schema raw result (29.3%), schema-prefill raw result (46.3%), SFT-only
result (46.2%), and SFT+RL result (64.9%) are external references. The released repository does not
publish the paper SFT dataset/checkpoint or a formal schema-prefill serialization, so those values
must not be described as locally reproduced unless their missing artifacts become available.

`download_model.sh` pins public `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c` and validates every LFS shard by SHA-256. The Direct
and iterative arms use the shared launcher in `../qwen3_8b_sql_controls/` with `MODEL_SIZE=4b`.

`train_atomic_sft1_qlora.sh` is a two-RTX-3090 scale control, not the paper's full-parameter SFT:

- same 4,471 atomic-version26 SFT1 targets as the completed Qwen3-8B run;
- two epochs, global batch 16, learning rate `1e-4`;
- 4-bit all-linear QLoRA, rank 16, alpha 32, dropout 0.05;
- Qwen3 thinking template, history masking, cutoff 6,400;
- exact 4B/8B tokenizer-file equality and full model-shard hashes are launch gates.

Run the two-step integration smoke first, using any two idle physical GPUs:

```bash
RUN_KIND=smoke GPU_IDS=5,7 \
  bash /home/dengyan/tabular_rl_outputs/runtime/qwen3_4b_atomic_sft1_20260808/train_atomic_sft1_qlora.sh
```

Only after smoke success, use a fresh full output directory:

```bash
RUN_KIND=full GPU_IDS=5,6 \
  bash /home/dengyan/tabular_rl_outputs/runtime/qwen3_4b_atomic_sft1_20260808/train_atomic_sft1_qlora.sh
```

`queue_atomic_eval_after_training.sh` is a fail-closed NewGNN handoff. It waits for the full
training status to report step 560, freezes and hashes `checkpoint-560`, verifies GPUs 5 and 6 are
idle, and then starts the atomic base and adapter evaluations in two new result directories. It
does not select a checkpoint from dev performance and will not resume an existing result directory.

`queue_qwen3_8b_sql_full_after_table_jobs.sh` is the corresponding detached table_rl handoff for
the two missing 8B SQL-interface controls. It waits for both current 4B full controls on GPUs 0 and
1 to complete successfully, rechecks the pinned input/runtime hashes and idle GPUs, then launches
8B Direct SQL and iterative SQL into new result directories. The queue never merges the earlier
smoke runs or the failed preflight-only runs.

The final report must compare atomic base→SFT as a paired in-protocol effect, while placing paper
numbers and Direct/iterative/four-phase controls in explicitly labeled, non-equivalent columns.
