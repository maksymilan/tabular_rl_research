# Arctic-Text2SQL-R1-7B and SQL-R1-7B reproduction

These wrappers preserve the upstream repositories and adapt their BIRD evaluation
commands to the `table_rl` server's two RTX 3090 GPUs.

Current status:

- Both released checkpoints are downloaded and verified against official
  Hugging Face SHA-256 metadata.
- The 8-question chain test completed at 7/8 for Arctic and 6/8 for SQL-R1.
  These are `bird-set` denotation results for pipeline validation, not paper
  comparisons.
- The full BIRD-dev queue was started on 2026-07-23. It runs Arctic greedy
  first, then SQL-R1's eight-sample majority-vote configuration.

Pinned upstream artifacts:

- `Snowflake/Arctic-Text2SQL-R1-7B`:
  `2d73facaf7e15831d0b918d219e0d565f113f592`
- `MPX0222forHF/SQL-R1-7B`:
  `db409e8372ca5e463126b07e905b5245caf14ea6`
- `snowflakedb/ArcticTraining`:
  `b28638e6c577ece34cc147ab3d45d27a7f66b33b`
- `DataArcTech/SQL-R1`:
  `ccfb511e81fdb3a8e87bd3da7f3d71882cedd532`

The default remote layout is:

```text
/home/dengyan/models/text2sql_reproduction/
  Arctic-Text2SQL-R1-7B/
  SQL-R1-7B/
/home/dengyan/tabular_rl_outputs/text2sql_reproduction/
  data/
  envs/
  logs/
  manifests/
  results/
  scripts/
  third_party/
```

Run the data preparation once:

```bash
bash scripts/prepare_arctic_bird.sh
```

Then run an 8-example greedy smoke test:

```bash
bash scripts/run_arctic_bird.sh
bash scripts/run_sqlr1_bird.sh
```

For full BIRD-dev greedy decoding, set `EVAL_SCOPE=full`. For SQL-R1's
paper-style self-consistency, additionally set `N=8 TEMPERATURE=0.8`.
Arctic's headline result is greedy; its optional majority-vote result also uses
eight candidates at temperature 0.8.

The serialized full queue is:

```bash
nohup bash scripts/run_full_bird_queue.sh \
  > logs/full_bird_queue_launcher.log 2>&1 < /dev/null &
```

Monitor it with:

```bash
cat logs/full_bird_queue.status
tail -f logs/full_bird_queue.log
```

Expected paper comparison points under `bird-set` are 68.9% for Arctic greedy
on BIRD-dev and 66.6% for SQL-R1-7B with eight-candidate execution
self-consistency. The SQL-R1 model card's 67.1% figure belongs to a different
model-size row in the paper, so it is not used as the 7B target here.

## Download path selected on table_rl

Direct access to the official Hugging Face endpoint timed out. Arctic was
downloaded from the ModelScope mirror and SQL-R1 from `hf-mirror.com` using a
portable, resumable aria2 client. Every safetensors shard and `tokenizer.json`
was then checked against the official Hugging Face file size and SHA-256
metadata. Because both mirror paths succeeded, the local-computer relay was not
used.

## Training boundary

The upstream Arctic repository explicitly does not release its training code or
data-generation pipeline. Its released-checkpoint evaluation is reproducible,
but exact retraining is not.

SQL-R1 does release the 4000/1000 parquet split, SynSQL databases, reward code,
and vendored verl trainer. An isolated compatibility environment was built with
Python 3.11, PyTorch 2.4, vLLM 0.6.3, FlashAttention 2.6.3, and NCCL 2.20.5.
The data/reward path, TP=2 rollout initialization, FSDP initialization, and
two-GPU NCCL all-reduce all pass.

The pinned upstream full-parameter hybrid trainer cannot finish the first
rollout on two 24 GB cards. Even after reducing the smoke run to two prompts of
555/610 tokens, a 640+128 context, `n=2`, and full CPU offload, FSDP state-dict
materialization needs a 130 MiB CUDA block while only 77 MiB is free. The
paper's 4096+2048, `n=8`, 8x80 GB run is therefore not reproducible on this
server without either larger GPUs or a materially different memory strategy
such as parameter-efficient training or a newer trainer architecture.

The one-step diagnostic wrapper and deterministic short-subset builder are kept
for auditing this boundary:

```bash
python scripts/prepare_sqlr1_smoke_subset.py \
  --input third_party/SQL-R1/example_data/train.parquet \
  --output data/sqlr1_train_smoke_short2.parquet \
  --tokenizer /home/dengyan/models/Qwen2.5-7B-Instruct
bash scripts/run_sqlr1_train_smoke_2gpu.sh
```
