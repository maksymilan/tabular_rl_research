# OmniSQL-7B BIRD reproduction

This directory reproduces the BIRD-dev results reported for
`seeklhy/OmniSQL-7B`. The paper target is:

- greedy decoding: 63.9% execution accuracy;
- eight samples at temperature 0.8, selected by execution-result majority:
  66.1% execution accuracy.

The completed `table_rl` reproduction obtained:

- greedy decoding: 983/1,534 = 64.08%;
- eight-sample execution-result majority: 1,010/1,534 = 65.84%.

The absolute differences from the paper are +0.18 and -0.26 percentage
points, respectively.

The run uses the upstream prompt and evaluation contracts:

- upstream source: `RUCKBReasoning/OmniSQL` at
  `a66d732010c89fac4353488d1c59e0b06f92b742`;
- model: `seeklhy/OmniSQL-7B` at
  `af4eed67f561bbeea555c017dae4b38b93bac2eb`;
- upstream dataset archive: `seeklhy/OmniSQL-datasets` at
  `e1db8bb10ef11f92eacd8d87801d716ba0e9e9be`;
- BIRD-dev: 1,534 examples;
- prompt: upstream preprocessed `data/dev_bird.json`, including DDL, column
  descriptions, sampled/retrieved database values, and BIRD evidence;
- generation: bfloat16, 8,192-token model context, up to 2,048 output tokens;
- score: duplicate-insensitive, order-insensitive BIRD execution-result set
  equality.

`fetch_remote_zip_member.py` downloads only a selected member from the
22.2 GB upstream ZIP by using byte ranges. This preserves the exact upstream
prompt file without downloading unrelated training data.

On `table_rl`, sync the two scripts and run:

```bash
bash /home/dengyan/tabular_rl_outputs/reproductions/omnisql/source/run_bird_official.sh smoke
bash /home/dengyan/tabular_rl_outputs/reproductions/omnisql/source/run_bird_official.sh full
```

The wrapper invokes the pinned upstream `infer.py` and `evaluate_bird.py`
directly. Existing non-empty prediction files are reused, so evaluation can be
rerun without repeating inference.

Completed remote artifacts are under:

```text
/home/dengyan/tabular_rl_outputs/reproductions/omnisql/results/bird_dev1534
```
