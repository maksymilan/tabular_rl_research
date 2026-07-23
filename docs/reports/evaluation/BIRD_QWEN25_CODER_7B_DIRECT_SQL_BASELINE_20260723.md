# Qwen2.5-Coder-7B Direct-SQL Baselines on BIRD Dev

Date: 2026-07-23

## Decision

Under the same BIRD-reference direct-SQL contracts as the completed Qwen2.5-7B-Instruct controls,
Qwen2.5-Coder-7B-Instruct is higher at greedy pass@1 and at every sampled K=4 metric:

| Decode | Ordinary Instruct | Coder-Instruct | Delta |
| --- | ---: | ---: | ---: |
| Greedy | 650/1534 = 42.37% | **748/1534 = 48.76%** | **+98 / +6.39 pp** |
| Sampled K=4 pass@1 | 624/1534 = 40.68% | **727/1534 = 47.39%** | **+103 / +6.71 pp** |
| Sampled K=4 pass@2 | 713/1534 = 46.48% | **820/1534 = 53.46%** | **+107 / +6.98 pp** |
| Sampled K=4 pass@4 | 785/1534 = 51.17% | **904/1534 = 58.93%** | **+119 / +7.76 pp** |

All four paired gains are statistically supported by exact two-sided McNemar tests. This establishes
Coder-7B-Instruct as the stronger zero-shot direct-SQL starting point under these contracts. It
does not establish how much of the gain will remain after identical SFT; that requires the planned
controlled SFT comparison.

## Evaluation Contract

Both comparisons use:

- BIRD dev, all 1,534 tasks from `bird_dev_20240627.jsonl`;
- the same full-schema prompt and BIRD external knowledge;
- `max_tokens=1024`;
- thinking disabled and no execution feedback;
- a 20-second generated-SQL execution deadline;
- BIRD reference set denotation, recorded as `bird-set`.

Greedy uses `n=1`, `temperature=0`, `top_p=1`, and pass@1. Sampled K=4 uses `n=4`,
`temperature=0.7`, `top_p=0.95`, and pass@1/2/4.

The 20-second deadline intentionally matches the existing 7B-Instruct artifacts. It is not the
separate 10-second setting documented for a strict reproduction of the pinned Arctic evaluator.

### Hidden generation-config control

The first completed Coder greedy run used the checkpoint's native
`repetition_penalty=1.10` and scored 743/1534 = 48.44%. Audit of the vLLM startup log exposed that
the ordinary Instruct checkpoint uses `repetition_penalty=1.05`; this model-level default was not
recorded in the historical manifest.

The final Coder runs therefore pass `repetition_penalty=1.05` explicitly and record it in both the
manifest and every result record. The two checkpoints have the same native `top_k=20`, while
temperature and top-p are overridden by the request. The controlled Coder greedy result is five
tasks higher than its native-1.10 result; the paired native-versus-controlled difference is not
significant (`p=0.6353`).

Every task identity, database ID, question, gold SQL, and complete model input matches between
the ordinary-Instruct and controlled-Coder artifacts. Every file contains exactly one unique record
for each example index 0 through 1533. Both K=4 files contain exactly four samples for every task.

## Paired Results

### Greedy

| Outcome | Tasks |
| --- | ---: |
| Both correct | 530 |
| Coder correct, ordinary Instruct wrong | 218 |
| Ordinary Instruct correct, Coder wrong | 120 |
| Both wrong | 666 |

The exact two-sided McNemar p-value is `1.0836963448292955e-7`.

### Sampled K=4

| Metric | Both correct | Coder only | Ordinary only | Both wrong | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: |
| pass@1 | 502 | 225 | 122 | 685 | 3.4877e-8 |
| pass@2 | 594 | 226 | 119 | 595 | 8.8234e-9 |
| pass@4 | 673 | 231 | 112 | 518 | 1.2308e-10 |

These K=4 tests pair tasks within one completed stochastic generation run. They quantify the
observed task-level difference but do not model variation across repeated decoding seeds.

## Difficulty Breakdown

Greedy:

| Difficulty | Tasks | Ordinary Instruct | Coder-Instruct | Delta |
| --- | ---: | ---: | ---: | ---: |
| Simple | 925 | 473/925 = 51.14% | **527/925 = 56.97%** | **+5.84 pp** |
| Moderate | 464 | 146/464 = 31.47% | **171/464 = 36.85%** | **+5.39 pp** |
| Challenging | 145 | 31/145 = 21.38% | **50/145 = 34.48%** | **+13.10 pp** |

Sampled K=4 Coder result and delta from ordinary Instruct:

| Difficulty | Coder pass@1 | Delta | Coder pass@2 | Delta | Coder pass@4 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Simple | 512/925 = 55.35% | +5.62 pp | 564/925 = 60.97% | +5.30 pp | 613/925 = 66.27% | +6.59 pp |
| Moderate | 168/464 = 36.21% | +7.76 pp | 197/464 = 42.46% | +8.62 pp | 225/464 = 48.49% | +8.84 pp |
| Challenging | 47/145 = 32.41% | +10.34 pp | 59/145 = 40.69% | +12.41 pp | 66/145 = 45.52% | +11.72 pp |

The largest descriptive gains occur on challenging tasks. Difficulty rows are diagnostic; the
paired full-dev comparisons remain primary.

## Failure Audit

There were no API errors, context overflows, incomplete responses, missing-SQL responses, or task
timeouts in either controlled Coder run. The launcher was deliberately interrupted once after 22
tasks when the hidden repetition-penalty difference was discovered; that separate native-config
partial artifact is excluded from every reported metric above.

Greedy per-sample outcomes:

| Outcome | Ordinary Instruct | Coder-Instruct | Coder minus ordinary |
| --- | ---: | ---: | ---: |
| Correct | 650 | 748 | +98 |
| SQL execution error | 370 | 236 | -134 |
| Executed but wrong result | 514 | 550 | +36 |

Sampled K=4 outcomes across all 6,136 generated samples:

| Outcome | Ordinary Instruct | Coder-Instruct | Coder minus ordinary |
| --- | ---: | ---: | ---: |
| Correct | 2,523 | 2,855 | +332 |
| SQL execution error | 1,552 | 1,093 | -459 |
| Executed but wrong result | 2,061 | 2,188 | +127 |

The gain is associated with a large reduction in SQL execution errors. Coder produces substantially
more executable queries, while semantic wrong-result errors remain the dominant bottleneck.

## Artifacts and Verification

Ordinary Instruct:

- greedy: `data/results/qwen2.5_7b_bird_direct_sql_base_greedy_dev1534_bird_ex/`;
- sampled K=4: `data/results/qwen2.5_7b_bird_direct_sql_base_passk4_dev1534_bird_ex/`.

Controlled Coder-Instruct:

- greedy: `data/results/qwen2.5_coder_7b_bird_direct_sql_base_greedy_rp105_dev1534_bird_ex/`;
- sampled K=4:
  `data/results/qwen2.5_coder_7b_bird_direct_sql_base_passk4_rp105_dev1534_bird_ex/`.

Auxiliary native-config Coder greedy:

`data/results/qwen2.5_coder_7b_bird_direct_sql_base_greedy_dev1534_bird_ex/`

Coder model:

`/home/dengyan/models/Qwen2.5-Coder-7B-Instruct`

The downloaded model is pinned to Hugging Face revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`; all four safetensor hashes were verified before
evaluation.

Controlled artifact SHA-256 values:

| Artifact | Manifest | All records | Summary |
| --- | --- | --- | --- |
| Greedy | `d2116a95fc8c306f5655034a08f0e52ce6b99227ee5220a4ea4ee98eb3d3a38c` | `483869fb7f63c8a7f965335a07cd451cd798ffaf2e0dcfbf2a79fb3d714001c3` | `35d0c09cb91df9f9a65277d3b1d0907dd11cc8de991e0cba2ceb71ae3af15a3d` |
| K=4 | `89fce39a0ec437a3d2807f7d808bb5f797ae487f1174977930e8b6bce503a36f` | `f38ff48e0e51d7a840788c3fb6a4c15c15eeacba85335d96cafaf8b2f511c09e` | `3174c217932eb7187fd226116bc862a685bf3d5eb246822df4443010f45445cc` |

Both launchers completed normally, stopped the remote vLLM server, released port 8018, and returned
GPU 0 to its idle 2 MiB state.
