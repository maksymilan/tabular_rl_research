# OmniSQL-7B BIRD reproduction (2026-07-23)

## Outcome

The released `seeklhy/OmniSQL-7B` checkpoint reproduces the paper's
BIRD-dev result on `table_rl`.

| Strategy | Paper | Reproduction | Correct | Difference |
|---|---:|---:|---:|---:|
| Greedy, temperature 0 | 63.9% | 64.08% | 983/1,534 | +0.18 pp |
| Execution majority, n=8, temperature 0.8 | 66.1% | 65.84% | 1,010/1,534 | -0.26 pp |

Both scores use the upstream duplicate-insensitive and order-insensitive
BIRD execution-result set equality. There were no API or inference transport
failures because inference was offline.

## Pinned inputs

- Upstream code:
  `RUCKBReasoning/OmniSQL@a66d732010c89fac4353488d1c59e0b06f92b742`.
- Model:
  `seeklhy/OmniSQL-7B@af4eed67f561bbeea555c017dae4b38b93bac2eb`.
- Model artifact:
  15,231,272,152 bytes,
  SHA256 `dbdd444b3233a3decb600547c8d3cda0e0118c7cfe8085b68a05c519b0e80b01`.
- The original release revision `9917e42a7a3173c91585cfa5962990325321be33`
  points to the same model artifact.
- Upstream evaluation-data revision:
  `e1db8bb10ef11f92eacd8d87801d716ba0e9e9be`.
- Exact upstream `data/dev_bird.json` prompt SHA256:
  `65186bb8046e20f30c0fe0c560d7cabef6baa0741a2602dd4f6936d55ddf5b50`.
- Existing server BIRD `dev.json` and `dev_tables.json` are byte-identical
  to the files in the upstream OmniSQL archive.

The exact upstream prompt was extracted from the 22.2 GB ZIP with byte ranges.
It includes DDL, column descriptions, sampled/retrieved database values, and
BIRD evidence. All 1,534 prompt outputs match the server gold SQL sequence.
The longest rendered chat prompt is 5,092 tokens, below the upstream 6,144
input-token allocation.

## Generation and evaluation

- bfloat16, tensor parallelism 2;
- model context 8,192, maximum output 2,048;
- greedy: `temperature=0`, `n=1`;
- sampling: `temperature=0.8`, `n=8`;
- SQL extraction: last lowercase fenced `sql` block, exactly as upstream;
- majority: execute every candidate, group valid candidates by exact
  `frozenset(rows)`, select the first maximum-vote group;
- final EX: `set(predicted_rows) == set(gold_rows)`;
- SQL execution timeout: 10 seconds, 20 scorer workers.

The upstream `infer.py` and `evaluate_bird.py` were invoked directly. The
reproduction environment used two RTX 3090 24 GiB GPUs, vLLM 0.19.1,
PyTorch 2.10.0+cu126, and Transformers 5.12.0. The paper reports vLLM 0.6.3
on A800 GPUs; this is the main environment difference.

## Artifact audit

- Greedy: 1,534/1,534 SQL strings were non-empty.
- Sampling: 12,272 candidates; 11,902 executable and 370 invalid.
- Sampling contained one empty extracted SQL. It was also selected for the
  single all-invalid/tied case handled by the upstream seeded fallback.
- Both GPUs returned to 2 MiB used and 0% utilization after completion.

Remote artifacts:

```text
/home/dengyan/models/text2sql_reproduction/OmniSQL-7B-af4eed67
/home/dengyan/tabular_rl_outputs/reproductions/omnisql/results/bird_dev1534
```

The result directory contains full prompts, model responses, all candidate
SQL strings, selected greedy/majority SQL files, evaluation logs, per-example
audit records, and structured score summaries. Exact hashes and environment
metadata are recorded in `reproductions/omnisql/manifest.json`.

## Download path

Direct Hugging Face access timed out. The Hugging Face mirror served metadata
and files, but the Hub client's long transfer failed on a TLS handshake and
discarded its partial file. A resumable range downloader was therefore used
against the mirror object URL; it refreshed signed URLs after stalls and
validated the final pinned SHA256. The local-computer relay was not needed.
