# AGENTS.md — shared memory for coding agents (Claude Code + Codex)

Single source of truth both agents read (Codex loads this natively; `CLAUDE.md` imports it).
Repo-tracked + git-versioned. **Do not put secrets here** — the external-LLM key lives in `api.md`
(never commit/echo it). Update this file when a durable decision, milestone, or pitfall lands.

## Project

RL training environment for an LLM tool-use agent over relational tables. Research focus: **tool
design + dense process-reward design** for table reasoning. The model answers a question by calling
abstract tools (relational + perception + memory); the harness translates each call into composed
SQL over SQLite and verifies results. Training data is built by **compiling Spider/BIRD gold SQL into
tool-call trajectories** (not LLM-guessed), so every trajectory is execution-verified.

## Current state (2026-06-15)

- **v0 SFT done** (Qwen2.5-3B LoRA): exec-acc **63.8%** on full Spider dev vs text-to-SQL parity line
  **64–66%** vs base+few-shot **4%**. Conclusion: SFT = behavior cloning of interface + canonical
  plans (NOT a contribution by itself); model is feedback-blind (e.g. filters `country='French'`,
  gets 0 rows, ignores it). The RL/process-reward delta is the actual research story.
- **v1 data done**: 6773 train + 998 dev execution-verified+legal trajectories, **268 with
  `add_to_memory`** (scalar subqueries), think-filled by the external LLM (99.7%). Compile coverage
  99.9%, execution-verified **98.4%** on 2000q. System prompt updated with `add_to_memory` +
  `value_ref`/`in_table` predicate forms.
- **7B baselines done** on all 1,034 Spider dev examples with Qwen2.5-7B-Instruct:
  zero-shot direct SQL **716/1034 = 69.25%**; strict two-shot tool use **40/1034 = 3.87%**.
  The tool run reached only 78 legal final answers. Its dominant failure is formatting:
  950 final `protocol_error`s, and 954 trajectories omitted a complete
  `<tool_call>...</tool_call>` block at least once (usually the closing tag). Do not reinterpret
  this score as pure table-reasoning failure or silently relax the parser when comparing to SFT.
  Full model inputs/outputs and per-case success/failure JSON live under
  `data/results/qwen2.5_7b_baselines/` (gitignored).
- **7B / v1 SFT done** (2026-06-13 13:03 Asia/Shanghai): Qwen2.5-7B QLoRA completed 2 epochs /
  830 optimizer steps in 13:01:20. Final train loss is **0.2495**, validation loss **0.2271**.
  Adapter output is `checkpoints/qwen2.5-7b-spider-v1-qlora`. The viable 24G configuration is
  `cutoff_len=8192`, LoRA rank/alpha 16/32, `paged_adamw_8bit`, bf16, gradient checkpointing,
  and effective batch 16. See `src/sft/EXPERIMENTS.md`.
- **7B / v1 zero-shot tool evaluation done** on all 1,034 Spider dev examples: **707/1034 =
  68.38%**, with 985 legal final answers (95.26%). Failures: 276 wrong answers, 42 execution
  errors, 8 protocol errors, and 1 max-steps case. On the 998 examples covered by verified v1 dev
  trajectories, accuracy is **705/998 = 70.64%**. The 36 examples outside v1 coverage score only
  **2/36 = 5.56%**. The 29-example `add_to_memory` subset scores **9/29 = 31.03%**, versus
  **696/969 = 71.83%** on the covered non-memory subset; this confirms the known scalar-memory
  semantics/provenance weakness. Full model I/O and separate success/failure artifacts live under
  `data/results/qwen2.5_7b_sft_v1/tool_zero_shot_dev1034/` (gitignored).
- **V2a memory + provenance DONE & verified (2026-06-14)**: the scalar-memory/provenance repair the
  old "Next" called for is implemented and replaces the v1 memory shape. `add_to_memory` is now
  harness-grounded — the model emits ONLY `{type:"derived_value", source_step_id}`; it never authors
  value/key/provenance. The harness extracts the scalar from the cited step, builds a deterministic
  `derivation`+`key`+`content`, assigns `memory_id = mem_<source_step_id>` and
  `authority:"harness_grounded"`. Predicates reference the `memory_id` via `value_ref`. Every step
  carries harness-authored `references`/`produces`; `emitter.backward_slice` reverse-derives the
  answer's dependency set (through memory too). Strict scalar-source validation rejects
  NULL/0-row/multi-row/multi-col → manifest `memory_reject_*` buckets (v2a dropped 4). New SHARED
  module `src/harness/memory_semantics.py` is called by BOTH emitter and rollout. Observation
  envelope is now `{step_id, status, output}` so the model copies a `step_id` as `source_step_id`.
  Verified: Spider dev replay **998/998**, strict per-tool schema 30,534 calls / **0** fail,
  backward-slice invariant **7767/7767**, unit 98+6+4, exec-verified **98.4%**, **v1 files byte-identical
  (untouched)**. Data (gitignored): `data/trajectories/spider_{train,dev}_v2{,_think}.jsonl` (6769+998,
  `schema_version:"v2a"`), SFT `data/sft/spider_v2_*` (6767+998, `protocol_hash:eedbb946aa0f2cb7`).
  Full write-up: `draft/v2a_memory_report.md`. Scope: V2a = scalar `derived_value` ONLY;
  `evidence_pointer` (V2b) and `plan`/`hypothesis` (V2c) are deferred, separately-ablated mixtures.
- **V2b memory removal + unified references DONE & verified (2026-06-23)**: `add_to_memory` is GONE as
  a tool/concept (it was a redundant wrapper — `memory_id == mem_<source_step_id>`, and the aggregate
  step already parked its scalar). A predicate's `value_ref` now cites the producing `step_id`
  DIRECTLY; the harness grounds the scalar from history with strict validation at resolve time
  (online: an illegal `value_ref` → `execution_error`, never a silent value). New SHARED
  `src/harness/provenance.py::build_references` builds typed `references` edges (`type=data|value`,
  structured `target`) for BOTH emitter and rollout; `backward_slice(reference_type=…)` is
  parameterized (default data+value). `memory_semantics.py` → `scalar_grounding.py` (just
  `extract_scalar` + neutral `ground_scalar_reference`; no memory_id/key/content/derivation).
  `supporting_memory_ids` deleted; `refine_memory` placeholder removed. PROTOCOL_VERSION v2a→**v2b**
  (hash `95c58d18ea4cca28`), `schema_version` v2-ctx→**v3**. Verified: run_all unit **98/98** + compile
  1998/2000, emit 296/300 verified, **online replay 296/296 = 100%** (incl. 5 scalar-subquery value
  edges), SFT build 296/296 with **0 memory residue**. SSOT + impl log: `draft/provenance_redesign.md`
  §5. Deferred (in SSOT §2): C (perception grounding edges + reward), F (subtable consolidation —
  empirically triggered by the first perception-SFT failure modes), D-3/D-4 (regenerate perception
  data + SFT).
- **V2c-plan/context scaffolding started (2026-07-02)**: `plan(ops)` is now a model-visible,
  harness-managed task-control tool for creating/updating/deleting subgoals. Plan state is not
  factual evidence: it cannot support `value_ref` or final answers and is excluded from data/value
  provenance slices. Plan items separate `status` (completion) from `result` (model-authored
  subtask answer/conclusion such as boolean/scalar/text); `result` is still control state, not
  grounded memory. New shared `src/harness/environment_state.py` maintains resident context with
  plan items plus per-table/handle schema, inspected column domains, reads, and produced handles.
  Online rollout/RL env and offline SFT rendering can include this `state` snapshot in observation
  envelopes. The raw SQL→trajectory emitter now produces the verified relational backbone only; it
  does **not** mechanically inject `describe_table` / `inspect_column` / `read_subtable`. External-
  model enrichment owns natural plan wording/updates and perception-step insertion, while harness
  checks remain the trust boundary. `src/sft/enrich_plan.py` is the standalone plan-enrichment pass:
  it calls the external model for initial `plan` + updates, splices plan steps into a verified
  trajectory, then replays through the harness so `EnvironmentState` and terminal correctness are
  checked. Its `--dry-run-template` mode is only for local smoke tests, not final training data.
- **7B / V2a and V2-ctx evaluations done (2026-06-15)**: V2a scores **66.83%** and V2-ctx scores
  **62.77%** on the full 1,034-example Spider dev set, versus v1 **68.38%** and direct SQL
  **69.25%**. V2-ctx preserves the large-database context invariant but exposes a planning weakness:
  current trajectories teach perception as a fixed ritual rather than evidence that can change the
  next action. In all 7,767 V2-ctx train+dev trajectories, every one of the 6,361
  `read_subtable` calls is penultimate and immediately followed by `answer_from_context`.
- **Qwen3.5-9B pilot SFT (2026-06-25)**: base model is now local on NewGNN at
  `/home/dengyan/models/Qwen3.5-9B`; Claude's baseline scripts live in `/home/dengyan/run_qwen35_*.sh`.
  Baselines already run on Spider dev: direct SQL no-thinking **751/1034 = 72.6%**, direct SQL
  thinking **756/1034 = 73.1%**, tool 2-shot no-thinking **422/1034 = 40.8%**, tool 2-shot thinking
  **401/1034 = 38.8%**. Qwen3.5 has a 248k vocab, so cross-entropy logits OOM on long records even
  when activation memory fits: the 178-record v8 pilot at 8192 cutoff OOMed after 3/46 steps. The
  working pilot uses Qwen3.5-tokenized <=4096 records (`175/178` kept; dropped indices 140,176,177)
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, output under
  `/data/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-spider-v8-pilot-ready-4k-qlora`. It
  completed 44 steps / 2 epochs in **39:21**, final train loss **0.5535**, peak GPU memory about
  **23.5/24.6 GiB** on one RTX 3090. A matched 4-epoch rerun on the same 175 records completed
  88 steps in **1:17:28**, final train loss **0.4179** and last logged loss **0.2633**, output under
  `/data/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-spider-v8-pilot-ready-4k-epoch4-qlora`.
  Full Spider dev tool zero-shot evals (no thinking, vLLM `--enforce-eager`, max len 4096) are done:
  2 epochs **377/1034 = 36.5%** and 4 epochs **440/1034 = 42.6%**. The 4-epoch run fixes some
  simple/mid-complexity tool chains (e.g. the youngest-singer songs case), but complex schemas with
  joins/set difference still produce many `steps=0` API/protocol failures; use the saved failure
  artifacts for the next correction-data loop. Local helper/configs:
  `src/sft/make_qwen35_4k_subset.py` and `src/sft/configs/qwen3.5_9b_qlora_v8_pilot_ready_4k.yaml`.
- **RL pilot environment prepared (2026-06-27)**: first-stage RL work should start from `src/rl/`.
  `env.py` wraps the same closed-loop model↔harness protocol used by eval; the trainer only supplies
  assistant text, while the environment owns parsing, tool execution, observation messages, and
  terminal scoring. `reward.py` is an auditable pilot reward (correctness dominates; legal answers,
  valid tool calls, evidence reads get small bonuses; repeated calls, tool/protocol/API failures and
  max-steps get penalties). Before any PPO/GRPO run, use `build_reward_report.py` on rollout/pass@k
  artifacts and manually audit a sample; use `select_pilot_tasks.py` to choose mixed-success or
  legal-but-wrong tasks. Do not train RL directly on API/protocol failure-heavy buckets.
- **Next data iteration (SFT v10 two-lane plan, 2026-06-28)**: do not add a reflection tool,
  `invalidate` state, memory, or any model-visible sidecar. Use two data lanes: (A) clean canonical
  data from gold-SQL verified trajectories with observation/think enrichment; (B) recovery data only
  from current SFT rollout failures on **training-set examples** (never Spider dev / held-out eval),
  where the external LLM rewrites a failed attempt into a validated recovery trajectory under
  harness feedback. Recovery repair defaults to **10 attempts**.
  Keep the mixture mostly clean canonical (about 1600-1700) plus a smaller recovery slice
  (about 300-400). Fixed plan and commands: `draft/sft_v10_two_lane_data_plan.md`. Candidate
  selector: `src/sft/select_recovery_candidates.py`; train rollout input builder:
  `src/sft/build_rollout_examples.py`. Current protocol index: `tool_design/current_trajectory_protocol.md`.
- **Claude implementation handoff**: read `draft/trajectory_data_generation_v2.md` before changing
  trajectory generation or starting another SFT/RL run. It records the audited blockers, canonical
  `add_to_memory(key, source_step_id)` ownership model, deterministic semantic derivations, online
  harness-authored provenance, diversity sources, implementation order, and acceptance gates.
  BIRD Mini-Dev is downloaded under `data/bird_mini_dev/` and should initially remain evaluation-only.
- **Memory decision (2026-06-14)**: memory remains broader than scalar values, but is typed by
  authority. `derived_value` and non-scalar `evidence_pointer` are harness-grounded from tool
  history; large lists/tables stay in `data_view`. `plan` and `hypothesis` are model-authored
  working state and cannot serve as factual evidence or `value_ref`. Plan SFT can be constructed
  programmatically from the abstract remaining gold Plan, then diversified with verified rollouts;
  an external LLM is optional and never authoritative. See `final_tool_design.md` §1.1 and the v2
  handoff document §3.
- **Long-context OOM result**: the 9,113-token longest record OOMs at cutoff 10,240. At cutoff
  8,192, standard AdamW OOMs only after its first optimizer-state allocation, so a one-step smoke
  is misleading. Rank 16 + paged 8-bit AdamW passed two worst-case updates at 23,646/24,576 MiB.
  The full run uses `spider_tools_v1_8k`: 6,763 records after dropping 8 records over 7,900 raw
  content tokens; longest kept is 7,701. Remote SFT env now has `bitsandbytes==0.46.1`.
- **v1 SFT data built**: `build_sft_data.py` now accepts `--input-pattern`, `--output-dir`,
  `--output-prefix`, and `--dataset-name`. The reproducible v1 command uses the think-filled
  trajectories and `--max-est-tokens 8900`, producing 6771 train + 998 dev records under
  `data/sft/spider_v1_*`. It deliberately drops `spider_train_3698` and `spider_train_3697`;
  Qwen2.5-7B tokenizer audit found them above/too close to the 10240 training cutoff. Final raw
  content-token maximum is 9113 train / 8166 dev (6 train records exceed 8192), so keep
  `cutoff_len: 10240`. The files are synced to `~/tabular_rl_project/data/sft/` on NewGNN.

## Memory v2 design (shared decision, 2026-06-14)

**Status: `derived_value` (scalar) is IMPLEMENTED & verified as V2a — see the Current-state V2a entry
and `draft/v2a_memory_report.md`.** `evidence_pointer`/`plan`/`hypothesis` remain design-only (V2b/V2c).
The implemented field names for the scalar case are `key`/`content`/`derivation` (not the design's
`definition`/`description`); the ownership rules below hold unchanged.

Canonical details live in `tool_design/final_tool_design.md` §1.1 and
`draft/trajectory_data_generation_v2.md` §3. Claude and Codex must follow these rules when changing
the compiler, emitter, harness, protocol, trajectory schema, SFT construction, or RL rewards.

- Memory is a typed task-level workspace, not an untyped scalar dictionary.
- `derived_value`: a scalar or small structured result extracted by the harness from a cited tool
  output. It has `harness_grounded` authority and may support the final answer. Only this type may
  be consumed through `condition_filter.value_ref`.
- `evidence_pointer`: a semantic pointer to a non-scalar intermediate result such as a filtered
  row set, grouped table, join result, or ranked subset. The actual rows remain in `data_view`;
  memory stores the table handle, structured operation definition, source step ids, and a compact
  description. Do not copy large row lists or tables into memory.
- `plan`: model-authored future goals/subgoals and their statuses. It is control state rather than
  evidence, cannot support the final answer, and cannot be used as `value_ref`.
- `hypothesis`: a model-authored tentative claim with `unverified` status. It should be paired with
  `refine_memory`: later tool evidence may cause the harness to mark it `confirmed`, `rejected`, or
  `revised`. The model may propose an update, but only the harness may grant `confirmed` status.

For grounded non-scalar memory, separate the fields by ownership:

- `definition`: structured operation semantics copied from the executed tool call, owned by the
  harness. Example: input table + `condition_filter` + exact conditions + output table.
- `description`: deterministic readable rendering of `definition`, owned by the harness. Templates
  may accurately describe what operation produced the result, but must not invent task-level
  interpretations.
- `purpose`: optional model-authored explanation of why the result may be useful. It is not factual
  authority.
- `source_step_ids` and the `data_view` handle: harness-authored provenance and data authority.

Thus code can reliably render descriptions such as "rows from customers where risk_score > 0.8"
or "employee counts grouped by department". Claims such as "these are the customers most worth
contacting" belong in `purpose` or `hypothesis`, not in the grounded description.

Data construction does not require an external LLM by default. Build `derived_value` and
`evidence_pointer` deterministically from verified Plan steps and tool outputs. Build initial plan
memory from an abstract form of the remaining gold Plan, without leaking answer values, then add
diversity from execution-verified rollouts. External LLM output is only a candidate source for
wording, alternate plans, or hypotheses; execution and provenance checks remain the acceptance
gate.

## Architecture (`src/harness/`)

- `executor.py` — relational core. Each table-producing tool registers a named SQL view; reading/
  scalar tools run a SELECT. Tools: condition_filter, project, join_tables (prefixes columns
  internally via left_prefix/right_prefix), group_aggregate, aggregate, extreme_value_select
  (merged old order_limit; table-producing), set_op, derive_column, window, preview
  (inlines table content for model perception), rows, gold.
- `plan.py` — Plan IR: `Step(id, tool, args)`. `run_plan` threads step ids → view names; a `values`
  map threads scalars (V2b: aggregate parks its scalar under its step id; a predicate's `value_ref`
  points at the producing step directly); `resolve_cond` resolves `value_ref` and `in_table` in condition trees.
- `compiler.py` — `Compiler(schema).compile(sql)`: sqlglot AST → Plan, walking FROM/JOIN → WHERE →
  GROUP → HAVING → ORDER/LIMIT → SELECT → DISTINCT. Scalar subquery → aggregate → predicate
  `value_ref` = that aggregate step's id (no add_to_memory). IN/NOT-IN subquery → membership via
  in_table. Unsupported → CompileError.
- `scalar_grounding.py` (V2b, SHARED by emitter + rollout) — `extract_scalar(history, source_step_id)`
  / `ground_scalar_reference`: strict scalar extraction (scalar tool / 1×1 table, non-NULL else
  `ScalarGroundingError`). The ONLY place a `value_ref`'s scalar is produced; no memory_id/key/content.
- `provenance.py` (V2b, SHARED by emitter + rollout) — `build_references(tool, args, resolve_step)`
  builds typed `references` edges (`type=data|value`, structured `target`) over the model-facing arg
  shape; `backward_slice(traj, reference_type=("data","value"))` is parameterized (grounding excluded
  by default). `col_lineage` + `grounding` edges land in V2c (阶段 C).
- `verify.py` — `round_trip`: compile → run → compare to gold SQL on the real DB.
- `emitter.py` — verified Plan → training trajectory. Each step gets harness-authored typed
  `references` (via `provenance.build_references`) + `produces`; a predicate's `value_ref` cites the
  producing step directly (no add_to_memory step). `provenance.backward_slice(traj)` reverse-derives
  the answer's dependency set; `validate()` = legality + reference-integrity gate.
  Trajectories carry `schema_version`.
- `run_all.py` (tests + compile coverage), `run_spider.py [N]` (execution-verified on real DBs),
  `gen_trajectories.py [train|dev]` (batch emit → `data/trajectories/spider_*.jsonl`, gitignored).

## SFT pipeline (`src/sft/`, `src/eval/`)

- `protocol.py` — SINGLE source of truth for the model↔harness protocol (system prompt + tool specs
  + `<think>`/`<tool_call>` rendering + parse). V2a: observation envelope `{step_id, status, output}`
  (`tool_output_message(step_id, output)`), `validate_arguments` strict per-tool schema, `PROTOCOL_VERSION`
  + `protocol_hash()`. Imported by both build_sft_data and rollout so SFT and eval can never drift.
- `build_sft_data.py` — trajectories → LLaMA-Factory sharegpt jsonl (loss on assistant turns only);
  manifest records `protocol_version`/`protocol_hash`.
- `fill_think.py` — replaces templated `think` with grounded reasoning from the external LLM (api.md).
  `splice_think.py` — reuse prior-version `think` for a new schema at ZERO API cost (memory steps get
  the template; renamed memory keys are substituted).
- `rollout.py` — closed-loop eval (live model ↔ harness) + `--replay`. V2b: online step_ids +
  `tool_history` + harness-derived typed `references` (`provenance.build_references`) + online
  `value_ref` grounding via `scalar_grounding.extract_scalar` (illegal ref → execution_error) —
  model-claimed provenance is never trusted. Doubles as the RL env.
- Configs: `src/sft/configs/qwen2.5_{3b_lora,7b_qlora}_sft.yaml`.

## Experiment dashboard

- `experiment_dashboard/` is the local React experiment console. It records the dataset, training
  configuration, loss history, elapsed time, evaluation summaries, and per-case JSON for each run.
- `experiment_dashboard/data/experiments.json` is the editable experiment registry;
  `data/trainer_states/*.json` are local snapshots of remote LLaMA-Factory trainer states.
- The registry includes the complete direct-SQL and two-shot tool baselines as first-class
  experiments. The backend dynamically enumerates every existing artifact for each experiment
  (including `all.jsonl`, manifests, summaries, trainer state, and smoke runs); do not restore a
  hard-coded frontend source list.
- JSON records use a structure-aware viewer (conversation, tool trajectory, direct-SQL comparison,
  or generic collapsible tree) with raw JSON as an alternate view.
- The dependency-free Python API scans repository JSONL/manifests, refreshes trainer state through
  `ssh table_rl`, serves the built React app, and proxies OpenAI-compatible requests to
  `VLLM_BASE_URL` (default `http://127.0.0.1:18001/v1`; tunnel port 18001→dell vLLM 8000).
- After every remote evaluation finishes, sync the result directory back into local
  `data/results/...` and make sure the dashboard registry points to it. The dashboard refresh button
  now syncs registered evaluation directories from table_rl as well as trainer state; use it or an
  explicit `scp -r` before expecting the frontend to show the latest run.
- The Playground system prompt is loaded directly from `src/sft/protocol.py`; do not duplicate or
  hand-maintain a second prompt in React. Its service controller may start only registry-backed
  LoRA adapters on an actually idle table_rl GPU. It tracks its own PID/metadata under remote
  `logs/dashboard_vllm.*` and must never stop an unowned process.
- Run the API with `.venv/bin/python experiment_dashboard/backend/server.py`; run the frontend with
  `cd experiment_dashboard/frontend && npm run dev`, or `npm run build` and use the API server alone.

## How to run (local, Mac)

```
.venv/bin/python src/harness/run_all.py          # unit tests + Spider compile coverage
.venv/bin/python src/harness/run_spider.py 2000  # execution-verified on real DBs
.venv/bin/python src/harness/gen_trajectories.py train   # (and dev)
.venv/bin/python src/sft/fill_think.py --split train --n 99999 --workers 16 --out <path>
.venv/bin/python src/sft/build_sft_data.py both \
  --input-pattern 'data/trajectories/spider_{split}_think.jsonl' \
  --output-prefix spider_v1 --dataset-name spider_tools_v1 --max-est-tokens 8900
```
Uses the project venv `.venv` (sqlglot 30.9). Spider DBs in `data/spider_data/` (gitignored).

## GPU server (`ssh table_rl`)

- **dell PowerEdge T640**, host 10.214.243.15 port 222, user dengyan, key auth.
  **2× RTX 3090 24G, driver 560.35.03 (=CUDA 12.6 max)**, 48 cores, 503G RAM.
  Disk `/dev/sda2` 4.9T (3.2T free), everything under `/home/dengyan`.
  Project at `~/tabular_rl_project`; Qwen3.5-9B at `~/models/Qwen3.5-9B`;
  checkpoints at `~/tabular_rl_outputs/checkpoints/`.
- Internet via reverse tunnel from the Mac: `nohup bash src/sft/tunnel_table_rl.sh > /tmp/tunnel_table_rl.log 2>&1
  & disown` (auto-reconnect loop; -R 28472→Mac clash 7897 gives server egress; -L 18001→dell vLLM 8000
  lets the Mac reach the model). Server side: `export http(s)_proxy=http://127.0.0.1:28472`.
- conda envs: `sft` (torch2.6.0+cu124, llamafactory0.9.5) and `vllm-qwen35` (torch2.10.0+cu126, vllm0.19.1).
- **NewGNN** (8× RTX 3090, driver 550, port 16014) kept for reference; tunnel ports 28471/18000; project
  same path. `/home` was 100% full as of 2026-06-25 — use `/data/dengyan/tabular_rl_outputs/` on NewGNN.

## PITFALLS — hard-won, do not re-step

**Ops / server:**
- **Version pins (table_rl / dell, driver 560 = cu126):** sft env uses torch2.6.0+cu124 (from
  NewGNN freeze, cu124 wheels still install fine on cu126 driver); vllm env uses torch2.10.0+cu126 +
  vllm0.19.1 + transformers5.12.0. Install with `pip install -r req.txt --extra-index-url
  https://download.pytorch.org/whl/cu124` (sft) or `cu126` (vllm). On dell, upgrade pip first
  (`conda run -n ENV pip install --upgrade pip`) — old conda pip ≤26 truncates the available-version
  list and fails to find 1.x packages like accelerate==1.11.0 or aiohappyeyeballs==2.6.2.
- **NewGNN version pins (legacy, driver 550 = cu124):** `vllm==0.8.5.post1` + `torch==2.6.0
  torchaudio==2.6.0 torchvision==0.21.0` + `transformers==4.51.3`. Still relevant if using NewGNN.
- **dell missing `libcuda.so` (triton/bitsandbytes JIT link fails):** dell's driver ships only
  `/usr/lib/x86_64-linux-gnu/libcuda.so.1` (no unversioned 64-bit `libcuda.so` symlink), so QLoRA
  training dies at "Quantizing model to 4 bit" — triton JIT-compiles `cuda_utils` and `gcc -lcuda`
  fails with `collect2: ld returned 1`. Fix (no sudo): `ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1
  ~/cuda_link/libcuda.so` then `export LIBRARY_PATH=$HOME/cuda_link:$LIBRARY_PATH` in every training
  launcher. (vLLM inference doesn't hit this; NewGNN already has the symlink so it never showed there.)
- vLLM/training must run with **`HF_HUB_OFFLINE=1`** (processes have no proxy env; model is cached).
- HF downloads: official huggingface.co via the clash tunnel + **`HF_HUB_DISABLE_XET=1`** (the Xet
  client ignores proxy env); do NOT use hf-mirror (hub ≥1.x rejects its HEAD). Command is
  `hf download` (`huggingface-cli` is deprecated/no-op in hub 1.x).
- **`pkill -f` self-kill:** a pattern matching the running command's own argv (or a literal in a
  comment on the same line) kills the ssh session (exit 255). Use a bracket: `pkill -f "vllm serv[e]"`
  and keep explanatory text in the chat, never on the command line.
- **`sed -i` anchoring:** anchor to line start (`^output_dir:`) — `output_dir: .*` also hit
  `overwrite_output_dir:`.
- **GPU contention:** public box — `nvidia-smi` before every job, set `CUDA_VISIBLE_DEVICES` to a
  truly-idle card (cards get grabbed between checks; in-process GPU 0 = the physical card you pinned).
  OOM from long samples / fragmentation → tune `cutoff_len` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  7B on 24G needs QLoRA (4-bit).
- **Idle GPU detection must be strict:** do not select a card by memory alone. Treat a GPU as idle
  only when memory is low (dashboard default ≤512 MiB), utilization is near zero (default ≤5%), and
  `nvidia-smi --query-compute-apps` shows no compute process for that GPU UUID. Re-check after a
  short sleep immediately before launching vLLM/training to reduce races with other users.
- **vLLM cleanup is mandatory:** after every inference/evaluation test, stop the vLLM server and
  verify with `nvidia-smi` that its GPU memory is released. Do not leave an idle vLLM process
  reserving GPUs. Before killing anything, confirm the PID belongs to user `dengyan` and its command
  is the vLLM instance started for this project.
- Tunnel dies when the Mac sleeps; `tunnel.sh` reconnects and `caffeinate` keeps the Mac awake.

**Code / compiler:**
- sqlglot 30 uses the `from_` arg key (not `from`).
- Column qualification for joins is **internalized in `join_tables`** (no separate rename steps);
  the model sees one `join_tables` call, prefixing is harness-internal.
- Scalar/list threading lives in `plan.resolve_cond`: `value_ref` (V2b: the producing `step_id`,
  grounded to its scalar at execution) and `in_table` (IN-subquery set). Trajectory **display keeps
  value_ref(step_id)/in_table**; execution resolves them — keep this split when editing emitter/run_plan.
- **Scalar trust boundary (V2b):** a predicate's `value_ref` cites only a `step_id`; the harness owns
  the value via `scalar_grounding.extract_scalar` (strict 1×1 non-NULL; online illegal ref →
  execution_error). NEVER let the model write a literal threshold value (the v1 bug that broke replay
  and is forgeable in RL). `references`/`produces` are harness-authored sidecars, never `tool_call.arguments`.
- **Verification gate is sound:** unsupported SQL → CompileError (counted, never mis-compiled);
  wrong decompositions → round-trip mismatch → dropped. New datasets lower coverage, never corrupt.
- **Ceiling:** execution-verified ~98.4% is the hard ceiling — 100% is impossible because
  `ORDER BY agg LIMIT k` with ties makes the gold SQL itself nondeterministic. ZERO real
  decomposition bugs remain; remaining failures are ties + 2 exotic 4-way self-joins.

## Conventions

- Two memory layers: this `AGENTS.md` (shared, repo, durable) + Claude's private auto-memory under
  `~/.claude/projects/<proj>/memory/` (Claude-only working notes). Put anything Codex should know
  HERE.
- Design discussion notes live in `draft/` (process_reward_density, subtable_vs_memory,
  table_text_handling, experiment_design, sft_plan, tool_design_summary, trajectory_data_generation_v2,
  v2a_memory_report).
- User preference: narrate every executed command + one-line why (the user checks and learns ops).
- Commit on the user's request; branch off master only if asked (repo convention is direct-to-master).
