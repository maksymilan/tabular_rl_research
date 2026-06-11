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

## Current state (2026-06-11)

- **v0 SFT done** (Qwen2.5-3B LoRA): exec-acc **63.8%** on full Spider dev vs text-to-SQL parity line
  **64–66%** vs base+few-shot **4%**. Conclusion: SFT = behavior cloning of interface + canonical
  plans (NOT a contribution by itself); model is feedback-blind (e.g. filters `country='French'`,
  gets 0 rows, ignores it). The RL/process-reward delta is the actual research story.
- **v1 data done**: 6773 train + 998 dev execution-verified+legal trajectories, **268 with
  `add_to_memory`** (scalar subqueries), think-filled by the external LLM (99.7%). Compile coverage
  99.9%, execution-verified **98.4%** on 2000q. System prompt updated with `add_to_memory` +
  `value_ref`/`in_table` predicate forms.
- **Next**: build the v1 SFT set from `data/trajectories/spider_{split}_think.jsonl`; 7B baseline
  (parallel-safe, no data dep); 7B / v1 SFT; then v1-c (`semantic_match` via literal-fuzzing +
  embedding backend) and RL (`scripts/eval/rollout.py` is the env loop).

## Architecture (`scripts/harness/`)

- `executor.py` — relational core. Each table-producing tool registers a named SQL view; reading/
  scalar tools run a SELECT. Tools: condition_filter, project, join_tables (prefixes columns
  internally via left_prefix/right_prefix), group_aggregate, aggregate, extreme_value_select
  (merged old order_limit; table-producing), set_op, derive_column, window, add_to_memory, preview
  (inlines table content for model perception), rows, gold.
- `plan.py` — Plan IR: `Step(id, tool, args)`. `run_plan` threads step ids → view names; a `values`
  map threads scalars; `resolve_cond` resolves `value_ref` (scalar parked in memory) and `in_table`
  (IN-subquery membership table) inside condition trees.
- `compiler.py` — `Compiler(schema).compile(sql)`: sqlglot AST → Plan, walking FROM/JOIN → WHERE →
  GROUP → HAVING → ORDER/LIMIT → SELECT → DISTINCT. Scalar subquery → aggregate → add_to_memory →
  predicate via value_ref. IN/NOT-IN subquery → membership via in_table. Unsupported → CompileError.
- `verify.py` — `round_trip`: compile → run → compare to gold SQL on the real DB.
- `emitter.py` — verified Plan → training trajectory (ReAct steps + terminal answer_from_context;
  tool_output inlines table content). `validate()` = legality gate.
- `run_all.py` (tests + compile coverage), `run_spider.py [N]` (execution-verified on real DBs),
  `gen_trajectories.py [train|dev]` (batch emit → `data/trajectories/spider_*.jsonl`, gitignored).

## SFT pipeline (`scripts/sft/`, `scripts/eval/`)

- `protocol.py` — SINGLE source of truth for the model↔harness protocol (system prompt + tool specs
  + `<think>`/`<tool_call>` rendering + parse). Imported by both build_sft_data and rollout so the
  SFT format and eval format can never drift.
- `build_sft_data.py` — trajectories → LLaMA-Factory sharegpt jsonl (loss on assistant turns only).
- `fill_think.py` — replaces templated `think` with grounded reasoning from the external LLM
  (api.md). Incremental/resumable (appends per trajectory; re-run skips done ids).
- `rollout.py` — closed-loop eval (live model ↔ harness) + `--replay`. Doubles as the future RL env.
- Configs: `scripts/sft/configs/qwen2.5_{3b_lora,7b_qlora}_sft.yaml`.

## How to run (local, Mac)

```
.venv/bin/python scripts/harness/run_all.py          # unit tests + Spider compile coverage
.venv/bin/python scripts/harness/run_spider.py 2000  # execution-verified on real DBs
.venv/bin/python scripts/harness/gen_trajectories.py train   # (and dev)
.venv/bin/python scripts/sft/fill_think.py --split train --n 99999 --workers 16 --out <path>
```
Uses the project venv `.venv` (sqlglot 30.9). Spider DBs in `data/spider_data/` (gitignored).

## GPU server (`ssh NewGNN`)

- host zju, user dengyan, key auth. 8× RTX 3090 24G, **driver 550 = CUDA 12.4 max**, NO direct net.
- **Writable only `/home/dengyan` and `/data/dengyan`; `/data` is ~full → keep everything in
  `/home/dengyan`.** Project at `~/tabular_rl_project`. Models in `~/.cache/huggingface/hub/`.
- Internet via reverse tunnel from the Mac: `nohup bash scripts/sft/tunnel.sh > /tmp/tunnel.log 2>&1
  & disown` (auto-reconnect loop; -R 28471→Mac clash 7897 gives the server egress; -L 18000→server
  vLLM 8000 lets the Mac reach the model). Server side: `export http(s)_proxy=http://127.0.0.1:28471`.

## PITFALLS — hard-won, do not re-step

**Ops / server:**
- **Version pins (critical):** driver 550 = cu124; latest torch (2.11+) ships cu13 wheels needing
  driver ≥580 and won't run. Pin the WHOLE family: `vllm==0.8.5.post1` (brings torch 2.6.0 cu124),
  and `torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0` (unpinned siblings pull cu13 →
  `libcudart.so.13 not found`), `transformers==4.51.3` (5.x breaks vllm 0.8.5: ProcessorMixin).
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
- Tunnel dies when the Mac sleeps; `tunnel.sh` reconnects and `caffeinate` keeps the Mac awake.

**Code / compiler:**
- sqlglot 30 uses the `from_` arg key (not `from`).
- Column qualification for joins is **internalized in `join_tables`** (no separate rename steps);
  the model sees one `join_tables` call, prefixing is harness-internal.
- Scalar/list threading lives in `plan.resolve_cond`: `value_ref` (scalar via add_to_memory) and
  `in_table` (IN-subquery set). Trajectory **display keeps value_ref/in_table**; execution resolves
  them — keep this split when editing emitter/run_plan.
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
  table_text_handling, experiment_design, sft_plan, tool_design_summary).
- User preference: narrate every executed command + one-line why (the user checks and learns ops).
- Commit on the user's request; branch off master only if asked (repo convention is direct-to-master).
