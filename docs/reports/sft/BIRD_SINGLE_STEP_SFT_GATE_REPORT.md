# BIRD single-step SFT learning gate

Date: 2026-07-17

## Question

Does splitting successful closed-loop trajectories into independently supervised next-action
records actually teach useful behavior, or does it only reduce training loss?

## Model and controls

- Base: Qwen2.5-7B-Instruct.
- SFT: the completed 40-episode BIRD adapter at
  `qwen2.5-7b-bird-scale500-train40-rolling4-full-6400-qlora`.
- Training data: 40 verified BIRD-train episodes split into 262 last-turn-only action targets.
- Historical compatibility: bounded four-turn rolling history, pre-R2 full observations, strict
  parser, and the exact system prompt saved by the earlier holdout run.

## Teacher-forced next-action test

The test uses 20 unseen BIRD-train episodes and 117 step states. They come from the frozen
22-episode grounded eval split after removing the two episode ids present in the old 40-episode
training set. Every model action is generated from the teacher state, so an earlier prediction
cannot corrupt a later input.

| Metric | Base control | 40-episode SFT |
|---|---:|---:|
| Strictly valid action | 0/117 (0.0%) | 116/117 (99.1%) |
| Teacher tool match | 0/117 (0.0%) | 91/117 (77.8%) |
| Exact tool + JSON arguments | 0/117 (0.0%) | 45/117 (38.5%) |
| First-step tool match | 0/20 (0.0%) | 20/20 (100.0%) |
| Later-step tool match | 0/97 (0.0%) | 71/97 (73.2%) |

The base control used the same records with the current read-limit wording; the SFT result was
additionally rerun with the exact historical prompt. The decisive exact-prompt base comparison is
the closed-loop test below, where it produces no legal action sequence.

Exact action agreement is intentionally strict and undercounts equivalent valid actions. For
example, the SFT model chooses `join_tables` on 7/10 join targets but has 0/10 exact JSON matches;
different valid join order/prefix choices are counted as mismatches. Closed-loop execution is the
behavioral test.

## BIRD dev closed-loop test

A deterministic 30-task BIRD-dev sample uses seed 20260717, 11 databases, and a 12/9/9
simple/moderate/challenging split. Both models receive the exact historical prompt and full
pre-R2 observations. They run with temperature 0, at most 30 actions, and at most three errors of
each recoverable type.

| Metric | Base | 40-episode SFT |
|---|---:|---:|
| Execution-correct | 0/30 (0.0%) | 6/30 (20.0%) |
| Legal terminal answer | 0/30 (0.0%) | 24/30 (80.0%) |
| Simple correct | 0/12 | 3/12 |
| Moderate correct | 0/9 | 1/9 |
| Challenging correct | 0/9 | 2/9 |

The base terminates all 30 episodes after three protocol errors (90 protocol events). The SFT
model has five clean successes and one recovered success. Its remaining outcomes are 18 legal but
wrong answers, two protocol terminals, two execution-error terminals, one context overflow, and
one max-step terminal.

## Conclusion

The single-step construction **does teach transferable behavior**. With only 262 targets, it moves
strict action legality from effectively zero to 99% under teacher states, produces legal closed-loop
answers on 80% of unseen BIRD-dev tasks, and solves 20% end to end. This cannot be explained by
memorizing training questions because the closed-loop test uses the separate BIRD dev split and
unseen databases.

The current limitation is semantic rather than formatting: 18/24 legal terminal answers are
wrong, and later-step exact action agreement is only 35%. The next 1,303-target grounded SFT run is
therefore justified, but it must be evaluated on this same fixed dev-30 gate and then a larger BIRD
dev sample before starting RL.

## Artifacts

- Step evaluation:
  `data/results/qwen2.5_7b_bird_train40_pre_r2_step_eval_unseen20_exact_prompt/`
- SFT closed-loop:
  `data/results/qwen2.5_7b_bird_train40_pre_r2_bird_dev_stratified30/`
- Base closed-loop:
  `data/results/qwen2.5_7b_base_pre_r2_bird_dev_stratified30/`
- Frozen dev indices:
  `data/eval_inputs/bird_dev_stratified30_seed20260717.indices.json`
- Generic step evaluator: `src/eval/evaluate_step_actions.py`

