#!/usr/bin/env python3
"""Build an isolated RL source copy with bounded performance probes."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count} in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def patch_rollout(path: Path) -> None:
    replace_once(
        path,
        "import json\nfrom collections import Counter",
        "import json\nimport time\nfrom collections import Counter",
        "rollout imports",
    )
    replace_once(
        path,
        "from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn\n",
        "from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn\nfrom rl.frameworks.trl.perf_trace import trace_event, tracing_enabled\n",
        "rollout trace import",
    )
    replace_once(
        path,
        "        settings = self.settings\n        sample_counts: Counter[int] = Counter()\n",
        "        settings = self.settings\n        collect_started = time.perf_counter()\n        generation_calls = 0\n        total_generated_tokens = 0\n        harness_apply_seconds = 0.0\n        sample_counts: Counter[int] = Counter()\n",
        "collect setup",
    )
    replace_once(
        path,
        "                if self.generate_batch_with_keys_override is not None:\n",
        "                generation_started = time.perf_counter()\n                if self.generate_batch_with_keys_override is not None:\n",
        "generation start",
    )
    replace_once(
        path,
        "                else:\n                    output = self._generate_with_trainer(active_prompts, trainer)\n                prompt_id_rows = output[\"prompt_ids\"]\n",
        "                else:\n                    output = self._generate_with_trainer(active_prompts, trainer)\n                generation_calls += 1\n                response_rows = output.get(\"completion_ids\") or []\n                generated_tokens = sum(len(row) for row in response_rows)\n                total_generated_tokens += generated_tokens\n                if tracing_enabled():\n                    trace_event(\n                        \"rollout_generation\",\n                        elapsed_seconds=time.perf_counter() - generation_started,\n                        active_rows=len(active_indices),\n                        prompt_tokens=sum(len(row) for row in local_prompt_ids),\n                        response_tokens=generated_tokens,\n                        turn_indices=[len(policy_turns[index]) for index in active_indices],\n                    )\n                prompt_id_rows = output[\"prompt_ids\"]\n",
        "generation timing",
    )
    replace_once(
        path,
        "                    env.apply_model_output(text)\n",
        "                    apply_started = time.perf_counter()\n                    env.apply_model_output(text)\n                    harness_apply_seconds += time.perf_counter() - apply_started\n",
        "harness apply timing",
    )
    replace_once(
        path,
        "            self._write_rollouts(episodes)\n            return episodes\n",
        "            self._write_rollouts(episodes)\n            if tracing_enabled():\n                trace_event(\n                    \"rollout_collect\",\n                    elapsed_seconds=time.perf_counter() - collect_started,\n                    generation_calls=generation_calls,\n                    episodes=len(episodes),\n                    total_generated_tokens=total_generated_tokens,\n                    harness_apply_seconds=harness_apply_seconds,\n                    transitions=sum(len(episode.policy_turns) for episode in episodes),\n                )\n            return episodes\n",
        "collect summary",
    )


def patch_transition(path: Path) -> None:
    replace_once(
        path,
        "from rl.frameworks.trl.mechanism import RLMechanism\n",
        "from rl.frameworks.trl.mechanism import RLMechanism\nfrom rl.frameworks.trl.perf_trace import cuda_snapshot, cuda_sync, trace_event, tracing_enabled\n",
        "trainer trace import",
    )
    replace_once(
        path,
        "        microbatch_plan: _TransitionMicrobatchPlan | None = None,\n    ):\n        \"\"\"Evaluate transition logprobs in compact, length-bucketed microbatches.\"\"\"\n",
        "        microbatch_plan: _TransitionMicrobatchPlan | None = None,\n        trace_label: str = \"token_logps\",\n    ):\n        \"\"\"Evaluate transition logprobs in compact, length-bucketed microbatches.\"\"\"\n",
        "token logprob signature",
    )
    replace_once(
        path,
        "        global_completion_width = microbatch_plan.completion_width\n        rows = []\n        preallocated = None\n        for range_index, (start, end) in enumerate(microbatch_plan.ranges):\n            micro_inputs = self._slice_batch(\n",
        "        global_completion_width = microbatch_plan.completion_width\n        rows = []\n        preallocated = None\n        for range_index, (start, end) in enumerate(microbatch_plan.ranges):\n            microbatch_started = time.perf_counter()\n            cuda_sync()\n            micro_inputs = self._slice_batch(\n",
        "token logprob start",
    )
    replace_once(
        path,
        "                rows.append(per_token_logps)\n        if preallocated is not None:\n",
        "                rows.append(per_token_logps)\n            cuda_sync()\n            if tracing_enabled():\n                trace_event(\n                    \"token_logps_microbatch\",\n                    label=trace_label,\n                    range_index=range_index,\n                    rows=end - start,\n                    prompt_width=int(micro_inputs[\"prompt_ids\"].shape[1]),\n                    completion_width=local_completion_width,\n                    padded_tokens=(end - start) * (int(micro_inputs[\"prompt_ids\"].shape[1]) + local_completion_width),\n                    elapsed_seconds=time.perf_counter() - microbatch_started,\n                    **cuda_snapshot(),\n                )\n        if preallocated is not None:\n",
        "token logprob event",
    )
    replace_once(
        path,
        "                microbatch_plan=microbatch_plan,\n            )\n            if self.beta != 0.0:\n",
        "                microbatch_plan=microbatch_plan,\n                trace_label=\"old_policy\",\n            )\n            if self.beta != 0.0:\n",
        "old policy label",
    )
    replace_once(
        path,
        "                        microbatch_plan=microbatch_plan,\n                    )\n            else:\n",
        "                        microbatch_plan=microbatch_plan,\n                        trace_label=\"reference\",\n                    )\n            else:\n",
        "reference label",
    )
    replace_once(
        path,
        "        started = time.perf_counter()\n        model.train()\n        inputs = self._prepare_inputs(inputs)\n",
        "        started = time.perf_counter()\n        step_index = int(getattr(self.state, \"global_step\", 0))\n        model.train()\n        inputs = self._prepare_inputs(inputs)\n",
        "training step setup",
    )
    replace_once(
        path,
        "        microbatch_plan = self._build_transition_microbatch_plan(\n            inputs,\n            prompt_lengths=prompt_lengths,\n            completion_lengths=completion_lengths,\n        )\n        self._assert_fsdp_microbatch_alignment(\n",
        "        plan_started = time.perf_counter()\n        microbatch_plan = self._build_transition_microbatch_plan(\n            inputs,\n            prompt_lengths=prompt_lengths,\n            completion_lengths=completion_lengths,\n        )\n        if tracing_enabled():\n            trace_event(\n                \"train_microbatch_plan\",\n                step=step_index,\n                rows=total,\n                microbatch_count=len(microbatch_plan.ranges),\n                prompt_width=microbatch_plan.prompt_width,\n                completion_width=microbatch_plan.completion_width,\n                elapsed_seconds=time.perf_counter() - plan_started,\n            )\n        self._assert_fsdp_microbatch_alignment(\n",
        "training plan trace",
    )
    replace_once(
        path,
        "        if self.policy_loss_coefficient != 0.0:\n            for range_index, (start, end) in enumerate(microbatch_plan.ranges):\n                micro_inputs = self._slice_batch(\n",
        "        if self.policy_loss_coefficient != 0.0:\n            for range_index, (start, end) in enumerate(microbatch_plan.ranges):\n                microbatch_started = time.perf_counter()\n                cuda_sync()\n                micro_inputs = self._slice_batch(\n",
        "training policy microbatch start",
    )
    replace_once(
        path,
        "                    self.accelerator.backward(scaled_loss)\n                detached_loss = detached_loss + scaled_loss.detach()\n",
        "                    backward_started = time.perf_counter()\n                    self.accelerator.backward(scaled_loss)\n                    cuda_sync()\n                    backward_seconds = time.perf_counter() - backward_started\n                if tracing_enabled():\n                    trace_event(\n                        \"train_policy_microbatch\",\n                        step=step_index,\n                        range_index=range_index,\n                        rows=end - start,\n                        prompt_width=int(micro_inputs[\"prompt_ids\"].shape[1]),\n                        completion_width=int(micro_inputs[\"completion_ids\"].shape[1]),\n                        padded_tokens=(end - start) * (int(micro_inputs[\"prompt_ids\"].shape[1]) + int(micro_inputs[\"completion_ids\"].shape[1])),\n                        synchronize=synchronize,\n                        backward_seconds=backward_seconds,\n                        elapsed_seconds=time.perf_counter() - microbatch_started,\n                        **cuda_snapshot(),\n                    )\n                detached_loss = detached_loss + scaled_loss.detach()\n",
        "training policy microbatch trace",
    )
    replace_once(
        path,
        "        self._step += 1\n        self._current_train_step_time += time.perf_counter() - started\n",
        "        self._step += 1\n        total_seconds = time.perf_counter() - started\n        if tracing_enabled():\n            trace_event(\n                \"train_step\",\n                step=step_index,\n                rows=total,\n                microbatch_count=len(microbatch_plan.ranges),\n                elapsed_seconds=total_seconds,\n                policy_backward_seconds=policy_finished - policy_started,\n                rank_coefficients_seconds=rank_coefficients_finished - policy_finished,\n                rank_backward_seconds=rank_backward_finished - rank_coefficients_finished,\n                **cuda_snapshot(),\n            )\n        self._current_train_step_time += total_seconds\n",
        "training step summary",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--trace-module", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    if target.exists():
        raise SystemExit(f"target already exists: {target}")
    shutil.copytree(source, target, symlinks=True)
    destination = target / "src/rl/frameworks/trl/perf_trace.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.trace_module.resolve(), destination)
    patch_rollout(target / "src/rl/frameworks/trl/rollout.py")
    patch_transition(target / "src/rl/frameworks/trl/transition_grpo.py")
    print(target)


if __name__ == "__main__":
    main()
