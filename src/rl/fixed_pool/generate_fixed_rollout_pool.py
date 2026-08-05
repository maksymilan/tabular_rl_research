#!/usr/bin/env python3
"""Generate one immutable SFT2 K=4 rollout pool with exact policy token evidence."""
from __future__ import annotations

import asyncio
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from frameworks.trl.fixed_rollout_pool import (  # noqa: E402
    serialize_episode,
    write_rows_atomic,
)
from frameworks.trl.rollout import RolloutSettings, TableAgentRolloutCollector  # noqa: E402
from frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn  # noqa: E402
from protocol import PROTOCOL_VERSION, student_runtime_system_prompt, tool_schema_hash  # noqa: E402
from rollout_scoring import episode_example, score_completed_rollout  # noqa: E402
from task_loader import load_rl_task_records  # noqa: E402
from tool_environment import create_tool_use_env  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rollout_seed(
    base_seed: int,
    task_id: str,
    sample_index: int,
    turn_index: int,
) -> int:
    """Return a scheduling-independent seed for one authored policy turn."""
    digest = hashlib.sha256(
        f"{base_seed}\0{task_id}\0{sample_index}\0{turn_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


class VLLMLoRAGenerator:
    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path,
        temperature: float,
        top_p: float,
        max_tokens: int,
        max_model_len: int,
        seed: int,
        gpu_memory_utilization: float,
    ) -> None:
        self.llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=16,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
        )
        self.lora_request = LoRARequest("sft2-fixed-pool", 1, str(adapter_path))
        self.settings = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
        }

    def __call__(
        self,
        prompts: list[str],
        request_keys: list[tuple[str, int, int]],
    ) -> dict[str, Any]:
        if len(prompts) != len(request_keys):
            raise RuntimeError("prompts and stable rollout request keys do not align")
        params = [
            SamplingParams(
                n=1,
                temperature=self.settings["temperature"],
                top_p=self.settings["top_p"],
                max_tokens=self.settings["max_tokens"],
                logprobs=0,
                seed=stable_rollout_seed(
                    int(self.settings["seed"]),
                    task_id,
                    sample_index,
                    turn_index,
                ),
            )
            for task_id, sample_index, turn_index in request_keys
        ]
        outputs = self.llm.generate(
            prompts,
            params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        result = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "logprob_token_ids": [],
        }
        for output in outputs:
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            sampled_logprobs = []
            sampled_token_ids = []
            if completion.logprobs is None or len(completion.logprobs) != len(token_ids):
                raise RuntimeError("vLLM omitted sampled-token logprobs")
            for token_id, candidates in zip(token_ids, completion.logprobs, strict=True):
                candidate = candidates.get(token_id)
                if candidate is None:
                    raise RuntimeError(f"sampled token {token_id} missing from vLLM logprobs")
                sampled_logprobs.append([float(candidate.logprob)])
                sampled_token_ids.append([int(token_id)])
            result["prompt_ids"].append(list(output.prompt_token_ids))
            result["completion_ids"].append(token_ids)
            result["logprobs"].append(sampled_logprobs)
            result["logprob_token_ids"].append(sampled_token_ids)
        return result


class AsyncVLLMRolloutPool:
    """Advance independent trajectories as soon as each vLLM request finishes."""

    def __init__(
        self,
        *,
        tokenizer,
        settings: RolloutSettings,
        model_path: Path,
        adapter_path: Path,
        seed: int,
        gpu_memory_utilization: float,
    ) -> None:
        self.tokenizer = tokenizer
        self.settings = settings
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.seed = seed
        self.gpu_memory_utilization = gpu_memory_utilization
        self.renderer = TableAgentRolloutCollector(tokenizer, settings)
        self.engine: AsyncLLM | None = None
        self.lora_request = LoRARequest("sft2-fixed-pool-dynamic", 1, str(adapter_path))

    async def start(self) -> None:
        if self.engine is not None:
            raise RuntimeError("dynamic rollout engine is already started")
        engine_args = AsyncEngineArgs(
            model=str(self.model_path),
            dtype="bfloat16",
            max_model_len=self.settings.max_context_tokens,
            max_num_batched_tokens=self.settings.max_context_tokens,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=16,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self.engine = AsyncLLM.from_engine_args(engine_args)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    async def _generate_turn(
        self,
        prompt_text: str,
        *,
        task_id: str,
        sample_index: int,
        turn_index: int,
    ) -> Any:
        if self.engine is None:
            raise RuntimeError("dynamic rollout engine is not started")
        params = SamplingParams(
            n=1,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            max_tokens=self.settings.max_new_tokens,
            logprobs=0,
            seed=stable_rollout_seed(
                self.seed,
                task_id,
                sample_index,
                turn_index,
            ),
        )
        request_id = f"fixed-pool:{task_id}:{sample_index}:{turn_index}"
        final_output = None
        async for output in self.engine.generate(
            prompt_text,
            params,
            request_id,
            lora_request=self.lora_request,
        ):
            final_output = output
        if final_output is None or not final_output.finished:
            raise RuntimeError(f"vLLM request did not finish: {request_id}")
        return final_output

    async def collect_episode(
        self,
        metadata: dict[str, Any],
        sample_index: int,
    ) -> PolicyEpisode:
        settings = self.settings
        task_id = str(metadata["task_id"])
        env = create_tool_use_env(
            episode_example(metadata),
            tool_scheme=settings.tool_scheme,
            example_index=int(metadata["example_index"]),
            max_steps=settings.max_steps,
            max_batch_calls=settings.max_batch_calls,
            context_mode=settings.context_mode,
            history_turns=settings.history_turns,
            compact_observations=True,
            denotation_comparison=settings.denotation_comparison,
        )
        policy_turns: list[PolicyTurn] = []
        scored_turns: list[tuple[list[int], list[int]]] = []
        tokenization_warning = False
        try:
            while not env.done:
                prompt_text, local_prompt_ids = self.renderer._render(env.model_messages())
                if (
                    len(local_prompt_ids) + settings.max_new_tokens
                    > settings.max_context_tokens
                ):
                    env.done = True
                    env.failure_type = "context_overflow"
                    break
                output = await self._generate_turn(
                    prompt_text,
                    task_id=task_id,
                    sample_index=sample_index,
                    turn_index=len(policy_turns),
                )
                completion = output.outputs[0]
                response_ids = list(completion.token_ids)
                if not response_ids:
                    env.done = True
                    env.failure_type = "generation_oom"
                    break
                server_prompt_ids = list(output.prompt_token_ids)
                if server_prompt_ids != local_prompt_ids:
                    tokenization_warning = True
                if completion.logprobs is None or len(completion.logprobs) != len(
                    response_ids
                ):
                    raise RuntimeError("vLLM omitted sampled-token logprobs")
                sampled_values = []
                for token_id, candidates in zip(
                    response_ids,
                    completion.logprobs,
                    strict=True,
                ):
                    candidate = candidates.get(token_id)
                    if candidate is None:
                        raise RuntimeError(
                            f"sampled token {token_id} missing from vLLM logprobs"
                        )
                    sampled_values.append(float(candidate.logprob))
                sampled_logprobs = tuple(sampled_values)
                policy_turn = PolicyTurn(
                    prompt_ids=tuple(server_prompt_ids),
                    response_ids=tuple(response_ids),
                    sampling_logprobs=sampled_logprobs,
                )
                policy_turn.validate()
                policy_turns.append(policy_turn)
                scored_turns.append((server_prompt_ids, response_ids))
                text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                env.apply_model_output(text)

            sample = score_completed_rollout(
                env,
                scored_turns,
                metadata,
                sample_index=sample_index,
                reward_mode="result-only",
                process_config=None,
                process_admission_policy=settings.process_admission_policy,
                denotation_comparison=settings.denotation_comparison,
                counterfactual_suite=None,
            )
            sample.audit_record["example_index"] = int(metadata["example_index"])
            sample.audit_record["sample_index"] = sample_index
            if tokenization_warning:
                sample.audit_record["rollout_tokenization_warning"] = True
            episode = PolicyEpisode(sample=sample, policy_turns=policy_turns)
            episode.validate()
            return episode
        finally:
            env.close()

    async def generate_groups(
        self,
        pending: list[tuple[int, dict[str, Any]]],
        *,
        group_size: int,
        question_window: int,
        task_order: dict[str, int],
        groups_dir: Path,
    ) -> None:
        queue: asyncio.Queue[tuple[int, dict[str, Any]] | None] = asyncio.Queue()
        for item in pending:
            queue.put_nowait(item)
        for _ in range(question_window):
            queue.put_nowait(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    position, record = item
                    metadata = record["environment"]
                    task_id = str(metadata["task_id"])
                    episodes = await asyncio.gather(
                        *(
                            self.collect_episode(metadata, sample_index)
                            for sample_index in range(group_size)
                        )
                    )
                    rows = [
                        serialize_episode(
                            episode,
                            sequence=task_order[task_id] * group_size + sample_index,
                            environment=metadata,
                        )
                        for sample_index, episode in enumerate(episodes)
                    ]
                    group_path = groups_dir / f"{task_id}.json"
                    temporary = group_path.with_suffix(".json.next")
                    temporary.write_text(
                        json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
                    )
                    temporary.replace(group_path)
                    print(
                        json.dumps(
                            {
                                "task": task_id,
                                "position": position,
                                "correct": sum(
                                    row["sample"]["correct"] for row in rows
                                ),
                                "status": "generated",
                                "scheduler": "dynamic",
                                "question_window": question_window,
                            }
                        ),
                        flush=True,
                    )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(question_window)]
        await queue.join()
        await asyncio.gather(*workers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--scheduler",
        choices=("static", "dynamic"),
        default="static",
        help="static step-synchronous batches or a rolling question window",
    )
    parser.add_argument(
        "--question-window",
        type=int,
        default=8,
        help="number of K-sized question groups kept active by the dynamic scheduler",
    )
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=1,
        help=(
            "number of independent questions advanced together; the active vLLM batch "
            "contains task_batch_size * group_size environments"
        ),
    )
    parser.add_argument(
        "--task-id-file",
        type=Path,
        help=(
            "optional newline-delimited task ids assigned to this worker; full task order is "
            "still used for immutable sequence numbers"
        ),
    )
    parser.add_argument(
        "--exclude-task-id-file",
        type=Path,
        help="optional newline-delimited task ids assigned to other parallel workers",
    )
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="write only per-task atomic group files for a parallel worker",
    )
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="skip model loading/generation and assemble all existing groups",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if PROTOCOL_VERSION != "version26":
        raise SystemExit(f"fixed training pool requires version26, got {PROTOCOL_VERSION}")
    if args.task_batch_size < 1:
        raise SystemExit("--task-batch-size must be positive")
    if args.question_window < 1:
        raise SystemExit("--question-window must be positive")
    if args.no_finalize and args.finalize_only:
        raise SystemExit("--no-finalize and --finalize-only are mutually exclusive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.output_dir / "groups"
    groups_dir.mkdir(exist_ok=True)
    raw_tasks = [
        json.loads(line)
        for line in args.tasks.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        raw_tasks = raw_tasks[: args.limit]
    ordered_task_ids = [
        str(
            task.get("example_id")
            or task.get("instance_id")
            or task.get("trajectory_id")
        )
        for task in raw_tasks
    ]
    if any(task_id == "None" for task_id in ordered_task_ids):
        raise RuntimeError("fixed-pool tasks require explicit stable task ids")
    task_order = {task_id: index for index, task_id in enumerate(ordered_task_ids)}
    if len(task_order) != len(raw_tasks):
        raise RuntimeError("fixed-pool task ids must be unique")
    assigned_task_ids = None
    if args.task_id_file is not None:
        assigned_task_ids = {
            line.strip()
            for line in args.task_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        unknown = assigned_task_ids - set(task_order)
        if unknown:
            raise RuntimeError(f"task-id file contains unknown tasks: {sorted(unknown)}")
    excluded_task_ids: set[str] = set()
    if args.exclude_task_id_file is not None:
        excluded_task_ids = {
            line.strip()
            for line in args.exclude_task_id_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        unknown = excluded_task_ids - set(task_order)
        if unknown:
            raise RuntimeError(
                f"exclude-task-id file contains unknown tasks: {sorted(unknown)}"
            )
        if assigned_task_ids is not None and assigned_task_ids & excluded_task_ids:
            raise RuntimeError("included and excluded task-id files overlap")
    include_task_ids = (
        set(ordered_task_ids) if assigned_task_ids is None else assigned_task_ids
    ) - excluded_task_ids
    if args.finalize_only:
        # Finalization is a pure artifact assembly pass.  Loading full DatasetTask
        # records opens every SQLite source (some exceed 1 GiB) even though no model or
        # harness step is executed.  Retain only the two fields used below so this path
        # cannot become database-I/O bound or change existing trajectory content.
        records = [
            {
                "environment": {
                    "task_id": task_id,
                    "example_index": int(task.get("example_index", position)),
                }
            }
            for position, (task_id, task) in enumerate(
                zip(ordered_task_ids, raw_tasks, strict=True)
            )
            if task_id in include_task_ids
        ]
    else:
        records = load_rl_task_records(
            ROOT,
            split="train",
            examples_json=args.tasks,
            seed=args.seed,
            context_mode="rolling-legal-history",
            include_task_ids=include_task_ids,
        )
    assigned_records = records
    collector = None
    tokenizer = None
    settings = None
    if not args.finalize_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        settings = RolloutSettings(
            reward_mode="result-only",
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            history_turns=args.history_turns,
            temperature=args.temperature,
            top_p=args.top_p,
            denotation_comparison="bird-set",
        )
        if args.scheduler == "static":
            generator = VLLMLoRAGenerator(
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_new_tokens,
                max_model_len=args.max_context_tokens,
                seed=args.seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            collector = TableAgentRolloutCollector(
                tokenizer,
                settings,
                generate_batch_with_keys=generator,
            )
    pending: list[tuple[int, dict[str, Any]]] = []
    for record in assigned_records:
        position = task_order[str(record["environment"]["task_id"])] + 1
        metadata = record["environment"]
        task_id = str(metadata["task_id"])
        group_path = groups_dir / f"{task_id}.json"
        if group_path.is_file():
            existing = json.loads(group_path.read_text())
            if len(existing) != args.group_size:
                raise RuntimeError(f"incomplete existing group: {group_path}")
            changed = False
            for sample_index, row in enumerate(existing):
                audit = row["sample"]["audit_record"]
                expected_sequence = task_order[task_id] * args.group_size + sample_index
                if int(row["sequence"]) != expected_sequence:
                    raise RuntimeError(f"unexpected sequence in existing group: {group_path}")
                for key, value in (
                    ("example_index", int(metadata["example_index"])),
                    ("sample_index", sample_index),
                ):
                    previous = audit.get(key)
                    if previous is not None and int(previous) != value:
                        raise RuntimeError(
                            f"inconsistent {key} in existing group: {group_path}"
                        )
                    if previous is None:
                        audit[key] = value
                        changed = True
            if changed:
                temporary = group_path.with_suffix(".json.next")
                temporary.write_text(
                    json.dumps(existing, ensure_ascii=False, separators=(",", ":"))
                )
                temporary.replace(group_path)
            print(json.dumps({"task": task_id, "position": position, "status": "existing"}))
            continue
        pending.append((position, record))

    if pending and args.scheduler == "dynamic":
        if tokenizer is None or settings is None:
            raise RuntimeError("dynamic generation runtime is unavailable")

        async def run_dynamic() -> None:
            pool = AsyncVLLMRolloutPool(
                tokenizer=tokenizer,
                settings=settings,
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                seed=args.seed,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            await pool.start()
            try:
                await pool.generate_groups(
                    pending,
                    group_size=args.group_size,
                    question_window=args.question_window,
                    task_order=task_order,
                    groups_dir=groups_dir,
                )
            finally:
                pool.close()

        asyncio.run(run_dynamic())
    elif pending:
        for batch_start in range(0, len(pending), args.task_batch_size):
            task_batch = pending[batch_start : batch_start + args.task_batch_size]
            rollout_inputs = [
                record
                for _, record in task_batch
                for _ in range(args.group_size)
            ]
            if collector is None:
                raise RuntimeError("generation collector is unavailable")
            episodes = collector.collect(rollout_inputs, trainer=None)
            expected_episodes = len(task_batch) * args.group_size
            if len(episodes) != expected_episodes:
                raise RuntimeError(
                    f"collector returned {len(episodes)} episodes; expected {expected_episodes}"
                )
            for task_offset, (position, record) in enumerate(task_batch):
                metadata = record["environment"]
                task_id = str(metadata["task_id"])
                group_path = groups_dir / f"{task_id}.json"
                episode_start = task_offset * args.group_size
                task_episodes = episodes[episode_start : episode_start + args.group_size]
                rows = []
                for sample_index, episode in enumerate(task_episodes):
                    episode.sample.audit_record["example_index"] = int(
                        metadata["example_index"]
                    )
                    episode.sample.audit_record["sample_index"] = sample_index
                    rows.append(
                        serialize_episode(
                            episode,
                            sequence=task_order[task_id] * args.group_size + sample_index,
                            environment=metadata,
                        )
                    )
                temporary = group_path.with_suffix(".json.next")
                temporary.write_text(
                    json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
                )
                temporary.replace(group_path)
                print(
                    json.dumps(
                        {
                            "task": task_id,
                            "position": position,
                            "correct": sum(row["sample"]["correct"] for row in rows),
                            "status": "generated",
                            "task_batch_size": len(task_batch),
                            "scheduler": "static",
                        }
                    ),
                    flush=True,
                )

    if args.no_finalize:
        print(
            json.dumps(
                {
                    "status": "worker_complete",
                    "assigned_tasks": len(assigned_records),
                    "task_batch_size": args.task_batch_size,
                    "scheduler": args.scheduler,
                    "question_window": (
                        args.question_window if args.scheduler == "dynamic" else None
                    ),
                }
            ),
            flush=True,
        )
        return

    all_rows = []
    for record in records:
        task_id = str(record["environment"]["task_id"])
        group_path = groups_dir / f"{task_id}.json"
        if not group_path.is_file():
            raise RuntimeError(f"cannot finalize; missing group: {group_path}")
        all_rows.extend(json.loads(group_path.read_text()))
    all_rows.sort(key=lambda row: int(row["sequence"]))
    trajectories = args.output_dir / "trajectories.jsonl"
    write_rows_atomic(trajectories, all_rows)
    manifest = {
        "schema_version": "table-agent-fixed-rollout-pool-pending-v1",
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": PROTOCOL_VERSION,
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path),
        "adapter_sha256": sha256_file(args.adapter_path / "adapter_model.safetensors"),
        "tasks_path": str(args.tasks),
        "tasks_sha256": sha256_file(args.tasks),
        "tasks": len(records),
        "group_size": args.group_size,
        "trajectories": len(all_rows),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "max_context_tokens": args.max_context_tokens,
        "history_turns": args.history_turns,
        "seed": args.seed,
        "generation_task_batch_size": args.task_batch_size,
        "generation_scheduler": args.scheduler,
        "generation_question_window": (
            args.question_window if args.scheduler == "dynamic" else None
        ),
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "denotation_comparison": "bird-set",
        "student_prompt_sha256": hashlib.sha256(
            student_runtime_system_prompt(
                context_mode="rolling-legal-history", compact=False
            ).encode()
        ).hexdigest(),
        "tool_schema_sha256": tool_schema_hash(),
        "trajectories_sha256": sha256_file(trajectories),
        "correct_trajectories": sum(row["sample"]["correct"] for row in all_rows),
    }
    (args.output_dir / "manifest.pending.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
