"""One-base, three-adapter backend for a 24 GiB RTX 3090.

The base model is loaded once in 4-bit form.  The trainable ``student`` and the
two frozen teacher adapters share that base.  Adapter switching is serialized;
independent harness environments remain batched during rollout generation.
"""
from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence


SFT2_ADAPTER_SHA256 = "d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e"


def adapter_weight_path(path: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"adapter has no weight file: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MultiAdapterPolicyBackend:
    """Quantized causal LM with one trainable and two frozen LoRA identities."""

    STUDENT = "student"
    SFT2 = "sft2_teacher"
    EXP15 = "exp15_teacher"

    def __init__(
        self,
        *,
        model_path: Path,
        sft2_adapter_path: Path,
        exp15_adapter_path: Path,
        student_adapter_path: Path | None = None,
        max_context_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        learning_rate: float,
        weight_decay: float,
        gradient_checkpointing: bool = True,
    ) -> None:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        actual_sft2_sha = sha256_file(adapter_weight_path(sft2_adapter_path))
        if actual_sft2_sha != SFT2_ADAPTER_SHA256:
            raise ValueError(
                "SFT2 checkpoint-1682 adapter SHA256 mismatch: "
                f"{actual_sft2_sha} != {SFT2_ADAPTER_SHA256}"
            )
        if max_context_tokens < 1024 or max_new_tokens < 1:
            raise ValueError("invalid model context or generation budget")
        if temperature <= 0.0 or not 0.0 < top_p <= 1.0:
            raise ValueError("streaming rollout requires temperature>0 and top_p in (0,1]")

        self.torch = torch
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        student_source = student_adapter_path or sft2_adapter_path
        self.model = PeftModel.from_pretrained(
            base,
            student_source,
            adapter_name=self.STUDENT,
            is_trainable=True,
        )
        self.model.load_adapter(
            sft2_adapter_path,
            adapter_name=self.SFT2,
            is_trainable=False,
        )
        self.model.load_adapter(
            exp15_adapter_path,
            adapter_name=self.EXP15,
            is_trainable=False,
        )
        if gradient_checkpointing:
            # The non-reentrant implementation releases more intermediate
            # state and matches the configuration used by the established
            # single-GPU transition trainer.  It does not change the scored
            # sequence or objective.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.model.enable_input_require_grads()

        raw_base = getattr(self.model, "base_model", self.model)
        raw_base = getattr(raw_base, "model", raw_base)
        signature = inspect.signature(raw_base.forward)
        self.keep_argument = next(
            (
                name
                for name in ("logits_to_keep", "num_logits_to_keep")
                if name in signature.parameters
            ),
            None,
        )
        self._activate(self.STUDENT)
        # A pair of 155 MiB adapter files occupies substantially more once
        # materialized in GPU compute dtype.  Only one frozen teacher is ever
        # active, so keep both on CPU outside their short inference contexts.
        # The trainable student stays resident and optimizer-owned throughout.
        self._move_adapter(self.SFT2, torch.device("cpu"))
        self._move_adapter(self.EXP15, torch.device("cpu"))
        torch.cuda.empty_cache()
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("student adapter exposes no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    @staticmethod
    def _is_student_parameter(name: str) -> bool:
        return MultiAdapterPolicyBackend.STUDENT in name.split(".")

    def _activate(self, adapter: str) -> None:
        if adapter not in {self.STUDENT, self.SFT2, self.EXP15}:
            raise ValueError(f"unknown adapter: {adapter}")
        self.model.set_adapter(adapter)
        # PEFT's set_adapter may toggle trainability.  Reassert the invariant
        # after every switch so frozen teacher weights never enter autograd.
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self._is_student_parameter(name))
        self.active_adapter = adapter

    def _move_adapter(self, adapter: str, device) -> None:
        """Move one LoRA identity without touching the quantized base model."""
        for module in self.model.modules():
            mover = getattr(module, "_move_adapter_to_device_of_base_layer", None)
            if callable(mover):
                mover(adapter, device=device)

    @contextmanager
    def adapter(self, adapter: str):
        previous = getattr(self, "active_adapter", self.STUDENT)
        if adapter != self.STUDENT:
            device = next(self.model.parameters()).device
            self._move_adapter(adapter, device)
        self._activate(adapter)
        try:
            yield
        finally:
            self._activate(previous)
            if adapter != self.STUDENT and adapter != previous:
                self._move_adapter(adapter, self.torch.device("cpu"))
                self.torch.cuda.empty_cache()

    def render(self, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = list(self.tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) + self.max_new_tokens > self.max_context_tokens:
            raise ValueError(
                f"exact prefix exceeds context budget: {len(ids)} + "
                f"{self.max_new_tokens} > {self.max_context_tokens}"
            )
        return text, ids

    @staticmethod
    def _batch_seed(seeds: Sequence[int]) -> int:
        payload = "\0".join(str(int(seed)) for seed in seeds).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF

    def generate(
        self,
        adapter: str,
        prompts: list[str],
        *,
        seeds: Sequence[int],
        include_sampled_logprobs: bool = True,
    ) -> dict[str, Any]:
        """Generate one sampled continuation per prompt in one adapter batch."""
        torch = self.torch
        if len(prompts) != len(seeds) or not prompts:
            raise ValueError("generation prompts and seeds must be non-empty and aligned")
        encoded = self.tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        input_ids = encoded.input_ids.to(device)
        attention_mask = encoded.attention_mask.to(device)
        prompt_rows = [
            input_ids[row][attention_mask[row].bool()].tolist()
            for row in range(input_ids.shape[0])
        ]
        if any(len(row) + self.max_new_tokens > self.max_context_tokens for row in prompt_rows):
            raise ValueError("batched exact prefix exceeds context budget")

        torch.manual_seed(self._batch_seed(seeds))
        with self.adapter(adapter), torch.inference_mode():
            self.model.config.use_cache = True
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                # Retaining one full-vocabulary score tensor per generated
                # token is prohibitive for 16 parallel repair branches.  Exact
                # student raw log-probabilities are recomputed below by prefill;
                # repair branches are rescored only if selected for DPO.
                output_scores=False,
            )
            self.model.config.use_cache = False

        width = input_ids.shape[1]
        completion_matrix = output.sequences[:, width:]
        completion_rows = []
        eos = self.tokenizer.eos_token_id
        for row_index in range(completion_matrix.shape[0]):
            ids: list[int] = []
            for step_index in range(completion_matrix.shape[1]):
                token_id = int(completion_matrix[row_index, step_index])
                ids.append(token_id)
                if eos is not None and token_id == int(eos):
                    break
            completion_rows.append(ids)
        logprob_rows = []
        token_candidate_rows = []
        for prompt_ids, response_ids in zip(prompt_rows, completion_rows, strict=True):
            if include_sampled_logprobs:
                token_logps = self.score_response(
                    adapter,
                    prompt_ids,
                    response_ids,
                    requires_grad=False,
                )
                values = [[float(value)] for value in token_logps.detach().cpu().tolist()]
            else:
                # Repair selection uses the harness verifier; selected branches
                # are later scored exactly under policy/reference adapters.
                values = [[0.0] for _ in response_ids]
            logprob_rows.append(values)
            token_candidate_rows.append([[int(token_id)] for token_id in response_ids])
        return {
            "prompt_ids": prompt_rows,
            "completion_ids": completion_rows,
            "logprobs": logprob_rows,
            "logprob_token_ids": token_candidate_rows,
        }

    def generator_callback(self, adapter: str, *, seed: int, initial_call_index: int = 0):
        if initial_call_index < 0:
            raise ValueError("generator call index cannot be negative")

        def generate_batch(prompts: list[str]) -> dict[str, Any]:
            call_index = int(generate_batch.call_index)
            seeds = [
                self._batch_seed([seed, call_index, row_index])
                for row_index in range(len(prompts))
            ]
            generate_batch.call_index = call_index + 1
            return self.generate(
                adapter,
                prompts,
                seeds=seeds,
                include_sampled_logprobs=True,
            )

        generate_batch.call_index = int(initial_call_index)
        return generate_batch

    def score_response(
        self,
        adapter: str,
        prompt_ids: Sequence[int],
        response_ids: Sequence[int],
        *,
        requires_grad: bool,
    ):
        """Return raw-model log p for every exact response token."""
        torch = self.torch
        if not prompt_ids or not response_ids:
            raise ValueError("response scoring requires non-empty prompt and response")
        total = len(prompt_ids) + len(response_ids)
        if total > self.max_context_tokens:
            raise ValueError(f"scored sequence exceeds context budget: {total}")
        device = next(self.model.parameters()).device
        input_ids = torch.tensor(
            [list(prompt_ids) + list(response_ids)],
            dtype=torch.long,
            device=device,
        )
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "use_cache": False,
        }
        if self.keep_argument:
            kwargs[self.keep_argument] = len(response_ids) + 1
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        # PEFT stores this SFT2 LoRA in fp32.  The established TRL trainer runs
        # its forward/backward under bf16 autocast; without the same context,
        # PEFT casts every long hidden-state input to fp32 for the LoRA path and
        # nearly doubles the activation peak.  Match that proven compute policy
        # while retaining fp32 master adapter weights and optimizer state.
        autocast = torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        )
        # Even with activation checkpointing, QLoRA autograd retains enough
        # exact-prefix tensors to hit the last ~130 MiB on a 24 GiB card.  Move
        # tensors saved for backward to pinned CPU memory and restore them on
        # demand.  This is an exact storage-location change, not truncated BPTT
        # or a change to the OPD/DPO objective.
        saved_tensor_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if requires_grad and device.type == "cuda"
            else nullcontext()
        )
        with self.adapter(adapter), context, autocast, saved_tensor_context:
            try:
                logits = self.model(**kwargs).logits
            except torch.OutOfMemoryError as error:
                allocated = torch.cuda.memory_allocated(device) / (1024**2)
                reserved = torch.cuda.memory_reserved(device) / (1024**2)
                error.add_note(
                    "exact response scoring context: "
                    f"adapter={adapter}, requires_grad={requires_grad}, "
                    f"prompt_tokens={len(prompt_ids)}, "
                    f"response_tokens={len(response_ids)}, total_tokens={total}, "
                    f"allocated_mib={allocated:.1f}, reserved_mib={reserved:.1f}"
                )
                raise
            returned = int(logits.shape[1])
            response_logits = logits[
                0,
                returned - len(response_ids) - 1 : returned - 1,
            ]
            if response_logits.shape[0] != len(response_ids):
                raise RuntimeError("causal logits do not align with response tokens")
            labels = torch.tensor(response_ids, dtype=torch.long, device=device)
            chosen = response_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            token_logps = chosen.float() - torch.logsumexp(response_logits.float(), dim=-1)
        return token_logps if requires_grad else token_logps.detach()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self, *, gradient_clip: float) -> float:
        if gradient_clip <= 0.0:
            raise ValueError("gradient_clip must be positive")
        self._activate(self.STUDENT)
        norm = self.torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            gradient_clip,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(norm.detach().cpu())

    def save_student(self, output_dir: Path) -> Path:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"refusing to overwrite non-empty output: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        self._activate(self.STUDENT)
        self.model.save_pretrained(output_dir, selected_adapters=[self.STUDENT])
        nested = output_dir / self.STUDENT
        if nested.is_dir() and any(
            (nested / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            return nested
        if any(
            (output_dir / name).is_file()
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ):
            return output_dir
        raise RuntimeError("PEFT save completed without a student adapter weight file")

    def save_optimizer(self, path: Path) -> None:
        if path.exists():
            raise ValueError(f"refusing to overwrite optimizer state: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.optimizer.state_dict(), path)

    def load_optimizer(self, path: Path) -> None:
        if not path.is_file():
            raise ValueError(f"optimizer checkpoint is missing: {path}")
        state = self.torch.load(path, map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state)
        self.torch.cuda.empty_cache()
