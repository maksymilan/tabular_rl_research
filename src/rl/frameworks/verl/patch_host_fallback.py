#!/usr/bin/env python3
"""Patch a pinned Verl checkout so FSDP Ray workers inherit NCCL host fallback settings.

Verl's Ray worker factory builds a fresh ``runtime_env`` containing only rank-related variables.
On the table_rl host that drops the NCCL fallback settings required by its non-peer-accessible 3090s.
The patch is idempotent and deliberately fails if the upstream code shape changes.
"""
from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "# table_rl host-fallback NCCL environment v2"
LEGACY_MARKER = "# table_rl host-fallback NCCL environment"
NEEDLE = "        if worker_env is not None:\n"
FSDP_MARKER = "# table_rl: all ranks load the same local checkpoint; avoid unsupported P2P init broadcast"
FSDP_NEEDLE = "                sync_module_states=True,\n"
LAYERED_MARKER = "# table_rl: collect locally named LoRA tensors after each FSDP-layer summon"
LAYERED_NEEDLE = """            sub_lora_params = get_peft_model_state_dict(peft_model, state_dict=sub_state_dict)
            if not sub_lora_params:
                continue
            sub_lora_params = {
                f\"{clean_prefix}.{key}\": (
                    param.full_tensor().detach().cpu() if hasattr(param, \"full_tensor\") else param.detach().cpu()
                )
                for key, param in sub_lora_params.items()
            }
"""
LAYERED_REPLACEMENT = """            # table_rl: collect locally named LoRA tensors after each FSDP-layer summon.
            # ``get_peft_model_state_dict`` expects root-relative names, while this state dict is
            # local to ``submodule``.  Passing it through PEFT therefore produces an empty result.
            # LoRA tensors already have the canonical lora_A/lora_B suffixes, so prefixing the
            # summoned unit's root-relative name is sufficient and avoids a full-model all-gather.
            sub_lora_params = {
                f\"{clean_prefix}.{key.replace('_fsdp_wrapped_module.', '')}\": (
                    param.full_tensor().detach().cpu() if hasattr(param, \"full_tensor\") else param.detach().cpu()
                )
                for key, param in sub_state_dict.items()
                if \"lora_\" in key
            }
            if not sub_lora_params:
                continue
"""
REPLICATED_LORA_MARKER = "# table_rl: replicate the tiny LoRA adapter; the host cannot FSDP-all-gather"
REPLICATED_LORA_DTYPE_MARKER = "# table_rl: the restored adapter must match the FSDP parameter dtype"
REPLICATED_LORA_DTYPE_NEEDLE = """        if table_rl_replicated_lora and self.model_config.lora_rank > 0:
            ignored_lora_modules = [
"""
REPLICATED_LORA_DTYPE_REPLACEMENT = """        if table_rl_replicated_lora and self.model_config.lora_rank > 0:
            # table_rl: the restored adapter must match the FSDP parameter dtype.
            # PEFT checkpoints keep LoRA weights in fp32 even when the base model is bf16.
            for _name, _param in module.named_parameters():
                if _param.requires_grad and _param.dtype != param_dtype:
                    _param.data = _param.data.to(param_dtype)
            ignored_lora_modules = [
"""
REPLICATED_LORA_NEEDLE = """        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == \"fsdp\":
"""
REPLICATED_LORA_INSERT = """        # table_rl: replicate the tiny LoRA adapter; the host cannot FSDP-all-gather.
        # The base model remains FSDP-sharded. Each rank retains complete adapter weights and
        # averages its adapter gradients explicitly, equivalent to DDP for these parameters.
        table_rl_replicated_lora = os.environ.get(\"TABLE_RL_REPLICATED_LORA\") == \"1\"
        ignored_lora_modules = []
        if table_rl_replicated_lora and self.model_config.lora_rank > 0:
            ignored_lora_modules = [
                child for name, child in module.named_modules()
                if name.endswith(\".lora_A\") or name.endswith(\".lora_B\")
            ]
            if not ignored_lora_modules:
                raise RuntimeError(\"TABLE_RL_REPLICATED_LORA requested but no LoRA modules were found\")
            world_size = torch.distributed.get_world_size()

            def _average_replicated_lora_grad(grad):
                torch.distributed.all_reduce(grad)
                return grad.div_(world_size)

            for name, param in module.named_parameters():
                if \"lora_\" in name and param.requires_grad:
                    param.register_hook(_average_replicated_lora_grad)

        # Note: We force turn off CPUOffload because it causes incorrect results when using grad accumulation
        if self.engine_config.strategy == \"fsdp\":
"""
REPLICATED_LORA_FSDP_NEEDLE = """                cpu_offload=cpu_offload,
            )
"""
REPLICATED_LORA_FSDP_REPLACEMENT = """                cpu_offload=cpu_offload,
                ignored_modules=ignored_lora_modules or None,
            )
"""
REPLICATED_EXPORT_MARKER = "# table_rl: adapters are replicated, so exporting them needs no FSDP gather"
REPLICATED_EXPORT_NEEDLE = """    lora_params = OrderedDict()
    peft_model = getattr(module, \"_fsdp_wrapped_module\", module)
    if fsdp_version(module) > 0:
"""
REPLICATED_EXPORT_REPLACEMENT = """    lora_params = OrderedDict()
    peft_model = getattr(module, \"_fsdp_wrapped_module\", module)
    # table_rl: adapters are replicated, so exporting them needs no FSDP gather.
    if os.environ.get(\"TABLE_RL_REPLICATED_LORA\") == \"1\" and base_sync_done:
        lora_params = get_peft_model_state_dict(peft_model)
        return OrderedDict((name, param.detach().cpu()) for name, param in lora_params.items())
    if fsdp_version(module) > 0:
"""
REPLICATED_EXPORT_V2_MARKER = "# table_rl: enumerate replicated LoRA params without recursively calling FSDP.state_dict"
REPLICATED_EXPORT_V1_BLOCK = """    # table_rl: adapters are replicated, so exporting them needs no FSDP gather.
    if os.environ.get(\"TABLE_RL_REPLICATED_LORA\") == \"1\" and base_sync_done:
        lora_params = get_peft_model_state_dict(peft_model)
        return OrderedDict((name, param.detach().cpu()) for name, param in lora_params.items())
"""
REPLICATED_EXPORT_V2_BLOCK = """    # table_rl: enumerate replicated LoRA params without recursively calling FSDP.state_dict.
    # Calling PEFT's state-dict helper on the wrapped root visits the sharded base model and
    # reintroduces an unsupported full all-gather. The ignored adapter modules are complete locally.
    if os.environ.get(\"TABLE_RL_REPLICATED_LORA\") == \"1\" and base_sync_done:
        return OrderedDict(
            (
                name.replace(\"_fsdp_wrapped_module.\", \"\"),
                param.detach().cpu(),
            )
            for name, param in peft_model.named_parameters()
            if \"lora_\" in name
        )
"""
REPLICATED_EXPORT_V3_MARKER = "# table_rl: vLLM expects PEFT checkpoint keys without the adapter-name namespace"
REPLICATED_EXPORT_V2_KEY = """                name.replace(\"_fsdp_wrapped_module.\", \"\"),
"""
REPLICATED_EXPORT_V3_KEY = """                # table_rl: vLLM expects PEFT checkpoint keys without the adapter-name namespace.
                name.replace(\"_fsdp_wrapped_module.\", \"\").replace(\".default.\", \".\"),
"""
INSERT = """        # table_rl host-fallback NCCL environment
        # table_rl host-fallback NCCL environment v2
        env_vars.update({
            \"NCCL_P2P_DISABLE\": \"1\",
            \"NCCL_P2P_LEVEL\": \"LOC\",
            \"NCCL_IB_DISABLE\": \"1\",
        })
\n"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, required=True)
    args = parser.parse_args()
    target = args.verl_root / "verl" / "single_controller" / "ray" / "base.py"
    text = target.read_text(encoding="utf-8")
    if MARKER not in text and LEGACY_MARKER in text:
        legacy = """        # table_rl host-fallback NCCL environment
        for name in (\"NCCL_P2P_DISABLE\", \"NCCL_P2P_LEVEL\", \"NCCL_IB_DISABLE\"):
            value = os.environ.get(name)
            if value is not None:
                env_vars[name] = value
\n"""
        if legacy not in text:
            raise SystemExit(f"unsupported legacy Verl host-fallback patch: {target}")
        target.write_text(text.replace(legacy, INSERT, 1), encoding="utf-8")
        print(f"upgraded worker-env patch: {target}")
    elif MARKER not in text:
        if NEEDLE not in text:
            raise SystemExit(f"unsupported Verl worker-env shape: {target}")
        target.write_text(text.replace(NEEDLE, INSERT + NEEDLE, 1), encoding="utf-8")
        print(f"patched worker-env: {target}")

    fsdp_target = args.verl_root / "verl" / "workers" / "engine" / "fsdp" / "transformer_impl.py"
    fsdp_text = fsdp_target.read_text(encoding="utf-8")
    if FSDP_MARKER not in fsdp_text:
        if FSDP_NEEDLE not in fsdp_text:
            raise SystemExit(f"unsupported Verl FSDP init shape: {fsdp_target}")
        fsdp_text = fsdp_text.replace(
            FSDP_NEEDLE,
            f"                {FSDP_MARKER}\n                sync_module_states=False,\n",
            1,
        )
        fsdp_target.write_text(fsdp_text, encoding="utf-8")
        print(f"patched FSDP init: {fsdp_target}")
    else:
        print(f"already patched FSDP init: {fsdp_target}")

    if REPLICATED_LORA_MARKER not in fsdp_text:
        if REPLICATED_LORA_NEEDLE not in fsdp_text or REPLICATED_LORA_FSDP_NEEDLE not in fsdp_text:
            raise SystemExit(f"unsupported Verl replicated-LoRA FSDP shape: {fsdp_target}")
        fsdp_text = fsdp_text.replace(REPLICATED_LORA_NEEDLE, REPLICATED_LORA_INSERT, 1)
        fsdp_text = fsdp_text.replace(REPLICATED_LORA_FSDP_NEEDLE, REPLICATED_LORA_FSDP_REPLACEMENT, 1)
        fsdp_target.write_text(fsdp_text, encoding="utf-8")
        print(f"patched replicated LoRA FSDP: {fsdp_target}")
    else:
        print(f"already patched replicated LoRA FSDP: {fsdp_target}")

    fsdp_text = fsdp_target.read_text(encoding="utf-8")
    if REPLICATED_LORA_DTYPE_MARKER not in fsdp_text:
        if REPLICATED_LORA_DTYPE_NEEDLE not in fsdp_text:
            raise SystemExit(f"unsupported Verl replicated-LoRA dtype shape: {fsdp_target}")
        fsdp_target.write_text(
            fsdp_text.replace(REPLICATED_LORA_DTYPE_NEEDLE, REPLICATED_LORA_DTYPE_REPLACEMENT, 1),
            encoding="utf-8",
        )
        print(f"patched replicated LoRA dtype: {fsdp_target}")
    else:
        print(f"already patched replicated LoRA dtype: {fsdp_target}")

    layered_target = args.verl_root / "verl" / "utils" / "fsdp_utils.py"
    layered_text = layered_target.read_text(encoding="utf-8")
    if LAYERED_MARKER not in layered_text:
        if LAYERED_NEEDLE not in layered_text:
            raise SystemExit(f"unsupported Verl layered-summon shape: {layered_target}")
        layered_target.write_text(
            layered_text.replace(LAYERED_NEEDLE, LAYERED_REPLACEMENT, 1), encoding="utf-8"
        )
        print(f"patched layered LoRA collection: {layered_target}")
    else:
        print(f"already patched layered LoRA collection: {layered_target}")

    layered_text = layered_target.read_text(encoding="utf-8")
    if REPLICATED_EXPORT_MARKER not in layered_text and REPLICATED_EXPORT_V2_MARKER not in layered_text:
        if REPLICATED_EXPORT_NEEDLE not in layered_text:
            raise SystemExit(f"unsupported Verl replicated-LoRA export shape: {layered_target}")
        layered_target.write_text(
            layered_text.replace(REPLICATED_EXPORT_NEEDLE, REPLICATED_EXPORT_REPLACEMENT, 1), encoding="utf-8"
        )
        print(f"patched replicated LoRA export: {layered_target}")
    else:
        print(f"already patched replicated LoRA export: {layered_target}")

    layered_text = layered_target.read_text(encoding="utf-8")
    if REPLICATED_EXPORT_V2_MARKER not in layered_text:
        if REPLICATED_EXPORT_V1_BLOCK not in layered_text:
            raise SystemExit(f"unsupported Verl replicated-LoRA direct-export shape: {layered_target}")
        layered_target.write_text(
            layered_text.replace(REPLICATED_EXPORT_V1_BLOCK, REPLICATED_EXPORT_V2_BLOCK, 1), encoding="utf-8"
        )
        print(f"patched direct replicated LoRA export: {layered_target}")
    else:
        print(f"already patched direct replicated LoRA export: {layered_target}")

    layered_text = layered_target.read_text(encoding="utf-8")
    if REPLICATED_EXPORT_V3_MARKER not in layered_text:
        if REPLICATED_EXPORT_V2_KEY not in layered_text:
            raise SystemExit(f"unsupported Verl replicated-LoRA key shape: {layered_target}")
        layered_target.write_text(
            layered_text.replace(REPLICATED_EXPORT_V2_KEY, REPLICATED_EXPORT_V3_KEY, 1), encoding="utf-8"
        )
        print(f"patched replicated LoRA keys: {layered_target}")
    else:
        print(f"already patched replicated LoRA keys: {layered_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
