#!/usr/bin/env python3
"""Fail-closed precision helpers for QLoRA policy optimization."""
from __future__ import annotations

from collections import Counter
from typing import Any

import torch


FP32 = torch.float32
ADAM_MOMENT_NAMES = frozenset({"exp_avg", "exp_avg_sq", "max_exp_avg_sq"})


def _dtype_counts(named_tensors) -> dict[str, dict[str, int]]:
    tensor_counts: Counter[str] = Counter()
    parameter_counts: Counter[str] = Counter()
    for _, tensor in named_tensors:
        dtype = str(tensor.dtype)
        tensor_counts[dtype] += 1
        parameter_counts[dtype] += int(tensor.numel())
    return {
        "tensors": dict(sorted(tensor_counts.items())),
        "parameters": dict(sorted(parameter_counts.items())),
    }


def trainable_parameter_precision_audit(model) -> dict[str, Any]:
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    non_floating = [name for name, parameter in named if not parameter.is_floating_point()]
    non_fp32 = [
        name
        for name, parameter in named
        if parameter.is_floating_point() and parameter.dtype != FP32
    ]
    return {
        "schema_version": "trl-trainable-precision-v1",
        "trainable_tensors": len(named),
        "trainable_parameters": sum(parameter.numel() for _, parameter in named),
        "dtype_counts": _dtype_counts(named),
        "non_floating_tensors": non_floating,
        "non_fp32_tensors": non_fp32,
    }


def require_trainable_parameters_fp32(model) -> dict[str, Any]:
    audit = trainable_parameter_precision_audit(model)
    if audit["trainable_tensors"] < 1:
        raise RuntimeError("model has no trainable parameters")
    if audit["non_floating_tensors"]:
        raise TypeError(
            "trainable parameters must be floating point: "
            f"{audit['non_floating_tensors'][:5]}"
        )
    if audit["non_fp32_tensors"]:
        raise RuntimeError(
            "trainable parameters must remain FP32 for low-learning-rate QLoRA: "
            f"{audit['non_fp32_tensors'][:5]}"
        )
    return audit


def promote_trainable_parameters_to_fp32(model) -> dict[str, Any]:
    """Promote only trainable tensors, leaving the quantized/frozen base untouched."""
    before = trainable_parameter_precision_audit(model)
    promoted_tensors = 0
    promoted_parameters = 0
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if not parameter.is_floating_point():
                raise TypeError("a trainable parameter is not floating point")
            if parameter.dtype != FP32:
                promoted_tensors += 1
                promoted_parameters += int(parameter.numel())
                parameter.data = parameter.data.to(dtype=FP32)
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.to(dtype=FP32)
    after = require_trainable_parameters_fp32(model)
    return {
        "schema_version": "trl-trainable-precision-promotion-v1",
        "before": before,
        "after": after,
        "promoted_tensors": promoted_tensors,
        "promoted_parameters": promoted_parameters,
    }


def _unwrap_optimizer(optimizer):
    seen = set()
    current = optimizer
    while not hasattr(current, "state") and hasattr(current, "optimizer"):
        if id(current) in seen:
            break
        seen.add(id(current))
        current = current.optimizer
    return current


def optimizer_moment_precision_audit(optimizer) -> dict[str, Any]:
    optimizer = _unwrap_optimizer(optimizer)
    moment_tensors = []
    for state in optimizer.state.values():
        for name, value in state.items():
            if name in ADAM_MOMENT_NAMES and torch.is_tensor(value):
                moment_tensors.append((name, value))
    non_fp32 = [
        {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
        for name, value in moment_tensors
        if value.dtype != FP32
    ]
    return {
        "schema_version": "trl-optimizer-precision-v1",
        "moment_tensors": len(moment_tensors),
        "moment_elements": sum(value.numel() for _, value in moment_tensors),
        "dtype_counts": _dtype_counts(moment_tensors),
        "non_fp32_moments": non_fp32,
    }


def require_adam_moments_fp32(optimizer) -> dict[str, Any]:
    audit = optimizer_moment_precision_audit(optimizer)
    if audit["moment_tensors"] < 1:
        raise RuntimeError("Adam optimizer state is uninitialized; no moments were found")
    if audit["non_fp32_moments"]:
        raise RuntimeError(
            "Adam moments must remain FP32 for low-learning-rate QLoRA: "
            f"{audit['non_fp32_moments'][:5]}"
        )
    return audit
