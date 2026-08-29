#!/usr/bin/env python3
"""Durable, low-overhead gradient evidence for GRPO credit experiments.

The recorder is intentionally independent of the loss implementation.  The trainer supplies
the gradient produced by the mixed update and two counterfactual gradients (positive and
negative effective policy coefficients).  We persist exact scalar/layer statistics on every
optimizer update and a small deterministic sketch.  Full vectors are optional because writing
three tens-of-millions-parameter tensors per update can otherwise dominate an RL run's disk.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

try:
    import torch
except ImportError:  # pragma: no cover - GPU runtime dependency
    torch = None


SCHEMA_VERSION = "gradient-conflict-record-v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cosine(left, right) -> float:
    if torch is None:
        return 0.0
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    denominator = left_norm * right_norm
    if float(denominator) == 0.0:
        return 0.0
    # Accumulating a very high-dimensional LoRA vector in FP32 can exceed the
    # mathematical [-1, 1] range by a few ULPs (especially when one partition
    # nearly equals the mixed gradient).  Clamp the reported diagnostic rather
    # than emitting an impossible cosine.
    value = float(torch.dot(left, right) / denominator)
    return max(-1.0, min(1.0, value))


def _dot(left, right) -> float:
    if torch is None:
        return 0.0
    return float(torch.dot(left, right))


class GradientConflictRecorder:
    """Record exact gradient conflict scalars and optional full gradient vectors."""

    def __init__(
        self,
        model,
        output_dir: Path,
        *,
        save_vectors: bool = False,
        sketch_size: int = 4096,
    ) -> None:
        if torch is None:
            raise RuntimeError("gradient conflict recording requires torch")
        if sketch_size < 1:
            raise ValueError("sketch_size must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_vectors = bool(save_vectors)
        self.sketch_size = int(sketch_size)
        self.parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not self.parameters:
            raise ValueError("gradient conflict recorder found no trainable parameters")
        self.total_numel = sum(parameter.numel() for _, parameter in self.parameters)
        self.summary_path = self.output_dir / "summary.jsonl"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "save_vectors": self.save_vectors,
            "sketch_size": self.sketch_size,
            "trainable_parameter_count": len(self.parameters),
            "trainable_numel": self.total_numel,
            "parameters": [
                {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
                for name, parameter in self.parameters
            ],
        }
        manifest_path = self.output_dir / "parameter_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing != manifest:
                raise RuntimeError("gradient recorder parameter manifest changed across resume")
        else:
            _atomic_json(manifest_path, manifest)

    def capture(self, model) -> tuple[Any, list[dict[str, Any]]]:
        """Return a CPU FP32 flat vector and per-parameter norm metadata."""
        chunks = []
        layers = []
        current_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if len(current_parameters) != len(self.parameters):
            raise RuntimeError(
                "gradient recorder model trainable parameter count changed: "
                f"{len(current_parameters)} != {len(self.parameters)}"
            )
        for (name, parameter), current_parameter in zip(
            self.parameters,
            current_parameters,
            strict=True,
        ):
            if current_parameter.numel() != parameter.numel():
                raise RuntimeError(
                    f"gradient recorder parameter shape changed for {name}: "
                    f"{tuple(current_parameter.shape)} != {tuple(parameter.shape)}"
                )
            current = current_parameter.grad
            if current is None:
                vector = torch.zeros(parameter.numel(), dtype=torch.float32)
            else:
                vector = current.detach().float().reshape(-1).cpu()
            chunks.append(vector)
            layers.append(
                {
                    "name": name,
                    "norm": float(torch.linalg.vector_norm(vector)),
                    "nonzero": int(torch.count_nonzero(vector)),
                }
            )
        return torch.cat(chunks), layers

    def _sketch(self, vector):
        if vector.numel() <= self.sketch_size:
            return vector
        # Deterministic evenly spaced samples are cheap, reproducible, and avoid allocating a
        # projection matrix proportional to the QLoRA parameter count.  Exact conflict scalars
        # below are always computed from the complete vectors.
        indices = torch.linspace(
            0,
            vector.numel() - 1,
            self.sketch_size,
            dtype=torch.long,
        )
        return vector.index_select(0, indices)

    def record(
        self,
        *,
        step: int,
        combined,
        positive,
        negative,
        layers: dict[str, list[dict[str, Any]]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        combined = combined.float().cpu()
        positive = positive.float().cpu()
        negative = negative.float().cpu()
        combined_norm = float(torch.linalg.vector_norm(combined))
        positive_norm = float(torch.linalg.vector_norm(positive))
        negative_norm = float(torch.linalg.vector_norm(negative))
        positive_negative_dot = _dot(positive, negative)
        positive_negative_cosine = _cosine(positive, negative)
        positive_combined_cosine = _cosine(positive, combined)
        negative_combined_cosine = _cosine(negative, combined)
        sum_norm = float(torch.linalg.vector_norm(positive + negative))
        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "step": int(step),
            "combined_norm": combined_norm,
            "positive_norm": positive_norm,
            "negative_norm": negative_norm,
            "positive_negative_dot": positive_negative_dot,
            "positive_negative_cosine": positive_negative_cosine,
            "positive_combined_cosine": positive_combined_cosine,
            "negative_combined_cosine": negative_combined_cosine,
            "positive_negative_conflict_mass": max(0.0, -2.0 * positive_negative_dot),
            "net_over_sum_norm": sum_norm / (positive_norm + negative_norm)
            if positive_norm + negative_norm > 0.0
            else 0.0,
            "sketch_size": min(self.sketch_size, int(combined.numel())),
            "metadata": metadata or {},
        }
        if layers is not None:
            summary["layers"] = layers
        if self.save_vectors:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "step": int(step),
                "combined": combined,
                "positive": positive,
                "negative": negative,
                "metadata": metadata or {},
            }
            vector_path = self.output_dir / f"step_{int(step):06d}.pt"
            temporary = vector_path.with_name(f".{vector_path.name}.{os.getpid()}.tmp")
            torch.save(payload, temporary)
            temporary.replace(vector_path)
            summary["vector_path"] = vector_path.name
        else:
            sketch_path = self.output_dir / f"step_{int(step):06d}_sketch.pt"
            temporary = sketch_path.with_name(f".{sketch_path.name}.{os.getpid()}.tmp")
            torch.save(
                {
                    "schema_version": SCHEMA_VERSION,
                    "step": int(step),
                    "combined": self._sketch(combined),
                    "positive": self._sketch(positive),
                    "negative": self._sketch(negative),
                },
                temporary,
            )
            temporary.replace(sketch_path)
            summary["sketch_path"] = sketch_path.name
        with self.summary_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return summary
