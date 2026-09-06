# Evaluation scenarios

This directory contains experiment-specific evaluators and launch wrappers
kept for reproducibility (boundary, Arm B, early-stop, and handoff runs).
Reusable evaluation logic is centralized in `rl.evaluation.runners`; new
scenarios should call those runners with explicit dataset, split, checkpoint,
GPU partition, and contract arguments. Historical scenarios may reference
archived contracts and are not current Atomic v26 entry points.
