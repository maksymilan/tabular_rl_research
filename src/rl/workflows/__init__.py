"""Stable workflow layer for RL, evaluation, and frozen SFT operations.

The workflow layer owns *what* operation is requested and its parameters.  The
underlying trainer/evaluator modules own *how* it runs.  New experiments should
construct a :class:`WorkflowConfig` instead of copying a launcher.
"""
from .config import WorkflowConfig, WorkflowKind, build_plan

__all__ = ["WorkflowConfig", "WorkflowKind", "build_plan"]
