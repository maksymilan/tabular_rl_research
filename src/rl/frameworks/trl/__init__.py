"""TRL-backed transition-level optimization for the table-agent environment."""

from .mechanism import RLMechanism, register_credit_handler

__all__ = ["RLMechanism", "register_credit_handler"]
