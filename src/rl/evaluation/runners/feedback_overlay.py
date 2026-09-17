"""Opt-in error-only adapter around a pinned pass@k runner.

No runtime source files are changed. The original parser/executor always decides
acceptance and the original runner still owns errors, budgets and scoring.
Per-sample ContextVars isolate concurrent evaluator workers.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from functools import wraps

from rl.runtime.error_feedback import (
    FEEDBACK_VERSION, error_feedback_payload, merge_error_feedback, rejected_action,
)


def install_feedback_overlay(runner, protocol) -> None:
    if getattr(runner, "error_feedback_version", None):
        raise RuntimeError("feedback overlay is already installed")
    scope = ContextVar("rejected_action_feedback", default=None)
    original_validate = protocol.validate_model_arguments

    @wraps(original_validate)
    def validate(tool, arguments, *args, **kwargs):
        try:
            return original_validate(tool, arguments, *args, **kwargs)
        except Exception as exc:
            # This call is downstream of the original strict carrier/envelope parser.
            exc.attempted_tool = tool
            exc.attempted_arguments = deepcopy(arguments)
            raise

    protocol.validate_model_arguments = validate
    original_execute = runner.execute_tool

    @wraps(original_execute)
    def execute(harness, tool, arguments, context, *args, **kwargs):
        active = scope.get()
        if active is not None:
            active["history"] = context.get("history", {})
        return original_execute(harness, tool, arguments, context, *args, **kwargs)

    runner.execute_tool = execute
    original_format = runner.format_tool_error

    @wraps(original_format)
    def format_error(exc, harness, tool, arguments):
        message = original_format(exc, harness, tool, arguments)
        active = scope.get()
        if active is not None:
            parsed = {"tool": tool, "arguments": arguments}
            payload = error_feedback_payload(
                exc, parsed=parsed,
                state=active["state"], history=active["history"],
            )
            if rejected_action(exc, parsed)[0] not in ("scalar_compute", "join_tables"):
                # Preserve useful existing feedback for unrelated tools. For the
                # two targeted tools, operation-specific facts replace legacy
                # generic hints (e.g. scalar_compute cannot use in_table).
                payload["error"]["message"] = message
            active["payloads"].append(payload)
        return message  # Keep raw event/legacy accounting unchanged.

    runner.format_tool_error = format_error

    def wrap_context(original):
        @wraps(original)
        def context(system, overview, question, state, last_error, *args, **kwargs):
            active = scope.get()
            if active is not None:
                active["state"] = state or {}
                if last_error and active["payloads"]:
                    last_error = merge_error_feedback(last_error, active["payloads"][-1])
            return original(system, overview, question, state, last_error, *args, **kwargs)
        return context

    runner.rolling_legal_history_messages = wrap_context(runner.rolling_legal_history_messages)
    runner.model_context_messages = wrap_context(runner.model_context_messages)
    original_sample = runner.run_sample

    @wraps(original_sample)
    def sample(*args, **kwargs):
        active = {"state": {}, "history": {}, "payloads": []}
        token = scope.set(active)
        try:
            record = original_sample(*args, **kwargs)
            events = record.get("error_events", [])
            if len(events) != len(active["payloads"]):
                raise RuntimeError("feedback/event coverage mismatch; runner contract changed")
            for event, payload in zip(events, active["payloads"]):
                event["model_visible_feedback"] = merge_error_feedback({
                    "step_id": event["step_id"], "status": "error",
                    "error": {"type": event["error_type"], "message": event["message"]},
                }, payload)
            record["error_feedback_version"] = FEEDBACK_VERSION
            return record
        finally:
            scope.reset(token)

    runner.run_sample = sample
    original_writer = runner.ArtifactWriter

    @wraps(original_writer)
    def writer(result_dir, manifest, *args, **kwargs):
        return original_writer(result_dir, {**manifest, "error_feedback_version": FEEDBACK_VERSION},
                               *args, **kwargs)

    runner.ArtifactWriter = writer
    runner.error_feedback_version = FEEDBACK_VERSION
