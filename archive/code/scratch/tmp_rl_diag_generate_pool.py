#!/usr/bin/env python3
"""Isolated fixed-pool generator wrapper for a real RL two-phase diagnostic.

The production generator is intentionally unchanged.  This wrapper only
overrides the result-reward profile and records every vLLM batch size/timing,
while forcing the same non-eager engine setting used by the online A/B probe.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path


SOURCE = Path(os.environ["DIAG_GENERATOR_SOURCE"]).resolve()
spec = importlib.util.spec_from_file_location("diag_fixed_pool_generator", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load generator source: {SOURCE}")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


# Keep the production generator's tokenization, harness, and serialization
# paths; only bind this diagnostic to the requested four-level terminal reward.
_OriginalSettings = module.RolloutSettings


class DiagnosticSettings(_OriginalSettings):
    def __init__(self, *args, **kwargs):
        kwargs["result_reward_profile"] = "four-level"
        super().__init__(*args, **kwargs)


module.RolloutSettings = DiagnosticSettings


# The production fixed-pool helper uses eager execution for historical
# reproducibility.  This isolated RL probe intentionally tests the non-eager
# path and records its actual behavior; no production file is changed.
_OriginalLLM = module.LLM


def diagnostic_llm(*args, **kwargs):
    kwargs["enforce_eager"] = False
    return _OriginalLLM(*args, **kwargs)


module.LLM = diagnostic_llm


_OriginalGenerator = module.VLLMLoRAGenerator


class InstrumentedGenerator(_OriginalGenerator):
    def __init__(self, *args, **kwargs):
        self._diag_log = Path(os.environ.get("DIAG_BATCH_LOG", "vllm_batches.jsonl"))
        self._diag_log.parent.mkdir(parents=True, exist_ok=True)
        self._diag_start = time.time()
        super().__init__(*args, **kwargs)
        with self._diag_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": "engine_ready",
                "time": time.time(),
                "elapsed_since_wrapper_start": time.time() - self._diag_start,
                "enforce_eager": False,
            }) + "\n")

    def __call__(self, prompts, request_keys):
        started = time.time()
        result = super().__call__(prompts, request_keys)
        row = {
            "event": "generate_batch",
            "time": time.time(),
            "elapsed_seconds": time.time() - started,
            "batch_size": len(prompts),
            "request_keys": [list(key) for key in request_keys],
            "completion_tokens": [len(ids) for ids in result["completion_ids"]],
        }
        with self._diag_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return result


module.VLLMLoRAGenerator = InstrumentedGenerator


if __name__ == "__main__":
    module.main()
