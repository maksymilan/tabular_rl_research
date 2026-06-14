#!/usr/bin/env python3
"""Fill the `think` field of compiled trajectories with grounded reasoning from an external LLM.

The compiled trajectories ship templated thinks ("Filter to the rows ..."). For SFT we want the
agent's real per-step reasoning, conditioned on what is actually visible at that step. This calls
the OpenAI-compatible endpoint in api.md ONCE per trajectory (cheaper + more coherent than per
step): it sends the question + schema + the executed tool calls and their results, and asks for one
reasoning sentence per step. Output is validated (a JSON array of exactly N strings); on any failure
the trajectory keeps its templated thinks (never corrupts data). Token usage is summed and reported
so the API cost is known before a full run.

Run (smoke):  .venv/bin/python src/sft/fill_think.py --n 20 --out data/sft/think_smoke.jsonl
              [--model deepseek-v4-pro] [--workers 4] [--split train]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SYSTEM = (
    "You annotate the reasoning of a table-reasoning agent. The agent answers a natural-language "
    "question by calling tools over relational tables; each step is one tool call. Given the "
    "question, the schema, and the executed tool calls with their results, write the agent's "
    "internal reasoning for EACH step: one concise first-person sentence saying why this specific "
    "action is the right next move toward the answer, grounded in the real table/column/value names "
    "and the data seen so far. For the final answer_from_context step, say how the evidence rows "
    "answer the question. Do not merely restate the tool name; explain the intent. Do not invent "
    "facts not shown. Output ONLY a JSON array of exactly {n} strings (one per step), nothing else."
)


def load_api() -> tuple[str, str]:
    key = base = ""
    for line in open(os.path.join(ROOT, "api.md")):
        if line.startswith("API_KEY="):
            key = line.split("=", 1)[1].strip()
        elif line.startswith("BASE_URL="):
            base = line.split("=", 1)[1].strip()
    return key, base


def overview_brief(ov: dict) -> str:
    return "; ".join(f"{t['table_name']}({','.join(c['name'] for c in t['columns'])})"
                     for t in ov["tables"])


def step_lines(steps: list[dict]) -> str:
    lines = []
    for i, s in enumerate(steps, 1):
        tc, out = s["tool_call"], s["tool_output"]
        args = json.dumps(tc["arguments"], ensure_ascii=False, separators=(",", ":"))
        if len(args) > 300:
            args = args[:300] + "…"
        if "table" in out:
            desc = f"-> {out['table']}: {out.get('row_count', '?')} rows; cols[{','.join(out.get('columns', []))}]"
            rows = out.get("rows") or []
            if rows:
                desc += f"; e.g. {json.dumps(rows[0], ensure_ascii=False)[:120]}"
        elif "final_answer" in out:
            desc = f"-> answer {json.dumps(out['final_answer'], ensure_ascii=False)[:160]}"
        else:
            desc = f"-> {out.get('row_count', '?')} rows {json.dumps(out.get('result_sample', []), ensure_ascii=False)[:120]}"
        lines.append(f"{i}. {tc['tool']}({args}) {desc}")
    return "\n".join(lines)


def build_messages(traj: dict) -> list[dict]:
    n = len(traj["steps"])
    user = (f"Question: {traj['question']}\n\n"
            f"Tables: {overview_brief(traj['initial_state']['dataset_overview'])}\n\n"
            f"Steps:\n{step_lines(traj['steps'])}\n\n"
            f"Write a JSON array of exactly {n} reasoning strings, one per step in order.")
    return [{"role": "system", "content": SYSTEM.format(n=n)}, {"role": "user", "content": user}]


def call(base: str, key: str, model: str, messages: list[dict], timeout: int = 120):
    body = json.dumps({"model": model, "messages": messages, "temperature": 0.3}).encode()
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d.get("usage", {})


def parse_thinks(text: str, n: int):
    t = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        arr = json.loads(t)
    except json.JSONDecodeError:
        return None
    if isinstance(arr, list) and len(arr) == n and all(isinstance(x, str) and x.strip() for x in arr):
        return arr
    return None


def fill_one(base, key, model, traj):
    for attempt in range(2):
        try:
            text, usage = call(base, key, model, build_messages(traj))
            thinks = parse_thinks(text, len(traj["steps"]))
            return thinks, usage, (None if thinks else f"bad_format: {text[:80]!r}")
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                return None, {}, f"api_error: {type(e).__name__}: {e}"
            time.sleep(2)
    return None, {}, "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="train", choices=["train", "dev"])
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    key, base = load_api()

    lines = open(os.path.join(ROOT, "data", "trajectories", f"spider_{args.split}.jsonl")).read().splitlines()
    stride = max(1, len(lines) // args.n)                       # spread the sample across lengths
    trajs = [json.loads(lines[i]) for i in range(0, len(lines), stride)][: args.n]

    pin = pout = ok = bad = 0
    t0 = time.time()

    # incremental + resumable: append each filled trajectory as it completes (crash-safe,
    # monitorable via wc -l), and skip any trajectory_id already present in the output file.
    out_path = (args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)) if args.out else None
    done_ids = set()
    if out_path and os.path.exists(out_path):
        for ln in open(out_path):
            try:
                done_ids.add(json.loads(ln).get("trajectory_id"))
            except json.JSONDecodeError:
                pass
    todo = [t for t in trajs if t.get("trajectory_id") not in done_ids]
    print(f"  {len(done_ids)} already filled, {len(todo)} to go (workers={args.workers})", flush=True)

    def work(tr):
        thinks, usage, err = fill_one(base, key, args.model, tr)
        return tr, thinks, usage, err

    out_f = open(out_path, "a") if out_path else None
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for tr, thinks, usage, err in (f.result() for f in as_completed([pool.submit(work, t) for t in todo])):
            pin += usage.get("prompt_tokens", 0)
            pout += usage.get("completion_tokens", 0)
            if thinks:
                ok += 1
                for s, th in zip(tr["steps"], thinks):
                    s["think"] = th
            else:
                bad += 1
            if out_f:
                out_f.write(json.dumps(tr, ensure_ascii=False, default=str) + "\n")
                out_f.flush()
            done += 1
            if done % 100 == 0:
                print(f"  ...{done}/{len(todo)}  ok={ok} fallback={bad}", flush=True)
    if out_f:
        out_f.close()
    if not todo:
        return 0
    dt = time.time() - t0
    total_traj = len(lines) + len(open(os.path.join(ROOT, "data", "trajectories", "spider_dev.jsonl")).readlines())
    scale = total_traj / max(1, len(trajs))
    # AIHubMix deepseek pricing varies; report tokens exactly + estimate under a labeled rate.
    in_rate, out_rate = 0.30, 1.20   # USD per 1M tokens (PLACEHOLDER — confirm on aihubmix)
    cost = (pin * in_rate + pout * out_rate) / 1e6

    print(f"\n=== think-fill smoke: {len(trajs)} trajectories, model={args.model}, {args.workers} workers ===")
    print(f"  filled ok: {ok}   kept-template: {bad}   wall: {dt:.0f}s ({dt/len(trajs):.1f}s/traj)")
    print(f"  tokens: prompt={pin}  completion={pout}  total={pin+pout}  (avg {(pin+pout)//len(trajs)}/traj)")
    print(f"  est cost THIS smoke: ${cost:.4f}  @ ${in_rate}/1M in + ${out_rate}/1M out (PLACEHOLDER rate)")
    print(f"  EXTRAPOLATION to full {total_traj} trajectories (train+dev):")
    print(f"    tokens ~{int((pin+pout)*scale):,}   est cost ~${cost*scale:.2f}")
    if out_path:
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
