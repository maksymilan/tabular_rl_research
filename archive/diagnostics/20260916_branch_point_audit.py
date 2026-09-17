"""Prefix-divergence audit over a frozen K=8 corpus.

For every question, compares the K trajectories turn by turn and reports where
they actually branch.  `plan` turns are excluded because they are harness-owned
control state, not executable evidence.  No gold is read.
"""
import collections
import json
import re
import sys


HANDLE = re.compile(r"\bt\d+\b")


def actions(turns):
    """Executable action digests of one trajectory, plus the tool-name-only view."""
    strict, tools, loose = [], [], []
    for turn in turns:
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        if not tool:
            break
        if tool == "plan":
            continue
        args = json.dumps(parsed.get("arguments"), sort_keys=True, ensure_ascii=False)
        strict.append(f"{tool}|{args}")
        loose.append(f"{tool}|{HANDLE.sub('H', args)}")
        tools.append(str(tool))
    return strict, tools, loose


def lcp(a, b):
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


first_action_identical = 0
first_action_identical_loose = 0
questions = 0
pair_lcp = collections.Counter()
pair_lcp_tool = collections.Counter()
pair_lcp_loose = collections.Counter()
cw_divergence = collections.Counter()
cw_divergence_loose = collections.Counter()
cw_pairs = 0
cw_share3 = 0
cw_share3_loose = 0
any_correct_any_wrong = 0
with_branch_has_correct = collections.Counter()
traj_len = []
clean_correct = clean_total = dirty_correct = dirty_total = 0
mixed_clean_correct = mixed_clean_total = mixed_dirty_correct = mixed_dirty_total = 0
selector_total = 0
selector_hits = collections.Counter()
clean_present = 0
clean_unique = 0


def modal_correct(sample_list):
    groups = collections.defaultdict(list)
    for s in sample_list:
        groups[json.dumps(s.get("pred_sample"), sort_keys=True, ensure_ascii=False)].append(bool(s.get("correct")))
    labels = sorted(groups.items(), key=lambda kv: -len(kv[1]))[0][1]
    return sum(labels) * 2 > len(labels)

for path in sys.argv[1:]:
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            samples = rec.get("samples") or []
            if not samples:
                continue
            questions += 1
            per_sample = []
            for sample in samples:
                strict, tools, loose = actions(sample.get("turns") or [])
                traj_len.append(len(strict))
                clean = int(sample.get("errors") or 0) == 0 and bool(sample.get("legal"))
                if clean:
                    clean_total += 1
                    clean_correct += bool(sample.get("correct"))
                else:
                    dirty_total += 1
                    dirty_correct += bool(sample.get("correct"))
                per_sample.append((strict, tools, loose, bool(sample.get("correct")), clean))
            if len({s[1][0] for s in per_sample if s[1]}) == 1:
                first_action_identical += 1
            if len({s[2][0] for s in per_sample if s[2]}) == 1:
                first_action_identical_loose += 1

            correct = [s for s in per_sample if s[3]]
            wrong = [s for s in per_sample if not s[3]]
            clean_samples = [s for s in samples if int(s.get("errors") or 0) == 0 and bool(s.get("legal"))]
            if clean_samples:
                clean_present += 1
            if len(clean_samples) == 1:
                clean_unique += 1
            selector_total += 1
            selector_hits["sample0"] += bool(samples[0].get("correct"))
            selector_hits["modal"] += modal_correct(samples)
            if clean_samples:
                selector_hits["clean_first"] += bool(clean_samples[0].get("correct"))
                selector_hits["clean_modal"] += modal_correct(clean_samples)
            else:
                selector_hits["clean_first"] += bool(samples[0].get("correct"))
                selector_hits["clean_modal"] += modal_correct(samples)
            if len(clean_samples) == 1:
                selector_hits["clean_unique"] += bool(clean_samples[0].get("correct"))
            else:
                selector_hits["clean_unique"] += modal_correct(samples)
            if correct and wrong:
                any_correct_any_wrong += 1
                best = 0
                best_loose = 0
                for c in correct:
                    for w in wrong:
                        value = lcp(c[0], w[0])
                        cw_divergence[value] += 1
                        cw_divergence_loose[lcp(c[2], w[2])] += 1
                        cw_pairs += 1
                        best = max(best, value)
                        best_loose = max(best_loose, lcp(c[2], w[2]))
                if best >= 3:
                    cw_share3 += 1
                if best_loose >= 3:
                    cw_share3_loose += 1
                with_branch_has_correct[min(best, 3)] += 1
                for s in per_sample:
                    if s[4]:
                        mixed_clean_total += 1
                        mixed_clean_correct += s[3]
                    else:
                        mixed_dirty_total += 1
                        mixed_dirty_correct += s[3]

            # all pairs (for the general "is there any shared prefix" question)
            for i in range(len(per_sample)):
                for j in range(i + 1, len(per_sample)):
                    pair_lcp[min(lcp(per_sample[i][0], per_sample[j][0]), 6)] += 1
                    pair_lcp_tool[min(lcp(per_sample[i][1], per_sample[j][1]), 6)] += 1
                    pair_lcp_loose[min(lcp(per_sample[i][2], per_sample[j][2]), 6)] += 1

total_pairs = sum(pair_lcp.values())
print(f"questions: {questions}; mean executable steps per trajectory: {sum(traj_len)/len(traj_len):.2f}")
print(f"questions where all K trajectories share the same first executable action: "
      f"{first_action_identical}/{questions} = {first_action_identical/questions:.1%}")
print(f"  same first action after normalizing derived-table handles: "
      f"{first_action_identical_loose/questions:.1%}")
print("\npairwise shared-prefix length (strict action identity), share of all C(K,2) pairs:")
for k in sorted(pair_lcp):
    label = f"{k}" if k < 6 else "6+"
    print(f"  shared exactly {label} step(s): {pair_lcp[k]/total_pairs:6.1%}")
print("\npairwise shared-prefix length (tool name only):")
for k in sorted(pair_lcp_tool):
    label = f"{k}" if k < 6 else "6+"
    print(f"  shared exactly {label} step(s): {pair_lcp_tool[k]/total_pairs:6.1%}")
print("\npairwise shared-prefix length (arguments with handles normalized to H):")
for k in sorted(pair_lcp_loose):
    label = f"{k}" if k < 6 else "6+"
    print(f"  shared exactly {label} step(s): {pair_lcp_loose[k]/total_pairs:6.1%}")
print(f"\nquestions with >=1 correct and >=1 wrong sample: {any_correct_any_wrong}")
print(f"  correct-wrong pairs: {cw_pairs}; divergence index distribution:")
for k in sorted(cw_divergence):
    print(f"    diverge after {k} shared step(s): {cw_divergence[k]/cw_pairs:6.1%}")
print("  same, handles normalized:")
for k in sorted(cw_divergence_loose):
    print(f"    diverge after {k} shared step(s): {cw_divergence_loose[k]/cw_pairs:6.1%}")
print(f"  questions where some correct-wrong pair shares >=3 steps: {cw_share3}"
      f" = {cw_share3/max(any_correct_any_wrong,1):.1%}")
print(f"  same with normalized handles: {cw_share3_loose}"
      f" = {cw_share3_loose/max(any_correct_any_wrong,1):.1%}")
print("\ngold-free selector signal (no harness error and legal termination):")
print(f"  overall: clean {clean_correct}/{clean_total} correct = {clean_correct/max(clean_total,1):.1%}"
      f"   dirty {dirty_correct}/{dirty_total} = {dirty_correct/max(dirty_total,1):.1%}")
print(f"  inside mixed questions: clean {mixed_clean_correct}/{mixed_clean_total}"
      f" = {mixed_clean_correct/max(mixed_clean_total,1):.1%}"
      f"   dirty {mixed_dirty_correct}/{mixed_dirty_total}"
      f" = {mixed_dirty_correct/max(mixed_dirty_total,1):.1%}")
print(f"\nquestions with >=1 clean sample: {clean_present}/{selector_total} = {clean_present/selector_total:.1%}"
      f"; exactly one clean sample: {clean_unique}/{selector_total} = {clean_unique/selector_total:.1%}")
print("end-to-end selectors over the same K=8 candidates (accuracy on all questions):")
for name in ("sample0", "modal", "clean_first", "clean_modal", "clean_unique"):
    print(f"  {name:14s}: {selector_hits[name]}/{selector_total} = {selector_hits[name]/selector_total:.4f}")
