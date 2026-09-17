import collections
import json
import statistics
import sys

paths = sys.argv[1:]

per_task = {}
tools_by_group = collections.defaultdict(collections.Counter)
fail_by_group = collections.defaultdict(collections.Counter)
steps_by_group = collections.defaultdict(list)
errors_by_group = collections.Counter()
term_by_group = collections.defaultdict(collections.Counter)
recovery_by_group = collections.Counter()
group_n = collections.Counter()


def tool_of(turn):
    parsed = turn.get("parsed") or {}
    action = parsed.get("action")
    if isinstance(action, dict):
        name = action.get("tool") or action.get("name")
        if name:
            return str(name)
    for key in ("tool", "name"):
        if parsed.get(key):
            return str(parsed[key])
    return "unparsed"


for path in paths:
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            samples = rec.get("samples") or []
            correct = rec.get("sample_correct_count")
            legal = rec.get("sample_legal_count")
            if correct == 0 and legal == 8:
                group = "never_solved_legal"
            elif correct == 8:
                group = "always_solved"
            elif 1 <= correct <= 6:
                group = "band_1_6"
            else:
                group = "other"
            group_n[group] += 1
            for sample in samples:
                fail_by_group[group][str(sample.get("failure_type"))] += 1
                steps_by_group[group].append(int(sample.get("steps") or 0))
                errors_by_group[group] += len(sample.get("error_events") or [])
                turns = sample.get("turns") or []
                for turn in turns:
                    tools_by_group[group][tool_of(turn)] += 1
                    if turn.get("feedback_recovery"):
                        recovery_by_group[group] += 1
                if turns:
                    term_by_group[group][tool_of(turns[-1])] += 1

print("group task counts:", dict(group_n))
for group in ("always_solved", "band_1_6", "never_solved_legal", "other"):
    if not group_n[group]:
        continue
    steps = steps_by_group[group]
    n_samples = len(steps)
    print(f"\n=== {group} ({group_n[group]} tasks, {n_samples} samples) ===")
    print("  failure_type:", dict(fail_by_group[group].most_common()))
    print(
        f"  steps: mean={statistics.mean(steps):.1f} median={statistics.median(steps)} "
        f"max={max(steps)} share>=25={sum(1 for s in steps if s>=25)/n_samples:.1%} "
        f"share==30={sum(1 for s in steps if s>=30)/n_samples:.1%}"
    )
    print(f"  harness error events: {errors_by_group[group]} total, {errors_by_group[group]/n_samples:.2f}/sample")
    print(f"  recovery turns: {recovery_by_group[group]} ({recovery_by_group[group]/n_samples:.2f}/sample)")
    total_turns = sum(tools_by_group[group].values()) or 1
    print("  tools per sample:", {k: round(v / n_samples, 2) for k, v in tools_by_group[group].most_common(16)})
    print("  terminal tool mix:", {k: round(v / n_samples, 3) for k, v in term_by_group[group].most_common(8)})
