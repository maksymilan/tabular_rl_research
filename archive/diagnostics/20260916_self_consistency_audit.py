"""Self-consistency audit over a frozen K=8 screening corpus.

Reads per-sample `pred_sample` (the submitted denotation) plus the per-sample
`correct` flag.  Never reads gold: the label of a prediction cluster is taken
from the samples inside it.
"""
import collections
import json
import sys


def canon(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


total = 0
single_ok = 0
modal_ok = 0
any_ok = 0
band_total = band_single = band_modal = 0
cluster_counts = []
correct_is_modal = collections.Counter()

for path in sys.argv[1:]:
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            samples = rec.get("samples") or []
            if not samples:
                continue
            groups = collections.defaultdict(list)
            for sample in samples:
                groups[canon(sample.get("pred_sample"))].append(bool(sample.get("correct")))
            # deterministic: largest cluster wins, ties resolved by first appearance
            ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
            modal_key, modal_labels = ordered[0]
            modal_correct = sum(modal_labels) * 2 > len(modal_labels)
            single_correct = bool(samples[0].get("correct"))
            any_correct = any(bool(s.get("correct")) for s in samples)
            n_correct = int(rec.get("sample_correct_count") or 0)

            total += 1
            single_ok += single_correct
            modal_ok += modal_correct
            any_ok += any_correct
            cluster_counts.append(len(groups))
            if 1 <= n_correct <= 6:
                band_total += 1
                band_single += single_correct
                band_modal += modal_correct
            correct_key = [
                key for key, labels in groups.items() if sum(labels) * 2 > len(labels)
            ]
            if n_correct > 0:
                correct_is_modal[
                    "correct_class_is_modal"
                    if any(key in correct_key for key in correct_key)
                    and modal_correct
                    else "correct_exists_but_not_modal"
                    if not modal_correct
                    else "modal_correct"
                ] += 1

print(f"questions: {total}")
print(f"  single-sample (sample 0) accuracy: {single_ok/total:.4f}")
print(f"  modal-cluster (self-consistency) : {modal_ok/total:.4f}")
print(f"  any-correct (pass@8 oracle)      : {any_ok/total:.4f}")
print(f"  mean distinct predictions / question: {sum(cluster_counts)/len(cluster_counts):.2f}")
print(f"1-6 band (n={band_total}):")
print(f"  single-sample: {band_single/band_total:.4f}   modal: {band_modal/band_total:.4f}")
print("among questions with any correct sample:", dict(correct_is_modal))
