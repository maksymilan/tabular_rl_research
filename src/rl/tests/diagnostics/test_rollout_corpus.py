from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from rl.diagnostics.io import iter_jsonl, read_jsonl, write_json, write_jsonl, atomic_write_bytes
from rl.diagnostics.rollout_corpus import action_sequence, audit_run, compact_rollout, question_key
from rl.diagnostics.distance_credit import StateFingerprint, classify_pair, state_distance


class RolloutCorpusTests(unittest.TestCase):
    def record(self, index=0):
        return {'db_id': 'test_db', 'question': 'How many things?', 'example_index': 42,
                'policy_global_step': 0, 'trajectory_id': f'rl_42_sample_{index}',
                'correct': index < 4, 'legal': True, 'protocol_hash': 'test-protocol',
                'turns': [{'parsed': {'tool': 'answer_from_context', 'think': 'hidden',
                                     'arguments': {'evidence': {'table': 'x'}, 'reason': 'hidden'}},
                           'gold_sample': [[999]], 'pred_sample': [[4]]}], 'gold_sql': 'secret'}

    def test_stream_equivalence_and_line_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            write_jsonl(path, [{'a': 1}, {'b': 2}])
            self.assertEqual(list(iter_jsonl(path)), read_jsonl(path))
            atomic_write_bytes(path, b'{"a":1}\nBAD\n')
            with self.assertRaisesRegex(ValueError, r':2: invalid JSON'):
                list(iter_jsonl(path))

    def test_partition_identity_survives_cohort_renumbering(self):
        row = self.record()
        other = {**row, 'example_index': 999, 'policy_global_step': 3}
        self.assertEqual(question_key(row), question_key(other))
        self.assertNotEqual(question_key(row), question_key({**row, 'db_id': 'other_db'}))

    def test_full_groups_partial_groups_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest = Path(directory) / 'rows.jsonl', Path(directory) / 'manifest.json'
            write_json(manifest, {'group_size': 8, 'protocol_hash': 'test-protocol'})
            rows = [self.record(i) for i in range(8)]
            write_jsonl(path, rows)
            result = audit_run(path, manifest, reviewed_keys={question_key(rows[0])})
            self.assertEqual(result['summary']['correct_of_8_histogram']['4'], 1)
            self.assertEqual(result['review_rows'], [])
            write_jsonl(path, rows[:-1])
            result = audit_run(path, manifest, reviewed_keys=set())
            self.assertEqual(result['summary']['complete_k8_groups'], 0)
            self.assertEqual(len(result['summary']['incomplete_groups']), 1)
            write_jsonl(path, rows + rows[:1])
            with self.assertRaisesRegex(ValueError, 'duplicate rollout identity'):
                audit_run(path, manifest, reviewed_keys=set())

    def test_compact_review_omits_gold_and_reasoning(self):
        row = self.record()
        out = compact_rollout(row)
        self.assertNotIn('gold', str(out))
        self.assertNotIn('hidden', str(out))
        self.assertEqual(out['turns'][0]['terminal_pred_sample'], [[4]])
        other = copy.deepcopy(row)
        other['turns'][0]['parsed']['think'] = 'changed'
        other['turns'][0]['parsed']['arguments']['reason'] = 'changed'
        self.assertEqual(action_sequence(row), action_sequence(other))

    def test_unvalidated_distance_can_miss_semantic_change(self):
        common = dict(turn_index=1, tool='condition_filter', leaves=frozenset({'patients'}),
                      columns=frozenset({'id'}), row_count=1, grain='filter', has_error=False)
        left = StateFingerprint(lineage='age < 30', **common)
        right = StateFingerprint(lineage='age > 30', **common)
        # Regression counterexample, not a claim that these states are equivalent.
        self.assertEqual(state_distance(left, right), 0.0)

    def test_distance_prototype_never_admits_reward(self):
        positive, negative = self.record(0), self.record(7)
        result = classify_pair(positive, negative).to_dict()
        self.assertFalse(result['actor_update_allowed'])
        self.assertFalse(result['causal_attribution_certified'])
        self.assertIsNone(result['penalty_coefficient'])
        self.assertIsNone(result['confidence'])
        changed = {**negative, 'example_index': 200, 'gold_sql': 'different'}
        self.assertEqual(classify_pair(positive, changed).to_dict(), result)


if __name__ == '__main__':
    unittest.main()
