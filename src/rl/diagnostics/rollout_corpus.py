"""Factual cross-run rollout audit; no error classifier, distance or actor reward.

Exact action-prefix overlap is a syntactic statistic, NOT SAAM state equivalence,
semantic correctness, or causal blame. Manual-review sampling excludes previously
reviewed questions and a task-level reserved partition. Missing probabilities are
never imputed. Correctness and legality are reported as logged, not relabeled.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from .io import canonical_json, iter_jsonl, read_json, sha256_json
from .validation import require, task_id


def question_key(row: dict[str, Any]) -> str:
    """Leakage partition key, deliberately independent of run and example index."""
    return sha256_json([row['db_id'], row['question']])


def action_sequence(row: dict[str, Any]) -> list[str]:
    result = []
    for turn in row.get('turns', []):
        parsed = turn.get('parsed') or {}
        if not parsed.get('tool'):
            # A missing action must not silently shorten an exact prefix.
            result.append(canonical_json({'unparsed_turn': turn.get('turn_index')}))
            continue
        args = dict(parsed.get('arguments') or {})
        if parsed['tool'] == 'answer_from_context':
            args.pop('reason', None)
        result.append(canonical_json({'tool': parsed['tool'], 'arguments': args}))
    return result


def common_prefix(left: list[str], right: list[str]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def compact_rollout(row: dict[str, Any]) -> dict[str, Any]:
    """Allowlist observed actions/feedback; omit reasoning and all gold/ref fields."""
    result = {k: row.get(k) for k in (
        'example_index', 'db_id', 'question', 'trajectory_id', 'policy_global_step',
        'policy_micro_step', 'correct', 'legal', 'failure_type', 'errors', 'error_events',
    )}
    result['task_key'] = question_key(row)
    result['model_visible_task'] = '\n'.join(
        str(message.get('content', '')) for message in row.get('initial_model_input', [])
        if message.get('role') == 'user'
    )
    result['turns'] = []
    for index, turn in enumerate(row.get('turns') or []):
        parsed = turn.get('parsed') or {}
        output = turn.get('tool_output') or {}
        result['turns'].append({
            'turn_index': index,
            'tool': parsed.get('tool'),
            'arguments': {k: v for k, v in (parsed.get('arguments') or {}).items() if k != 'reason'},
            'observation': {k: output[k] for k in (
                'table', 'columns', 'row_count', 'kind', 'derivation', 'rows',
                'frequent_values', 'distinct_count', 'has_null', 'value', 'scalar', 'error',
            ) if k in output},
            'terminal_pred_sample': turn.get('pred_sample'),
            'execution_error_type': turn.get('execution_error_type'),
        })
    return result


def audit_run(rollouts: Path, manifest: Path, *, reviewed_keys: set[str]) -> dict[str, Any]:
    config = read_json(manifest, require_object=True)
    require(config.get('group_size') == 8, 'this audit requires logged K=8')
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    task_rows: dict[str, dict[str, Any]] = {}
    failures, statuses, rewards, tools = Counter(), Counter(), Counter(), Counter()
    protocol_counts = Counter()
    row_counts = 0
    lengths = defaultdict(list)
    seen_ids = set()
    review_groups = []
    # Keep only small metadata per trajectory. Full rows are streamed a second time
    # for a bounded review selection, never loaded as a multi-GB Python object graph.
    for row in iter_jsonl(rollouts):
        row_counts += 1
        require(isinstance(row.get('correct'), bool), 'missing/invalid logged correct')
        require(isinstance(row.get('legal'), bool), 'missing/invalid logged legal')
        require(row.get('protocol_hash') == config['protocol_hash'], 'row protocol mismatch')
        protocol_counts[row['protocol_hash']] += 1
        task_id(row, fields=('example_index',))  # validate the source-local identifier
        key = question_key(row)
        partition = 'reviewed_development' if key in reviewed_keys else (
            'reserved_unread' if int(key[:8], 16) % 5 == 0 else 'discovery'
        )
        task_rows[key] = {'task_key': key, 'example_index': row['example_index'],
                          'db_id': row['db_id'], 'question': row['question'], 'partition': partition}
        group_key = (row['policy_global_step'], row.get('policy_micro_step', 0), key)
        trajectory_key = (*group_key, row['trajectory_id'])
        require(trajectory_key not in seen_ids, f'duplicate rollout identity {trajectory_key}')
        seen_ids.add(trajectory_key)
        c, legal = row['correct'], row['legal']
        statuses[f'{"correct" if c else "wrong"}_{"legal" if legal else "illegal"}'] += 1
        has_error = bool(row.get('error_events')) or bool(row.get('errors'))
        statuses[f'{"correct" if c else "wrong"}_{"with" if has_error else "without"}_error'] += 1
        failures[str(row.get('failure_type') or 'none')] += 1
        rewards[str((row.get('result_reward') or {}).get('value'))] += 1
        seq = action_sequence(row)
        lengths['correct' if c else 'wrong'].append(len(seq))
        for turn in row.get('turns', []):
            tools[str((turn.get('parsed') or {}).get('tool') or 'unparsed')] += 1
        groups[group_key].append({'trajectory_id': row['trajectory_id'], 'correct': c,
                                 'actions': seq, 'legal': legal, 'task_key': key})
    hist, incomplete, prefixes = Counter(), [], Counter()
    wrong_in_mixed = 0
    for gkey, rows in groups.items():
        n_correct = sum(row['correct'] for row in rows)
        if len(rows) != 8:
            incomplete.append({'policy_global_step': gkey[0], 'policy_micro_step': gkey[1],
                               'task_key': gkey[2], 'rows': len(rows)})
            continue
        hist[n_correct] += 1
        positives = [row for row in rows if row['correct']]
        negatives = [row for row in rows if not row['correct']]
        for negative in negatives if positives else []:
            wrong_in_mixed += 1
            prefixes[max(common_prefix(negative['actions'], p['actions']) for p in positives)] += 1
        if task_rows[gkey[2]]['partition'] != 'discovery' or not negatives:
            continue
        # Hash-based sampling, not cherry-picking for apparent format/distance success.
        negative = min(negatives, key=lambda r: sha256_json([gkey, r['trajectory_id']]))
        positive = min(positives, key=lambda r: sha256_json([gkey, r['trajectory_id']])) if positives else None
        review_groups.append({'group_key': list(gkey), 'example_index': task_rows[gkey[2]]['example_index'],
                              'correct_of_8': n_correct, 'negative_id': negative['trajectory_id'],
                              'positive_id': positive['trajectory_id'] if positive else None,
                              'selection_basis': 'deterministic_hash_not_probability',
                              'rank': sha256_json(gkey)})
    # Separate paired and all-wrong strata; at most one observation per task.
    selected, chosen_tasks = [], set()
    for paired, limit in ((True, 12), (False, 4)):
        added = 0
        for group in sorted(review_groups, key=lambda r: r['rank']):
            key = group['group_key'][2]
            if bool(group['positive_id']) != paired or key in chosen_tasks:
                continue
            selected.append(group)
            chosen_tasks.add(key)
            added += 1
            if added >= limit:
                break
    targets = {(g['group_key'][0], g['group_key'][1], g['group_key'][2], tid)
               for g in selected for tid in [g['positive_id'], g['negative_id']] if tid}
    review_rows = []
    for row in iter_jsonl(rollouts):
        key = (row['policy_global_step'], row.get('policy_micro_step', 0), question_key(row), row['trajectory_id'])
        if key in targets:
            review_rows.append(compact_rollout(row))
    identity_fields = ('adapter_path', 'initial_adapter_sha256', 'protocol_hash',
                       'student_prompt_sha256', 'base_model_identity', 'examples_json_sha256',
                       'credit_assignment', 'result_advantage_profile', 'span_balance_alpha',
                       'prompts_per_update', 'optimizer_steps', 'records', 'group_size',
                       'trainer_parallelism', 'rollout_settings')
    summary = {
        'identity': {k: config.get(k) for k in identity_fields},
        'rollouts': row_counts, 'unique_tasks': len(task_rows),
        'complete_k8_groups': sum(hist.values()), 'incomplete_groups': incomplete,
        'correct_of_8_histogram': {str(k): hist[k] for k in range(9)},
        'statuses': dict(statuses), 'failure_types': dict(failures), 'reward_values': dict(rewards),
        'turn_lengths': {k: {'mean': mean(v), 'median': median(v), 'max': max(v)} for k, v in lengths.items()},
        'tool_counts': dict(tools), 'wrong_in_mixed': wrong_in_mixed,
        'best_exact_action_prefix_histogram': dict(sorted(prefixes.items())),
        'logged_policy_steps': sorted(set(g[0] for g in groups)),
        'partition_tasks': dict(Counter(r['partition'] for r in task_rows.values())),
    }
    return {'summary': summary, 'tasks': list(task_rows.values()),
            'review_selection': selected, 'review_rows': review_rows}
