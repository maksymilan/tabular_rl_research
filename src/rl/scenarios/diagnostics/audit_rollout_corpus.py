#!/usr/bin/env python3
"""Audit immutable K=8 rollout snapshots; no remote actions or training changes."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'src'))

from rl.diagnostics.io import canonical_json, iter_jsonl, read_json, write_json, write_jsonl
from rl.diagnostics.reporting import ManifestBuilder
from rl.diagnostics.rollout_corpus import audit_run, question_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f'refusing to overwrite {args.output}')
    sources = read_json(args.sources, require_object=True)
    builder = ManifestBuilder('rollout-corpus-static-audit-v1', diagnostic_only=True,
                              actor_update_allowed=False, semantic_labels_validated=False)
    builder.add_input('source_inventory', args.sources)
    builder.add_input('implementation_cli', Path(__file__))
    builder.add_input('implementation_corpus', Path(__file__).resolve().parents[2] / 'diagnostics' / 'rollout_corpus.py')
    builder.add_input('implementation_io', Path(__file__).resolve().parents[2] / 'diagnostics' / 'io.py')
    reviewed_ids = set(map(str, sources['reviewed_example_ids']))
    reviewed_source = Path(sources['reviewed_identity_source'])
    reviewed_keys = {question_key(row) for row in iter_jsonl(reviewed_source)
                     if str(row['example_index']) in reviewed_ids}
    builder.add_input('reviewed_identity_source', reviewed_source)
    all_tasks, summary = {}, {}
    identity_anchor = None
    for source in sources['runs']:
        name = source['name']
        rollouts, manifest = Path(source['rollouts']), Path(source['manifest'])
        result = audit_run(rollouts, manifest, reviewed_keys=reviewed_keys)
        identity = result['summary']['identity']
        identity_key = tuple(identity.get(k) for k in (
            'protocol_hash', 'student_prompt_sha256', 'initial_adapter_sha256'))
        if not all(identity_key):
            raise ValueError(f'{name}: missing model/protocol/prompt identity')
        if identity_anchor is not None and identity_key != identity_anchor:
            raise ValueError(f'{name}: mixed model/protocol/prompt identities; audit separately')
        identity_anchor = identity_key
        for row in result['tasks']:
            key = row['task_key']
            if key in all_tasks:
                assert all_tasks[key]['partition'] == row['partition']
            else:
                all_tasks[key] = {**row, 'runs': [], 'example_index_by_run': {}}
            all_tasks[key]['runs'].append(name)
            all_tasks[key]['example_index_by_run'][name] = row['example_index']
        dest = args.output / name
        write_json(dest / 'summary.json', result['summary'])
        write_json(dest / 'review_selection.json', result['review_selection'])
        write_jsonl(dest / 'review_rows.jsonl', result['review_rows'])
        builder.add_input(name + '_rollouts', rollouts, records=result['summary']['rollouts'],
                          include_size=True, remote_source=source.get('remote_source'))
        builder.add_input(name + '_run_manifest', manifest)
        for file in ('summary.json', 'review_selection.json', 'review_rows.jsonl'):
            builder.add_output(name + '_' + file, dest / file)
        summary[name] = result['summary']
        print(canonical_json({'run': name, 'rollouts': result['summary']['rollouts'],
                              'tasks': result['summary']['unique_tasks']}), flush=True)
    write_jsonl(args.output / 'task_partitions.jsonl', sorted(all_tasks.values(), key=lambda r: r['task_key']))
    builder.add_output('task_partitions', args.output / 'task_partitions.jsonl', records=len(all_tasks))
    builder.set('total_unique_tasks', len(all_tasks)).write(args.output / 'manifest.json')


if __name__ == '__main__':
    main()
