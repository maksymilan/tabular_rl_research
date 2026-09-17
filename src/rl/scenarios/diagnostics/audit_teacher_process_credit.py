#!/usr/bin/env python3
"""Explicitly authorized, bounded, post-hoc external-teacher credit pilot."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'src'))

from rl.diagnostics.io import canonical_json, read_jsonl
from rl.diagnostics.teacher_credit import prepare_pilot, run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-v4-pro')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--execute-authorized-api', action='store_true')
    args = parser.parse_args()
    if args.execute_authorized_api and (args.output / 'requests.jsonl').is_file():
        # The preflight artifact is immutable; execution must consume exactly its requests.
        cases = read_jsonl(args.output / 'requests.jsonl')
        print(canonical_json({'preflight':'reusing_immutable_requests', 'cases':len(cases),
                              'max_requests':len(cases)*args.repeats}), flush=True)
    else:
        cases = prepare_pilot(args.selection, args.output, model=args.model, repeats=args.repeats)
        print(canonical_json({'preflight':'passed', 'cases':len(cases), 'max_requests':len(cases)*args.repeats}), flush=True)
    if args.execute_authorized_api:
        print(canonical_json(run_pilot(cases, args.output, repeats=args.repeats)), flush=True)


if __name__ == '__main__':
    main()
