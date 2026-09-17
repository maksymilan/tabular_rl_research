"""Compare first-error capped credit on an immutable raw rollout corpus."""
import argparse
import json
from pathlib import Path

from rl.diagnostics.saam_error_cap import audit_error_cap
from rl.fixed_pool.io import load_jsonl, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--soft-alpha",
        dest="soft_alphas",
        type=float,
        action="append",
        default=[],
        help="Offline attenuation alpha for later deterministic Harness errors; repeatable.",
    )
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    report = audit_error_cap(load_jsonl(args.rollouts), tokenizer, soft_alphas=args.soft_alphas)
    report.update(input=str(args.rollouts), input_sha256=sha256_file(args.rollouts), tokenizer=args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
