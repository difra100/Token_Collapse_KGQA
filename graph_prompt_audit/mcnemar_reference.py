"""Compare a selected reference condition with a matched condition family.

For example, use ``--reference notext_real --prefix notext_`` to perform the
correct paired tests among text-ablated conditions, rather than comparing those
conditions with text-present ``real``.
"""

import argparse
import json
from pathlib import Path

from .mcnemar import compare, load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", default="real")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--exclude", nargs="*", default=[])
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    reference_name = args.reference
    reference = load(run_dir / f"{reference_name}.jsonl")
    excluded = set(args.exclude) | {reference_name}
    results = {}
    for path in sorted(run_dir.glob(f"{args.prefix}*.jsonl")):
        condition = path.stem
        if condition not in excluded:
            results[condition] = compare(reference, load(path))

    output = {
        "reference": f"{reference_name}.jsonl",
        "test": "two-sided exact McNemar (binomial test on discordant paired correctness)",
        "run_dir": str(run_dir),
        "conditions": results,
    }
    destination = args.output or run_dir / f"mcnemar_vs_{reference_name}.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "conditions": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
