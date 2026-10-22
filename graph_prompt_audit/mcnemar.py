"""Exact paired McNemar tests for saved GraphQA prediction records.

Each JSONL row must contain ``id``, ``pred``, and ``label``.  The script uses
the WebQSP Hit@1 rule used by the audit runners: after lower-casing and basic
normalization, a prediction is correct when it contains at least one
pipe-separated gold answer.  For each condition it compares paired correctness
against ``real.jsonl`` using the two-sided exact binomial McNemar test.
"""

import argparse
import json
import re
import string
from pathlib import Path

from scipy.stats import binomtest


def normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", re.sub(r"\b(a|an|the)\b", " ", text)).strip()


def hit(prediction: str, label: str) -> bool:
    prediction = normalize(prediction.replace("|", " "))
    return any(normalize(answer) in prediction for answer in label.split("|"))


def load(path: Path) -> dict[str, tuple[bool, str]]:
    rows: dict[str, tuple[bool, str]] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            identifier = str(row["id"])
            if identifier in rows:
                raise ValueError(f"duplicate id {identifier!r} in {path}:{line_number}")
            rows[identifier] = (hit(row["pred"], row["label"]), row["pred"])
    if not rows:
        raise ValueError(f"no prediction records in {path}")
    return rows


def compare(real: dict[str, tuple[bool, str]], condition: dict[str, tuple[bool, str]]) -> dict:
    if set(real) != set(condition):
        raise ValueError(
            "prediction id sets differ: "
            f"real-only={len(set(real) - set(condition))}, "
            f"condition-only={len(set(condition) - set(real))}"
        )
    identifiers = sorted(real)
    real_better = sum(real[i][0] and not condition[i][0] for i in identifiers)
    condition_better = sum(not real[i][0] and condition[i][0] for i in identifiers)
    discordant = real_better + condition_better
    p_value = binomtest(real_better, discordant, p=0.5).pvalue if discordant else 1.0
    real_score = 100.0 * sum(real[i][0] for i in identifiers) / len(identifiers)
    condition_score = 100.0 * sum(condition[i][0] for i in identifiers) / len(identifiers)
    churn = 100.0 * sum(real[i][1] != condition[i][1] for i in identifiers) / len(identifiers)
    return {
        "num_examples": len(identifiers),
        "real_hit_at_1": real_score,
        "condition_hit_at_1": condition_score,
        "delta_points": condition_score - real_score,
        "real_better": real_better,
        "condition_better": condition_better,
        "discordant": discordant,
        "p_value_two_sided_exact": p_value,
        "prediction_churn_percent": churn,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="condition stems to omit, e.g. no_graph_tokens",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    real_path = run_dir / "real.jsonl"
    real = load(real_path)
    excluded = set(args.exclude) | {"real"}
    results = {}
    for path in sorted(run_dir.glob("*.jsonl")):
        condition = path.stem
        if condition in excluded:
            continue
        results[condition] = compare(real, load(path))

    output = {
        "reference": "real.jsonl",
        "test": "two-sided exact McNemar (binomial test on discordant paired correctness)",
        "run_dir": str(run_dir),
        "conditions": results,
    }
    destination = args.output or run_dir / "mcnemar_vs_real.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(destination), "conditions": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
