"""Regenerate data/ (200 batches, seed 42, dev/test split).

    python scripts/generate_dataset.py [--seed 42] [--n 200] [--out data]
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.dataset import generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    evidence, truth, splits = generate(a.seed, a.n)
    os.makedirs(a.out, exist_ok=True)
    with gzip.open(os.path.join(a.out, "batches.json.gz"), "wt") as f:
        json.dump(evidence, f, separators=(",", ":"))
    with open(os.path.join(a.out, "ground_truth.json"), "w") as f:
        json.dump(truth, f, indent=1)      # EVALUATOR ONLY
    with open(os.path.join(a.out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)
    print(f"wrote {len(evidence)} batches to {a.out}/ (dev={len(splits['dev'])}, test={len(splits['test'])})")


if __name__ == "__main__":
    main()
