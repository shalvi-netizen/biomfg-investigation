"""Dataset generation: 200 batches, fixed seed, shuffled cause assignment, dev/test split."""
import numpy as np

from .batch import make_batch
from .failures import CAUSES


def generate(seed=42, n_batches=200):
    rng = np.random.default_rng([seed, 999])
    causes = [CAUSES[i % len(CAUSES)] for i in range(n_batches)]
    rng.shuffle(causes)
    evidence, truth = [], []
    for i, cause in enumerate(causes):
        e, t = make_batch(seed, i, cause=cause)
        evidence.append(e)
        truth.append(t)
    # stratified split: half of each cause to dev, half to test
    dev, test = [], []
    for c in CAUSES:
        ids = [t["batch_id"] for t in truth if t["true_cause"] == c]
        half = len(ids) // 2
        dev += ids[:half]
        test += ids[half:]
    return evidence, truth, {"dev": sorted(dev), "test": sorted(test)}
