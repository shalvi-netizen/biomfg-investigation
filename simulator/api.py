"""Public API for the agents.

    from simulator.api import get_evidence, run_test, test_table, batch_ids

`ground_truth()` is for the evaluator only; agents must never call it.
Data is read from data/ if present, otherwise regenerated from the fixed seed.
"""
import copy
import gzip
import json
import os
import zlib

import numpy as np

from .failures import CAUSES
from .tests import TESTS, effective_p

SEED = 42
N_BATCHES = 200
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class Simulator:
    def __init__(self, data_dir=DEFAULT_DATA_DIR, seed=SEED, n_batches=N_BATCHES):
        self.seed = seed
        self._evidence, self._truth, self._splits = {}, {}, {}
        ev_path = os.path.join(data_dir, "batches.json.gz")
        if os.path.exists(ev_path):
            with gzip.open(ev_path, "rt") as f:
                for rec in json.load(f):
                    self._evidence[rec["batch_id"]] = rec
            with open(os.path.join(data_dir, "ground_truth.json")) as f:
                self._truth = {r["batch_id"]: r for r in json.load(f)}
            with open(os.path.join(data_dir, "splits.json")) as f:
                self._splits = json.load(f)
        else:
            from .dataset import generate
            ev, truth, splits = generate(seed, n_batches)
            self._evidence = {e["batch_id"]: e for e in ev}
            self._truth = {t["batch_id"]: t for t in truth}
            self._splits = splits

    def batch_ids(self, split=None):
        if split is None:
            return sorted(self._evidence)
        return list(self._splits[split])

    def get_evidence(self, batch_id):
        """Everything observable about a batch, never the cause."""
        return copy.deepcopy(self._evidence[batch_id])

    def run_test(self, batch_id, test_name, replicate=0):
        """Noisy result of a diagnostic test plus its cost.

        Deterministic per (batch, test, replicate): re-running the same replicate returns the
        same result (still costs again); a new replicate is an independent draw.
        """
        if test_name not in TESTS:
            raise KeyError(f"unknown test {test_name!r}; options: {sorted(TESTS)}")
        truth = self._truth[batch_id]
        idx = int(batch_id[1:])
        rng = np.random.default_rng([self.seed, 7919, idx, zlib.crc32(test_name.encode()), replicate])
        p = effective_p(test_name, truth["true_cause"], truth["severity"])
        positive = bool(rng.random() < p)
        t = TESTS[test_name]
        return {"batch_id": batch_id, "test": test_name, "result": "positive" if positive else "negative",
                "cost": t["cost"], "turnaround_hours": t["turnaround_hours"]}

    def test_table(self):
        """Costs and P(positive | cause) for every test, for information-gain calculations."""
        return {"causes": list(CAUSES),
                "tests": {n: {k: v for k, v in t.items() if k != "floor"} for n, t in TESTS.items()}}

    def ground_truth(self, batch_id):
        """EVALUATOR ONLY."""
        return dict(self._truth[batch_id])


_default = None


def _sim():
    global _default
    if _default is None:
        _default = Simulator()
    return _default


def get_evidence(batch_id):
    return _sim().get_evidence(batch_id)


def run_test(batch_id, test_name, replicate=0):
    return _sim().run_test(batch_id, test_name, replicate)


def test_table():
    return _sim().test_table()


def batch_ids(split=None):
    return _sim().batch_ids(split)


def ground_truth(batch_id):
    return _sim().ground_truth(batch_id)
