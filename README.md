# biomfg-investigation

Synthetic CHO fed-batch simulator for a bio-manufacturing root-cause investigation project.
CPU-only Python. Agents read evidence, choose diagnostic tests, and name the root cause.

## Quick start

```
pip install -r requirements.txt
python scripts/validate.py --data data          # sanity checks
python scripts/generate_dataset.py              # regenerate data/ (seed 42)
```

```python
from simulator.api import batch_ids, get_evidence, run_test, test_table

ids = batch_ids("dev")                      # or "test"
ev = get_evidence(ids[0])                   # everything except the cause
r = run_test(ids[0], "offline_ph_check")    # {"result": "positive", "cost": 1, ...}
table = test_table()                        # costs + P(positive | cause), for information gain
```

`simulator.api.ground_truth()` is for the evaluator only.

## Layout

- `simulator/healthy.py` - healthy-batch generator (logistic growth, setpoint noise, batch-to-batch variation)
- `simulator/failures.py` - five failure signatures with severity-scaled, deliberately overlapping effects
- `simulator/tests.py` - nine diagnostic tests: cost, turnaround, P(result | cause)
- `simulator/batch.py`, `dataset.py` - assemble batches, QC, alarms, raw materials, missing data, split
- `simulator/api.py` - `get_evidence`, `run_test`, `test_table`, `batch_ids`
- `data/` - 200 batches, seed 42, stratified 100 dev / 100 test (`batches.json.gz`, `splits.json`, `ground_truth.json`)
- `docs/schema.md` - data schema; `docs/signatures.png` - healthy-style signature plot per failure cause

## Validation (`scripts/validate.py`)

- A simple hand-written rule classifier scores about 60% on both dev and test (chance is 20%, target 50-70%),
  so the problem is neither trivial nor impossible. This is a first sanity check, not a full validation.
- Empirical `run_test` positive rates follow the probability table, slightly lower for weak failures.
- Signatures overlap on purpose: lactate rises in contamination and mixing failure, viability drops in
  temperature excursion and contamination, online pH moves in sensor drift and in real contamination,
  plus red-herring alarms, nuisance temperature blips, sensor dropouts, missing QC values and a CoA
  that can pass a contaminated lot.

## Honesty note (for the pitch)

All numeric ranges are illustrative CHO-style values loosely based on typical literature ranges.
They are **not calibrated to a real process**. This is synthetic data for evaluating agent reasoning,
not a validated process model.
