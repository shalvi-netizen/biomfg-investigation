"""Sanity-check the simulator.

    python scripts/validate.py --data data [--plot signatures.png]

1. A deliberately simple rule-based classifier (target: ~50-70% accuracy; chance = 20%).
2. Empirical run_test positive rates vs the probability table.
3. Optional signature plot: healthy vs each failure.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulator.api import Simulator  # noqa: E402
from simulator.failures import CAUSES  # noqa: E402
from simulator.tests import TESTS, effective_p  # noqa: E402


def _arr(ev, key):
    return np.array([np.nan if v is None else v for v in ev["time_series"][key]], dtype=float)


def rule_classifier(ev):
    """Simple hand-written rules. Uses only get_evidence output."""
    temp, ph = _arr(ev, "temperature"), _arr(ev, "ph_online")
    viab, lac, do = _arr(ev, "viability"), _arr(ev, "lactate"), _arr(ev, "dissolved_oxygen")
    vcd = _arr(ev, "viable_cell_density")
    alarms = {a["code"] for a in ev["equipment_events"]["alarms"]}
    cal = ev["equipment_events"]["calibration"]
    raw, qc = ev["raw_materials"], ev["qc_results"]

    if np.nanmax(temp) > 37.8 and "HIGH_TEMP" in alarms:
        return "temperature_excursion"
    if (qc["bioburden_cfu_ml"] or 0) > 5 or (qc["endotoxin_eu_ml"] or 0) > 0.25 or qc["mycoplasma"] == "pos":
        return "contaminated_raw_material"
    if np.nanmax(np.abs(ph[-72:] - 7.0)) > 0.25 and cal["ph_probe_days_since_cal"] > cal["calibration_interval_days"]:
        return "ph_sensor_drift"
    if viab[:6][~np.isnan(viab[:6])].mean() < 90:
        return "low_viability_inoculum"
    if np.nanstd(do[48:]) > 6:
        return "mixing_failure"
    if raw["lot_history"]["flagged"]:
        return "contaminated_raw_material"
    if np.nanmax(temp) > 37.8:
        return "temperature_excursion"
    return "mixing_failure"


def check_rules(sim):
    out = {}
    for split in ("dev", "test"):
        ids = sim.batch_ids(split)
        pred = [rule_classifier(sim.get_evidence(b)) for b in ids]
        true = [sim.ground_truth(b)["true_cause"] for b in ids]
        out[split] = (np.mean([p == t for p, t in zip(pred, true)]), pred, true)
    return out


def check_tests(sim, reps=5):
    """Empirical positive rate per (test, cause) over all batches and replicates."""
    hits = defaultdict(list)
    for b in sim.batch_ids():
        c = sim.ground_truth(b)["true_cause"]
        for t in TESTS:
            for r in range(reps):
                hits[(t, c)].append(sim.run_test(b, t, r)["result"] == "positive")
    return hits


def plot_signatures(sim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from simulator.batch import make_batch

    rng_cause = {c: [b for b in sim.batch_ids() if sim.ground_truth(b)["true_cause"] == c][:4] for c in CAUSES}
    sig = [("temperature", "Temperature (C)"), ("ph_online", "pH online"), ("viability", "Viability (%)"),
           ("lactate", "Lactate (g/L)"), ("dissolved_oxygen", "DO (%)"), ("viable_cell_density", "VCD (1e6/mL)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    colors = dict(zip(CAUSES, ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]))
    for ax, (key, label) in zip(axes.ravel(), sig):
        for c in CAUSES:
            for i, b in enumerate(rng_cause[c]):
                ev = sim.get_evidence(b)
                ax.plot(ev["time_series"]["hours"], _arr(ev, key), color=colors[c], alpha=0.5, lw=0.8,
                        label=c if i == 0 else None)
        ax.set_title(label)
        ax.set_xlabel("hour")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    sim = Simulator(data_dir=a.data)

    print("== Rule-based classifier (chance = 20%, target 50-70%) ==")
    res = check_rules(sim)
    for split, (acc, _, _) in res.items():
        print(f"  {split}: {acc:.1%}")
    _, pred, true = res["dev"]
    print("  dev confusion (true -> predicted counts):")
    for c in CAUSES:
        cnt = Counter(p for p, t in zip(pred, true) if t == c)
        print(f"    {c:28s}", {k: v for k, v in cnt.items()})

    print("\n== run_test positive rate: empirical vs table (effective at mean severity) ==")
    hits = check_tests(sim)
    sev = {c: np.mean([sim.ground_truth(b)["severity"] for b in sim.batch_ids()
                       if sim.ground_truth(b)["true_cause"] == c]) for c in CAUSES}
    worst = 0
    for t in TESTS:
        row = []
        for c in CAUSES:
            emp = np.mean(hits[(t, c)])
            tab = TESTS[t]["p_positive"][c]
            worst = max(worst, abs(emp - effective_p(t, c, sev[c])))
            row.append(f"{emp:.2f}/{tab:.2f}")
        print(f"  {t:22s}", "  ".join(row))
    print("  (columns:", ", ".join(CAUSES), "; each cell = empirical/table)")
    print(f"  max |empirical - severity-adjusted table| = {worst:.3f}")

    if a.plot:
        plot_signatures(sim, a.plot)
        print(f"\nsaved plot to {a.plot}")


if __name__ == "__main__":
    main()
