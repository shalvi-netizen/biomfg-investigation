"""Diagnostic tests: cost, turnaround, and P(positive | cause).

"positive" = the test finds an abnormality consistent with the defect it targets.
These tables are what an agent can use to compute expected information gain.
Cause order: see failures.CAUSES.
"""
from .failures import CAUSES

# cause order: ph_sensor_drift, temperature_excursion, contaminated_raw_material,
#              low_viability_inoculum, mixing_failure
_ROWS = {
    "offline_ph_check": dict(
        cost=1, hours=0.5, p=[0.90, 0.05, 0.05, 0.05, 0.15],
        desc="Offline pH sample disagrees with online pH by >0.15."),
    "ph_calibration_check": dict(
        cost=2, hours=1, p=[0.85, 0.10, 0.08, 0.08, 0.10],
        desc="pH probe fails a calibration check."),
    "temp_logger_audit": dict(
        cost=2, hours=1, p=[0.05, 0.90, 0.05, 0.05, 0.08],
        desc="Independent temperature logger shows an excursion."),
    "trypan_blue_count": dict(
        cost=3, hours=1, p=[0.25, 0.80, 0.70, 0.90, 0.35],
        desc="Manual trypan blue count confirms low viability."),
    "seed_train_review": dict(
        cost=2, hours=2, p=[0.05, 0.05, 0.10, 0.85, 0.05],
        desc="Seed-train records show a poor-quality inoculum."),
    "impeller_do_check": dict(
        cost=4, hours=3, p=[0.06, 0.06, 0.10, 0.05, 0.85],
        desc="Mixing-time / DO mapping finds a mixing fault."),
    "raw_material_retest": dict(
        cost=10, hours=24, p=[0.04, 0.04, 0.80, 0.04, 0.04],
        desc="Retest of the retained raw-material lot fails spec."),
    "endotoxin_assay": dict(
        cost=12, hours=24, p=[0.04, 0.04, 0.55, 0.04, 0.04],
        desc="Endotoxin assay is positive (LAL)."),
    "bioburden_assay": dict(
        cost=15, hours=48, p=[0.03, 0.05, 0.60, 0.03, 0.03],
        desc="Bioburden culture grows organisms (slow, expensive)."),
}

TESTS = {
    name: {
        "cost": r["cost"],
        "turnaround_hours": r["hours"],
        "description": r["desc"],
        "p_positive": dict(zip(CAUSES, r["p"])),
        # false-positive floor used when severity weakens a detectable cause
        "floor": min(r["p"]),
    }
    for name, r in _ROWS.items()
}


def effective_p(test_name, cause, severity):
    """Table probability, shrunk toward the floor for weak (low-severity) failures."""
    t = TESTS[test_name]
    p, floor = t["p_positive"][cause], t["floor"]
    if p <= floor:
        return p
    return floor + (p - floor) * (0.55 + 0.45 * severity)
