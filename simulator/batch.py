"""Assemble one batch: process data + QC + equipment events + raw materials + hidden truth."""
import numpy as np

from .failures import CAUSES, inject
from .healthy import SIGNALS, sample_params, simulate

NUISANCE_ALARMS = ["DOOR_OPEN", "FEED_PUMP_WARN", "COMMS_LOSS", "FILTER_DP_HIGH"]
SUPPLIERS = ["SupA", "SupB", "SupC"]


def _round(x, nd=3):
    return [None if (v is None or v != v) else round(float(v), nd) for v in x]


def _apply_missing(rng, series):
    """Random dropouts, and occasionally a whole multi-hour channel gap."""
    n = len(series)
    out = series.copy()
    out[rng.random(n) < rng.uniform(0.005, 0.03)] = np.nan
    if rng.random() < 0.15:
        g = int(rng.integers(6, 37))
        s = int(rng.integers(0, max(n - g, 1)))
        out[s:s + g] = np.nan
    return out


def build_batch(batch_id, rng, cause, severity):
    p = sample_params(rng)
    facts = inject(cause, p, rng, severity)
    ts = simulate(p, rng)
    n = len(ts["hours"])

    # nuisance temperature blip (no alarm) in some batches, healthy or not
    if cause != "temperature_excursion" and rng.random() < 0.30:
        s = int(rng.integers(24, n - 6))
        ts["temperature"][s:s + int(rng.integers(1, 4))] += rng.uniform(0.3, 1.3)
        if rng.random() < 0.40:
            facts["planned_alarms"].append((s, "HIGH_TEMP"))     # spurious alarm too

    # alarms: planned + red herrings
    alarms = [{"hour": int(h), "code": c} for h, c in facts["planned_alarms"]]
    for _ in range(int(rng.choice([0, 0, 1, 1, 2]))):
        alarms.append({"hour": int(rng.integers(0, n)), "code": str(rng.choice(NUISANCE_ALARMS))})
    alarms.sort(key=lambda a: a["hour"])

    # equipment calibration / maintenance
    overdue = facts["ph_cal_overdue"]
    calibration = {
        "ph_probe_days_since_cal": int(rng.integers(25, 61) if overdue else rng.integers(1, 14)),
        "temp_probe_days_since_cal": int(rng.integers(5, 120)),
        "do_probe_days_since_cal": int(rng.integers(1, 40)),
        "calibration_interval_days": 14,
    }
    maintenance = [{"item": "impeller_service",
                    "days_ago": int(rng.integers(200, 400) if facts["impeller_overdue"] else rng.integers(10, 150))},
                   {"item": "bioreactor_cleaning_cycle", "days_ago": int(rng.integers(1, 10))}]

    # raw materials
    flagged = facts["lot_flagged"]
    raw = {
        "lot_id": f"LOT-{int(rng.integers(1000, 9999))}",
        "supplier": str(rng.choice(SUPPLIERS)),
        "supplier_coa": {
            "result": "marginal" if facts["coa_marginal"] else "pass",
            "endotoxin_eu_ml": round(float(rng.uniform(0.15, 0.4) if facts["coa_marginal"] else rng.uniform(0.01, 0.12)), 3),
        },
        "lot_history": {
            "n_prior_batches": int(rng.integers(0, 30)),
            "n_prior_deviations": int(rng.integers(1, 4) if flagged else rng.integers(0, 2)),
            "flagged": bool(flagged),
        },
    }

    # QC (end of batch); each value may be pending/missing
    contam = cause == "contaminated_raw_material"
    detect = 0.35 + 0.55 * severity
    bio = float(rng.uniform(1, 60)) if contam and rng.random() < 0.55 * detect + 0.15 else (
        float(rng.uniform(1, 5)) if rng.random() < 0.03 else 0.0)
    endo = float(rng.uniform(0.3, 4.0)) if contam and rng.random() < 0.55 * detect + 0.10 else float(
        rng.lognormal(np.log(0.05), 0.5))
    myco = "pos" if (contam and rng.random() < 0.12) else "neg"
    penalty = {"ph_sensor_drift": 0.05, "temperature_excursion": 0.25, "contaminated_raw_material": 0.30,
               "low_viability_inoculum": 0.20, "mixing_failure": 0.20}[cause] * severity
    titer = ts["_ivcd"] * 0.00085 * (1 - penalty) * (1 + rng.normal(0, 0.05))
    purity = 97.5 - (3.5 * severity if contam else 0) + rng.normal(0, 0.6)
    qc = {"titer_g_l": round(float(titer), 3), "purity_pct": round(float(purity), 2),
          "endotoxin_eu_ml": round(endo, 3), "bioburden_cfu_ml": round(bio, 2), "mycoplasma": myco}
    for key in ("endotoxin_eu_ml", "bioburden_cfu_ml", "mycoplasma", "purity_pct"):
        if rng.random() < 0.40:
            qc[key] = None

    # observed time series with missing data
    process = {"hours": [int(h) for h in ts["hours"]]}
    for s in SIGNALS:
        process[s] = _round(_apply_missing(rng, np.array(ts[s], dtype=float)), 3)

    evidence = {
        "batch_id": batch_id,
        "process": "CHO fed-batch",
        "duration_days": p["n_days"],
        "time_series": process,
        "qc_results": qc,
        "equipment_events": {"alarms": alarms, "calibration": calibration, "maintenance_log": maintenance},
        "raw_materials": raw,
    }
    truth = {
        "batch_id": batch_id, "true_cause": cause, "severity": round(float(severity), 3),
        "onset_hour": facts["onset_hour"],
        "_ph_true_end": round(float(np.nanmean(ts["_ph_true"][-48:])), 4),
    }
    return evidence, truth


def make_batch(seed, index, cause=None, severity=None):
    rng = np.random.default_rng([seed, index])
    cause = cause or CAUSES[index % len(CAUSES)]
    severity = float(rng.uniform(0.10, 1.0)) if severity is None else severity
    return build_batch(f"B{index:04d}", rng, cause, severity)
