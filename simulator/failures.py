"""Failure signatures. Each injector edits the healthy params and returns hidden facts.

Signatures overlap on purpose:
  - lactate rises early in contamination AND (more mildly) in mixing failure
  - viability drops in temperature excursion AND contamination
  - low cell density in low inoculum AND mixing failure
  - online pH moves in pH sensor drift AND real pH shifts in contamination
Severity (0.25-1.0) scales the effect, so weak cases are ambiguous.
"""
CAUSES = [
    "ph_sensor_drift",
    "temperature_excursion",
    "contaminated_raw_material",
    "low_viability_inoculum",
    "mixing_failure",
]


def inject(cause, params, rng, sev):
    n = params["n_days"] * 24
    facts = {
        "ph_cal_overdue": bool(rng.random() < 0.25),       # red herring in other causes
        "impeller_overdue": bool(rng.random() < 0.15),
        "lot_flagged": bool(rng.random() < 0.25),
        "coa_marginal": bool(rng.random() < 0.10),
        "planned_alarms": [],                              # (hour, code)
        "contam_onset": None,
        "onset_hour": None,
    }

    if cause == "ph_sensor_drift":
        onset = int(rng.integers(24, 120))
        rate = rng.choice([-1, 1]) * (0.0012 + 0.0025 * sev)      # pH/h
        params["ph_sensor_drift"] = (onset, rate, 0.15 + 0.55 * sev)
        params["viab_penalty"].append((onset + 24, 96, 1.0 + 3.0 * sev))   # mild cell effect
        params["growth_scale"] *= 1 - 0.04 * sev
        facts["ph_cal_overdue"] = bool(rng.random() < 0.70)
        facts["onset_hour"] = onset

    elif cause == "temperature_excursion":
        onset = int(rng.integers(72, min(216, n - 48)))
        dur = int(rng.integers(6, 31))
        amp = 0.4 + 1.4 * sev                                     # up to ~39.5 C
        params["temp_events"].append((onset, dur, amp))
        params["viab_penalty"].append((onset, 48, 4.0 + 22.0 * sev))
        params["growth_scale"] *= 1 - 0.15 * sev
        if rng.random() < 0.55:                                   # alarm sometimes missing
            facts["planned_alarms"].append((onset + int(rng.integers(0, 4)), "HIGH_TEMP"))
        facts["onset_hour"] = onset

    elif cause == "contaminated_raw_material":
        onset = int(rng.integers(24, 72))
        params["lactate_boost"].append((onset, 48, 0.006 + 0.014 * sev))
        params["ph_true_shift"].append((onset + 24, 120, -(0.05 + 0.20 * sev)))
        vstart = int(rng.integers(100, 180))
        params["viab_penalty"].append((vstart, 60, 3.0 + 18.0 * sev))
        facts["lot_flagged"] = bool(rng.random() < 0.45)
        facts["coa_marginal"] = bool(rng.random() < 0.30)        # CoA can pass a bad lot
        facts["contam_onset"] = onset
        facts["onset_hour"] = onset

    elif cause == "low_viability_inoculum":
        params["x0_scale"] = 1 - 0.55 * sev
        params["viab0_drop"] = 1.0 + 12.0 * sev
        params["growth_scale"] *= 1 - 0.08 * sev
        facts["onset_hour"] = 0

    elif cause == "mixing_failure":
        onset = int(rng.integers(48, 150))
        params["do_disturb"] = (onset, 2.0 + 7.0 * sev)
        params["k_scale"] *= 1 - 0.30 * sev
        params["lactate_boost"].append((onset, 72, 0.002 + 0.006 * sev))   # overlaps contamination
        params["viab_penalty"].append((onset + 24, 96, 1.0 + 6.0 * sev))
        if rng.random() < 0.60:
            params["agit_sag"] = (onset, int(rng.integers(12, 60)), 10 + 25 * sev)
            if rng.random() < 0.50:
                facts["planned_alarms"].append((onset, "AGITATION_LOW"))
        facts["impeller_overdue"] = bool(rng.random() < 0.60)
        facts["onset_hour"] = onset

    else:
        raise ValueError(f"unknown cause {cause}")
    return facts
