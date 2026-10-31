"""Healthy CHO fed-batch generator (hourly, 10-14 days).

All numeric ranges are illustrative CHO-style values, not calibrated to a real process.
Failure modes plug in through the `effects` keys of the params dict (see failures.py).
"""
import numpy as np

SIGNALS = [
    "temperature", "ph_online", "dissolved_oxygen", "agitation", "pressure",
    "viable_cell_density", "viability", "glucose", "lactate",
]


def _ar1(rng, n, sigma, phi=0.6):
    """Autocorrelated noise with stationary std `sigma`."""
    x = np.zeros(n)
    x[0] = rng.normal(0, sigma)
    e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def sample_params(rng):
    """Batch-to-batch variation: no two healthy batches look identical."""
    return {
        "n_days": int(rng.integers(10, 15)),
        "mu": rng.normal(0.031, 0.003),            # 1/h growth rate
        "k": rng.normal(18.0, 2.5),                # carrying capacity, 1e6 cells/mL
        "x0": rng.normal(0.50, 0.05),              # seeding density, 1e6 cells/mL
        "lag_h": rng.uniform(6, 24),
        "viab_peak": rng.normal(97.0, 1.0),        # %
        "viab_decline_start_day": rng.uniform(7.0, 9.5),
        "viab_decline_per_day": rng.uniform(1.2, 2.5),
        "temp_set": 36.8 + rng.normal(0, 0.05),
        "ph_set": 7.0 + rng.normal(0, 0.02),
        "do_set": 40.0 + rng.normal(0, 2.0),
        "agit_set": 120.0 + rng.normal(0, 2.0),
        "pressure_set": 0.30 + rng.normal(0, 0.01),
        "q_glc": rng.normal(0.0040, 0.0005),       # g/L/h per 1e6 cells/mL
        "lac_yield": rng.normal(0.9, 0.1),
        # --- failure hooks (neutral by default) ---
        "x0_scale": 1.0,
        "growth_scale": 1.0,
        "k_scale": 1.0,
        "viab0_drop": 0.0,                         # % below peak at t=0, recovers (tau 72 h)
        "viab_penalty": [],                        # (start_h, ramp_h, depth_pct)
        "lactate_boost": [],                       # (start_h, ramp_h, g/L/h extra)
        "temp_events": [],                         # (start_h, dur_h, amplitude_C)
        "ph_sensor_drift": None,                   # (start_h, rate_per_h, cap)
        "ph_true_shift": [],                       # (start_h, ramp_h, amount)
        "do_disturb": None,                        # (start_h, amplitude)
        "agit_sag": None,                          # (start_h, dur_h, depth_rpm)
    }


def simulate(params, rng):
    """Return dict of hourly arrays (no missing data) plus hidden true pH."""
    p = params
    n = p["n_days"] * 24
    t = np.arange(n, dtype=float)

    # total cell density: logistic growth with lag
    te = np.clip(t - p["lag_h"], 0, None)
    k = p["k"] * p["k_scale"]
    x0 = max(p["x0"] * p["x0_scale"], 0.05)
    mu = p["mu"] * p["growth_scale"]
    xt = k / (1 + (k / x0 - 1) * np.exp(-mu * te))

    # viability
    v = p["viab_peak"] - p["viab0_drop"] * np.exp(-t / 72.0)
    v -= p["viab_decline_per_day"] * np.clip(t / 24 - p["viab_decline_start_day"], 0, None)
    for start, ramp, depth in p["viab_penalty"]:
        v -= depth * np.clip((t - start) / max(ramp, 1), 0, 1)
    v += _ar1(rng, n, 0.4)
    v = np.clip(v, 20, 100)
    vcd = xt * v / 100 * (1 + rng.normal(0, 0.03, n))

    # glucose (daily bolus feed from day 3) and lactate
    glc = np.zeros(n)
    lac = np.zeros(n)
    glc[0], lac[0] = 6.0 + rng.normal(0, 0.2), 0.3 + abs(rng.normal(0, 0.05))
    for i in range(1, n):
        rate = p["q_glc"] * xt[i - 1]
        g = glc[i - 1] - rate
        if i % 24 == 0 and i >= 72 and g < 4.0:
            g += 2.0
        glc[i] = max(g, 0.2)
        y = p["lac_yield"] if i < 7 * 24 else p["lac_yield"] - 1.1
        extra = 0.0
        for start, ramp, g_per_h in p["lactate_boost"]:
            extra += g_per_h * np.clip((i - start) / max(ramp, 1), 0, 1)
        lac[i] = np.clip(lac[i - 1] + y * rate + extra, 0.1, 9.0)
    glc += rng.normal(0, 0.06, n)
    lac += rng.normal(0, 0.04, n)

    # temperature
    temp = p["temp_set"] + _ar1(rng, n, 0.05)
    for start, dur, amp in p["temp_events"]:
        w = np.clip(1 - np.abs((t - start - dur / 2) / (dur / 2 + 1e-9)) ** 4, 0, None)
        temp += amp * w

    # pH: true value (hidden, what an offline sample would read) vs online sensor
    ph_true = p["ph_set"] - 0.015 * (lac - 1.0) + _ar1(rng, n, 0.02)
    for start, ramp, amt in p["ph_true_shift"]:
        ph_true += amt * np.clip((t - start) / max(ramp, 1), 0, 1)
    ph_online = ph_true + rng.normal(0, 0.015, n)
    if p["ph_sensor_drift"] is not None:
        start, rate, cap = p["ph_sensor_drift"]
        drift = np.clip(np.abs(rate) * np.clip(t - start, 0, None), 0, abs(cap))
        ph_online += np.sign(rate) * drift

    # dissolved oxygen, agitation, pressure
    do = p["do_set"] - 0.25 * xt + _ar1(rng, n, 1.2)
    if p["do_disturb"] is not None:
        start, amp = p["do_disturb"]
        m = t >= start
        osc = amp * np.sin(2 * np.pi * (t - start) / rng.uniform(9, 17)) * m
        do += osc - 0.5 * amp * np.clip((t - start) / 48, 0, 1) * m
    agit = p["agit_set"] + _ar1(rng, n, 0.8)
    if p["agit_sag"] is not None:
        start, dur, depth = p["agit_sag"]
        agit -= depth * ((t >= start) & (t < start + dur))
    pressure = p["pressure_set"] + _ar1(rng, n, 0.006)

    return {
        "hours": t.astype(int),
        "temperature": temp, "ph_online": ph_online, "dissolved_oxygen": np.clip(do, 1, 100),
        "agitation": agit, "pressure": pressure,
        "viable_cell_density": np.clip(vcd, 0, None), "viability": v,
        "glucose": glc, "lactate": lac,
        "_ph_true": ph_true, "_ivcd": float(np.trapezoid(np.clip(vcd, 0, None), t)),
    }
