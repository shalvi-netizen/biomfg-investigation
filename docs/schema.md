# Data schema

Process: CHO cell-culture fed-batch, 10-14 days, hourly sampling.
Causes (5): `ph_sensor_drift`, `temperature_excursion`, `contaminated_raw_material`,
`low_viability_inoculum`, `mixing_failure`.

## `get_evidence(batch_id)` returns

```
batch_id, process, duration_days
time_series (hourly; null = missing):
  hours, temperature (C), ph_online, dissolved_oxygen (%), agitation (rpm),
  pressure (bar), viable_cell_density (1e6/mL), viability (%),
  glucose (g/L), lactate (g/L)
qc_results (end of batch; any value may be null = pending/missing):
  titer_g_l, purity_pct, endotoxin_eu_ml, bioburden_cfu_ml, mycoplasma ("pos"/"neg")
equipment_events:
  alarms: [{hour, code}]            # includes unrelated red-herring alarms
  calibration: ph_probe_days_since_cal, temp_probe_days_since_cal,
               do_probe_days_since_cal, calibration_interval_days (14)
  maintenance_log: [{item, days_ago}]
raw_materials:
  lot_id, supplier, supplier_coa {result: pass|marginal, endotoxin_eu_ml},
  lot_history {n_prior_batches, n_prior_deviations, flagged}
```

The cause and severity are never in the evidence.

## `run_test(batch_id, test_name, replicate=0)` returns

`{batch_id, test, result: "positive"|"negative", cost, turnaround_hours}`

Deterministic per (batch, test, replicate): repeating the same replicate gives the same
answer (and costs again); a new `replicate` is an independent draw.
`test_table()` returns every test's cost, turnaround and `P(positive | cause)`.

| test | cost | turnaround (h) | targets |
|---|---|---|---|
| offline_ph_check | 1 | 0.5 | pH sensor drift |
| ph_calibration_check | 2 | 1 | pH sensor drift |
| temp_logger_audit | 2 | 1 | temperature excursion |
| seed_train_review | 2 | 2 | low-viability inoculum |
| trypan_blue_count | 3 | 1 | viability (several causes) |
| impeller_do_check | 4 | 3 | mixing failure |
| raw_material_retest | 10 | 24 | contaminated raw material |
| endotoxin_assay | 12 | 24 | contaminated raw material |
| bioburden_assay | 15 | 48 | contaminated raw material |

The table gives the strong-failure probability. Weak (low-severity) failures are detected
less often, so empirical positive rates run slightly below the table.

## Hidden (evaluator only): `ground_truth(batch_id)`

`true_cause`, `severity` (0.1-1.0), `onset_hour`. Stored in `data/ground_truth.json`.
Agents must not read this file.
