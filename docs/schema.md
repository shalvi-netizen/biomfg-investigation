# Data schema

## Time series (hourly, per batch)
temperature, ph_online, dissolved_oxygen, agitation, pressure,
viable_cell_density, viability, glucose, lactate

## QC results
titer, purity, endotoxin, bioburden, mycoplasma

## Equipment events
alarms, calibration_dates, maintenance_log

## Raw materials
lot_id, supplier_coa, lot_history

## Hidden (evaluator only)
true_cause, severity

## Functions
get_evidence(batch_id) -> everything except the cause
run_test(batch_id, test_name) -> (noisy_result, cost)
