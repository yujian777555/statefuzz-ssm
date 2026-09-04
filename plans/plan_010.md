# Plan 010

## Research Goal

Move from calibrated capability measurement to statistically meaningful SSM long-context boundary discovery.

## Tasks

### Task 1: Expand calibrated evaluation

Files:
- src/statefuzz/generator/calibrated.py
- src/statefuzz/runner/mamba_runner.py

Implement:
- multiple retrieval instances
- multiple seeds
- confidence intervals
- longer context search schedule

Goal:
Estimate degradation boundary instead of lower bound only.

### Task 2: Improve adaptive boundary search

Files:
- src/statefuzz/search/engine.py

Implement:
- exponential expansion when passing
- binary search after degradation
- avoid reporting boundary without observed failure

Output:
capability curve with valid boundary evidence.

### Task 3: Strengthen failure mechanism analysis

Files:
- src/statefuzz/analyzer/capability.py
- src/statefuzz/analyzer/hidden_state.py

Add:
- layer-wise state statistics
- state retention metrics
- failure evidence aggregation

## Verify

Run:

python -m pytest -q

Success criteria:

- real model experiments remain reproducible
- boundary estimates distinguish lower_bound and true_failure_boundary
- reports contain evidence for any claimed mechanism
- results/result_round_010.json generated
