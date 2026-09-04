# Plan 011

## Round 010 Review

Codex successfully improved the scientific evaluation protocol.

Completed:
- multi-seed calibrated experiments
- confidence intervals
- longer context capability curve
- valid lower-bound capability reporting
- 88 tests passing

Current finding:
The current Mamba-130M experiment does not yet observe degradation up to 2048 tokens. The result is scientifically valid, but it is only a lower bound, not the final memory boundary.

## Research Goal

StateFuzz should discover real long-context capability boundaries of SSMs, not only verify that short contexts work.

The next stage should prioritize discovering failure regimes.

## Tasks

### Task 1: Expand stress task design

Implement stronger SSM-specific probes:

- long-range retrieval
- interference injection
- state collision construction
- state pollution recovery

Files:
- src/statefuzz/generator/
- src/statefuzz/search/engine.py

Goal:
Create tasks where degradation can actually emerge.

### Task 2: Extend real model evaluation

Files:
- src/statefuzz/runner/mamba_runner.py
- src/statefuzz/analyzer/capability.py

Run:
- larger context windows
- multiple task families
- multiple seeds

Goal:
Generate true capability curves rather than lower bounds only.

### Task 3: Improve mechanism diagnosis

Files:
- src/statefuzz/analyzer/hidden_state.py
- src/statefuzz/analyzer/failure_classifier.py

Add:
- layer-wise state similarity
- temporal retention analysis
- failure evidence aggregation

Goal:
Explain why degradation occurs.

## Verify

Run:

python -m pytest -q

Generate:

results/result_round_011.json

## Success

Round 011 succeeds when:

1. At least one calibrated task can reveal degradation.
2. Capability boundary search identifies a non-trivial failure region.
3. Report contains mechanism-level evidence.
