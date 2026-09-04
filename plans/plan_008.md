# Round 008 Plan

## Research Objective

Continue evolving StateFuzz from boundary search prototype into a scientific discovery framework for SSM long-context capability analysis.

## Tasks

### 1. Replace synthetic-only discovery with model-backed experiments

Files:
- src/statefuzz/runner/mamba_runner.py
- src/statefuzz/search/engine.py

Implement:
- real model execution path
- hidden state collection during inference
- reproducible experiment configuration

Goal:
Move from deterministic synthetic probes toward evidence from actual SSM models.

### 2. Improve boundary discovery algorithm

Files:
- src/statefuzz/search/engine.py
- src/statefuzz/analyzer/capability.py

Implement:
- adaptive context length search
- boundary refinement around failure points
- separation between observed failure and estimated capability boundary

Output:
- effective memory boundary
- confidence/evidence records

### 3. Upgrade failure diagnosis

Files:
- src/statefuzz/analyzer/failure_classifier.py
- src/statefuzz/analyzer/hidden_state.py

Implement evidence-based classification:
- state collision
- state forgetting
- state pollution

Avoid relying only on output mismatch.

### 4. Generate paper-quality artifacts

Files:
- results/result_round_008.json

Required artifacts:
- capability curve data
- failure examples
- hidden-state evidence
- reproducible configurations

## Verify

Run:

python -m pytest -q

Success criteria:
- all tests pass
- at least one real model-backed discovery experiment works
- result artifact contains interpretable capability boundary evidence

## Success

Round 008 succeeds when StateFuzz can demonstrate automated discovery of an SSM capability boundary rather than only measuring supplied observations.
