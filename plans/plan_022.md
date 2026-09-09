# Plan 022 - Consolidate Mechanism Evidence and Prepare Paper Artifacts

## Goal
Move from causal candidate evidence to a publication-ready mechanism package without overclaiming.

## Tasks

### Task 1: Freeze scientific claims
Files:
- results/result_round_021.json
- docs/paper_claims.md (new)

Write the exact allowed claim:

> Under structured-repetition remote-memory stress conditions, Mamba-130M recurrent state content causally influences remote-memory behavior.

Do not claim:
- all SSMs;
- universal memory length;
- all long-context failures.

### Task 2: Build final experiment summary
Files:
- src/statefuzz/analyzer/failure_risk.py
- tests/analyzer/test_failure_risk.py

Add helpers:

```python
summarize_mechanism_evidence(results)
```

Output:
- behavior failure risk;
- memory-consistent recovery;
- wrong-memory rejection;
- random perturbation rejection;
- cross-pair support.

### Task 3: Add final robustness pair if feasible
Frozen protocol:
- structured_repetitive
- Mamba-130M
- fresh seeds

Evaluate one additional pair only if required for paper completeness.
Do not search pairs.

### Task 4: Generate paper figures data
Files:
- results/paper_figures_round_022.json

Required figures:
1. failure-risk curve;
2. intervention recovery bar chart;
3. memory-specificity gap;
4. state intervention schematic.

### Task 5: Full verification
Run:

```bash
python -m pytest -q -o addopts=''
```

## Verify
Success requires:

- mechanism evidence remains specific to tested condition;
- no overclaiming universal SSM memory limit;
- all tests pass;
- paper artifact JSON generated.

## Success
Round 022 succeeds when the repository contains a stable experiment package suitable for drafting Method, Experiments, and Mechanism sections.

Next executor: codex
