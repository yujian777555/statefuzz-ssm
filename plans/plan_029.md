# Plan 029 - Paper Consolidation and Final Validation

## Goal

Convert the completed StateFuzz pipeline into a submission-ready evidence package.

Round 028 established that mitigation is possible but not universal:
- context_anchor improves code context dependency in tested settings;
- no single mitigation dominates all workloads.

The next step is not uncontrolled experimentation, but final validation and paper preparation.

## Tasks

### Task 1: Freeze final scientific claims

Files:
- `docs/paper_claims.md`
- `docs/reviewer_attack.md`

Update claims around:
- discovery of stress families;
- diagnosis through recurrent-state analysis;
- scoped mitigation evidence.

Avoid:
- universal SSM memory claims;
- universal mitigation claims;
- architecture ranking claims without statistical support.

### Task 2: Build final experiment manifest

Files:
- `results/`
- `docs/`

Create a single manifest containing:
- models;
- tasks;
- seeds;
- stress families;
- metrics;
- artifacts;
- limitations.

### Task 3: Generate paper-ready tables

Prepare:

Table 1:
StateFuzz pipeline comparison.

Table 2:
Stress family transfer results.

Table 3:
Mechanism intervention results.

Table 4:
Mitigation results and limitations.

### Task 4: Final reproducibility check

Verify:
- pytest passes;
- all result files referenced exist;
- no invalid claims remain in documentation.

## Verify

Run complete test suite and validate all paper artifacts.

## Success

Round 029 succeeds when:

1. The repository contains a coherent submission package.
2. Every major claim maps to an experiment artifact.
3. Limitations are explicitly documented.
4. The paper can be written without additional core experiments.
