# Plan 023 - Paper Validation and Reviewer Attack Round

## Goal
Transform the scoped mechanism package from Round 022 into a submission-ready evidence package by attacking the strongest remaining reviewer objections.

## Current Evidence
Round 022 establishes a scoped claim:

> Under structured-repetition remote-memory stress conditions, Mamba-130M recurrent state content causally influences remote-memory behavior.

Supported evidence:
- red/blue: failure risk 1.0, correct memory recovery 1.0, wrong-memory rejection 1.0, random rejection 1.0.
- cat/dog: state-content specificity, but not a failure-recovery case because original long context succeeds.
- 156/156 tests passed.

## Remaining Risks
1. Only Mamba-130M has causal intervention evidence.
2. The stress condition may be model-specific rather than architecture-level.
3. Need stronger separation between discovery framework contribution and single-model finding.
4. Need reproducible paper artifacts and final experiment tables.

## Tasks

### Task 1: Create reviewer checklist artifact
Files:
- Create: docs/reviewer_attack.md

Include:
- What claim is supported.
- What claim is intentionally not made.
- Possible reviewer objections.
- Evidence answering each objection.

### Task 2: Add model-scope robustness plan
Files:
- Update docs/paper_claims.md

Do not run uncontrolled model expansion. Define optional future validation:
- Mamba2 if available.
- one additional SSM checkpoint.

The paper claim must remain scoped until those experiments exist.

### Task 3: Validate figure/table generation
Files:
- Validate results/paper_figures_round_022.json
- Add tests if needed.

Required figures:
1. StateFuzz pipeline.
2. Failure-risk transition.
3. Correct/wrong/random recurrent-state intervention.
4. Mechanism summary.

### Task 4: Prepare final experiment manifest
Create:
- results/final_experiment_manifest.json

Include:
- model names.
- seeds.
- frozen conditions.
- metrics.
- excluded claims.

## Verify
Run:
python -m pytest -q -o addopts=''

## Success
Round 023 succeeds when:
- reviewer limitations are explicitly documented;
- paper artifacts are reproducible;
- no unsupported universal SSM claim remains;
- final experiment manifest exists;
- all tests pass.
