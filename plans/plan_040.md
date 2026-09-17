# Round40: Reviewer Simulation and Final Claim Audit

## Goal
Perform a reviewer-level audit of the reconciled paper evidence before final submission preparation.

## Constraints
- No new model inference.
- No checkpoint download.
- No new stress evaluation.
- Do not expand claims beyond frozen evidence.

## Tasks

### 1. Reviewer attack simulation
Create a structured audit covering:
- novelty risk;
- causal claim validity;
- model scope limitations;
- baseline fairness;
- missing ablations;
- reproducibility concerns.

Output:
`results/reviewer_audit_round_040.json`

### 2. Claim-evidence matrix
Map every major paper claim to:
- supporting artifact;
- evidence strength;
- allowed wording;
- forbidden overstatement.

Output:
`results/claim_evidence_matrix_round_040.json`

### 3. Final paper consistency check
Verify:
- abstract matches evidence;
- figures match claims;
- limitations are explicit;
- Round18/20/21 causal evidence and Round35/36/37 transfer evidence are not conflated.

Output:
`results/paper_consistency_round_040.json`

### 4. Verification
Run:
`python -m pytest -q -o addopts=''`

## Completion condition
Return status with phase:
`reviewer_audit_complete`

Next executor: codex
