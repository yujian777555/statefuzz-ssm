# Plan 024 - Submission Refinement and Final Validation

## Goal
Convert the scoped evidence package into a submission-ready research artifact without expanding unsupported claims.

## Tasks

### Task 1: Paper narrative consistency audit
Files:
- docs/paper_claims.md
- docs/reviewer_attack.md

Verify:
- [ ] Every claim is scoped to tested conditions.
- [ ] No statement implies all SSMs share the observed behavior.
- [ ] Distinguish framework contribution from Mamba case study.

Success:
- Reviewer-facing claim table is internally consistent.

### Task 2: Final experiment manifest validation
Files:
- results/final_experiment_manifest.json
- results/paper_figures_round_022.json

Verify:
- [ ] Every figure maps to an existing experiment artifact.
- [ ] Seeds, models, prompts, and metrics are reproducible.
- [ ] Unsupported extrapolations are removed.

Success:
- Complete artifact traceability.

### Task 3: Optional robustness extension decision
Do not run automatically.

Evaluate whether an additional model (Mamba2 or larger Mamba) materially changes the paper claim.

Rules:
- If added, it is a robustness experiment only.
- Do not redefine the main contribution around it.

Success:
- Clear decision: sufficient evidence or targeted extension.

### Task 4: Prepare submission structure
Create:
- docs/paper_outline.md

Include:
- Abstract claim
- Introduction motivation
- StateFuzz method
- Discovery experiment
- Mechanism experiment
- Limitations
- Reproducibility checklist

## Verify

Run:

```bash
python -m pytest -q -o addopts=''
```

## Success

Round 024 succeeds when:

1. Evidence package remains scoped and reproducible.
2. Paper narrative clearly separates method contribution and discovered phenomenon.
3. All figures/tables have traceable experiment sources.
4. No unsupported universal SSM claims remain.
