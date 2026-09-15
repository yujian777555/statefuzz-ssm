# Round 36: Paper Evidence Packaging

## Objective

Transform existing StateFuzz evidence into a paper-ready evidence package. No new model exploration and no protocol changes.

## Scope

Use existing validated artifacts only:

- Mamba stress analysis
- Pythia historical comparison
- Zamba2 Round34 hybrid stress surface
- Round35 stress surface matrix

## Task 1: Generate unified evidence table

Create:

`results/evidence_table_round_036.json`

Required fields:

- model
- architecture
- checkpoint_reference
- stress_family
- token_budget
- actual_tokens
- seed
- signed_margin
- normalized_margin_retention
- failure_probability

Do not invent missing historical fields. Preserve null values.

## Task 2: Generate paper figures data

Create:

`results/figure_data_round_036.json`

Required views:

1. Architecture × Stress surface matrix
2. Margin retention curves
3. Stress family comparison

The output is data only; plotting style decisions remain outside the experiment pipeline.

## Task 3: Draft evidence summary

Create:

`results/paper_evidence_summary_round_036.json`

Include:

- supported claims
- unsupported claims
- limitations

Important:

Do not claim universal failure boundaries because Round35 observed zero confirmed failure boundaries.

## Verification

Run:

```bash
python -m pytest
python scripts/round036_evidence_packaging.py --verify
```

Completion requires all generated artifacts to be reproducible from existing result files.
