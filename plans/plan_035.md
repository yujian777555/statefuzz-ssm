# Round 35: Stress Surface Analysis

## Goal

Convert accumulated StateFuzz results into a cross-architecture stress surface analysis. Do not introduce new model selection or modify frozen evaluation protocols.

## Inputs

Use existing evidence only:

- Mamba baseline results already collected
- Pythia baseline results already collected
- Round34 Zamba2 results:
  - results/result_round_034.json
  - results/hybrid_stress_round_034.json

## Objective

Build architecture × stress family × token budget robustness maps.

## Required outputs

Create:

- results/stress_surface_round_035.json
- results/result_round_035.json

Every aggregated metric must be recomputable from raw logits and signed margins.

## Analysis dimensions

Fixed stress families:

- structured_repetitive
- periodic_pattern
- interleaved_distractor
- lexically_diverse

Fixed budgets where available:

- 256
- 768
- 1792
- 3584

## Metrics

Compute:

- mean signed margin
- normalized margin retention
- failure probability
- first observed failure boundary (if available)

Do not infer failure when no failure is observed.

## Research question

Determine whether StateFuzz discovers architecture-specific stress surfaces rather than assuming monotonic context degradation.

## Verification

Run:

```bash
python -m pytest
python scripts/round035_stress_surface_analysis.py --verify
```

No checkpoint changes are allowed.