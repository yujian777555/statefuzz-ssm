# Plan 030 - Large Scale Validation

## Goal

Validate whether StateFuzz findings transfer beyond small models and determine whether the discovered memory stress behavior is a scale-specific artifact or a broader property of stateful sequence models.

The goal is not to claim that larger models always fail, but to measure stress-response robustness across scale.

## Tasks

### Task 1: SSM scale validation

Models (priority order):

- Mamba-130M baseline
- Mamba-370M
- Mamba-790M if resources permit
- Mamba2 variant if environment supports it

Use existing stress families only:

- structured_repetitive
- periodic_pattern
- interleaved_distractor
- semantic_distractor

Record:

- model metadata
- parameter scale
- context length
- failure risk
- memory intervention metrics

### Task 2: Transformer scale control

Add one comparable Transformer baseline if feasible.

Purpose:

- compare behavioral stress response;
- avoid claiming equivalent internal mechanisms.

Do not interpret Transformer results as state mechanism evidence.

### Task 3: Mechanism transfer validation

For selected failures run:

- correct state intervention;
- wrong state intervention;
- random state intervention.

Verify whether state-content specificity remains at larger scales.

### Task 4: Real workload subset

Run one representative realistic workload, preferably long document retrieval.

Do not expand workload count unnecessarily.

## Verify

Run:

- pytest -q

Check that every experiment records:

- checkpoint identity;
- seed;
- tokenizer statistics;
- actual context length.

## Success

Round 030 succeeds if:

1. At least one larger SSM model is evaluated.
2. Stress family behavior can be compared across scales.
3. Mechanism evidence is preserved or limitations are clearly documented.
4. Claims remain scoped and avoid unsupported universal SSM statements.
