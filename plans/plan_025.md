# Plan 025 - Stress Family Generalization

## Goal

Upgrade StateFuzz from a single discovered case (`structured_repetitive + red/blue`) into a framework that can discover and characterize multiple remote-memory stress families.

The goal is not to find arbitrary failures, but to test whether the current mechanism generalizes beyond one hand-observed condition.

## Tasks

### Task 1: Add stress family abstraction

Files:
- `src/statefuzz/generator/`
- related generator tests

Add a common interface for stress families:

- structured_repetitive
- periodic_pattern
- interleaved_distractor
- semantic_distractor

Each family must report:
- construction parameters;
- actual token length;
- reproducible seed.

### Task 2: Run controlled discovery

Keep fixed:

- Mamba-130M
- existing remote-memory counterfactual protocol
- held-out seeds

Do not manually select failures.

For every stress family record:
- failure risk curve;
- first-risk region;
- memory signal decay;
- recurrent-state separation.

### Task 3: Compare specificity

Determine whether failures are:

1. general long-context degradation;
2. repetition-specific stress;
3. architecture-specific vulnerability.

Keep lexically diverse filler as negative control.

### Task 4: Prepare paper artifacts

Generate:

- stress family comparison table;
- failure-risk plots;
- mechanism summary artifact.

## Verify

Run full pytest.

Confirm every experiment has:
- frozen configuration;
- independent seeds;
- actual tokenizer counts.

## Success

Round 025 succeeds if:

1. At least two stress families can be evaluated through the same StateFuzz pipeline.
2. Results distinguish stress-specific effects from generic context degradation.
3. The paper claim can move from a single case study toward a general diagnostic framework.
4. No universal SSM memory claim is introduced without evidence.
