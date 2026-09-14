# Round 34 Plan: Cross-architecture Stress Generalization

## Motivation

Round32 attempted Hybrid SSM-Attention causal path localization. Round33 showed that the current cache replay intervention protocol is invalid for Hybrid generation state manipulation. This finding should be treated as a protocol limitation, not the main research direction.

The paper focus returns to the original StateFuzz objective:

> Automatically discover and diagnose memory failure modes in stateful sequence models.

Hybrid models remain an extension case, not the central mechanism claim.

## Goal

Evaluate whether StateFuzz discovers consistent but architecture-specific memory stress surfaces across SSM and Hybrid architectures.

## Models

Primary:
- Mamba family
- Mamba2 family

Extension:
- Zamba2-1.2B-Instruct-v2

Do not claim Hybrid internal causal attribution in this round.

## Experiments

Use identical stress families:
- structured repetition
- periodic interference
- distractor injection
- lexically diverse long context

Measure:
- signed margin degradation
- failure probability
- first failure boundary
- stress sensitivity curve

## Required analysis

Compare:

architecture -> stress fingerprint -> failure boundary

Do not only report accuracy. Report the discovered failure mechanism pattern.

## Constraints

- Keep seeds explicit.
- Preserve raw logits.
- Preserve token budgets.
- Do not use invalid cache intervention.
- Separate infrastructure failure from scientific failure.

## Success criteria

A successful round should answer:

1. Does StateFuzz transfer across architectures?
2. Are failure surfaces architecture dependent?
3. Can the same stress family reveal different memory weaknesses?

## Stop condition

If no stable architecture-dependent pattern is found after controlled evaluation, report negative evidence rather than extending experiments.
