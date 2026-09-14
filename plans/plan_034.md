# Round 34 Plan: Cross-architecture Stress Generalization (Executable Version)

## Objective

Return to the core StateFuzz contribution:

Automatically discover and diagnose memory failure modes in stateful sequence models.

Round33 showed that Hybrid cache replay intervention is invalid. Do not perform cache swap or claim Hybrid causal attribution in this round.

## Execution Environment

All experiments must run on the experiment VM where checkpoints are stored.

Before execution verify:

```bash
hostname
python --version
nvidia-smi
```

If a checkpoint path is unavailable, stop and report infrastructure failure.

## Models

### Mamba baseline

Checkpoint: MUST use the exact VM path recorded in `model_paths.json` before execution.

Required fields:
- model_id
- checkpoint_path
- sha/version if available

### Mamba2 baseline

Checkpoint: MUST use the exact VM path recorded in `model_paths.json` before execution.

### Hybrid extension

Checkpoint:

```
/202532803004/models/Zamba2-1.2B-Instruct-v2
```

Role: extension only. No internal memory attribution.

## Fixed Evaluation Protocol

Seeds:

```
69,70,71,72
```

If additional seeds are needed, add them before execution and record them.

Stress families:

1. structured_repetitive
2. periodic_interference
3. distractor_injection
4. lexically_diverse

Each stress family must map explicitly to:

```
stress_name -> generator_function -> output_file
```

## Token Budgets

Required budgets:

```
256
768
1792
3584
```

Tolerance:

```
+/- 8 tokens
```

Actual token count must be recorded. No approximate lengths accepted.

## Candidate Selection

For each stress family:

- Keep candidate generation deterministic.
- Record candidate pair IDs.
- Preserve red/blue candidate values.
- Do not manually select successful examples after observing results.

## Required Raw Evidence

Every evaluation record must contain:

```json
{
  "seed": int,
  "model_id": string,
  "stress_family": string,
  "token_budget": int,
  "actual_tokens": int,
  "candidate_a": string,
  "candidate_b": string,
  "logits": {
    "a": float,
    "b": float
  },
  "signed_margin": float
}
```

Derived metrics must be recomputable from raw logits.

## Output Files

Required:

```
results/result_round_034.json
results/stress_generalization_round_034.json
```

## Verify Commands

Before reporting completion:

```bash
python -m pytest
python scripts/round034_stress_generalization.py --verify
```

Verify:

- all seeds present
- all models present
- all stress families present
- token budgets within tolerance
- raw logits preserved
- no cache intervention used

## Analysis

Compare:

```
architecture
    -> stress fingerprint
    -> failure boundary
    -> memory weakness pattern
```

Do not only report accuracy.

## Success Criteria

Answer:

1. Does StateFuzz transfer across architectures?
2. Are failure surfaces architecture dependent?
3. Can identical stress families expose different memory weaknesses?

## Stop Conditions

If no stable pattern appears:

Report negative evidence.

Do not expand models, seeds, or stress families without a new plan.
