# Round 37: Strategic Model Expansion Before Paper Drafting

## Objective

Strengthen the StateFuzz paper evidence by adding strategically selected architecture representatives before writing the final manuscript.

Do not perform uncontrolled model expansion.

## Motivation

Current evidence:
- Mamba: pure SSM representative
- Zamba2-1.2B: Hybrid SSM-Attention representative
- Pythia: historical Attention comparison

Remaining gap:
- Modern pure Attention baseline
- Mamba2 evolution within the SSM family

## Phase A: Model Discovery

Before execution:
1. Scan available VM checkpoints.
2. Record model_id, checkpoint_path, architecture, revision.
3. Do not download or select models based on results.

Output:
results/model_inventory_round_037.json

## Phase B: Frozen Model Selection

Planner freezes:
configs/model_paths_round_037.json

Preferred targets:
1. Mamba2 checkpoint (required if available)
2. One modern Attention baseline (recommended if available)

If unavailable:
- explicitly record unavailable
- do not substitute silently

## Phase C: Evaluation

Run only existing StateFuzz stress surface protocol.

Fixed:
- seeds: 69,70,71,72
- budgets: 256,768,1792,3584
- stress families:
  - structured_repetitive
  - periodic_pattern
  - interleaved_distractor
  - lexically_diverse

Metrics:
- signed_margin
- normalized_margin_retention
- failure_probability

## Success Criteria

Success means:
- broader architecture coverage
- stronger comparative evidence

Not allowed:
- claiming model superiority
- changing protocol after observing results

## Deliverables

results/result_round_037.json
results/stress_surface_round_037.json
