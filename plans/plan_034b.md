# Round34-B Plan: Frozen Cross-Architecture Evaluation

## Objective

Execute the frozen evaluation phase only after Round34-A model discovery.

Discovered checkpoint:
- Model: Zyphra/Zamba2-1.2B-Instruct-v2
- VM path: /202532803004/models/Zamba2-1.2B-Instruct-v2

## Rules

1. Do not replace checkpoint.
2. Do not download new models.
3. Do not tune model selection after observing results.
4. All experiments must use fixed configuration recorded before execution.

## Required Planner Freeze Before Execution

Create:

configs/model_paths.json

containing:
- model path
- model architecture
- parameter count
- revision field (null if unavailable)

## Codex Tasks

Only after model_paths.json exists:

1. Load the frozen checkpoint.
2. Run validation inference.
3. Execute predefined stress evaluation.
4. Save raw logits and intermediate evidence.

## Required Outputs

results/result_round_034.json
results/hybrid_stress_round_034.json

## Verification

Run the repository test suite and attach pass count.

No experiment result should be accepted without raw evidence files.
