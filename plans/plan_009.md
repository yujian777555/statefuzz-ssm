# Plan 009

## Round 008 Review

Codex successfully connected the framework to a real Mamba execution path.

Completed:
- real model loading and hidden-state capture path
- runner integration
- real execution artifact generation
- 77 tests passing

Important finding:
The first real-model experiment does not yet measure a valid positive memory boundary.
The tested Mamba-130M checkpoint fails at the minimum context because the current retrieval prompt is not suitable for the pretrained model.

This is a research finding, not a framework failure.

## Research Objective

StateFuzz must distinguish:

1. Model capability failure
2. Task/prompt mismatch
3. True long-context memory degradation

## Tasks

### Task 1: Build calibrated evaluation tasks

Files:
- src/statefuzz/generator/
- src/statefuzz/runner/mamba_runner.py

Create tasks where success is measurable before long-context scaling.

Verify:
- short context baseline succeeds
- increasing context isolates memory degradation

### Task 2: Improve adaptive search

Files:
- src/statefuzz/search/engine.py

Add:
- baseline calibration
- boundary search only after valid short-context success
- failure reason classification

Verify:
- avoid reporting boundary=0 from invalid prompts

### Task 3: Upgrade diagnosis artifacts

Files:
- src/statefuzz/analyzer/

Output:
- task validity
- capability boundary confidence
- failure mechanism evidence

Success:

Generate a real-model capability report that separates:
- prompt failure
- retrieval failure
- state degradation

## Success Criteria

- Real Mamba experiment has valid baseline
- Boundary estimate is meaningful
- Failure explanation contains hidden-state evidence
- results/result_round_009.json generated
