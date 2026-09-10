# Plan 026 - Architecture and Scale Validation

## Goal

Determine whether the stress-family findings are specific to Mamba-130M or represent a broader architecture-dependent effective-memory phenomenon.

Round 025 showed that StateFuzz can evaluate multiple stress families through a unified interface:

- structured_repetitive
- periodic_pattern
- interleaved_distractor
- semantic_distractor

Round 026 expands the model axis.

## Tasks

### Task 1: Add model matrix support

Files:
- `src/statefuzz/runner/`
- model configuration modules

Support a frozen experiment manifest containing:

SSM:
- Mamba-130M
- larger Mamba checkpoint if resources allow
- Mamba2 if available

Transformer controls:
- Pythia-160M or equivalent existing control

Do not mix model sizes without reporting them.

### Task 2: Run cross-model stress evaluation

Use only predeclared stress families from Round 025.

Record:
- failure-risk curve;
- actual token budgets;
- memory signal;
- state diagnostics where applicable.

For Transformer models:
- behavior comparison only;
- do not treat KV cache as equivalent to SSM recurrent state.

### Task 3: Analyze architecture dependence

Determine whether results indicate:

1. shared long-context degradation;
2. architecture-dependent stress sensitivity;
3. SSM-specific candidate behavior.

Avoid universal claims from limited checkpoints.

### Task 4: Generate paper artifacts

Create:
- architecture comparison table;
- stress-family x model matrix;
- failure-risk plots.

## Verify

Run:

```bash
python -m pytest -q
```

Validate:
- frozen seeds;
- actual token counts;
- reproducible manifests.

## Success

Round 026 succeeds if:

1. At least two architectures are compared under identical stress conditions.
2. The paper can distinguish stress effects from model effects.
3. Any SSM-specific statement is supported only by direct evidence.
4. Results are ready for a main paper experiment section.
