# Plan 027 - Real Workload Validation

## Goal

Move StateFuzz evidence from synthetic remote-memory tasks toward realistic long-context scenarios while preserving the discovered stress-analysis methodology.

The goal is not to replace controlled experiments, but to show practical relevance.

## Tasks

### Task 1: Add realistic memory tasks

Files:
- `src/statefuzz/tasks/`
- task tests

Add controlled real-workload inspired tasks:

1. Long document retrieval
2. Code context dependency
3. Agent-style conversation memory

Each task must define:
- hidden memory fact;
- distractor context;
- query position;
- measurable success metric.

### Task 2: Evaluate architecture differences

Reuse:
- Mamba-130M
- Pythia-160M

Do not compare internal cache mechanisms across architectures.

Only compare behavioral outcomes.

### Task 3: Connect discovered stress families

For each workload:

Measure whether StateFuzz-discovered stress patterns predict degradation.

Record:
- family type;
- failure rate;
- memory distance;
- recovery/intervention compatibility if applicable.

### Task 4: Prepare final paper evidence

Generate:

- realistic workload table;
- synthetic-to-real transfer analysis;
- limitations section artifact.

## Verify

Run:

```bash
python -m pytest -q -o addopts=''
```

Ensure:
- reproducible seeds;
- no post-hoc task selection;
- all claims remain scoped.

## Success

Round 027 succeeds if:

1. At least one realistic workload reproduces the stress-family trend.
2. StateFuzz findings transfer beyond synthetic counterfactual tasks.
3. The paper can argue practical relevance without overstating universality.
