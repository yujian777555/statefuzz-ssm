# Latest Plan

See `plans/plan_005.md`.

## Important Direction Change

The project direction has shifted from a general SSM testing pipeline to a research framework for automatically discovering SSM long-context capability boundaries.

Codex must prioritize research value over infrastructure expansion.

New research goal:

> Automatically discover where SSM long-context memory fails, diagnose the hidden-state mechanism behind failures, and generate evidence that can guide model improvement.

## Research Priorities

1. Build SSM-specific stress generators:
   - memory decay patterns
   - state collision patterns
   - state pollution patterns

2. Build model execution interface:
   - run probes on SSM models
   - capture hidden states

3. Build failure diagnosis:
   - identify failure categories
   - analyze hidden-state dynamics
   - produce interpretable artifacts

Avoid spending future rounds only on generic testing infrastructure.

The target output is a paper-quality experimental platform, not only a benchmark.

Next executor: codex
