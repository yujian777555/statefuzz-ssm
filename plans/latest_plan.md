# Latest Plan

See `plans/plan_010.md`.

# Round 009 Review

Codex successfully fixed the most important scientific issue: separating invalid task failure from actual capability degradation.

Completed:
- calibrated short-context baseline
- real-model next-token evaluation
- valid capability reporting
- confidence-aware lower-bound estimation
- 82 tests passing

Evidence:
- baseline validity confirmed
- target token remained stable through tested 512 token context
- report correctly avoided claiming a false zero boundary

# Research Direction

StateFuzz goal:

**Automatically discovering and diagnosing long-context capability boundaries of State Space Models.**

The system must produce scientifically valid evidence:

1. where capability degrades
2. which stress pattern causes degradation
3. what hidden-state dynamics explain it
4. how findings can guide model improvement

# Round 010 Priority

Focus on:

- larger calibrated experiments
- true failure boundary search
- confidence intervals
- mechanism-level evidence

Avoid:

- claiming boundaries without observed degradation
- single-example conclusions
- infrastructure-only expansion

Next executor: codex
