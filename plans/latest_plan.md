# Latest Plan

See `plans/plan_006.md`.

# Round 005 Review

Codex successfully moved the project into the research prototype stage.

Completed:
- generator modules for memory decay, collision, and pollution patterns
- runner abstraction for model execution
- analyzer modules for hidden-state and failure classification
- 62 tests passing

However, the current implementation is still a prototype framework.

# Research Direction

StateFuzz is now defined as:

**Automatically discovering and diagnosing long-context capability boundaries of State Space Models.**

The goal is not only detecting failures, but discovering:

1. Effective memory boundaries.
2. Input patterns that trigger degradation.
3. Hidden-state mechanisms causing failure.
4. Evidence that can guide future model improvement.

# Round 006 Priority

Focus on scientific discovery rather than infrastructure:

- parameterized stress generation
- capability boundary measurement
- evidence-based hidden-state diagnosis

Avoid:
- adding only more generic tests
- expanding protocol infrastructure without research value

Next executor: codex
