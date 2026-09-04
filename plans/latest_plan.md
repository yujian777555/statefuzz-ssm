# Latest Plan

See `plans/plan_011.md`.

# Round 010 Review

Codex successfully improved the scientific evaluation protocol.

Completed:
- multi-seed calibrated experiments
- confidence intervals
- longer context capability curve
- valid lower-bound capability reporting
- 88 tests passing

Important research finding:
The current experiment establishes a reproducible lower bound of 2048 tokens, but no true degradation boundary has been observed yet.

# Research Direction

StateFuzz goal:

**Automatically discovering and diagnosing long-context capability boundaries of State Space Models.**

The next phase must move from measuring stable capability to discovering actual failure regimes.

# Round 011 Priority

Focus on:

- stronger SSM-specific stress tasks
- real degradation discovery
- mechanism-level diagnosis
- capability boundary evidence

Avoid:

- reporting lower bounds as exact boundaries
- weak synthetic failures
- infrastructure-only expansion

Next executor: codex
