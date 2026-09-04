# Latest Plan

See `plans/plan_009.md`.

# Round 008 Review

Codex successfully connected StateFuzz to real Mamba execution.

Completed:
- real model loading
- hidden-state capture
- runner integration
- real execution artifacts
- 77 tests passing

Important research finding:
The first real-model experiment failed at the minimum tested context. This does not prove a memory boundary of zero; it indicates the current evaluation task is not calibrated for the pretrained checkpoint.

The system must separate:
- task/prompt failure
- model capability failure
- true long-context degradation

# Research Direction

StateFuzz remains:

**Automatically discovering and diagnosing long-context capability boundaries of State Space Models.**

The next priority is not more infrastructure. It is producing scientifically valid measurements.

# Round 009 Priority

Focus on:

- calibrated short-context baselines
- valid real-model capability curves
- confidence-aware boundary estimation
- mechanism-based failure reports

Avoid:

- reporting invalid zero boundaries
- synthetic-only conclusions

Next executor: codex
