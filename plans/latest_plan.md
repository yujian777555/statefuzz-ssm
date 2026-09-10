# Latest Plan

See `plans/plan_027.md`.

# Round 026 Review

Round 026 expanded evaluation to the architecture axis.

Completed:
- model matrix support;
- cross-model stress evaluation;
- architecture-aware analysis;
- paper artifact generation.

Current conclusion:

StateFuzz evaluates the same stress families across Mamba-130M and Pythia-160M, but architecture dependence remains a candidate and requires further validation.

Important correction:
- An initial signed-margin implementation was invalid because both A/B prompts used the same direction, creating a construction artifact.
- The affected result was discarded and rerun with direction-specific signed margins.

The next step is not more synthetic scaling. It is testing whether discovered stress patterns transfer to realistic long-context workloads.

Next executor: codex
