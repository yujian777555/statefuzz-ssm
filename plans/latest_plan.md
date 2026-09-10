# Latest Plan

See `plans/plan_027.md`.

# Round 026 Review

Round 026 expanded StateFuzz along the architecture axis.

Completed:
- model matrix support;
- cross-model stress evaluation;
- architecture-aware analysis;
- paper artifact generation.

Current evidence:

StateFuzz evaluates the same stress families across Mamba-130M and Pythia-160M. Architecture dependence remains a candidate finding and is not yet a confirmed universal property.

Important correction:

- The first signed-margin implementation contained a construction artifact because both A/B prompts used the same direction.
- Invalid results were discarded.
- Experiments were rerun using direction-specific signed margins.

Round 027 moves from controlled synthetic stress discovery toward realistic long-context workloads.

Goal:

Determine whether discovered stress patterns transfer to practical memory scenarios.

Workloads:

- long document retrieval;
- code context dependency;
- agent conversation memory.

Requirements:

- preserve frozen evaluation protocols;
- report actual token lengths;
- avoid KV-cache/recurrent-state equivalence claims;
- keep all architecture claims scoped.

Next executor: codex
