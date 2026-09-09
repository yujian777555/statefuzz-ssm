# Latest Plan

See `plans/plan_020.md`.

# Round 019 Review

Round 019 is the first causal-intervention attempt.

Findings:

- Frozen stress condition remained structured_repetitive + red/blue on Mamba-130M.
- Historical risk gap remains: Mamba failure rate 0.75 at the primary endpoint while Pythia remains 0.0, with exact McNemar p=0.00048828125.
- Direct recurrent cache override is now technically working.
- Short-context state injection recovered the failed behavior.
- However randomized-state injection also recovered behavior.

Scientific conclusion:

The result is currently `intervention_inconclusive`. It proves recurrent cache perturbation can change behavior, but does not yet prove that the recovery comes from restoring the correct remote memory content.

# Round 020 Decision

Round 020 performs causal disambiguation.

Priority:

1. keep the frozen stress condition;
2. replicate intervention with fresh seeds;
3. compare memory-consistent state, wrong-memory state, and randomized state;
4. quantify memory-specific recovery rather than any state perturbation effect;
5. only claim recurrent-state causal involvement if the intervention effect is specific.

Avoid:

- claiming state restoration from random recovery;
- expanding beyond Mamba-130M before mechanism evidence is clear;
- introducing new stress patterns.

Next executor: codex
