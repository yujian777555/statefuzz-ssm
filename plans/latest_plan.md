# Latest Plan

See `plans/plan_022.md`.

# Round 021 Review

Round 021 strengthened the causal mechanism evidence.

Findings:

- Fresh seeds `[53..60]` replicated Round 020 on red/blue.
- Correct memory-consistent state recovered 8/8 failed cases.
- Wrong-memory state recovered 0/8.
- Randomized state recovered 0/8.
- An additional frozen pair (`cat/dog`) showed state-content specificity, although it was not a failure-recovery case because the original long context already succeeded.

Scientific interpretation:

The evidence supports that recurrent state content, rather than arbitrary perturbation, influences remote-memory behavior under the tested Mamba-130M stress condition.

Round 022 should consolidate evidence, generate paper artifacts, and avoid expanding claims beyond the tested scope.

Next executor: codex
