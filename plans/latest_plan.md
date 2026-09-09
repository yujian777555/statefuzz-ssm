# Latest Plan

See `plans/plan_021.md`.

# Round 020 Review

Round 020 successfully resolved the main ambiguity from Round 019.

Evidence:

- Fresh seeds `[45..52]` were used.
- Frozen condition remained `structured_repetitive + red/blue` on `state-spaces/mamba-130m-hf`.
- Original long-context cache failed for all 8 seeds.
- Memory-consistent value-B short state recovered all 8/8 seeds.
- Wrong-memory value-A state recovered 0/8 seeds.
- Randomized matched state recovered only 1/8 seeds.

The key result:

The recovery effect is specific to memory-consistent recurrent state, not generic cache perturbation.

Scientific status:

`recurrent_state_causal_candidate_confirmed`

The paper claim can now move beyond correlation, but remains scoped:

> Under a controlled structured-repetition remote-memory stress condition, Mamba-130M recurrent state content causally affects remote-memory behavior.

Do not generalize to all SSMs.

# Round 021 Decision

Goal: strengthen mechanism evidence.

Tasks:

1. replicate causal intervention on fresh seeds;
2. test wrong-memory and randomized controls again;
3. evaluate one additional frozen value pair (`cat/dog`) to avoid red/blue overfitting;
4. quantify memory-specific recovery gap.

Only after this round should the paper mechanism section be finalized.

Next executor: codex
