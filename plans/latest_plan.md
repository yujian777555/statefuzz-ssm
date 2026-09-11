# Latest Plan

See `plans/plan_031.md`.

# Round 030 Review

Round 030 did not produce larger-model scientific results because checkpoint access was blocked.

Verified blockers:
- `state-spaces/mamba-370m-hf` download timed out;
- no local Mamba-370M/790M/Mamba2 checkpoint was found;
- ModelScope fallback was unavailable because package installation/DNS failed;
- full tests still pass 165/165.

Scientific interpretation:

Round 030 is an infrastructure/checkpoint blocker, not evidence that scale removes the phenomenon. No scale law or universal SSM conclusion is supported.

# Round 031 Decision

Instead of repeatedly retrying unavailable larger Mamba checkpoints, test a hybrid SSM-attention architecture using a frozen StateFuzz protocol.

Primary candidate:
- `Zyphra/Zamba2-1.2B` (base hybrid Mamba2 + Transformer architecture)

Round 031 priorities:

1. hard checkpoint-availability gate with no infinite retries;
2. hybrid model capability metadata;
3. short-context remote-memory validity before stress testing;
4. frozen stress-family fingerprint over structured repetition, periodic, interleaved distractor, semantic distractor, and lexically diverse control;
5. compare within-model normalized degradation, not raw quality;
6. separately inspect SSM recurrent state and attention-cache observability;
7. optional single realistic-workload spot check;
8. no architecture-causal claim because model scale/training data are unmatched.

If the hybrid checkpoint is unavailable, record the blocker cleanly rather than fabricating or substituting a post-hoc model.

Next executor: codex
