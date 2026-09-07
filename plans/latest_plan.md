# Latest Plan

See `plans/plan_017.md`.

# Round 016 Review

Round 016 successfully added the Transformer control path, actual-token budget fitting, and filler-style control, but it **did not establish a general SSM-specific memory boundary**.

Confirmed from the actual result:

- primary models loaded successfully: `state-spaces/mamba-130m-hf` and `EleutherAI/pythia-160m`;
- Pythia exposes a 2048-token context limit and was evaluated behaviorally without treating KV cache as Mamba recurrent state;
- full pytest passes 137/137;
- under `lexically_diverse`, Mamba has no replicated transition through about 1.8k actual tokens for any predeclared value pair;
- under `structured_repetitive`, `cat/dog` remains a lower bound and `one/two` is only candidate-unreplicated;
- the only replicated Mamba transition is `structured_repetitive + red/blue`, with discovery interval `[1531,1791]` actual tokens;
- Pythia remains a lower bound through the same covered range for that condition, giving one `ssm_specific_candidate` combination;
- because the effect is not yet independently confirmed as a frozen condition and the filler result is not broadly replicated, the overall paper claim correctly remains `architecture_specificity_inconclusive`;
- the 1280 target budget produced `budget_unreachable` gaps for some seeds and cannot define a paper boundary.

# Critical Scientific Interpretation

The Round 015 `[1115,1388]` one/two result should no longer be described as a universal Mamba memory limit. Round 016 shows that the observed effective-memory transition depends on the stress condition, including filler distribution, value pair, seed set, and exact prompt construction.

The new scientifically defensible hypothesis is narrower and more useful:

> StateFuzz discovered a candidate structured-repetition stress pattern (`template_1`, red/blue) under which Mamba-130M loses counterfactual remote-memory preference before a similarly sized Pythia control.

This is a discovery result. It requires an independent confirmation round before causal state intervention or an SSM-specific paper claim.

# Round 017 Research Decision

Round 017 is an **independent stressor-confirmation + cross-round reconciliation** round.

Priority:

1. freeze the Round 016 discovery condition `structured_repetitive + red/blue`;
2. use new confirmation seeds `[21,22,23,24,25,26,27,28]`;
3. independently re-estimate the Mamba replicated transition with complete seed sets;
4. run Pythia through the entire confirmed Mamba failure region up to its context limit;
5. keep `lexically_diverse + red/blue` as a predeclared negative control;
6. fix/relax token-budget fitting only enough to avoid incomplete seed sets, while reporting exact actual counts;
7. explicitly document that effective memory is a condition-dependent response surface rather than one scalar model limit;
8. proceed to recurrent-state causal intervention only if the SSM stressor independently confirms.

Avoid:

- calling Mamba's universal boundary 1115-1388 or 1531-1791;
- hiding the failure of one/two to replicate universally;
- post-hoc selecting a new value pair after Round 017 outcomes;
- calling repetitive-filler sensitivity a general long-context failure;
- beginning causal mechanism claims before the frozen stressor confirms.

Next executor: codex
