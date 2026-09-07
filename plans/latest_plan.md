# Latest Plan

See `plans/plan_015.md`.

# Round 014 Review

Codex successfully corrected the Round 013 probability-mass confound and produced the strongest behavioral result so far.

Confirmed from the actual Round 014 result:

- primary metric is now within-prompt signed candidate margin;
- frozen task remains `template_id=1`, values ` one` / ` two`;
- held-out seeds `[9,10,11,12]` were evaluated without task re-selection;
- nominal 512 / actual 1115 tokens: all four seeds pass;
- nominal 1024 / actual 2220 tokens: all four seeds fail;
- nominal 2048 / actual 4443 tokens: all four seeds remain failed;
- the Round 013 nominal-512 candidate was correctly retracted;
- direct recurrent-state discriminability contracts strongly as context grows;
- full pytest passes (123/123).

# Critical Scientific Interpretation

The Round 014 artifact currently reports nominal 1024 / actual 2220 tokens as a replicated zero-crossing. This is a real replicated *tested failure point*, but it is not yet an exact memory boundary. With the current grid, the scientifically supported transition interval is between the last all-pass point (1115 actual tokens) and first all-fail point (2220 actual tokens).

A second important signal is directional asymmetry:

- at the failing contexts, the `one`-correct direction still has a positive candidate margin;
- the `two`-correct direction becomes negative;
- therefore the failure is not symmetric disappearance of all remote-memory information;
- a plausible hypothesis is that the counterfactual memory signal weakens until a lexical candidate bias dominates one direction.

At the same time, A/B direct recurrent states become progressively more similar. This supports a descriptive hypothesis of **counterfactual recurrent-state convergence associated with memory-signal decay**, but it is not yet causal evidence for a named SSM mechanism.

# Round 015 Research Decision

Round 015 is an **independent confirmation + boundary localization + bias decomposition** round.

The next executor must:

1. decompose candidate preference into remote-memory signal versus shared lexical bias;
2. fix paper-facing boundary semantics to an actual-token interval;
3. independently replicate the finding on new seeds `[13,14,15,16]`;
4. refine the pass/fail interval only after independent replication;
5. test the same template on predeclared `red/blue`, `cat/dog`, and `one/two` pairs without post-hoc selection;
6. quantify recurrent-state convergence alongside behavioral memory-signal decay.

Avoid:

- calling 2220 an exact boundary from the coarse grid;
- reusing seeds 9-12 as independent confirmation;
- interpreting one-direction failure as complete memory erasure;
- choosing a new best value pair after seeing long-context results;
- naming state collision/forgetting/pollution without intervention evidence.

Next executor: codex
