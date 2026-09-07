# Latest Plan

See `plans/plan_016.md`.

# Round 015 Review

Round 015 independently replicated and strengthened the remote-memory finding.

Confirmed from the actual result and implementation:

- discovery seeds `[9,10,11,12]` and confirmatory seeds `[13,14,15,16]` are separated;
- the frozen primary `template_id=1`, ` one` / ` two` task was reused without re-selection;
- the primary transition is localized to `[1115, 1388]` actual tokenizer tokens: all confirmatory seeds pass at 1115 and all fail at 1388;
- `memory_signal` decays strongly with context while lexical bias remains large enough to dominate the failing direction;
- direct Mamba counterfactual recurrent-state discriminability contracts with the behavioral memory signal;
- predeclared alternative value pairs produce `multi_pair_generalization`;
- full pytest passes 127/127.

# Critical Scientific Interpretation

The current result is now a real independently replicated phenomenon in `state-spaces/mamba-130m-hf`, but it is not yet an SSM-specific result.

The most important unresolved alternative explanation is architectural/task specificity:

> a similarly sized Transformer base LM may show the same counterfactual remote-memory decay under the same completion and filler protocol.

A second unresolved confound is filler structure: the current filler is highly templated/repetitive, so the effect could be driven by repeated-distribution behavior rather than state-space architecture.

Therefore the next round must attempt to falsify SSM specificity before pursuing a named mechanism or more precise Mamba-only boundary.

# Round 016 Research Decision

Round 016 is an **architecture-specificity + filler-control falsification round**.

Priority:

1. add a behavior-only Hugging Face causal-LM runner for a predeclared Transformer control;
2. compare models at actual tokenizer token budgets rather than nominal generator lengths;
3. use new seeds `[17,18,19,20]`;
4. compare `state-spaces/mamba-130m-hf` with predeclared `EleutherAI/pythia-160m`;
5. evaluate both `structured_repetitive` and `lexically_diverse` filler styles;
6. preserve the predeclared `red/blue`, `cat/dog`, and `one/two` pairs without post-hoc selection;
7. classify the result as SSM-specific candidate, architecture differential, shared base-LM decay, task/filler-specific, or inconclusive;
8. keep Mamba recurrent-state evidence separate from Transformer behavioral evidence.

Avoid:

- calling the Round 015 result SSM-specific before a valid Transformer control covers the Mamba bracket;
- comparing nominal context values across different tokenizers;
- swapping in a different Transformer after seeing an unfavorable control result;
- interpreting Transformer KV cache as equivalent to Mamba recurrent state;
- preserving the prior paper story if architecture/filler controls falsify it.

Next executor: codex
