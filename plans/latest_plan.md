# Latest Plan

See `plans/plan_014.md`.

# Round 013 Review

Codex successfully proved that the new task is genuinely remote-memory dependent before long-context search.

Confirmed from the actual Round 013 result and implementation:

- template/value selection was performed on calibration seeds 7/8;
- the frozen selected task is `template_1`, values ` one` / ` two`;
- held-out seeds pass the short-context counterfactual-dependence validity gate;
- candidate targets come from explicit remote values, not model argmax;
- A/B prompts are tokenizer-length matched;
- direct Mamba recurrent/cache states are compared between counterfactual memories;
- Round 013 reports a candidate first degradation at nominal `context_tokens=512`, and already marks it as requiring replication.

# New Critical Finding

The current Round 013 boundary metric is still scientifically confounded.

`src/statefuzz/analyzer/memory_dependence.py::_metric_for_pair()` defines its primary score from cross-prompt differences in **absolute softmax probabilities**:

- `P(A | prompt_A) - P(A | prompt_B)`
- `P(B | prompt_B) - P(B | prompt_A)`

At long context, candidate probability mass can shrink because unrelated vocabulary tokens receive more mass even while A/B relative discrimination remains intact. Therefore a score drifting toward 0.5 does not necessarily mean the model has forgotten which remote value is correct.

The Round 013 artifact already shows exactly why this must be checked: at long context the raw probability contrasts become tiny while candidate logit evidence remains nontrivial. The reported nominal-512 boundary must therefore remain a candidate until recomputed with a within-prompt pairwise metric.

# Round 014 Research Decision

Round 014 is a **metric validity + replication** round.

The primary behavioral quantity must become within-prompt A/B discrimination:

- on prompt A: `logit(A) - logit(B)`;
- on prompt B: `logit(B) - logit(A)`;
- main failure event: a signed margin crosses zero;
- paper-facing observed boundary: the zero crossing is replicated across all primary held-out seeds and the previous context passes.

Pairwise conditional probabilities may be reported, but cross-prompt absolute probability differences become diagnostics only.

The frozen task must not be re-selected using held-out data.

# Round 014 Priority

Focus on:

- within-prompt candidate-pair margins;
- threshold-free replicated zero-crossing boundary semantics;
- actual tokenizer token counts at every curve point;
- expanded held-out seeds `[9, 10, 11, 12]`;
- explicit survival/retraction decision for the Round 013 nominal-512 candidate boundary;
- recurrent-state discriminability aligned with the corrected behavioral curve;
- conservative mechanism interpretation.

Avoid:

- defending the 512 boundary because it appeared in Round 013;
- using cross-prompt raw probability mass as the main memory metric;
- re-selecting template/value pairs on held-out seeds;
- reporting nominal generator context as if it were actual tokenizer length;
- naming `state_collision`, `state_forgetting`, or `state_pollution` without causal intervention evidence.

Next executor: codex
