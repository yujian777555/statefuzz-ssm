# Latest Plan

See `plans/plan_012.md`.

# Round 011 Review

Codex produced the first replicated real-model failure region under interference. This is meaningful progress, but the result is not yet sufficient to claim an SSM state mechanism.

Observed from the actual `src/` and `results/` implementation:

- no-interference controls pass at the same nominal short context where interference cases fail;
- the failure is replicated across seeds and changes the model argmax;
- however, interference is currently appended to the prompt, so interference strength also increases the actual input length;
- the runner captures layer hidden activations, while the result explicitly records that direct SSM recurrent state was not captured;
- current mechanism labels can overclassify behavioral prediction errors as `state_pollution`.

# Research Decision

Round 012 is a causality/control round, not an expansion round.

Before claiming `state_pollution`, `state_collision`, or a memory boundary, StateFuzz must separate:

1. semantic interference from extra sequence length;
2. behavioral output failure from hidden-activation evidence;
3. hidden activations from actual recurrent SSM state;
4. 2-D interference frontier from a true context-length memory boundary.

# Round 012 Priority

Focus on:

- length-matched paired controls
- conservative mechanism labels
- tokenizer-aware evidence
- direct recurrent/cache-state capture when actually exposed by the model
- interference stress frontier with replicated real-model evidence

Avoid:

- preserving a failure claim by weakening controls
- calling short-context interference a memory boundary
- calling hidden activations direct SSM state
- defaulting unexplained prediction errors to `state_pollution`

Next executor: codex
