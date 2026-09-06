# Latest Plan

See `plans/plan_013.md`.

# Round 012 Review

Codex successfully completed the causal-control round.

Confirmed from the actual implementation and result:
- replacement-based control/stress pairs are tokenizer-length matched;
- 48/48 evaluated instances had no token-length mismatch;
- the Round 011 argmax failure disappears under the matched control;
- therefore the Round 011 failure must be rejected as an appended-length/lexical confound;
- direct Mamba recurrent/cache state is now captured from 24 recurrent layers through the real Transformers cache path;
- no SSM mechanism claim is preserved when controlled behavioral failure is absent.

This is strong scientific progress because StateFuzz falsified its own previous candidate failure instead of preserving a weak claim.

# New Critical Finding

The current calibrated next-token task is still not a valid long-range-memory task.

In the real Round 012 run, the target token is the local continuation token ` the`, selected from the control prompt behavior near the suffix `The next symbol is`.

That target can be predicted from the local suffix without reading a remote memory value. Therefore, even perfect stability at 32k tokens would not establish long-range memory retention.

The next scientific requirement is **counterfactual remote-memory dependence**:

> If only a remote stored value changes while prompt length, local suffix, and structure stay matched, the model's target preference must change accordingly at short context.

Only a task that passes this held-out short-context test may be used for long-context boundary discovery.

# Round 013 Priority

Focus on:
- counterfactual remote-memory task pairs;
- tokenizer-aware candidate target scoring;
- an explicit memory-dependence validity metric;
- calibration-seed versus held-out-seed separation;
- true remote-memory boundary search only after validity passes;
- direct recurrent-state comparison between counterfactual memory values.

Avoid:
- choosing the model's local argmax as the memory target;
- treating local continuation robustness as memory capacity;
- searching long contexts before proving short-context remote dependence;
- claiming a state mechanism without aligned behavioral evidence.

Next executor: codex
