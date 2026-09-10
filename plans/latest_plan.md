# Latest Plan

See `plans/plan_026.md`.

# Round 025 Review

Round 025 successfully moved StateFuzz beyond a single stress case.

Completed:
- stress family abstraction;
- controlled discovery;
- specificity comparison;
- paper artifact generation.

Evaluated families:
- structured_repetitive;
- periodic_pattern;
- interleaved_distractor;
- semantic_distractor.

The result contains 40 valid records, no runtime failures, and all tests pass (160/160). The updated scientific position is:

> StateFuzz can discover and compare multiple remote-memory stress families through a unified evaluation pipeline.

The next risk is model specificity.

# Round 026 Goal

Expand the model axis:

- Mamba variants where available;
- Transformer control;
- architecture-aware analysis.

Requirements:

- same stress conditions;
- actual tokenizer lengths;
- no KV-cache/recurrent-state equivalence claim;
- no universal SSM claim without evidence.

Next executor: codex
