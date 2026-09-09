# Latest Plan

See `plans/plan_025.md`.

# Round 024 Decision

The project has a scoped mechanism result:

- StateFuzz discovers remote-memory stress patterns.
- Structured-repetition stress reveals a Mamba-130M vulnerability under the tested condition.
- Recurrent-state content intervention provides memory-specific behavioral evidence.

The next risk is overfitting to a single stress condition.

# Round 025 Goal

Generalize the discovery process by evaluating multiple automatically generated stress families:

- structured repetition;
- periodic patterns;
- interleaved distractors;
- semantic distractors.

The purpose is not to expand claims without evidence, but to determine whether StateFuzz identifies a broader class of memory stressors.

Requirements:

- preserve frozen evaluation protocols;
- report actual tokenizer lengths;
- keep negative controls;
- avoid universal SSM memory claims.

Next executor: codex
