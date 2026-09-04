# Latest Plan

See `plans/plan_005.md`.

# Important Research Direction Change

The project is no longer only a generic SSM testing framework.

The research goal has changed to:

**StateFuzz: Automatically discovering and diagnosing long-context capability boundaries of State Space Models.**

The objective is not only to find failing inputs, but to build a scientific analysis platform that can answer:

1. What is the effective memory boundary of SSM models?
2. Which input patterns trigger failures?
3. What hidden-state mechanism causes the failure?
4. How can discovered weaknesses guide future model improvement?

# New Research Contributions

The implementation should prioritize:

## 1. Capability Boundary Discovery

Automatically search for limits of:

- long-range retrieval
- memory decay
- interference resistance
- state persistence

## 2. SSM-specific Failure Discovery

Focus on:

- state forgetting
- state collision
- state pollution

## 3. Mechanistic Diagnosis

Collect and analyze hidden states:

- state similarity
- state norm changes
- layer/channel abnormal behavior

The output should explain not only that a model failed, but why it failed.

# Engineering Priority Change

Avoid spending future rounds only on generic infrastructure, protocol code, or testing utilities.

Prioritize research modules:

- generator/
- runner/
- analyzer/
- capability measurement/

The final goal is a research platform for understanding and improving SSM long-context behavior.

Planner analysis:

Codex should treat this as a strategic shift from benchmark construction to scientific discovery of SSM memory mechanisms.

Next executor: codex
