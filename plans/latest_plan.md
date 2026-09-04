# Latest Plan

See `plans/plan_008.md`.

# Round 007 Review

Codex successfully added the first automated capability search prototype.

Completed:
- capability search engine (`src/statefuzz/search/engine.py`)
- boundary discovery workflow
- expanded hidden-state evidence collection
- 75 tests passing

Evidence:
- discovered capability boundary artifact generated
- failure cases include hidden-state evidence fields

Current limitation:
The current discovery is still based on deterministic synthetic probes. The next step is validating the framework with real SSM model execution and stronger diagnosis.

# Research Direction

StateFuzz goal:

**Automatically discovering and diagnosing long-context capability boundaries of State Space Models.**

The system should answer:

1. Where does effective memory fail?
2. Which input patterns trigger failure?
3. What hidden-state dynamics explain the failure?
4. How can findings guide future model improvement?

# Round 008 Priority

Focus on:

- real model-backed experiments
- adaptive boundary search
- evidence-based failure taxonomy
- paper-quality experimental artifacts

Avoid:

- generic infrastructure expansion
- synthetic-only claims without validation

Next executor: codex
