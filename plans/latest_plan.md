# Latest Plan

See `plans/plan_028.md`.

# Round 027 Review

Round 027 validated transfer from synthetic stress discovery to realistic controlled workloads.

Completed:
- long document retrieval;
- code context dependency;
- agent conversation memory;
- cross-model behavioral evaluation;
- stress-family transfer analysis.

Results:
- 48 valid records;
- 164/164 tests passed;
- transfer_supported=true.

The realistic workload artifact shows:
- Mamba-130M and Pythia-160M can both be evaluated through the same workload pipeline;
- four stress families transfer into realistic workload templates;
- degradation exists, but architecture-level superiority is not established.

Scientific interpretation:

StateFuzz has moved beyond synthetic-only analysis. The current evidence supports practical relevance of discovered stress factors, while keeping claims scoped to controlled realistic workloads.

# Round 028 Goal

Move from:

Discover → Diagnose

into:

Discover → Diagnose → Mitigate

Evaluate lightweight interventions that can reduce discovered memory failures without changing model weights.

Candidate interventions:
- memory refresh/reinjection;
- state/context anchoring;
- retrieval-assisted reminder.

Success requires measurable improvement on frozen stress conditions and at least one realistic workload.

Next executor: codex
