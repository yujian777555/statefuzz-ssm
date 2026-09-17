# Latest Plan

See `plans/plan_039.md`.

Round38 paper draft artifacts are present, but the first reviewer-level audit found a material narrative omission: the draft centers Round35/36 zero-failure stress-surface cohorts while omitting the stronger validated Round18 failure-discovery evidence and Round20/21 recurrent-state causal intervention evidence.

Round39 objective:
- reconcile Round18/20/21 with later Round35/36/37 evidence instead of treating them as contradictory;
- restore the narrow Mamba-130M failure-discovery and causal-diagnosis results to the paper mainline;
- retain broader stress-family, realistic-task, and Zamba2 results as transfer/scope evidence;
- keep Round28 mitigation as secondary evidence;
- replace the placeholder Related Work using the frozen literature seeds in `plans/plan_039.md`;
- redesign figures around discovery -> diagnosis -> transfer;
- run a reviewer-level claim/evidence audit;
- run no new model inference and download no checkpoints.

Important claim boundary:
- do not use `architecture-specific` as an unqualified causal claim;
- do not generalize the Mamba recurrent-state result to all SSMs;
- do not claim all evaluated cohorts have zero failures: that statement is only valid for the later packaged surface cohorts.

Follow `docs/PLANNER_EXECUTION_CONTRACT.md` and `plans/plan_039.md` exactly.

Next executor: codex
