# Latest Plan

See `plans/plan_018.md`.

# Round 017 Review

Round 017 did not independently replicate the Round 016 candidate as a clean all-seed pass-to-fail boundary, so the prior `ssm_specific_candidate` must not be promoted to a confirmed hard boundary.

Confirmed from the actual Round 017 result:

- discovery seeds `[17,18,19,20]` and confirmation seeds `[21,22,23,24,25,26,27,28]` are independent;
- strict architecture confirmation is `candidate_not_replicated`;
- structured red/blue Mamba shows seed-heterogeneous failure progression rather than a single simultaneous crossing:
  - ~1661 actual tokens: 2/8 seeds fail;
  - ~1791 actual tokens: 6/8 fail;
  - ~1921 actual tokens: 8/8 fail;
- Pythia remains a lower bound through ~1921 actual tokens;
- lexically-diverse Mamba red/blue remains a lower bound through ~1929 tokens;
- all boundary-relevant seed sets are complete after the improved budget fitter;
- full pytest passes 142/142;
- paper claim status is correctly `architecture_candidate_not_independently_replicated`.

# Critical Scientific Interpretation

The Round 017 strict negative conclusion is real and must remain visible. However, the raw data also show that the all-seed boundary criterion is too coarse for heterogeneous seeds: failure risk rises from partial to majority to universal failure as distance increases, while the matched Pythia control remains all-pass over the same tested region.

Therefore the next scientific question is no longer:

> Is there one exact token at which all Mamba seeds fail?

It is:

> Does the probability of counterfactual sign loss rise reproducibly for Mamba under the frozen structured-repetition stressor, and is that failure risk significantly higher than Pythia at a predeclared transition endpoint?

This keeps the Round 017 non-replication intact while testing a more realistic seed-distributed transition model.

# Round 018 Research Decision

Round 018 is a **prospective probabilistic transition + paired architecture-risk confirmation** round.

Priority:

1. add a dedicated failure-risk analyzer with Wilson binomial intervals and exact paired McNemar testing;
2. preserve the historical strict all-seed boundary result separately;
3. use a third fresh cohort `[29..44]` (16 seeds);
4. freeze structured red/blue as the primary stress condition;
5. predeclare target budget 1792 as the only primary architecture endpoint;
6. evaluate complete Mamba/Pythia curves at `[256,1024,1408,1536,1664,1792,1920]`;
7. model each seed as its own pass/fail trajectory and report first-crossing intervals/censoring;
8. run lexically-diverse red/blue as a predeclared negative control;
9. keep Mamba recurrent-state comparisons descriptive only;
10. permit recurrent-state intervention in the next round only if the fresh paired architecture-risk gap is prospectively confirmed.

Avoid:

- rewriting Round 017 as a successful hard-boundary replication;
- selecting 1920 or another endpoint after seeing Round 018 outcomes;
- treating multiple context points as independent confirmatory tests;
- using fake confidence values instead of statistical intervals/tests;
- calling one Mamba/Pythia pair evidence about all SSMs/Transformers;
- starting causal mechanism claims before the prospective risk gap is confirmed.

Next executor: codex
