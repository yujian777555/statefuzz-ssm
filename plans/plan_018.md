# Plan 018 - Probabilistic Failure-Risk Transition and Fresh Architecture Confirmation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the overly rigid all-seed boundary criterion with a statistically explicit seed-level failure-risk transition, and independently test whether the frozen structured-repetition stress condition produces a reproducible Mamba-vs-Pythia risk gap on fresh seeds.

**Architecture:** Preserve the validated counterfactual signed-margin task and all historical strict-boundary results. Add a separate statistical analyzer that treats each seed as a repeated trajectory over increasing actual-token budgets, estimates per-budget failure rates and per-seed first-crossing intervals, and performs a predeclared paired architecture comparison at one primary token budget. Run a third, fresh seed cohort under the frozen structured `red/blue` condition, with lexically-diverse filler as a negative control. Do not reinterpret the new risk-band metric as an exact universal model memory limit.

**Tech Stack:** Python standard library (`math`, `statistics`), PyTorch, Hugging Face Transformers, pytest, existing StateFuzz generator/runner/analyzer modules. Do not add SciPy solely for this round.

**Spec:** `plans/plan_017.md`, `results/result_round_016.json`, and `results/result_round_017.json`.

## Global Constraints

- Round 017 result is authoritative: `architecture_candidate_not_independently_replicated` under the strict all-seed zero-crossing criterion.
- Do not rewrite or delete the strict criterion; the probabilistic/risk-band analysis is an additional paper-facing view for heterogeneous seeds.
- Frozen primary condition remains `template_id=1`, values `(" red"," blue")`, `filler_style="structured_repetitive"`, `target_position=0.0`.
- Fresh Round 018 seeds are exactly `[29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44]`.
- Primary models remain exactly `state-spaces/mamba-130m-hf` and `EleutherAI/pythia-160m`.
- Primary confirmatory endpoint is target budget **1792 tokens**. This endpoint is frozen before Round 018 execution because Round 016/017 already identified it as the heterogeneous transition region.
- Secondary budgets are descriptive and must not be used to replace the primary endpoint after results are observed.
- Main failure event remains `min_signed_margin <= 0.0`.
- Every observation must retain actual tokenizer token count and target budget.
- Pythia KV cache is not Mamba recurrent state. Cross-architecture primary statistics remain behavioral.
- Lexically-diverse filler is a predeclared negative control, not an alternate task-selection search.
- No causal state-mechanism claim is permitted in this round.
- Do not call the resulting transition band a universal scalar `effective_memory_boundary` for Mamba-130M.

---

## Analysis

Round 017 rejected the Round 016 stressor under the deliberately strict rule requiring a clean replicated all-pass point immediately followed by an all-fail point. However, the raw fresh-seed trajectory is structured rather than null:

- Mamba structured red/blue at ~1661 actual tokens: 2/8 seeds fail;
- at ~1791: 6/8 fail;
- at ~1921: 8/8 fail;
- Pythia remains all-pass/lower-bound through ~1921;
- lexically-diverse Mamba remains a lower bound through ~1929;
- all boundary-relevant seed sets are complete;
- pytest passes 142/142.

The existing `candidate_unreplicated` label is therefore correct for an **all-seed step boundary**, but it discards the scientifically important possibility that effective-memory failure is a seed-dependent transition distribution. Round 018 must test that possibility prospectively rather than retrofitting a preferred claim to Round 017.

The key paper question becomes:

> Under a frozen structured-repetition stress condition and a predeclared token endpoint, is the probability of remote-memory sign loss reproducibly higher for Mamba-130M than for the matched Pythia-160M control?

---

### Task 1: Add a failure-risk statistics module without changing historical boundary semantics

**Files:**
- Create: `src/statefuzz/analyzer/failure_risk.py`
- Modify: `src/statefuzz/analyzer/__init__.py` only if the package currently exports analyzers there.
- Test: `tests/analyzer/test_failure_risk.py`

**Interfaces:**

Add:

```python
wilson_interval(failures: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]

summarize_failure_risk_curve(
    records: Iterable[Mapping[str, Any]],
    expected_seeds: Iterable[int],
) -> dict[str, Any]

summarize_seed_transition_intervals(
    records: Iterable[Mapping[str, Any]],
    expected_seeds: Iterable[int],
) -> dict[str, Any]

exact_paired_mcnemar(
    mamba_failures: Mapping[int, bool],
    transformer_failures: Mapping[int, bool],
) -> dict[str, Any]

compare_architecture_failure_risk(
    mamba_records: Iterable[Mapping[str, Any]],
    transformer_records: Iterable[Mapping[str, Any]],
    expected_seeds: Iterable[int],
    primary_target_budget: int,
) -> dict[str, Any]
```

- [ ] **Step 1: Write failing Wilson interval tests**

Test at least `0/8`, `6/8`, `8/8`, and invalid counts. Assert bounds remain in `[0,1]` and contain the observed proportion. Do not hard-code a fake confidence field elsewhere.

- [ ] **Step 2: Write failing risk-curve tests**

Given complete records at budgets `[1000,1500,1800]`, require per-budget:

```python
{
    "target_budget_tokens": ...,
    "failure_count": ...,
    "seed_count": ...,
    "failure_rate": ...,
    "wilson_95": [low, high],
    "complete_seed_set": True,
}
```

A missing seed makes that budget incomplete; incomplete budgets stay in the artifact but cannot be primary statistics.

- [ ] **Step 3: Write failing seed-transition tests**

For each seed sorted by actual tokens, compute:
- `last_pass_actual_tokens`;
- `first_fail_actual_tokens`;
- `transition_interval_actual=[last_pass, first_fail]` when both exist;
- `right_censored=True` when no failure is observed;
- `left_censored=True` if the first observed point already fails.

Also detect per-seed nonmonotonicity: fail followed by later pass. Do not silently average nonmonotonic trajectories into one boundary.

- [ ] **Step 4: Write failing exact paired McNemar tests**

Use the exact two-sided binomial test over discordant pairs only. Implement with `math.comb`; no SciPy dependency.

Return:

```python
{
    "mamba_only_fail": b,
    "transformer_only_fail": c,
    "discordant_pairs": b + c,
    "exact_two_sided_p": ...,
    "direction": "mamba_higher" | "transformer_higher" | "tie",
}
```

Verify known cases such as `b=6,c=0` and `b=8,c=0` numerically.

- [ ] **Step 5: Implement minimal functions and run tests**

```bash
python -m pytest tests/analyzer/test_failure_risk.py -q
```

Expected: PASS.

---

### Task 2: Add strict validation for prospective paired architecture comparisons

**Files:**
- Modify: `src/statefuzz/analyzer/failure_risk.py`
- Test: `tests/analyzer/test_failure_risk.py`

`compare_architecture_failure_risk()` must:

- require exactly the same expected seed set for both models at the primary target budget;
- require matched/valid candidate records for every seed;
- use the failure event `min_signed_margin <= 0`;
- keep target budget and actual token ranges for each model;
- reject a comparison if the actual-token ranges do not represent the requested budget within the experiment tolerance;
- return model-specific failure rates/Wilson intervals plus the exact paired McNemar result.

- [ ] **Step 1: Test mismatched seed sets and invalid records**

Expected: explicit invalid comparison, not silent dropping.

- [ ] **Step 2: Test a clean architecture-risk gap**

Example 16 seeds where Mamba fails 10 and Pythia fails 0 should report `direction="mamba_higher"` and the exact p-value.

- [ ] **Step 3: Test a shared-failure case**

If both models fail on the same seeds, paired discordance should be low and must not be labeled an architecture risk gap merely because raw rates are high.

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest tests/analyzer/test_failure_risk.py -q
```

---

### Task 3: Run a third fresh cohort on the frozen structured stress condition

**Files:**
- Reuse: `src/statefuzz/generator/remote_memory.py`
- Reuse: `src/statefuzz/runner/mamba_runner.py`
- Reuse: `src/statefuzz/runner/hf_causal_lm_runner.py`
- Reuse: `src/statefuzz/analyzer/memory_dependence.py`
- Write raw/aggregate output into: `results/result_round_018.json`

**Frozen experiment:**

```text
template_id = 1
values = (" red", " blue")
filler_style = structured_repetitive
target_position = 0.0
seeds = [29..44]   # 16 fresh seeds
models = [mamba-130m, pythia-160m]
tolerance_tokens = 16
```

**Target budgets:**

```python
[256, 1024, 1408, 1536, 1664, 1792, 1920]
```

**Primary statistical endpoint:** `1792` only.

- [ ] **Step 1: Short-context validity gate at 256**

For both models, all 16 seeds must have:
- matched A/B token counts;
- valid distinct single-token candidates;
- positive signed margins.

If either model fails the gate, report `prospective_control_invalid` and do not compute the primary architecture statistic.

- [ ] **Step 2: Evaluate every predeclared budget**

Store all existing behavioral fields. Mamba additionally retains direct recurrent-state comparison; Pythia uses `state_source="not_applicable"`.

Do not stop when the first seed fails. The whole curve is necessary to characterize a transition distribution.

- [ ] **Step 3: Require complete seed sets for the primary endpoint**

If any seed is unavailable at target 1792, the primary architecture test is invalid/censored. Do not substitute 1664 or 1920 after seeing results.

- [ ] **Step 4: Compute prospective primary comparison**

At target 1792 run `compare_architecture_failure_risk()`.

Report:
- Mamba failure count/rate + Wilson 95%;
- Pythia failure count/rate + Wilson 95%;
- exact paired McNemar discordant counts and p-value;
- actual token ranges for both models.

Secondary budgets are descriptive and carry no independent confirmatory p-value unless clearly labeled exploratory.

---

### Task 4: Characterize the Mamba transition as a seed distribution, not a single hard boundary

**Files:**
- Reuse: `src/statefuzz/analyzer/failure_risk.py`
- Write into: `results/result_round_018.json`

- [ ] **Step 1: Build the 16-seed failure-risk curve**

For every budget report failure count/rate/Wilson interval.

- [ ] **Step 2: Build seed-specific first-crossing intervals**

For each seed report its last pass / first fail actual-token interval or censoring.

- [ ] **Step 3: Validate monotonicity**

Report:

```json
{
  "monotonic_seed_count": ...,
  "nonmonotonic_seed_count": ...,
  "right_censored_seed_count": ...
}
```

If substantial nonmonotonicity appears, do not describe the phenomenon as a monotone memory transition without further modeling.

- [ ] **Step 4: Report transition quantiles descriptively**

For uncensored monotonic seeds, report median and interquartile range of `first_fail_actual_tokens`. If censoring is substantial, omit naive quantiles and report censoring instead. Do not fabricate survival-analysis confidence intervals.

Historical strict all-seed boundary fields remain available separately; do not overwrite them with these statistics.

---

### Task 5: Repeat the primary region with lexically-diverse filler as a negative control

**Files:**
- Reuse existing generator/runners/analyzers.
- Write into: `results/result_round_018.json`.

**Frozen negative control:** same `red/blue`, template, target position, and seeds `[29..44]`, but `filler_style="lexically_diverse"`.

Evaluate budgets:

```python
[256, 1664, 1792, 1920]
```

Run Mamba as required; Pythia at 1792 is recommended if compute permits, but Mamba negative-control validity is mandatory.

- [ ] **Step 1: Validate at 256**

- [ ] **Step 2: Compute Mamba failure rates at 1664/1792/1920**

- [ ] **Step 3: Classify stress specificity**

Return exactly one:
- `structured_repetition_risk_specific`: structured Mamba risk is elevated at the primary endpoint while diverse Mamba remains all/near-all pass over matched coverage;
- `general_remote_memory_risk`: diverse filler also shows a comparable transition;
- `negative_control_inconclusive`: invalid/incomplete coverage.

This classification is behavioral and condition-specific.

---

### Task 6: Preserve recurrent-state evidence as descriptive support only

**Files:**
- Reuse: `src/statefuzz/analyzer/hidden_state.py`
- Optionally add a small aggregation helper there only if needed; test in `tests/analyzer/test_hidden_state.py`.
- Write into: `results/result_round_018.json`.

For Mamba structured filler, aggregate recurrent-state comparison by target budget and behavioral status:
- median minimum SSM cosine similarity;
- median maximum relative L2 distance;
- same metrics split into passing versus failing seeds where both groups exist.

The scientific question is descriptive:

> Do seeds that behaviorally fail in the transition band also show reduced A/B recurrent-state discriminability?

Do not use these observational aggregates as a causal mechanism test. Do not name `state_collision`, `state_forgetting`, or `state_pollution`.

---

### Task 7: Predeclare the claim decision before interpreting Round 018

**Files:**
- Implement decision helper in `src/statefuzz/analyzer/failure_risk.py`.
- Test: `tests/analyzer/test_failure_risk.py`.

Add:

```python
classify_prospective_architecture_risk(
    primary_comparison: Mapping[str, Any],
    structured_curve: Mapping[str, Any],
    negative_control: Mapping[str, Any],
) -> str
```

Return exactly one:

1. `architecture_risk_gap_confirmed` when:
   - primary endpoint is valid and complete;
   - exact paired McNemar `p < 0.05`;
   - direction is `mamba_higher`;
   - Pythia does not exhibit a comparable shared paired failure pattern;
2. `shared_failure_risk` when both models fail comparably at the primary endpoint;
3. `architecture_risk_gap_not_confirmed` when the valid primary test does not meet criterion 1;
4. `prospective_control_invalid` when the primary endpoint cannot be validly compared.

Negative-control classification is reported separately and must constrain wording even when an architecture gap is confirmed.

Do not invent a different p-value threshold or endpoint after seeing results.

---

### Task 8: Produce Round 018 artifact and handoff

**Files:**
- Write: `results/result_round_018.json`
- Update: `status.json`

Required top-level fields:

```json
{
  "round": 18,
  "historical_strict_result": {
    "round_017_status": "architecture_candidate_not_independently_replicated"
  },
  "prospective_design": {
    "seeds": [29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44],
    "primary_target_budget_tokens": 1792,
    "failure_event": "min_signed_margin <= 0"
  },
  "structured_mamba_risk_curve": {...},
  "structured_pythia_risk_curve": {...},
  "seed_transition_distribution": {...},
  "primary_architecture_comparison": {...},
  "diverse_filler_negative_control": {...},
  "recurrent_state_descriptive_evidence": {...},
  "paper_claim_status": "...",
  "tests": {...}
}
```

`paper_claim_status` must be exactly one of:

1. `architecture_risk_gap_confirmed`;
2. `shared_failure_risk`;
3. `architecture_risk_gap_not_confirmed`;
4. `prospective_control_invalid`.

If `architecture_risk_gap_confirmed` is obtained and the negative control supports structured-repetition specificity, the next Planner round may proceed to a narrowly targeted Mamba recurrent-state intervention at the frozen structured-red/blue transition condition.

Paper language if confirmed:

> Under a predeclared structured-repetition remote-memory stress condition, Mamba-130M exhibits a reproducibly higher probability of counterfactual sign loss than a similarly sized Pythia control in the transition region; the transition is seed-distributed rather than a universal hard token boundary.

Do not generalize from one model pair to all SSMs or all Transformers.

---

## Verify

Run:

```bash
python -m pytest tests/analyzer/test_failure_risk.py -q
python -m pytest tests/analyzer/test_architecture_specificity.py -q
python -m pytest tests/analyzer/test_memory_dependence.py -q
python -m pytest tests/analyzer/test_hidden_state.py -q
python -m pytest tests/generator/test_remote_memory.py -q
python -m pytest tests/runner/test_hf_causal_lm_runner.py -q
python -m pytest tests/search/test_engine.py -q
python -m pytest -q
```

## Success

Round 018 succeeds when:

1. The historical Round 017 strict non-replication remains explicitly preserved.
2. A separate failure-risk curve is computed with statistically valid binomial intervals rather than fake confidence scores.
3. Fresh seeds `[29..44]` are used prospectively.
4. Target budget 1792 is the only predeclared primary architecture endpoint.
5. Mamba and Pythia are compared seed-paired with an exact test and complete seed sets.
6. Seed-specific transition intervals/censoring are reported instead of forcing a universal hard boundary.
7. Lexically-diverse filler is evaluated as a predeclared negative control.
8. Recurrent-state evidence remains descriptive, not causal.
9. Full pytest passes and `results/result_round_018.json` is generated.
