# Plan 014 - Replace Probability-Mass Boundary with Replicated Pairwise Memory Boundary

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-evaluate the Round 013 candidate memory boundary with a metric that measures within-prompt A/B memory discrimination, then report only a replicated zero-crossing boundary or a lower bound.

**Architecture:** Keep the frozen Round 013 remote-memory task (`template_1`, values ` one` / ` two`) and held-out evaluation protocol. Replace the current cross-prompt raw-probability boundary metric with within-prompt candidate-pair margins/conditional probabilities, propagate exact tokenizer lengths through the search artifact, and align recurrent-state evidence with the corrected behavioral curve.

**Tech Stack:** Python, PyTorch, Transformers Mamba, pytest, existing StateFuzz generator/runner/search/analyzer modules.

**Spec:** `plans/plan_013.md` plus the actual Round 013 result in `results/result_round_013.json`.

## Global Constraints

- Do not re-select the template or candidate pair using held-out seeds.
- Frozen task for this round: `template_id=1`, `value_a=" one"`, `value_b=" two"`.
- Calibration seeds remain `[7, 8]`; primary held-out seeds are `[9, 10, 11, 12]`.
- A/B prompts must remain tokenizer-length matched.
- Candidate targets must come from the remote values, never the model argmax.
- Do not preserve the Round 013 `memory_boundary=512` claim unless it survives the corrected metric and replication gate.
- Direct recurrent state is mechanism evidence only; it does not by itself justify `state_collision`, `state_forgetting`, or `state_pollution`.

---

## Analysis

Round 013 achieved the most important validity milestone so far:

- the selected remote-memory task was chosen only on calibration seeds 7/8;
- held-out seeds passed short-context counterfactual dependence;
- A/B prompts were tokenizer-length matched;
- direct Mamba recurrent/cache state was captured;
- the result reported a candidate boundary at nominal `context_tokens=512` and correctly marked it as requiring replication.

However, the current boundary metric is not yet suitable for the paper claim.

`src/statefuzz/analyzer/memory_dependence.py::_metric_for_pair()` currently computes:

```python
direction_a = P(A | prompt_A) - P(A | prompt_B)
direction_b = P(B | prompt_B) - P(B | prompt_A)
score = 0.5 + 0.25 * (direction_a + direction_b)
```

This compares **absolute softmax probability mass across different prompts**. As context grows, probability mass can move to unrelated vocabulary tokens even when the relative preference between the two memory candidates remains intact. The Round 013 long-context records already show this warning pattern: probability contrasts collapse toward zero while candidate logit-related evidence remains nontrivial.

Therefore Round 014 is a metric-validity and replication round. The 512 boundary is a candidate, not a result to defend.

---

### Task 1: Add within-prompt pairwise memory metrics

**Files:**
- Modify: `src/statefuzz/analyzer/memory_dependence.py`
- Test: `tests/analyzer/test_memory_dependence.py`

**Interfaces:**
- Consumes: the existing candidate entries produced by `MambaRunner.score_candidate_tokens()`.
- Produces:
  - `compute_pairwise_memory_metrics(record: Mapping[str, Any]) -> dict[str, Any]`
  - existing `compute_counterfactual_memory_score()` updated to expose the new fields without deleting raw diagnostic fields.

- [ ] **Step 1: Write failing tests for within-prompt margins**

Add tests with synthetic logits where all vocabulary probability mass may change across prompts but A/B relative logits remain identical.

Required assertions:

```python
metrics = compute_pairwise_memory_metrics(record)
assert metrics["direction_a_margin"] == pytest.approx(logit_a_on_a - logit_b_on_a)
assert metrics["direction_b_margin"] == pytest.approx(logit_b_on_b - logit_a_on_b)
assert metrics["min_signed_margin"] == pytest.approx(
    min(metrics["direction_a_margin"], metrics["direction_b_margin"])
)
assert metrics["pairwise_preference_valid"] is True
```

Add a second test where absolute candidate probabilities shrink by orders of magnitude but the within-prompt A/B logit margins stay positive; `pairwise_preference_valid` must remain true.

- [ ] **Step 2: Run the focused tests and confirm failure**

```bash
python -m pytest tests/analyzer/test_memory_dependence.py -q
```

Expected before implementation: FAIL because `compute_pairwise_memory_metrics` does not exist.

- [ ] **Step 3: Implement the metric**

For prompt A:

```python
margin_a = logit(A | prompt_A) - logit(B | prompt_A)
pair_prob_a = sigmoid(margin_a)
```

For prompt B:

```python
margin_b = logit(B | prompt_B) - logit(A | prompt_B)
pair_prob_b = sigmoid(margin_b)
```

Return at minimum:

```python
{
    "direction_a_margin": margin_a,
    "direction_b_margin": margin_b,
    "direction_a_pair_probability": pair_prob_a,
    "direction_b_pair_probability": pair_prob_b,
    "pairwise_memory_score": (pair_prob_a + pair_prob_b) / 2.0,
    "min_signed_margin": min(margin_a, margin_b),
    "pairwise_preference_valid": margin_a > 0.0 and margin_b > 0.0,
}
```

Do **not** compare raw logits for the same token across different prompts as the primary metric. Keep the old raw probability contrast fields only as diagnostics and name them clearly as legacy/absolute-probability evidence.

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest tests/analyzer/test_memory_dependence.py -q
```

Expected: PASS.

---

### Task 2: Add threshold-free replicated behavioral boundary semantics

**Files:**
- Modify: `src/statefuzz/search/engine.py`
- Test: `tests/search/test_engine.py`

**Interfaces:**
- Consumes evaluator records containing `min_signed_margin`, `pairwise_memory_score`, `actual_input_tokens`, and held-out seed.
- Produces:
  - `search_replicated_remote_memory_boundary(...) -> dict[str, Any]`

- [ ] **Step 1: Write failing search tests**

Cover exactly these cases:

1. all seeds have positive signed margin at every context -> `boundary_kind="lower_bound"`;
2. one seed fails but others pass -> `boundary_kind="candidate_unreplicated"`;
3. all held-out seeds pass at context N and all fail at context M>N -> `boundary_kind="replicated_zero_crossing"` and boundary=M;
4. tokenizer counts differ -> case is invalid and cannot create a boundary.

- [ ] **Step 2: Implement the replication gate**

Behavioral failure for a seed is:

```python
min_signed_margin <= 0.0
```

A paper-eligible observed boundary requires **all primary held-out seeds** to fail at the same tested context while the immediately preceding tested context has all seeds passing.

Return:

```python
{
    "boundary_kind": "replicated_zero_crossing" | "candidate_unreplicated" | "lower_bound" | "invalid_task",
    "nominal_boundary_context": int | None,
    "actual_boundary_token_range": [min_tokens, max_tokens] | None,
    "previous_passing_context": int | None,
    "cases": [...],
}
```

The previous probability threshold `0.55` must not define the primary boundary in this function.

- [ ] **Step 3: Run focused search tests**

```bash
python -m pytest tests/search/test_engine.py -q
```

Expected: PASS.

---

### Task 3: Propagate actual tokenizer length into every remote-memory observation

**Files:**
- Modify: `src/statefuzz/runner/mamba_runner.py`
- Modify: `src/statefuzz/search/engine.py`
- Test: `tests/runner/test_calibrated_runner.py`
- Test: `tests/search/test_engine.py`

**Interfaces:**
- Existing `score_remote_memory_pair(...)` must expose a scalar `actual_input_tokens` when A/B counts match.

- [ ] **Step 1: Add tests**

For matched A/B prompts:

```python
result = runner.score_remote_memory_pair(pair)
assert result["matched"] is True
assert result["actual_input_tokens"] == result["prompt_token_counts"][0]
```

For a mismatched pair, `actual_input_tokens` must be `None` and the search case must be invalid.

- [ ] **Step 2: Implement and propagate the field**

Do not report only nominal generator `context_tokens`. Every curve point and boundary artifact must include actual tokenizer counts.

- [ ] **Step 3: Run focused tests**

```bash
python -m pytest tests/runner/test_calibrated_runner.py tests/search/test_engine.py -q
```

Expected: PASS.

---

### Task 4: Re-run the frozen Round 013 task on four held-out seeds

**Files:**
- Modify only the existing experiment/execution path needed to generate the result; do not create a second framework.
- Write: `results/result_round_014.json`

**Frozen experiment:**

```text
template_id = 1
values = (" one", " two")
calibration seeds = [7, 8]
held-out seeds = [9, 10, 11, 12]
target_position = 0.0
primary nominal contexts = [64, 128, 256, 512, 1024, 2048]
```

- [ ] **Step 1: Verify the short-context task remains valid on all four held-out seeds**

At nominal context 64, require for every held-out seed:

```python
direction_a_margin > 0
direction_b_margin > 0
A/B tokenizer counts match
```

If this fails, report `task_validity=false` and do not search a boundary.

- [ ] **Step 2: Evaluate all six primary context points**

For every seed/context record store:

- actual tokenizer count;
- both within-prompt signed margins;
- both pairwise conditional probabilities;
- `pairwise_memory_score`;
- `min_signed_margin`;
- old absolute-probability contrasts as diagnostics only;
- direct recurrent-state summary/comparison.

- [ ] **Step 3: Re-evaluate the Round 013 candidate boundary**

`results/result_round_014.json` must contain:

```json
{
  "round_013_candidate_boundary_nominal": 512,
  "round_013_boundary_survives_corrected_metric": true,
  "corrected_boundary": {...}
}
```

Set the boolean from data. Do not force it to true.

If no replicated zero-crossing occurs through 2048, report a lower bound rather than preserving 512.

- [ ] **Step 4: Optional bounded extension only if no boundary is found**

If all four held-out seeds still have positive `min_signed_margin` through nominal 2048 and the run is operationally stable, evaluate nominal context 4096 once with all four held-out seeds. Do not add 8192 in this round. If resource/runtime prevents 4096, record `censored_by_resource_limit=true` and keep the 2048 lower bound.

---

### Task 5: Align recurrent-state discriminability with the corrected behavioral curve

**Files:**
- Modify: `src/statefuzz/analyzer/hidden_state.py`
- Test: `tests/analyzer/test_hidden_state.py`

**Interfaces:**
- Extend `compare_recurrent_states(reference, counterfactual)` with normalized L2 distance per layer in addition to cosine similarity.

- [ ] **Step 1: Add tests for normalized state distance**

Per layer compute:

```python
relative_l2_distance = ||A - B||_2 / max((||A||_2 + ||B||_2) / 2, eps)
```

Test identical states -> 0.0, scaled/different states -> positive finite value.

- [ ] **Step 2: Add aggregate fields**

For `ssm_states` and `conv_states`, return:

- min/median/max cosine similarity;
- min/median/max relative L2 distance;
- strongest divergent layer by cosine;
- strongest divergent layer by relative L2 distance.

- [ ] **Step 3: Add behavioral/mechanistic alignment artifact**

For every context aggregate across held-out seeds:

```json
{
  "behavior": {
    "median_min_signed_margin": ...,
    "failure_seed_count": ...
  },
  "state": {
    "median_min_ssm_cosine": ...,
    "median_max_ssm_relative_l2": ...
  }
}
```

Do not infer a named mechanism solely from correlation.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/analyzer/test_hidden_state.py -q
```

Expected: PASS.

---

### Task 6: Add threshold sensitivity as secondary analysis, not primary boundary logic

**Files:**
- Modify: `src/statefuzz/analyzer/memory_dependence.py`
- Write fields into: `results/result_round_014.json`

- [ ] **Step 1: Compute pairwise-score sensitivity**

For descriptive comparison only, report the first nominal context where the mean pairwise score falls below each of:

```python
[0.60, 0.70, 0.80, 0.90]
```

Call this `pairwise_score_sensitivity`, not `memory_boundary`.

- [ ] **Step 2: Preserve the primary zero-crossing rule**

The main paper-facing boundary remains the replicated sign-loss criterion on within-prompt candidate margins. Threshold sensitivity is supplementary evidence.

---

### Task 7: Produce the Round 014 scientific artifact and handoff

**Files:**
- Write: `results/result_round_014.json`
- Update: `status.json`

The result must include:

- changed files;
- full pytest result;
- frozen task identity and seed split;
- actual tokenizer counts;
- corrected pairwise metric definition/version;
- six-point held-out curve (plus optional 4096 point under the bounded rule above);
- Round 013 candidate boundary survival/retraction decision;
- replicated boundary/lower-bound classification;
- recurrent-state discriminability curve;
- threshold sensitivity table;
- conservative scientific interpretation.

Scientific interpretation must explicitly choose one of:

1. `replicated_zero_crossing_observed`;
2. `round_013_boundary_retracted_corrected_metric_lower_bound`;
3. `candidate_failure_not_replicated`;
4. `task_invalid_on_expanded_heldout_seeds`.

Do not use a named SSM failure mechanism unless a later intervention/ablation establishes causality.

---

## Verify

Run:

```bash
python -m pytest -q
```

Required focused tests:

```bash
python -m pytest tests/analyzer/test_memory_dependence.py -q
python -m pytest tests/analyzer/test_hidden_state.py -q
python -m pytest tests/runner/test_calibrated_runner.py -q
python -m pytest tests/search/test_engine.py -q
```

## Success

Round 014 succeeds when all of the following are true:

1. Boundary logic no longer uses cross-prompt absolute softmax probability differences as the primary memory metric.
2. Within-prompt A/B signed margins and pairwise conditional probabilities are recorded for every held-out observation.
3. The frozen Round 013 task is tested on held-out seeds `[9, 10, 11, 12]` without re-selection.
4. Actual tokenizer token counts are reported for every curve point and boundary.
5. The Round 013 nominal-512 candidate boundary is either replicated under the corrected metric or explicitly retracted.
6. A main boundary is reported only for replicated signed-margin zero crossing; otherwise report a lower bound/candidate-unreplicated result.
7. Direct recurrent-state discriminability is aligned with, but not used to overclaim beyond, behavioral evidence.
8. Full pytest passes and `results/result_round_014.json` is generated.
