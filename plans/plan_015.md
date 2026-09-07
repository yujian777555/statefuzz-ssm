# Plan 015 - Confirm Memory-Signal Decay, Separate Lexical Bias, and Localize the Boundary Interval

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently replicate the corrected Mamba remote-memory failure, distinguish memory-signal decay from candidate lexical bias, and report a bracketed actual-token boundary rather than a coarse-grid failure point.

**Architecture:** Keep the Round 013 frozen remote-memory protocol and the Round 014 within-prompt signed-margin metric. Add an interpretable memory-signal/bias decomposition, confirm the phenomenon on new seeds never used for discovery, locally refine the pass/fail transition, test predeclared value pairs for generality, and align the behavioral signal with direct recurrent-state convergence without making a causal mechanism claim.

**Tech Stack:** Python, PyTorch, Transformers Mamba, pytest, existing StateFuzz generator/runner/analyzer/search modules.

**Spec:** `plans/plan_014.md` and `results/result_round_014.json`.

## Global Constraints

- Primary frozen task remains `template_id=1`, `value_a=" one"`, `value_b=" two"`, `target_position=0.0`.
- Discovery/previous held-out seeds `[9, 10, 11, 12]` must not be treated as independent confirmation in this round.
- New primary confirmatory seeds are `[13, 14, 15, 16]`.
- Do not re-select the primary template/value pair using Round 015 outcomes.
- The primary behavioral failure event remains within-prompt signed-margin sign loss; legacy absolute probability contrasts stay diagnostic only.
- Every paper-facing context value must include actual tokenizer token counts.
- A coarse first-failure point is not an exact boundary. Report a bracket `[last replicated pass, first replicated fail]` in actual tokens.
- Direct recurrent-state convergence is mechanistic evidence, not proof of `state_collision`, `state_forgetting`, or `state_pollution`.

---

## Analysis

Round 014 corrected the Round 013 metric and produced the first replicated sign-loss region:

- nominal 512 / actual 1115 tokens: all four seeds `[9,10,11,12]` pass;
- nominal 1024 / actual 2220 tokens: all four seeds fail;
- nominal 2048 / actual 4443 tokens: all four seeds remain failed;
- therefore the Round 013 nominal-512 boundary was correctly retracted;
- the corrected result currently labels nominal 1024 as `replicated_zero_crossing`.

However, two scientific issues remain.

First, the current artifact stores `actual_boundary_token_range=[2220,2220]`. That is the observed failing token count, not the mathematical boundary. With the present grid, the supported statement is only that the replicated transition lies between the last all-pass point (1115 actual tokens) and the first all-fail point (2220 actual tokens).

Second, the failure is directionally asymmetric. At 2220 and 4443 actual tokens the `one`-correct prompt still has a positive A-vs-B margin, while the `two`-correct prompt has a negative B-vs-A margin. At the same time, direct counterfactual recurrent states become much more similar as context increases. This is consistent with a hypothesis in which remote-memory signal shrinks until a lexical/prior preference dominates one direction. Round 015 must test this hypothesis rather than calling the result generic forgetting.

---

### Task 1: Add memory-signal versus lexical-bias decomposition

**Files:**
- Modify: `src/statefuzz/analyzer/memory_dependence.py`
- Test: `tests/analyzer/test_memory_dependence.py`

**Interfaces:**
- Consumes the same two-candidate logits used by `compute_pairwise_memory_metrics()`.
- Produces `decompose_pairwise_preference(record: Mapping[str, Any]) -> dict[str, Any]`.

For a record define:

```python
d_a = logit(A | prompt_A) - logit(B | prompt_A)
d_b = logit(A | prompt_B) - logit(B | prompt_B)
memory_signal = (d_a - d_b) / 2.0
lexical_bias = (d_a + d_b) / 2.0
bias_dominance_margin = memory_signal - abs(lexical_bias)
```

Equivalent relations to existing fields:

```python
d_a == direction_a_margin
d_b == -direction_b_margin
memory_signal == (direction_a_margin + direction_b_margin) / 2.0
lexical_bias == (direction_a_margin - direction_b_margin) / 2.0
```

Interpretation:
- `memory_signal > 0`: counterfactual remote value still shifts A/B preference in the expected direction;
- `abs(lexical_bias)` estimates the shared candidate preference independent of which remote value is present;
- both directions are behaviorally correct exactly when `bias_dominance_margin > 0` (apart from ties).

- [ ] **Step 1: Write failing decomposition tests**

Test at least:

```python
# Strong memory overcomes bias.
direction_a_margin = 4.0
direction_b_margin = 2.0
# memory_signal=3, lexical_bias=1, bias_dominance_margin=2
```

and:

```python
# Memory remains non-zero but lexical bias dominates one direction.
direction_a_margin = 3.0
direction_b_margin = -1.0
# memory_signal=1, lexical_bias=2, bias_dominance_margin=-1
```

Assert the second case is not described as zero memory; it is `bias_dominated=True` with positive `memory_signal`.

- [ ] **Step 2: Implement `decompose_pairwise_preference()`**

Return at minimum:

```python
{
    "raw_preference_prompt_a": d_a,
    "raw_preference_prompt_b": d_b,
    "memory_signal": memory_signal,
    "lexical_bias": lexical_bias,
    "bias_dominance_margin": bias_dominance_margin,
    "bias_dominated": bias_dominance_margin <= 0.0,
}
```

Integrate these fields into `compute_counterfactual_memory_score()` without deleting existing Round 014 metrics.

- [ ] **Step 3: Run focused tests**

```bash
python -m pytest tests/analyzer/test_memory_dependence.py -q
```

Expected: PASS.

---

### Task 2: Fix boundary semantics to an observed transition interval

**Files:**
- Modify: `src/statefuzz/search/engine.py`
- Test: `tests/search/test_engine.py`

**Interfaces:**
- Extend the replicated remote-memory boundary path; do not create a separate incompatible search framework.
- Produce an explicit actual-token transition bracket.

- [ ] **Step 1: Write failing interval tests**

For an all-seed pass at actual 1115 tokens followed by an all-seed fail at actual 2220 tokens, require:

```python
assert result["boundary_kind"] == "replicated_zero_crossing"
assert result["actual_boundary_interval"] == [1115, 2220]
assert result["last_replicated_pass_actual_tokens"] == 1115
assert result["first_replicated_fail_actual_tokens"] == 2220
```

Do not allow `[2220, 2220]` unless an actual adaptive/local search establishes a zero-width interval, which is not expected here.

Add a non-monotonic test: if all seeds fail at one tested context and later all seeds pass, return `boundary_kind="nonmonotonic"` and do not issue a single paper boundary.

- [ ] **Step 2: Implement interval and monotonicity semantics**

Every tested context remains in the returned curve even after the first failure. Validate that no later all-pass point invalidates the monotone transition interpretation.

- [ ] **Step 3: Run search tests**

```bash
python -m pytest tests/search/test_engine.py -q
```

Expected: PASS.

---

### Task 3: Independently replicate the primary phenomenon on new seeds

**Files:**
- Use the existing real-model execution path.
- Write records into: `results/result_round_015.json`.

**Frozen confirmatory experiment:**

```text
model = state-spaces/mamba-130m-hf
template_id = 1
values = (" one", " two")
target_position = 0.0
confirmatory seeds = [13, 14, 15, 16]
primary coarse nominal contexts = [512, 1024, 2048]
```

- [ ] **Step 1: Revalidate the frozen task at short context**

Run nominal context 64 for seeds `[13,14,15,16]` and require both signed margins positive for every seed. If this fails, report `confirmatory_task_invalid` and do not claim replication.

- [ ] **Step 2: Evaluate the coarse confirmatory contexts**

At each context store:
- actual input token count;
- both signed margins;
- `memory_signal`;
- `lexical_bias`;
- `bias_dominance_margin`;
- pairwise score;
- direct SSM/conv recurrent-state comparison.

- [ ] **Step 3: Decide replication before local refinement**

Primary Round 014 pattern is replicated only if new seeds show:

```text
all seeds pass at nominal 512
all seeds fail by nominal 1024 or 2048
no later all-pass reversal
```

If only some seeds fail, classify `candidate_unreplicated` and do not force a shared boundary.

---

### Task 4: Locally refine the replicated transition

**Files:**
- Modify/search through existing experiment path and `src/statefuzz/search/engine.py` as needed.
- Test: `tests/search/test_engine.py`.

Only execute this task if Task 3 confirms a replicated transition.

- [ ] **Step 1: Evaluate fixed intermediate nominal contexts**

Evaluate all confirmatory seeds at:

```python
[640, 768, 896]
```

Together with 512 and 1024, determine the last all-pass and first all-fail context.

- [ ] **Step 2: Add at most two adaptive midpoint contexts**

If the actual-token interval is still wider than 300 tokens, evaluate the nominal midpoint between the last all-pass and first all-fail context, then repeat once at most.

Stop after two adaptive additions even if the interval remains wider. Report the surviving actual-token bracket honestly.

- [ ] **Step 3: Store the localized interval**

Result fields must include:

```json
{
  "boundary_kind": "replicated_zero_crossing",
  "actual_boundary_interval": [LAST_PASS_ACTUAL, FIRST_FAIL_ACTUAL],
  "nominal_boundary_interval": [LAST_PASS_NOMINAL, FIRST_FAIL_NOMINAL],
  "localization_censored": false
}
```

If runtime/resource limits prevent refinement, set `localization_censored=true` and retain the coarse bracket.

---

### Task 5: Test whether the effect generalizes beyond `one/two`

**Files:**
- Reuse: `src/statefuzz/generator/remote_memory.py`
- Reuse/modify only if necessary: `src/statefuzz/analyzer/memory_dependence.py`
- Write results into: `results/result_round_015.json`

Use the same `template_id=1` and the predeclared value pairs that already existed in Round 013 calibration:

```python
[(" red", " blue"), (" cat", " dog"), (" one", " two")]
```

Do not select the best pair after seeing long-context results.

- [ ] **Step 1: Short-context validity gate per pair on confirmatory seeds**

At nominal 64, mark each pair `valid` only if all four new seeds have both signed margins positive and tokenizer-matched A/B prompts.

- [ ] **Step 2: Evaluate valid pairs at nominal `[512, 1024, 2048]`**

For every valid pair record the full signal/bias decomposition.

- [ ] **Step 3: Classify generality**

Report exactly one:

- `multi_pair_generalization`: at least two valid pairs show replicated pass-to-fail transitions;
- `pair_specific_effect`: only the frozen one/two pair shows the transition;
- `insufficient_valid_pairs`: fewer than two pairs pass the short-context validity gate.

Do not convert this into a new template-selection phase.

---

### Task 6: Quantify recurrent-state counterfactual convergence alongside memory signal

**Files:**
- Modify: `src/statefuzz/analyzer/hidden_state.py`
- Test: `tests/analyzer/test_hidden_state.py`
- Write aggregate into: `results/result_round_015.json`

**Interfaces:**
- Add a compact aggregation helper such as:
  - `summarize_counterfactual_state_convergence(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]`

- [ ] **Step 1: Aggregate per context**

For SSM recurrent states separately from convolution states, report across confirmatory seeds:
- median minimum cosine similarity;
- median maximum relative L2 distance;
- inter-seed range;
- most frequent strongest-divergent layer.

- [ ] **Step 2: Align with behavioral decomposition**

At every context include:

```json
{
  "behavior": {
    "median_memory_signal": ...,
    "median_abs_lexical_bias": ...,
    "failure_seed_count": ...
  },
  "state": {
    "median_min_ssm_cosine": ...,
    "median_max_ssm_relative_l2": ...
  }
}
```

The intended descriptive question is whether A/B recurrent-state discriminability contracts as `memory_signal` weakens.

Do not assign a causal mechanism name from this correlation.

- [ ] **Step 3: Run focused tests**

```bash
python -m pytest tests/analyzer/test_hidden_state.py -q
```

Expected: PASS.

---

### Task 7: Produce Round 015 scientific artifact and handoff

**Files:**
- Write: `results/result_round_015.json`
- Update: `status.json`

The artifact must include:
- changed files and full pytest result;
- frozen primary task identity;
- discovery seeds versus confirmatory seeds clearly separated;
- corrected actual-token boundary interval semantics;
- coarse and refined confirmatory curves;
- signal/bias decomposition for every observation;
- pair-generalization table;
- recurrent-state convergence alignment;
- explicit decision on whether Round 014 independently replicated;
- conservative paper claim status.

`paper_claim_status` must be exactly one of:

1. `independently_replicated_bracketed_boundary_multi_pair`;
2. `independently_replicated_bracketed_boundary_pair_specific`;
3. `candidate_boundary_not_independently_replicated`;
4. `confirmatory_task_invalid`.

The scientific interpretation should prefer language such as **counterfactual recurrent-state convergence associated with memory-signal decay**. Do not use `state_collision`, `state_forgetting`, or `state_pollution` as a causal conclusion.

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
python -m pytest tests/search/test_engine.py -q
python -m pytest tests/runner/test_calibrated_runner.py -q
```

## Success

Round 015 succeeds when all of the following are true:

1. The Round 014 result is no longer reported as an exact `2220-token boundary`; it becomes an actual-token interval until localized.
2. The `one/two` directional asymmetry is decomposed into memory signal and lexical bias.
3. The primary finding is evaluated on new confirmatory seeds `[13,14,15,16]`.
4. If replicated, the transition is locally refined with bounded extra evaluations.
5. Predeclared alternative value pairs are tested without post-hoc selection.
6. Recurrent-state convergence is quantitatively aligned with behavioral memory-signal decay but not overclaimed causally.
7. Full pytest passes and `results/result_round_015.json` is generated.
