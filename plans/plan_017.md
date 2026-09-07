# Plan 017 - Confirm the Structured-Repetition SSM Stressor and Reconcile Cross-Round Instability

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently confirm or reject the Round 016 `structured_repetitive + red/blue` SSM-specific candidate under new seeds, while explicitly reconciling why the Round 015 `one/two` boundary did not remain a universal boundary under the Round 016 token-budget protocol.

**Architecture:** Treat Round 016 as discovery of a specific stress condition, not proof of a universal Mamba memory limit. Freeze that condition, evaluate it on fresh confirmation seeds with behavior-matched Pythia control over the full Mamba failure region, keep lexically-diverse filler as a negative control, and aggregate discovery/confirmation evidence with actual-token boundary intervals. Only after this confirmation may the next round attempt recurrent-state intervention.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest, existing StateFuzz generator/runner/analyzer/search modules.

**Spec:** `plans/plan_016.md`, `results/result_round_015.json`, and `results/result_round_016.json`.

## Global Constraints

- Do not claim a universal Mamba memory boundary from Round 015 or Round 016.
- Round 016 discovery condition is frozen as `template_id=1`, `value_a=" red"`, `value_b=" blue"`, `filler_style="structured_repetitive"`, `target_position=0.0`.
- Round 016 discovery seeds `[17,18,19,20]` must not be reused as independent confirmation.
- New confirmatory seeds are `[21,22,23,24,25,26,27,28]`.
- Primary models remain exactly `state-spaces/mamba-130m-hf` and `EleutherAI/pythia-160m`; do not replace Pythia post-hoc.
- Main failure event remains within-prompt sign loss: `min_signed_margin <= 0`.
- All cross-model comparisons use actual tokenizer counts, never nominal generator lengths.
- Transformer KV cache is not Mamba recurrent state; cross-architecture primary evidence remains behavioral.
- `lexically_diverse + red/blue` is a predeclared negative/filler control, not a second discovery search.
- The earlier `one/two` result is historical evidence to reconcile, not a condition to rescue by re-selection.
- Do not start causal mechanism naming in this round. A state intervention round is allowed only if the stress condition independently confirms.

---

## Analysis

Round 016 correctly attempted to falsify architecture specificity. The result did **not** establish a general SSM-specific memory boundary:

- `lexically_diverse` produced no replicated Mamba transition through about 1.8k actual tokens for any predeclared pair;
- `structured_repetitive + cat/dog` remained a lower bound;
- `structured_repetitive + one/two` was only `candidate_unreplicated` under new seeds;
- `structured_repetitive + red/blue` produced the only replicated Mamba transition, with an actual-token interval `[1531,1791]`;
- Pythia-160M remained a lower bound through the same covered range for that condition, so this single condition is an `ssm_specific_candidate`;
- the overall paper claim correctly remained `architecture_specificity_inconclusive`;
- the 1280-token target budget produced `budget_unreachable` records for some seeds, so confirmation must avoid incomplete seed sets.

The correct interpretation is therefore not “Mamba can only remember ~1.3k tokens.” Instead:

> StateFuzz has discovered a candidate **structured-repetition stress pattern** under which Mamba-130M loses counterfactual remote-memory preference before a similarly sized Transformer control.

Round 017 must independently confirm or reject that stressor. It must also document that effective-memory boundaries are conditional on the stress distribution/value pair rather than universal scalar model properties.

---

### Task 1: Add discovery-to-confirmation aggregation semantics

**Files:**
- Modify: `src/statefuzz/analyzer/architecture_specificity.py`
- Test: `tests/analyzer/test_architecture_specificity.py`

**Interfaces:**

Add:

```python
confirm_architecture_stressor(
    discovery: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    transformer_confirmation: Mapping[str, Any],
) -> dict[str, Any]
```

- [ ] **Step 1: Write failing tests**

Cover exactly:

1. discovery Mamba bracket exists, fresh confirmation Mamba bracket overlaps it, and Transformer passes through confirmation first-fail -> `confirmed_ssm_stressor`;
2. confirmation Mamba has no replicated transition -> `candidate_not_replicated`;
3. Transformer also has an overlapping replicated transition -> `shared_stressor_decay`;
4. Transformer coverage stops before Mamba first-fail -> `control_censored`;
5. discovery and confirmation seed sets overlap -> reject with `ValueError` or explicit invalid result.

Require output fields:

```python
{
    "confirmation_classification": ...,
    "discovery_interval_actual": ...,
    "confirmation_interval_actual": ...,
    "transformer_interval_actual": ...,
    "transformer_lower_bound_actual": ...,
    "independent_seed_sets": bool,
    "coverage_sufficient": bool,
}
```

- [ ] **Step 2: Implement conservatively**

An Mamba confirmation bracket need not be numerically identical to Round 016; it must be a replicated pass-to-fail interval under the frozen condition. Treat non-overlapping discovery/confirmation brackets as `candidate_not_stable` rather than forcing confirmation.

- [ ] **Step 3: Run focused tests**

```bash
python -m pytest tests/analyzer/test_architecture_specificity.py -q
```

Expected: PASS.

---

### Task 2: Make token-budget fitting robust enough for complete confirmatory seed sets

**Files:**
- Modify: `src/statefuzz/generator/remote_memory.py`
- Test: `tests/generator/test_remote_memory.py`

**Function:**
- Extend existing `fit_remote_memory_pair_to_token_budget(...)` rather than creating another incompatible fitter.

- [ ] **Step 1: Add regression test for coarse token jumps**

Construct a fake monotonic token counter where adjacent filler-slot counts jump across the exact target by more than 8 tokens. Verify the fitter returns the nearest reachable prompt when it is inside the caller-provided tolerance and never returns an incomplete/incorrect pair.

- [ ] **Step 2: Improve nearest-budget search**

After binary bracketing, inspect both neighboring slot counts and a deterministic bounded neighborhood around them. Tokenization only is cheap; allow up to 128 counter calls if necessary.

Preserve explicit `budget_unreachable` semantics.

- [ ] **Step 3: Confirmation tolerance**

For the real Round 017 experiment use `tolerance_tokens=16`, but always report actual token counts. Do not pretend target budgets are exact.

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/generator/test_remote_memory.py -q
```

Expected: PASS.

---

### Task 3: Independently confirm the frozen Mamba stress condition

**Files:**
- Reuse: `src/statefuzz/runner/mamba_runner.py`
- Reuse: `src/statefuzz/analyzer/memory_dependence.py`
- Reuse: `src/statefuzz/search/engine.py::search_replicated_remote_memory_boundary`
- Write experiment records to: `results/result_round_017.json`

**Frozen condition:**

```text
model = state-spaces/mamba-130m-hf
template_id = 1
values = (" red", " blue")
filler_style = structured_repetitive
target_position = 0.0
confirmation seeds = [21,22,23,24,25,26,27,28]
```

**Actual-token target budgets:**

```python
[256, 1024, 1280, 1408, 1536, 1664, 1792, 1920]
```

Use `tolerance_tokens=16` and store the exact actual count per record.

- [ ] **Step 1: Short-context validity gate**

At target 256 all 8 seeds must have matched A/B prompts, valid single-token candidates, and positive signed margins. Otherwise classify `confirmatory_task_invalid` and stop this condition.

- [ ] **Step 2: Evaluate all budgets**

For each seed/budget store:
- actual input tokens;
- direction A/B margins;
- `memory_signal`;
- `lexical_bias`;
- `bias_dominance_margin`;
- pairwise score;
- direct Mamba recurrent-state comparison.

A budget point with fewer than all 8 valid seeds is incomplete and cannot define a boundary.

- [ ] **Step 3: Derive replicated interval**

Use only complete all-seed points. Record:
- last replicated all-pass actual-token range;
- first replicated all-fail actual-token range;
- actual boundary interval;
- nonmonotonic/candidate-unreplicated status if applicable.

- [ ] **Step 4: Compare with Round 016 discovery interval**

Feed Round 016 `[1531,1791]` discovery evidence and the new result to `confirm_architecture_stressor()`.

---

### Task 4: Run the Pythia control through the confirmed Mamba failure region

**Files:**
- Reuse: `src/statefuzz/runner/hf_causal_lm_runner.py`
- Reuse: `src/statefuzz/analyzer/architecture_specificity.py`
- Write into: `results/result_round_017.json`

**Model:** `EleutherAI/pythia-160m`

**Same frozen condition and seeds:** `[21..28]`.

**Budgets:**

```python
[256, 1024, 1280, 1408, 1536, 1664, 1792, 1920]
```

Do not exceed Pythia's resolved context limit of 2048.

- [ ] **Step 1: Apply the same short-context validity gate**

- [ ] **Step 2: Evaluate behavior at every complete budget**

Store the same behavioral metrics but `state_source="not_applicable"`.

- [ ] **Step 3: Architecture confirmation rule**

`confirmed_ssm_stressor` requires Pythia coverage through at least the new Mamba first replicated fail token count and no overlapping replicated Pythia boundary.

If Pythia also fails in the same region, classify `shared_stressor_decay`. Do not preserve an SSM-specific story.

---

### Task 5: Keep lexically-diverse red/blue as a negative control

**Files:**
- Reuse generator/runners/analyzers.
- Write into: `results/result_round_017.json`.

Evaluate **Mamba only** first under:

```text
template_id=1
values=(" red"," blue")
filler_style=lexically_diverse
seeds=[21..28]
```

Use target budgets `[256, 1408, 1792, 1920]`.

- [ ] **Step 1: Validate at 256**

- [ ] **Step 2: Evaluate the stress region**

If diverse filler remains all-pass through the structured first-fail region, report `structured_repetition_sensitive=true`.

If diverse filler also develops a replicated transition in a similar interval, change the interpretation to a broader remote-memory decay and do not call repetition a necessary trigger.

Do not add new filler styles in this round.

---

### Task 6: Explicitly reconcile Round 015 `one/two` with Round 016

**Files:**
- Modify: `src/statefuzz/analyzer/architecture_specificity.py` or add a small focused helper in `src/statefuzz/analyzer/memory_dependence.py`.
- Test the helper in the corresponding analyzer test file.
- Write reconciliation artifact to `results/result_round_017.json`.

Add a function such as:

```python
summarize_condition_dependence(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]
```

It must summarize, without averaging incompatible conditions:
- value pair;
- filler style;
- seed set;
- prompt-construction protocol/version;
- actual boundary interval or lower bound.

Round 017 must explicitly state:

> `effective_memory_boundary` is a condition-dependent response surface, not a single universal scalar for Mamba-130M.

The prior Round 015 `[1115,1388]` one/two result remains valid for its frozen protocol/seeds, but Round 016 shows it does not automatically generalize to every new seed/value/filler/token-budget condition.

Do not delete or rewrite historical results.

---

### Task 7: Optional size check only after primary confirmation

**Files:**
- Reuse Mamba runner and frozen structured red/blue condition.
- Append to `results/result_round_017.json` only if feasible.

Optional model: `state-spaces/mamba-370m-hf`.

Execute only if Tasks 1-6 are complete and the checkpoint can be loaded without destabilizing the environment.

Use seeds `[21,22,23,24]` and budgets `[256, 1536, 1792, 1920]`.

Report only a boundary/lower-bound observation. Do not infer a scaling law from two sizes.

---

### Task 8: Produce Round 017 artifact and handoff

**Files:**
- Write: `results/result_round_017.json`
- Update: `status.json`

Required top-level fields:

```json
{
  "round": 17,
  "discovery_condition": {...},
  "discovery_seeds": [17,18,19,20],
  "confirmation_seeds": [21,22,23,24,25,26,27,28],
  "mamba_confirmation": {...},
  "transformer_confirmation": {...},
  "diverse_filler_negative_control": {...},
  "architecture_confirmation": {...},
  "condition_dependence_reconciliation": {...},
  "paper_claim_status": "...",
  "tests": {...}
}
```

`paper_claim_status` must be exactly one of:

1. `confirmed_structured_repetition_ssm_stressor`;
2. `shared_structured_repetition_decay`;
3. `architecture_candidate_not_independently_replicated`;
4. `transformer_control_censored`;
5. `confirmatory_task_invalid`.

If status 1 is obtained, the next Planner round may proceed to recurrent-state causal intervention on this frozen stress condition.

Scientific language for status 1 should be:

> Under a predeclared structured-repetition remote-memory stress condition, Mamba-130M exhibits an independently replicated behavioral transition before a similarly sized Transformer control, while lexically diverse filler remains a negative control over the tested range.

Do **not** generalize this to “all SSMs” or “Mamba has a universal N-token memory limit.”

---

## Verify

Run:

```bash
python -m pytest tests/analyzer/test_architecture_specificity.py -q
python -m pytest tests/generator/test_remote_memory.py -q
python -m pytest tests/analyzer/test_memory_dependence.py -q
python -m pytest tests/search/test_engine.py -q
python -m pytest tests/runner/test_hf_causal_lm_runner.py -q
python -m pytest -q
```

## Success

Round 017 succeeds when:

1. The Round 016 structured-red/blue candidate is independently confirmed or explicitly rejected on seeds `[21..28]`.
2. Pythia covers the Mamba confirmation failure region or the result is explicitly censored.
3. No incomplete seed set defines a boundary.
4. Lexically diverse red/blue is evaluated as a predeclared negative control.
5. Round 015/016 differences are explained as condition dependence rather than hidden by a single scalar boundary.
6. Any SSM-specific statement is limited to the confirmed stress condition.
7. Full pytest passes and `results/result_round_017.json` is generated.
