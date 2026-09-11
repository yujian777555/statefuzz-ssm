# Plan 031 - Hybrid Architecture Memory-Stress Validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether StateFuzz memory-stress fingerprints transfer to a hybrid SSM-attention architecture, using a frozen protocol and without conflating architecture with model scale or training data.

**Architecture:** Add a hybrid-model capability layer and evaluate the same predeclared stress families used in Rounds 025-028. Primary hybrid checkpoint is `Zyphra/Zamba2-1.2B`. Behavior is the primary cross-architecture endpoint; internal cache/state diagnostics are recorded separately and only when the model exposes them reliably. Checkpoint availability is a hard gate because Round 030 was blocked by network/checkpoint access.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest, existing StateFuzz generator/runner/analyzer modules.

**Spec:** `results/result_round_030.json`, `results/scale_validation_round_030.json`, `results/stress_family_comparison_round_025.json`, `results/architecture_comparison_round_026.json`.

## Global Constraints

- Round 030 produced no larger-model result because checkpoints were unavailable; do not treat it as negative scientific evidence.
- Primary hybrid checkpoint: `Zyphra/Zamba2-1.2B`.
- Use the base checkpoint, not an instruction-tuned variant, to remain closer to the existing base-LM protocol.
- Do not repeatedly retry external downloads. Perform one deterministic checkpoint-resolution pass, then stop cleanly if unavailable.
- Do not claim hybrid superiority/inferiority from raw accuracy because scale/training data differ from Mamba-130M and Pythia-160M.
- Cross-architecture primary comparison is behavior only.
- Do not equate attention KV cache with SSM recurrent state.
- Do not call hybrid failures `routing failure`, `state collision`, or `attention dilution` without intervention evidence.
- Use actual tokenizer counts for every record.
- Freeze stress families before execution: `structured_repetitive`, `periodic_pattern`, `interleaved_distractor`, `semantic_distractor`, plus `lexically_diverse` negative control.
- New evaluation seeds: `[65,66,67,68]`.
- Preserve historical Round 020/021 Mamba mechanism claims as scoped case-study evidence; Round 031 does not rewrite them.

---

### Task 1: Add a hard checkpoint-availability gate

**Files:**
- Create: `src/statefuzz/runner/checkpoint_resolver.py`
- Test: `tests/runner/test_checkpoint_resolver.py`

**Interfaces:**

```python
resolve_checkpoint(
    model_id: str,
    local_candidates: list[str] | None = None,
    allow_single_remote_attempt: bool = True,
) -> dict[str, Any]
```

Required output fields:

```python
{
    "model_id": str,
    "available": bool,
    "source": "local_cache" | "explicit_path" | "remote" | "unavailable",
    "resolved_path": str | None,
    "error": str | None,
}
```

- [ ] Write failing tests for local hit, unavailable checkpoint, and single-attempt remote failure.
- [ ] Implement resolver without infinite retries.
- [ ] Ensure network failure produces a structured result rather than crashing the experiment.
- [ ] Run `python -m pytest tests/runner/test_checkpoint_resolver.py -q`.

**Execution gate:**

Before any expensive model work, resolve `Zyphra/Zamba2-1.2B`.

If unavailable after the one allowed remote attempt:

1. write `results/result_round_031.json` with `status="blocked_by_hybrid_checkpoint_availability"`;
2. record exact blocker;
3. do not fake hybrid results;
4. set the next scientific action to obtaining/mounting the checkpoint.

---

### Task 2: Add hybrid architecture capability metadata

**Files:**
- Modify: `src/statefuzz/runner/hf_causal_lm_runner.py`
- Create if cleaner: `src/statefuzz/analyzer/model_capabilities.py`
- Test: corresponding runner/analyzer tests.

**Interfaces:**

Add a model metadata helper that records:

```python
{
    "architecture_family": "hybrid_ssm_attention",
    "model_type": "zamba2",
    "num_layers": int,
    "layer_types": list[str] | None,
    "hybrid_layer_ids": list[int] | None,
    "max_position_embeddings": int | None,
    "cache_semantics": "hybrid",
}
```

Requirements:

- [ ] Detect layer-type metadata from config when exposed.
- [ ] Never infer an SSM state tensor from a generic KV cache field.
- [ ] Keep existing Mamba/Pythia behavior unchanged.
- [ ] Add regression tests proving Mamba, Transformer, and Hybrid are classified separately.

---

### Task 3: Validate the hybrid short-context task before long-context evaluation

**Files:**
- Reuse: `src/statefuzz/generator/remote_memory.py`
- Reuse: `src/statefuzz/analyzer/memory_dependence.py`
- Reuse/modify: `src/statefuzz/runner/hf_causal_lm_runner.py`

**Frozen primary value pair:** `(" red", " blue")` if both values are valid single-token candidates for the hybrid tokenizer.

If they are not, choose the first valid pair from the **predeclared** ordered list:

1. `(" red", " blue")`
2. `(" one", " two")`
3. `(" cat", " dog")`

Do not choose based on long-context results.

- [ ] Verify A/B token lengths match exactly.
- [ ] Verify candidate values are tokenizer-valid single tokens in the prediction context.
- [ ] At ~256 actual tokens, require positive within-prompt signed margins for all valid seeds before treating the task as a memory probe.
- [ ] Record the selected predeclared pair and the reason for any fallback.

If no pair passes short-context validity, stop with `hybrid_task_invalid` rather than running a meaningless stress sweep.

---

### Task 4: Run the frozen hybrid stress-family fingerprint

**Files:**
- Reuse existing stress-family generators from Round 025.
- Reuse runner/analyzers.
- Write: `results/hybrid_stress_round_031.json`.

**Model:** `Zyphra/Zamba2-1.2B`

**Seeds:** `[65,66,67,68]`

**Actual-token target budgets:**

```python
[256, 768, 1280, 1792, 2560, 3584]
```

Only use budgets below the resolved model limit. If fitting cannot hit a target, record exact actual tokens and explicit `budget_unreachable`.

For each family/budget/seed record:

- actual input tokens;
- A/B signed margins;
- memory signal;
- lexical bias;
- failure event (`min_signed_margin <= 0`);
- model metadata;
- stress family;
- runtime status.

Families:

1. `structured_repetitive`
2. `periodic_pattern`
3. `interleaved_distractor`
4. `semantic_distractor`
5. `lexically_diverse` negative control

Do not search for new stress families in this round.

---

### Task 5: Compare stress fingerprints rather than raw model quality

**Files:**
- Create: `src/statefuzz/analyzer/stress_fingerprint.py`
- Test: `tests/analyzer/test_stress_fingerprint.py`

**Interfaces:**

```python
build_stress_fingerprint(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]
compare_stress_fingerprints(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]
```

Per model/family summarize:

- short-context validity;
- failure rate by token budget;
- normalized margin retention relative to that model's own shortest valid budget;
- first observed risk region or lower bound;
- monotonicity violations;
- negative-control behavior.

Cross-model comparison must focus on **within-model normalized degradation**, not raw logits or raw accuracy.

Allowed classifications:

1. `hybrid_more_robust_candidate`
2. `hybrid_similar_stress_profile`
3. `hybrid_distinct_stress_profile`
4. `hybrid_more_sensitive_candidate`
5. `inconclusive_due_to_scale_training_confounds`

Always attach `architecture_causality_confirmed=false` in Round 031.

---

### Task 6: Inspect hybrid cache/state observability without mechanism claims

**Files:**
- Modify runner only if needed.
- Write diagnostics into `results/hybrid_stress_round_031.json`.

On one short and one long valid prompt, inspect returned cache/state object structure.

Record separately:

```python
{
    "attention_cache_observed": bool,
    "ssm_recurrent_state_observed": bool,
    "state_shapes": ...,
    "cache_shapes": ...,
}
```

Requirements:

- [ ] No intervention in this round unless APIs already expose a safe copy/restore path with no new architecture-specific hacks.
- [ ] Do not flatten the two memory systems into one metric.
- [ ] If both are observable, Round 032 may test memory-path localization.

---

### Task 7: Optional realistic-workload spot check

Run only if Tasks 1-6 succeed.

Use **one** existing workload from Round 027: `code_context_dependency`.

Use the strongest predeclared stress family according to the synthetic hybrid fingerprint **only for descriptive validation**, and also run the negative-control family.

Do not add new workload templates.

Record whether the synthetic stress ordering transfers qualitatively to this realistic controlled task.

---

### Task 8: Produce Round 031 result and handoff

**Files:**
- Write: `results/result_round_031.json`
- Write: `results/hybrid_stress_round_031.json` if model available.
- Update: `status.json`.

Required top-level result fields:

```json
{
  "round": 31,
  "status": "...",
  "hybrid_model": "Zyphra/Zamba2-1.2B",
  "checkpoint_resolution": {},
  "task_validity": {},
  "stress_fingerprint": {},
  "comparison_to_existing_models": {},
  "cache_state_observability": {},
  "realistic_spot_check": {},
  "claim_scope": "...",
  "tests": {}
}
```

Allowed final statuses:

- `hybrid_validation_complete`
- `blocked_by_hybrid_checkpoint_availability`
- `hybrid_task_invalid`
- `hybrid_runtime_incompatible`

Scientific language if validation completes:

> StateFuzz reveals whether a hybrid SSM-attention model exhibits a similar, weaker, stronger, or qualitatively different remote-memory stress fingerprint under the same controlled probe family.

Do **not** claim that the difference is caused by architecture alone because parameter count and training corpora are not matched.

---

## Verify

Run:

```bash
python -m pytest tests/runner/test_checkpoint_resolver.py -q
python -m pytest tests/analyzer/test_stress_fingerprint.py -q
python -m pytest -q -o addopts=''
```

## Success

Round 031 succeeds when:

1. Hybrid checkpoint availability is handled deterministically and honestly.
2. A valid short-context memory probe is established before long-context stress evaluation.
3. All five predeclared stress/control families receive a hybrid stress fingerprint if the checkpoint is available.
4. Cross-model comparison uses within-model normalized degradation rather than raw performance.
5. Attention cache and SSM state observability are kept separate.
6. No architecture-causal claim is made from unmatched checkpoints.
7. The round either produces a real hybrid result or an explicit availability/runtime blocker with no fabricated evidence.
