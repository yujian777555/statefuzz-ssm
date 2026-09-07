# Plan 016 - Falsify Architecture Specificity with Token-Budget-Matched Transformer Controls

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the independently replicated remote-memory signal decay found in Mamba-130M is SSM-specific, architecture-differential, or a shared base-LM/task phenomenon by running predeclared Transformer controls on tokenizer-matched actual-token budgets and multiple filler regimes.

**Architecture:** Preserve the validated counterfactual remote-memory protocol and within-prompt signed-margin metric from Rounds 013-015. Add a behavior-only Hugging Face causal-LM runner for non-Mamba baselines, build remote-memory prompts to explicit actual-token budgets rather than nominal generator lengths, evaluate Mamba and Transformer models on the same predeclared task/value pairs and new seeds, and classify architecture specificity conservatively. Mamba recurrent-state evidence remains available only for Mamba; Transformer hidden activations must not be called recurrent state.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest, existing StateFuzz generator/analyzer/search modules.

**Spec:** `plans/plan_015.md` and `results/result_round_015.json`.

## Global Constraints

- Do not re-select the primary task from Round 015 outcomes.
- Primary template remains `template_id=1`; predeclared value pairs remain `(" red"," blue")`, `(" cat"," dog")`, `(" one"," two")`.
- New architecture-comparison seeds are `[17, 18, 19, 20]`.
- Primary Mamba model remains `state-spaces/mamba-130m-hf`.
- Primary Transformer control is predeclared as `EleutherAI/pythia-160m` because it is a base causal LM of similar scale; do not replace it post-hoc with a Transformer that gives a preferred result.
- Optional secondary SSM scaling model: `state-spaces/mamba-370m-hf`, but only if available and operationally feasible after the primary Transformer comparison.
- `trust_remote_code` must remain false.
- Paper-facing comparisons must use actual tokenizer token counts, not nominal generator `context_tokens`.
- Main failure event remains loss of correct within-prompt candidate preference (`min_signed_margin <= 0`).
- Cross-architecture results are behavioral. Do not claim Transformer `past_key_values` are comparable to Mamba recurrent SSM state.
- Do not call the phenomenon SSM-specific unless a valid Transformer control has been run over a token range that covers the Mamba transition.

---

## Analysis

Round 015 substantially strengthened the current finding:

- Round 014 discovery seeds `[9,10,11,12]` and Round 015 confirmatory seeds `[13,14,15,16]` are separated;
- the primary one/two effect independently replicates;
- local refinement finds all confirmatory seeds passing at actual 1115 tokens and all failing at actual 1388 tokens, so the supported primary transition bracket is `[1115, 1388]` actual tokens;
- at longer contexts, counterfactual `memory_signal` shrinks toward zero while lexical bias remains large enough to dominate one direction;
- direct Mamba recurrent-state A/B discriminability also contracts with context;
- predeclared alternative value pairs produce `multi_pair_generalization`;
- pytest passes 127/127.

This is now strong evidence for a real phenomenon in `state-spaces/mamba-130m-hf`, but the paper still has a major untested alternative explanation:

> The same remote-memory decay may arise in any similarly sized base causal LM under this completion/filler protocol.

Until that alternative is tested, the project cannot support an SSM-specific claim. Round 016 is therefore an architecture-specificity falsification round, not another Mamba boundary-refinement round.

A second confound must also be tested: the current filler consists of repeated templated sentences. A model may degrade because of repetitive-distribution effects rather than state-space architecture. Therefore the architecture comparison must include a predeclared lexically diverse filler regime in addition to the legacy repeated filler.

---

### Task 1: Add a behavior-only Hugging Face causal-LM baseline runner

**Files:**
- Create: `src/statefuzz/runner/hf_causal_lm_runner.py`
- Modify: `src/statefuzz/runner/base.py`
- Modify: `src/statefuzz/runner/__init__.py`
- Test: `tests/runner/test_hf_causal_lm_runner.py`

**Interfaces:**
- Produce `HFCausalLMExperimentConfig` with at least `model_id`, `revision`, `device`, `dtype`, `seed`, `trust_remote_code`, `local_files_only`.
- Produce `HFCausalLMRunner.from_pretrained(config)`.
- Produce the same behavior-facing methods needed by the remote-memory protocol:
  - `single_token_id(text: str) -> int | None`
  - `count_tokens(prompt: str) -> int`
  - `score_candidate_tokens(prompt: str, candidate_token_ids: list[int]) -> dict[str, Any]`
  - `score_remote_memory_pair(pair, *, include_state_copies: bool = False) -> dict[str, Any]`
  - `model_metadata() -> dict[str, Any]`

Do not make the Transformer runner inherit semantic claims from `MambaRunner`. Behavior interfaces may be shared, but recurrent/cache-state evidence remains architecture-specific.

- [ ] **Step 1: Write failing protocol tests**

Use injected tiny/mock tokenizer/model objects so unit tests do not download weights.

Required assertions:

```python
runner = HFCausalLMRunner(model=fake_model, tokenizer=fake_tokenizer, experiment_config=config)
assert runner.count_tokens("...") > 0
assert runner.single_token_id(" one") is not None
result = runner.score_candidate_tokens("prompt", [token_a, token_b])
assert len(result["candidates"]) == 2
assert result["state_source"] == "not_applicable"
```

For `score_remote_memory_pair`, require tokenizer-length matching and `actual_input_tokens` just like the Mamba path.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
python -m pytest tests/runner/test_hf_causal_lm_runner.py -q
```

Expected before implementation: FAIL because the runner does not exist.

- [ ] **Step 3: Implement only behavior-facing scoring**

Use `AutoTokenizer` and `AutoModelForCausalLM` with `trust_remote_code=False`. Record logits/ranks/probabilities for explicit candidates. Do not capture/serialize Transformer KV cache as if it were SSM recurrent state.

`model_metadata()` must include when available:

```python
{
    "model_id": ...,
    "model_type": model.config.model_type,
    "num_parameters": sum(p.numel() for p in model.parameters()),
    "max_context_tokens": resolved_max_context_or_none,
    "architecture_role": "transformer_control",
}
```

- [ ] **Step 4: Run focused tests**

```bash
python -m pytest tests/runner/test_hf_causal_lm_runner.py -q
```

Expected: PASS.

---

### Task 2: Build remote-memory pairs to an actual tokenizer token budget

**Files:**
- Modify: `src/statefuzz/generator/remote_memory.py`
- Test: `tests/generator/test_remote_memory.py`

**Interfaces:**
- Add:
  - `fit_remote_memory_pair_to_token_budget(token_counter, target_tokens: int, tolerance_tokens: int = 8, **pair_kwargs) -> RemoteMemoryPair | structured result`

The function must vary only filler amount. It must never change the remote values, local suffix, template id, seed, or target position to hit the budget.

- [ ] **Step 1: Write failing budget-fitting tests**

With a deterministic fake token counter, require:

```python
result = fit_remote_memory_pair_to_token_budget(counter, target_tokens=1024, tolerance_tokens=8, ...)
assert abs(result.actual_tokens - 1024) <= 8
assert counter(result.pair.prompt_a) == counter(result.pair.prompt_b)
```

Also test an unreachable budget returns an explicit `budget_unreachable` result rather than silently accepting a large mismatch.

- [ ] **Step 2: Implement bounded slot search**

Use deterministic binary/local search over `filler_slots`; cap iterations. Return:

```python
{
    "status": "ok" | "budget_unreachable",
    "pair": pair_or_none,
    "target_tokens": target_tokens,
    "actual_tokens": actual_or_none,
    "absolute_error": error_or_none,
}
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/generator/test_remote_memory.py -q
```

Expected: PASS.

---

### Task 3: Add a lexically diverse filler regime without changing the remote-memory semantics

**Files:**
- Modify: `src/statefuzz/generator/remote_memory.py`
- Test: `tests/generator/test_remote_memory.py`

**Interfaces:**
- Extend `generate_remote_memory_pair(...)` and family generation with `filler_style`.
- Supported styles for this round must be exactly:
  - `structured_repetitive` = current behavior, preserved for backward compatibility;
  - `lexically_diverse` = deterministic varied filler sentences chosen from a fixed predeclared pool by seed/index.

- [ ] **Step 1: Add tests for semantic invariants**

For both styles assert:
- prompt A/B differ only in the remote value;
- identical local suffix;
- same filler style and slot count;
- no candidate values appear accidentally inside filler;
- deterministic output for same seed.

- [ ] **Step 2: Implement `lexically_diverse` filler**

Use a fixed list of neutral sentence templates/topics. Avoid words `red`, `blue`, `cat`, `dog`, `one`, `two` in the filler pool. Do not generate filler with the model.

- [ ] **Step 3: Run generator tests**

```bash
python -m pytest tests/generator/test_remote_memory.py -q
```

Expected: PASS.

---

### Task 4: Add cross-architecture comparison semantics

**Files:**
- Create: `src/statefuzz/analyzer/architecture_specificity.py`
- Test: `tests/analyzer/test_architecture_specificity.py`

**Interfaces:**
- Produce:
  - `summarize_model_behavior_curve(records) -> dict[str, Any]`
  - `compare_architecture_results(mamba_result, transformer_result) -> dict[str, Any]`

- [ ] **Step 1: Write classification tests**

Cover these exact cases:

1. Mamba has replicated sign-loss bracket; Transformer all passes through at least Mamba first-fail token count -> `ssm_specific_candidate`.
2. Both have replicated brackets and actual-token intervals overlap -> `shared_base_lm_decay`.
3. Both fail but non-overlapping intervals -> `architecture_differential`.
4. Transformer task invalid, unavailable, or its supported context ends before covering the Mamba bracket -> `inconclusive_control`.
5. Nonmonotonic curves -> `inconclusive_nonmonotonic`.

- [ ] **Step 2: Implement conservative coverage checks**

Never compare nominal contexts across tokenizers. Use actual-token intervals only.

The classification artifact must include:

```python
{
    "specificity_classification": ...,
    "mamba_boundary_interval_actual": ...,
    "transformer_boundary_interval_actual": ...,
    "common_evaluated_token_range": ...,
    "coverage_sufficient": bool,
}
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/analyzer/test_architecture_specificity.py -q
```

Expected: PASS.

---

### Task 5: Run the predeclared Mamba-vs-Transformer experiment on new seeds

**Files:**
- Use existing experiment path plus the new baseline runner.
- Write: `results/result_round_016.json`.

**Primary models:**

```text
SSM:         state-spaces/mamba-130m-hf
Transformer: EleutherAI/pythia-160m
```

**Seeds:** `[17,18,19,20]`

**Template:** `template_id=1`

**Predeclared value pairs:**

```python
[(" red", " blue"), (" cat", " dog"), (" one", " two")]
```

**Filler styles:**

```python
["structured_repetitive", "lexically_diverse"]
```

**Primary actual-token budgets:**

```python
[256, 512, 768, 1024, 1152, 1280, 1408, 1536, 1792]
```

The exact prompt may be within `±8` tokens of each target budget. Store the actual count for every observation.

- [ ] **Step 1: Resolve model availability and context support**

For each model record:
- load status;
- resolved revision if available;
- parameter count;
- model type;
- maximum supported context when exposed by config.

Try local cached weights first. If `EleutherAI/pythia-160m` is absent and normal Hugging Face access is available, a standard `from_pretrained` download with `trust_remote_code=False` is permitted. If it cannot be loaded, report `transformer_control_unavailable` and do not claim architecture specificity.

- [ ] **Step 2: Apply short-context validity gates per model/pair/style**

At target actual budget 256, all four seeds must have matched A/B prompts and positive signed margins. Invalid model/pair/style combinations are excluded with an explicit reason; they must not be replaced post-hoc.

- [ ] **Step 3: Evaluate behavior curves**

For every valid combination record:
- actual token count;
- both signed margins;
- memory signal / lexical bias / bias dominance;
- pairwise score;
- failure seed count;
- boundary interval classification.

For Mamba additionally record direct recurrent-state convergence. For Transformer record `state_source="not_applicable"`; hidden-state diagnostics may be stored separately but must not be merged with Mamba recurrent-state metrics.

- [ ] **Step 4: Stop at model context limit**

Do not exceed a model's supported context merely to match Mamba. Mark right-censored curves explicitly.

---

### Task 6: Test whether the Round 015 phenomenon survives filler-style change

**Files:**
- Analyze within `results/result_round_016.json` using existing/new analyzer helpers.

- [ ] **Step 1: Compare structured vs diverse filler within Mamba**

For each value pair report whether a replicated transition exists under each filler style and the actual-token intervals.

- [ ] **Step 2: Classify filler robustness**

Return exactly one:
- `filler_robust`: at least two predeclared pairs have replicated transitions under both filler styles;
- `repetitive_filler_specific`: transition appears under structured filler but not diverse filler over sufficient matched coverage;
- `filler_style_differential`: both styles degrade but intervals differ materially;
- `inconclusive_filler_control`: insufficient valid/covered combinations.

Do not silently keep only the style that supports the prior result.

---

### Task 7: Optional Mamba size scaling only after the primary control is complete

**Files:**
- Reuse the Mamba runner and actual-token-budget protocol.
- Append optional results to `results/result_round_016.json`.

**Optional model:** `state-spaces/mamba-370m-hf`

Execute only if:
- Tasks 1-6 are complete;
- checkpoint is available or can be loaded normally;
- primary Transformer comparison has not been skipped for resource reasons.

Use only the frozen `one/two`, `template_id=1`, `lexically_diverse`, seeds `[17,18,19,20]` for this optional scaling check.

Report the boundary interval or censoring; do not infer a scaling law from two model sizes.

---

### Task 8: Produce Round 016 scientific artifact and handoff

**Files:**
- Write: `results/result_round_016.json`
- Update: `status.json`

The artifact must include:
- changed files and full pytest result;
- Round 015 primary bracket `[1115,1388]` as prior evidence, not re-estimated from old seeds;
- exact model metadata/availability;
- new seeds `[17,18,19,20]`;
- token-budget fitting diagnostics;
- per-model/per-pair/per-filler behavioral curves;
- actual-token boundary intervals or censoring;
- architecture-specificity classification;
- filler-robustness classification;
- Mamba recurrent-state evidence separately from Transformer behavior;
- optional Mamba-370M result if run;
- conservative scientific interpretation.

`paper_claim_status` must be exactly one of:

1. `ssm_specific_candidate_with_transformer_control`;
2. `architecture_differential_not_ssm_unique`;
3. `shared_base_lm_memory_decay`;
4. `task_or_filler_specific_effect`;
5. `architecture_specificity_inconclusive`.

No result in Round 016 may use the phrase `SSM-specific failure` unless the valid Transformer control covers the Mamba bracket and supports classification 1.

---

## Verify

Run focused tests:

```bash
python -m pytest tests/runner/test_hf_causal_lm_runner.py -q
python -m pytest tests/generator/test_remote_memory.py -q
python -m pytest tests/analyzer/test_architecture_specificity.py -q
python -m pytest tests/analyzer/test_memory_dependence.py -q
python -m pytest tests/search/test_engine.py -q
```

Then run:

```bash
python -m pytest -q
```

## Success

Round 016 succeeds when all of the following are true:

1. A non-Mamba base causal LM can be evaluated with the same explicit candidate-pair behavioral protocol without pretending it exposes Mamba recurrent state.
2. Cross-model prompts are compared at explicit actual-token budgets, not nominal generator lengths.
3. The predeclared Pythia-160M control is run or its unavailability is explicitly reported; no substitute is chosen after seeing outcomes.
4. New seeds `[17,18,19,20]` are used for the architecture comparison.
5. Both legacy structured filler and predeclared lexically diverse filler are evaluated without post-hoc selection.
6. Architecture specificity is classified conservatively from actual-token coverage and replicated sign-loss intervals.
7. The paper claim is weakened if Transformer/filler controls falsify SSM uniqueness.
8. Full pytest passes and `results/result_round_016.json` is generated.
