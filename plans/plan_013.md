# Plan 013 - Prove Remote-Memory Dependence Before Searching Long-Context Boundaries

## Analysis

Round 012 successfully removed the Round 011 confound.

Actual result:
- replacement-based control/stress pairs are tokenizer-length matched;
- 48/48 instances are length matched;
- the prior argmax failure disappears under the matched control;
- direct Mamba recurrent/cache state is now captured from 24 recurrent layers;
- therefore the Round 011 `state_pollution` candidate must be rejected as a length/lexical confound.

This is scientifically useful, but it exposes a deeper problem in the current calibrated task.

The current target token is the control prompt's local next-token argmax (for the reported run, token text ` the`). The suffix `The next symbol is` can predict that token without using information stored far earlier in the sequence. Consequently, extending this task to 8k/32k tokens would mostly measure robustness of local continuation, not long-range memory.

Round 013 must therefore establish a **memory-dependence validity test** before any new memory-boundary claim.

The key falsifiable requirement is:

> Changing only a remote value, while keeping local suffix, token length, and surrounding structure matched, must change the model's preferred target in the corresponding direction at short context.

Only a task that passes this counterfactual short-context test may be used for long-context boundary search.

## Tasks

### Task 1 - Add counterfactual remote-memory task families

Create:
- `src/statefuzz/generator/remote_memory.py`

Add a dataclass or equivalent immutable record for a matched pair and implement:
- `generate_remote_memory_pair(...)`
- `generate_remote_memory_family(...)`

Requirements:
1. Produce paired prompts A/B with identical local suffix and identical structural template.
2. A/B differ only in the remote value that should determine the answer.
3. The remote value position must be controllable with `target_position` / filler placement.
4. Candidate value strings must be explicit metadata, not inferred from the model argmax.
5. Include several predefined natural-language completion templates suitable for a base causal LM, for example repetition/record-completion patterns rather than instruction-following prompts.
6. Do not append extra tokens to stressed/counterfactual variants. Pair construction must preserve slot count before tokenizer validation.

Do not hard-code a claim that any one template is valid. Round 013 must calibrate and select a valid template empirically.

### Task 2 - Add tokenizer-aware candidate scoring

Modify:
- `src/statefuzz/runner/mamba_runner.py`

Implement:
- `single_token_id(text: str) -> int | None`
- `score_candidate_tokens(prompt: str, candidate_token_ids: list[int]) -> dict`
- `score_remote_memory_pair(...) -> dict`

Requirements:
1. Candidate targets are determined from the generated remote values, not from the control argmax.
2. Reject or explicitly mark candidate values that are not representable as exactly one tokenizer token for the first prototype.
3. Record for each candidate:
   - probability
   - rank
   - logit or log-probability
   - top-1 token
4. Record exact tokenizer counts for prompt A and B and require equality.
5. Preserve `state_source` and recurrent-cache capture already implemented in Round 012.

### Task 3 - Define a memory-dependence validity metric

Create:
- `src/statefuzz/analyzer/memory_dependence.py`

Implement:
- `compute_counterfactual_memory_score(...)`
- `validate_remote_memory_task(...)`

The metric must test whether changing the remote value changes model preference in the expected direction.

At minimum compute a symmetric contrast such as:
- preference for A-target on prompt A versus prompt B;
- preference for B-target on prompt B versus prompt A;
- aggregate both directions into a bounded `memory_dependence_score`.

Return an artifact containing:
- `valid_short_context_task`
- `memory_dependence_score`
- per-direction probability/logit contrasts
- paired token counts
- seed count
- template id
- candidate value ids/text

Do not call a task memory-valid merely because both prompts have high confidence. It must show **counterfactual dependence on the remote value**.

### Task 4 - Prevent calibration overfitting

Add a two-stage protocol in the experiment path:

Calibration seeds:
- choose among the predefined template/value candidates using only calibration seeds.

Held-out seeds:
- freeze the selected template and candidate pair;
- validate `valid_short_context_task` on different seeds.

Suggested first split:
- calibration seeds: `[7, 8]`
- held-out seeds: `[9, 10]`

If no template/value pair passes held-out validation, report `task_validity=false` and stop. Do not search a long-context boundary.

### Task 5 - Search a true remote-memory boundary only after validation

Modify:
- `src/statefuzz/search/engine.py`

Implement:
- `search_remote_memory_boundary(...)`

Protocol:
1. Require a frozen short-context-valid remote-memory task.
2. Search context length while keeping the same template, target pair, local suffix, and evaluation metric.
3. Use `memory_dependence_score` as the behavioral quantity, not local next-token argmax stability.
4. Distinguish:
   - exact observed boundary;
   - lower bound when no degradation is observed;
   - invalid task when short-context counterfactual dependence fails.
5. Keep `target_position` as a research variable so later rounds can map distance/position dependence.

Initial real-model context points may be conservative for runtime (for example 64/128/256/512/1024/2048), but the result must be semantically valid before scaling farther.

### Task 6 - Compare direct recurrent states across counterfactual memories

Modify:
- `src/statefuzz/analyzer/hidden_state.py`
- `src/statefuzz/runner/mamba_runner.py` only as needed to expose safe state copies

Implement:
- `compare_recurrent_states(reference, counterfactual) -> dict`

For direct recurrent/cache states, compute per-layer evidence separately for SSM/recurrent state and convolution state when available:
- cosine similarity
- relative norm change
- aggregate min/median/max similarity
- strongest divergent layer

Important interpretation rule:
- state divergence at short context is evidence that the model encodes different remote values;
- convergence or loss of discriminability at long context is only mechanism evidence if it coincides with a validated behavioral memory-dependence drop;
- do not label `state_collision`, `state_forgetting`, or `state_pollution` without this behavioral/mechanistic alignment.

### Task 7 - Produce a Round 013 scientific artifact

Write:
- `results/result_round_013.json`

The result must include:
- selected template/value pair or explicit calibration failure;
- calibration-seed results;
- held-out short-context validity results;
- tokenizer-length matching evidence;
- remote-memory capability curve if and only if short-context validity passes;
- direct recurrent-state pair comparisons;
- conservative scientific interpretation.

If a real behavioral boundary is found, report it as a candidate boundary requiring replication. If none is found, report a lower bound. If no valid remote-memory task is found for `state-spaces/mamba-130m-hf`, report that honestly and do not fabricate a memory boundary.

## Verify

Run:

```bash
python -m pytest -q
```

Required tests:
- `tests/generator/test_remote_memory.py`
- `tests/analyzer/test_memory_dependence.py`
- runner tests for single-token candidate validation and paired candidate scoring
- search tests proving an invalid short-context task cannot produce a memory boundary
- recurrent-state comparison tests

## Success

Round 013 succeeds when all of the following are true:

1. Counterfactual remote-memory pairs are tokenizer-length matched.
2. Candidate answer tokens come from the remote values, never from the model's own control argmax.
3. Template selection is separated from held-out validation.
4. At least one of these outcomes is reported correctly:
   - a held-out-valid remote-memory task plus a real capability curve;
   - or an explicit `task_validity=false` result with no memory-boundary claim.
5. Direct recurrent-cache states are compared across counterfactual memory values.
6. Mechanism labels remain conservative unless behavioral and recurrent-state evidence align.
7. Full pytest passes and `results/result_round_013.json` is generated.
