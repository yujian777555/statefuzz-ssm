# Plan 012 — Confound-Controlled Mechanism Validation

## Planner diagnosis

Round 011 produced the first replicated real-model failure region, but it is not yet a defensible SSM-mechanism result.

Observed facts from `results/result_round_011.json` and `src/`:

- With the same nominal `context_tokens=64`, the no-interference control passes, while `interference_strength=0.5` fails for seeds 7 and 8.
- The failing configuration shifts the argmax target token and reduces target probability.
- The current result correctly marks the mechanism as preliminary and records `direct_ssm_recurrence_state_captured=false`.
- `generate_interference_prompt()` currently appends `round(context_tokens * interference_strength)` distractor units to the baseline prompt. Therefore interference strength changes both semantic interference and actual input length.
- `MambaRunner` captures transformer-style `hidden_states` / per-layer final activations, but does not yet expose the Mamba recurrent/cache state as separate evidence.
- `classify_failure()` still defaults many unmatched prediction errors to `state_pollution`; this is stronger than the available evidence.

Round 012 must therefore validate causality before expanding the benchmark.

# Tasks

## Task 1 — Remove the length confound from interference experiments

### Files
- `src/statefuzz/generator/calibrated.py`
- `src/statefuzz/runner/mamba_runner.py`
- `tests/generator/test_interference_calibrated.py`
- `tests/runner/test_calibrated_runner.py`

### Required changes

1. Add a paired prompt generator in `src/statefuzz/generator/calibrated.py`, e.g.:

```python
generate_length_matched_interference_pair(
    context_tokens: int,
    seed: int,
    interference_strength: float,
) -> tuple[str, str]
```

The control and stressed prompt must use the same number of content slots. Interference must replace normal filler rather than append extra filler.

2. Add tokenizer-aware verification in `MambaRunner`, e.g.:

```python
count_tokens(prompt: str) -> int
score_paired_next_token(
    control_prompt: str,
    stressed_prompt: str,
    target_token_id: int | None = None,
) -> dict[str, Any]
```

The paired evaluator must report both token counts and must not treat a pair as matched when tokenized lengths differ materially.

3. Extend next-token evidence with:

- decoded/printable target token
- decoded/printable predicted token
- target probability
- target rank (or equivalent rank evidence)
- top-1 margin relative to target

This is needed to determine whether the failure is genuine retention/interference or merely lexical continuation bias caused by the distractor string.

## Task 2 — Separate behavioral failures from mechanism claims

### Files
- `src/statefuzz/analyzer/failure_classifier.py`
- `src/statefuzz/analyzer/hidden_state.py`
- `tests/analyzer/test_hidden_state.py`

### Required changes

1. Stop mapping every unexplained wrong prediction to `state_pollution`.

2. Introduce conservative labels such as:

- `behavioral_interference`
- `interference_induced_token_shift`
- `insufficient_mechanism_evidence`

3. `state_collision`, `state_forgetting`, `state_pollution`, or `state_collapse` may only be emitted when the corresponding state evidence is explicitly present.

4. Remove or demote string-prefix heuristics as evidence for state collision.

5. Add paired hidden-activation comparison utilities, e.g.:

```python
compare_layer_states(reference, stressed)
find_first_divergent_layer(...)
```

Return at least:

- per-layer cosine similarity
- per-layer norm ratio/change
- first/strongest divergent layer

Call these measurements `hidden_activation_evidence` unless they are actual recurrent SSM states.

## Task 3 — Attempt direct recurrent/cache-state capture without overclaiming

### Files
- `src/statefuzz/runner/mamba_runner.py`
- `src/statefuzz/analyzer/hidden_state.py`
- `tests/runner/test_calibrated_runner.py`

### Required changes

1. Inspect the real model output/cache returned by the existing forward pass (`use_cache=True`).

2. If the model exposes recurrent/cache tensors, capture them separately from `hidden_states` using a clearly named API, e.g.:

```python
capture_recurrent_state()
capture_recurrent_state_summary()
```

3. Record which state source was actually captured:

```json
{
  "state_source": "direct_recurrent_cache" | "layer_hidden_activation" | "unavailable"
}
```

4. If direct recurrence state is unavailable, do not fabricate or infer it. Record `unavailable` and keep mechanism claims behavioral/preliminary.

5. Avoid storing full huge tensors in result JSON. Store deterministic summaries needed for comparison.

## Task 4 — Replace the single scalar boundary with a controlled stress frontier

### File
- `src/statefuzz/search/engine.py`
- `tests/search/test_engine.py`

### Required changes

Add a function such as:

```python
search_interference_frontier(...)
```

For each context length:

1. validate a matched no-interference control;
2. evaluate increasing interference strength at matched token budget;
3. find the minimum interference strength causing replicated failure;
4. report the frontier as `(context_tokens, minimum_failure_interference)`.

Do not label a failure at 64 tokens as an `effective_memory_boundary` merely because it is the smallest failing context in a 2-D search.

Reserve `memory boundary` for degradation caused by increasing context at a fixed, calibrated stress condition.

## Task 5 — Run a paired real-model validation experiment

Use the existing real model path and calibrated next-token protocol.

Minimum experiment:

- model: `state-spaces/mamba-130m-hf`
- seeds: at least 4 deterministic seeds if resources allow; otherwise preserve all completed seeds and explain the resource constraint
- context points: at least 64, 128, 256, 512
- interference strengths: 0.0 plus at least two nonzero strengths
- every stressed case must have a same-context, token-length-matched control

The output must distinguish:

1. behavioral failure;
2. hidden-activation divergence;
3. direct recurrent-state evidence, if actually available.

Write the experiment artifact to:

`results/result_round_012.json`

# Verify

Run:

```bash
python -m pytest -q
```

Verify additionally:

1. paired control/stress prompts have matched tokenized lengths (or explicit mismatch rejection);
2. no-interference controls still pass at the tested short contexts;
3. the same failure is replicated across multiple seeds before calling it a failure region;
4. `state_pollution` is not emitted without state-specific evidence;
5. result artifacts explicitly state the state evidence source;
6. 2-D interference failures are reported as a stress frontier, not misreported as a memory boundary.

# Success

Round 012 succeeds only if all of the following hold:

- all tests pass;
- semantic interference is experimentally separated from extra input length;
- the real-model failure region remains after length matching, or the previous failure is correctly reclassified as a confound;
- mechanism labels are conservative and evidence-backed;
- direct recurrent state is captured and identified if available, otherwise explicitly reported unavailable;
- `results/result_round_012.json` contains a reproducible paired-control stress frontier with multi-seed evidence.

The scientific objective is not to preserve a desired failure claim. It is to determine whether the Round 011 signal survives proper controls and whether it can legitimately be connected to SSM state dynamics.