# Plan 032 - Hybrid Memory-Path Causal Localization

> Execute on the same user experiment VM and follow `docs/PLANNER_EXECUTION_CONTRACT.md`.

**Goal:** determine whether remote-memory content in `Zyphra/Zamba2-1.2B-Instruct-v2` is causally influenced by the recurrent-state path, the full-attention KV path, or both.

**Round 031 basis:** the strict 256±8 gate passed; no valid stress cell failed through available points near 3.59k tokens; config reports 4096 max positions; `DynamicCache` exposed attention `keys/values` plus recurrent `conv_states/recurrent_states`. Therefore this is a **carrier-localization** study, not a natural-failure study.

## Frozen protocol

- VM checkpoint: `/202532803004/models/Zamba2-1.2B-Instruct-v2`
- offline loading only; no checkpoint substitution
- candidate pair: `(" red", " blue")`
- fresh seeds: `[69,70,71,72,73,74,75,76]`
- families: `structured_repetitive` and `lexically_diverse`
- budgets: `[256,1792,3584]`, tolerance `±8`
- target position: `0.0`
- cache/query split: final **8 token IDs** are held out as a common query tail
- primitive intervention coordinate: `raw_preference = logit(A)-logit(B)`
- preserve primitive candidate logits for every condition
- no architecture-superiority claim; scale/training/tuning confounds remain

## Task 1 - Cache intervention primitives

Create:
- `src/statefuzz/runner/hybrid_cache_intervention.py`
- `tests/runner/test_hybrid_cache_intervention.py`

Provide:

```python
cache_structure_signature(cache) -> dict
clone_cache_independent(cache)
assert_no_tensor_alias(source, clone) -> None
swap_cache_path(recipient_cache, donor_cache, path: str)
```

Path ownership is determined from actual runtime tensor fields:

```text
SSM path: conv_states, recurrent_states, ssm_states if present
Attention path: keys, values
```

`ssm` and `attention` swaps start from an independent recipient clone and replace only target-path tensors. `full` returns an independent donor clone. Require exact donor/recipient tensor-shape match and never mutate sources.

Unit tests must prove no tensor aliasing, non-target fields remain unchanged, full swap matches donor values, and shape mismatch fails deterministically.

## Task 2 - Hybrid runner helpers

Modify `src/statefuzz/runner/hybrid_causal_lm_runner.py` and its tests to expose focused helpers for:

```python
encode_prompt_ids(prompt)
run_body_to_cache(body_input_ids)
score_tail_from_cache(tail_input_ids, past_key_values, candidate_token_ids)
```

Return candidate logits. Do not modify canonical definitions in `memory_dependence.py`.

## Task 3 - Reconstruction gate

Create `scripts/round032_hybrid_path_localization.py`.

For every family × budget × seed:

1. fit full A/B prompts to target±8 using `fit_remote_memory_pair_to_token_budget`;
2. tokenize full prompts and require equal lengths;
3. require the final 8 token IDs of A and B to be exactly identical;
4. define body=`ids[:-8]`, tail=`ids[-8:]`;
5. score the full prompt directly;
6. score body -> cache -> tail;
7. require each A/B candidate logit to match direct scoring within `1e-3`.

If reconstruction fails, mark `cache_reconstruction_invalid` and do not intervene on that cell.

At 256 tokens all 8 fresh seeds in both families must satisfy native directionality:

```text
Prompt A: raw_preference > 0
Prompt B: raw_preference < 0
```

At 1792/3584 retain any natural failures; do not replace seeds.

## Task 4 - Hard intervention controls

Run both directions for each valid pair:

```text
A recipient <- B donor
B recipient <- A donor
```

Conditions:

```text
native
sham recipient clone
full donor cache
SSM donor only
Attention donor only
```

For each direction define recipient/donor raw A-minus-B preferences `r_recipient`, `r_donor`.

Controls must satisfy:

```text
abs(r_sham - r_recipient) <= 1e-3
abs(r_full_donor - r_donor) <= 1e-3
abs(r_donor - r_recipient) >= 0.25
```

Failure statuses:

```text
cache_clone_protocol_invalid
full_donor_transfer_invalid
counterfactual_separation_too_small
```

Control-invalid cells stay in raw output but cannot contribute to mechanism inference.

## Task 5 - Path-specific donor transfer

For `ssm_donor_only` and `attention_donor_only`, store recipient/donor/intervention candidate logits and raw preferences.

Primary metric:

```text
transfer_ratio = (r_intervention-r_recipient)/(r_donor-r_recipient)
```

Do not clamp it.

Interpretation:
- `0`: recipient-like
- `1`: donor-like
- `<0`: moved away from donor
- `>1`: donor-direction overshoot

Also store:

```text
toward_donor = (r_intervention-r_recipient)*(r_donor-r_recipient) > 0
```

## Task 6 - Analyzer and statistics

Create:
- `src/statefuzz/analyzer/path_localization.py`
- `tests/analyzer/test_path_localization.py`

Provide:

```python
compute_transfer_ratio(recipient, donor, intervention, min_separation=0.25)
summarize_path_localization(records, bootstrap_seed=32032, bootstrap_samples=10000)
```

**Seed is the independent unit.** Average A<-B and B<-A within each seed first. For every family × budget × path report valid seed count, mean, median, standard deviation, donor-movement count, and 95% bootstrap CI of the seed-level mean.

A path may be labeled `causal_influence_candidate` only when:

```text
protocol-valid seeds >= 7/8
donor movement in >= 7/8 seeds
95% bootstrap CI lower bound of mean transfer_ratio > 0
```

Allowed summaries:

```text
ssm_path_causal_influence_candidate
attention_path_causal_influence_candidate
dual_path_causal_influence_candidate
path_localization_inconclusive
protocol_invalid
```

Do not claim unique storage from these interventions.

Sanity tests:

```text
recipient=+2 donor=-2 intervention=+2 -> 0
recipient=+2 donor=-2 intervention=0  -> 0.5
recipient=+2 donor=-2 intervention=-2 -> 1
recipient=-2 donor=+2 intervention=0 -> 0.5
```

## Task 7 - Length/distribution comparison

Compare path transfer at 256, 1792 and 3584 for each of the two families. Use paired seed-level differences for 256 -> 3584; if bootstrapped, resample the 8 seed differences, not individual intervention directions.

This round may answer whether SSM state, Attention KV, or both have causal influence on donor-content preference and whether that influence changes near the context limit. It may not conclude that Hybrid architecture is globally more robust or that one path is the unique natural memory store.

## Task 8 - Artifacts

Write:

```text
results/hybrid_path_round_032.json
results/result_round_032.json
```

Raw artifact must contain protocol, environment, cache structure, native records, intervention records, summary, and protocol failures.

Result artifact must include checkpoint path, seeds, families, budgets, reconstruction summary, sham/full-donor controls, SSM and Attention path summaries, length dependence, claim scope, and tests.

Allowed final statuses:

```text
hybrid_path_localization_complete
hybrid_path_protocol_invalid
hybrid_short_probe_invalid
hybrid_runtime_incompatible
```

## Task 9 - Verification and handoff

Add `tests/test_round032_artifacts.py` that recomputes transfer ratios from primitive logits/preferences, verifies target±8, exact fresh seeds 69-76, both intervention directions, and exclusion of protocol-invalid cells from causal-support counts.

Run:

```bash
python -m pytest tests/runner/test_hybrid_cache_intervention.py -q
python -m pytest tests/runner/test_hybrid_causal_lm_runner.py -q
python -m pytest tests/analyzer/test_path_localization.py -q
python -m pytest -q -o addopts=''
```

Then update `status.json` to round 32 with `previous_actor="codex"`, `next="gpt"`, and `last_result="results/result_round_032.json"`.

## Success / claim boundary

A positive result supports only:

> In this tested Zamba2 checkpoint, content-specific intervention on the recurrent-state path and/or full-attention KV path causally shifts remote-memory answer preference toward a counterfactual donor state under the frozen probe.

It does not by itself prove Hybrid superiority, unique memory storage, or a natural failure mechanism.