# Plan 031 - Hybrid Architecture Memory-Stress Validation

> **For agentic workers:** Use the repository's normal executor workflow and execute this plan task-by-task. Do not substitute checkpoints post-hoc.

**Goal:** Test whether StateFuzz memory-stress fingerprints transfer to a hybrid SSM-attention model using the locally mounted Zamba2 checkpoint, while keeping instruction-tuning, scale, and training-data confounds explicit.

## Execution override: local checkpoint is now available

Use this checkpoint only:

```text
model_id: Zyphra/Zamba2-1.2B-Instruct-v2
local_path: /202532803004/models/Zamba2-1.2B-Instruct-v2
source: user-provided local ModelScope download
```

The directory has already been verified by the user to contain `config.json`, `model.safetensors`, `tokenizer.json`, tokenizer metadata, and generation/config files. Do **not** attempt Hugging Face or ModelScope downloads in this round. Load locally/offline.

Important scientific correction: this is an **instruction-tuned** hybrid checkpoint, whereas earlier Mamba/Pythia controls are base language models. Therefore Round 031 is a hybrid stress-transfer probe, not a clean architecture-causal comparison. Always set `architecture_causality_confirmed=false` and record `instruction_tuning_confound=true`.

## Frozen protocol

- Hybrid checkpoint: `/202532803004/models/Zamba2-1.2B-Instruct-v2`
- Stress families: `structured_repetitive`, `periodic_pattern`, `interleaved_distractor`, `semantic_distractor`
- Negative control: `lexically_diverse`
- New seeds: `[65,66,67,68]`
- Target token budgets: `[256,768,1280,1792,2560,3584]`, clipped to the model's actual supported limit
- Compare within-model normalized degradation, not raw cross-model logits/accuracy
- Keep Attention KV cache and Mamba2 recurrent state as separate memory systems
- Do not infer `routing failure`, `state collision`, or `attention dilution` without intervention evidence

---

## Task 1 - Offline-load and runtime compatibility gate

Files:
- modify runner/model-loading code only as needed;
- add regression tests for explicit local checkpoint paths.

Requirements:

1. Resolve the exact local path above before any network path.
2. Use `local_files_only=True` wherever the Transformers API supports it.
3. Record Transformers/PyTorch versions and detected config architecture/model type.
4. Inspect `config.json` / AutoConfig first; do not silently force a Mamba or generic Transformer class.
5. If custom/model-specific code is required, use only code already present locally or supported by the installed Transformers version. Do not fetch remote Python code during the experiment.
6. Run one very short forward pass on GPU/CPU as appropriate.

If the checkpoint cannot be instantiated, stop honestly with `hybrid_runtime_incompatible` and record the exact exception. Do not fabricate results.

---

## Task 2 - Hybrid capability metadata

Record at minimum:

```json
{
  "architecture_family": "hybrid_ssm_attention",
  "checkpoint_variant": "instruct_v2",
  "instruction_tuning_confound": true,
  "model_type": "...",
  "num_parameters": null,
  "num_layers": null,
  "layer_types": null,
  "attention_layer_ids": null,
  "ssm_layer_ids": null,
  "cache_semantics": "hybrid"
}
```

Fill fields from actual config/runtime evidence only. Do not guess missing architecture metadata.

---

## Task 3 - Establish a valid remote-memory probe on this tokenizer/model

Before long-context evaluation, test only the predeclared candidate pairs in this order:

1. `(" red", " blue")`
2. `(" one", " two")`
3. `(" cat", " dog")`

For each candidate:

- verify each answer is a valid single tokenizer token in the exact prediction context;
- verify A/B prompts have matched actual token length;
- use direction-specific signed margins;
- require positive signed margins for both counterfactual prompt directions on all valid seeds at ~256 actual tokens.

Do not select a pair based on long-context behavior. If no predeclared pair passes, finish with `hybrid_task_invalid`.

Because this is an instruction-tuned model, do **not** automatically apply a chat template unless the frozen probe requires it. Prefer the same plain causal-completion format used by prior StateFuzz rounds; document any unavoidable formatting difference.

---

## Task 4 - Run frozen stress fingerprint

Write:
- `results/hybrid_stress_round_031.json`

For each family × budget × seed record:

- actual token count;
- signed margin A;
- signed margin B;
- memory signal `(dA-dB)/2`;
- lexical bias `(dA+dB)/2`;
- failure event `min_signed_margin <= 0`;
- stress family;
- runtime status;
- model metadata.

No new stress-family search in this round.

---

## Task 5 - Compare fingerprints, not raw model quality

Build per-family summaries:

- failure risk by actual token budget;
- normalized margin retention relative to this model's own shortest valid budget;
- first observed risk region or lower bound;
- monotonicity violations;
- negative-control behavior.

Compare descriptively with existing Mamba-130M and Pythia-160M artifacts.

Allowed classifications:

- `hybrid_more_robust_candidate`
- `hybrid_similar_stress_profile`
- `hybrid_distinct_stress_profile`
- `hybrid_more_sensitive_candidate`
- `inconclusive_due_to_scale_training_tuning_confounds`

Every classification must include:

```json
{
  "architecture_causality_confirmed": false,
  "scale_confound": true,
  "training_data_confound": true,
  "instruction_tuning_confound": true
}
```

---

## Task 6 - Inspect the two memory paths separately

On one short and one long valid example, inspect returned cache/state structures.

Record separately:

```json
{
  "attention_cache_observed": false,
  "ssm_recurrent_state_observed": false,
  "attention_cache_shapes": null,
  "ssm_state_shapes": null
}
```

Only mark a field observed from actual runtime objects. Never label a generic cache tensor as an SSM recurrent state.

Do not add architecture-specific state interventions in Round 031 unless the model API already exposes a safe, validated copy/restore path. If both memory paths are observable, propose Round 032 memory-path localization instead.

---

## Task 7 - Optional realistic-workload spot check

Only if Tasks 1-6 succeed, run one existing Round 027 workload: `code_context_dependency`.

Compare the strongest descriptive synthetic stress family with the `lexically_diverse` negative control. Do not create new workload templates.

---

## Task 8 - Result, tests, and handoff

Write:
- `results/result_round_031.json`
- `results/hybrid_stress_round_031.json` when evaluation succeeds
- update `status.json` with `next="gpt"`

Required result fields include:

```json
{
  "round": 31,
  "hybrid_model": "Zyphra/Zamba2-1.2B-Instruct-v2",
  "local_checkpoint": "/202532803004/models/Zamba2-1.2B-Instruct-v2",
  "checkpoint_source": "local_modelscope_download",
  "instruction_tuning_confound": true,
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
- `hybrid_task_invalid`
- `hybrid_runtime_incompatible`

## Verify

Run focused new tests first, then:

```bash
python -m pytest -q -o addopts=''
```

## Success

Round 031 succeeds if the locally mounted checkpoint is loaded offline, a valid short-context memory probe is established, the frozen stress/control fingerprint is measured, the two hybrid memory systems are kept conceptually separate, and all claims remain descriptive rather than architecture-causal because scale/training/tuning are unmatched.
