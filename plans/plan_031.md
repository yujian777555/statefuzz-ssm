# Plan 031 - Hybrid Architecture Memory-Stress Validation

> **For agentic workers:** Execute this plan task-by-task on the **user-provided experiment VM/container**. Follow `docs/PLANNER_EXECUTION_CONTRACT.md`. Do not substitute checkpoints post-hoc and do not interpret VM-local paths as paths on the Planner side or on GitHub.

**Goal:** Test whether StateFuzz memory-stress fingerprints transfer to a hybrid SSM-attention model using the checkpoint already downloaded on the user's experiment VM, while keeping instruction-tuning, scale, and training-data confounds explicit.

**Primary checkpoint:** `Zyphra/Zamba2-1.2B-Instruct-v2`

**Exact VM path:** `/202532803004/models/Zamba2-1.2B-Instruct-v2`

**Important execution meaning:** every reference to "local", "local checkpoint", or `local_files_only=True` in this plan means **local to the user's experiment VM/container where Codex executes the experiment commands**. It does **not** mean the Planner/ChatGPT runtime, GitHub, Codex cloud storage, or an arbitrary local workstation.

**Known checkpoint contents verified by the user:**

```text
.gitattributes
README.md
config.json
configuration.json
generation_config.json
model.safetensors
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

The directory size reported by the user is approximately `2.3G`.

**Scientific correction:** this is an **instruction-tuned** hybrid checkpoint, while the historical Mamba-130M and Pythia-160M controls are base language models. Therefore Round 031 is a hybrid stress-transfer probe, not a clean architecture-causal comparison. Always record `architecture_causality_confirmed=false` and `instruction_tuning_confound=true`.

---

## Task 0 - Prove that Codex is actually executing on the correct VM

This is a hard gate. Do this before changing experiment code or loading the model.

Run **inside the Codex execution shell on the experiment VM/container**:

```bash
set -e

echo "=== execution identity ==="
hostname
whoami
pwd
python --version
which python

echo "=== GPU ==="
nvidia-smi || true

echo "=== checkpoint path ==="
CHECKPOINT=/202532803004/models/Zamba2-1.2B-Instruct-v2
ls -lah "$CHECKPOINT"
test -d "$CHECKPOINT"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
test -f "$CHECKPOINT/tokenizer.json"
du -sh "$CHECKPOINT"
```

### Required interpretation

- If these commands succeed, record the actual `hostname`, Python executable, Python version, GPU information, checkpoint path, and directory size in the Round 031 result artifact.
- If `/202532803004/models/Zamba2-1.2B-Instruct-v2` is **not visible from the Codex process**, stop immediately with status:

```text
vm_checkpoint_path_not_visible
```

- Do **not** fall back to Hugging Face, ModelScope, another machine, another container, or another checkpoint.
- A missing mount/path is an infrastructure error, not a model failure.

### Offline execution policy

Before model loading, set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Do not invoke `modelscope download`, `hf download`, `git clone` for model code, or any checkpoint download during the scientific run.

---

## Task 1 - Inspect the existing runner before implementation

Current relevant files already in the repository:

- `src/statefuzz/runner/hf_causal_lm_runner.py`
- `src/statefuzz/runner/mamba_runner.py`
- `src/statefuzz/runner/model_matrix.py`
- `tests/runner/test_hf_causal_lm_runner.py`
- `tests/runner/test_mamba_runner.py`
- `tests/runner/test_model_matrix.py`

Important current behavior: `HFCausalLMRunner` is explicitly a Transformer behavior-control runner. Its `from_pretrained()` path currently forbids `trust_remote_code=True` and reports `architecture_role="transformer_control"`. Do **not** silently reuse that metadata for Zamba2.

### Preferred implementation boundary

Create a dedicated hybrid runner rather than changing the semantic meaning of the Transformer control runner:

- Create: `src/statefuzz/runner/hybrid_causal_lm_runner.py`
- Create: `tests/runner/test_hybrid_causal_lm_runner.py`
- Modify: `src/statefuzz/runner/__init__.py`
- Modify: `src/statefuzz/runner/model_matrix.py` only if the matrix needs an explicit hybrid role.

Suggested interface:

```python
@dataclass(frozen=True)
class HybridCausalLMExperimentConfig:
    model_id: str
    checkpoint_path: str
    device: str = "cuda"
    dtype: str = "float16"
    seed: int = 0
    local_files_only: bool = True
    trust_remote_code: bool = False


class HybridCausalLMRunner:
    @classmethod
    def from_pretrained(
        cls,
        config: HybridCausalLMExperimentConfig,
    ) -> "HybridCausalLMRunner": ...

    def single_token_id(self, text: str) -> int | None: ...
    def count_tokens(self, prompt: str) -> int: ...
    def score_candidate_tokens(...): ...
    def score_remote_memory_pair(...): ...
    def model_metadata(self) -> dict[str, Any]: ...
```

Behavioral scoring should match the already validated signed-margin logic used in previous rounds.

---

## Task 2 - Offline configuration/runtime compatibility gate

Before loading the full weights, inspect configuration locally from the VM path.

Run this **on the VM**:

```bash
CHECKPOINT=/202532803004/models/Zamba2-1.2B-Instruct-v2
python - <<'PY'
import json
from pathlib import Path

p = Path('/202532803004/models/Zamba2-1.2B-Instruct-v2')
config = json.loads((p / 'config.json').read_text())
print('model_type =', config.get('model_type'))
print('architectures =', config.get('architectures'))
print('auto_map =', config.get('auto_map'))
print('max_position_embeddings =', config.get('max_position_embeddings'))
print('num_hidden_layers =', config.get('num_hidden_layers'))
print('hybrid_layer_ids =', config.get('hybrid_layer_ids'))
PY
```

Then record runtime versions:

```bash
python - <<'PY'
import torch
import transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
```

### Full model-load smoke test

The primary scientific execution must use the VM path, not a remote model ID:

```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

checkpoint = "/202532803004/models/Zamba2-1.2B-Instruct-v2"

config = AutoConfig.from_pretrained(
    checkpoint,
    local_files_only=True,
    trust_remote_code=False,
)

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint,
    local_files_only=True,
    trust_remote_code=False,
)

model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    local_files_only=True,
    trust_remote_code=False,
)
```

Use `trust_remote_code=False` first. If the installed Transformers version cannot instantiate the architecture, inspect the exception and the locally available files. Do not enable remote fetching. `trust_remote_code=True` may only be used if all required custom Python implementation files are already physically present in the checkpoint directory and execution remains fully offline; record this explicitly if it becomes necessary.

Move the model to the selected device only after successful instantiation. Run a one-token/very-short forward pass and verify finite logits.

### Stop conditions

Stop with `hybrid_runtime_incompatible` if any of the following occurs and cannot be fixed without remote code/model downloads:

- model class unavailable in installed Transformers;
- required custom modeling Python file is absent locally;
- tokenizer cannot load from the VM path;
- model weights cannot be instantiated;
- deterministic short forward pass fails;
- CUDA/runtime incompatibility prevents execution.

Record the full exception type and concise message. Do not count this as scientific failure.

---

## Task 3 - Hybrid capability metadata

Do not guess architecture fields. Populate them from `config.json`, AutoConfig, and actual runtime objects.

Required metadata schema:

```json
{
  "architecture_family": "hybrid_ssm_attention",
  "checkpoint_name": "Zyphra/Zamba2-1.2B-Instruct-v2",
  "checkpoint_path": "/202532803004/models/Zamba2-1.2B-Instruct-v2",
  "checkpoint_location": "user_experiment_vm",
  "checkpoint_source": "user_downloaded_modelscope_directory",
  "offline_loading": true,
  "instruction_tuning_confound": true,
  "model_type": null,
  "architectures": null,
  "num_parameters": null,
  "num_layers": null,
  "layer_types": null,
  "attention_layer_ids": null,
  "ssm_layer_ids": null,
  "max_context_tokens": null,
  "cache_semantics": "hybrid",
  "architecture_causality_confirmed": false
}
```

If the architecture exposes different field names, preserve both the raw config field name and normalized interpretation in the result artifact.

---

## Task 4 - Establish a valid short-context remote-memory probe

Do not run long-context stress tests until this passes.

### Frozen ordered candidate pairs

Try only in this order:

1. `(" red", " blue")`
2. `(" one", " two")`
3. `(" cat", " dog")`

For each pair:

1. tokenize both candidates with `add_special_tokens=False`;
2. require exactly one token per candidate and distinct token IDs;
3. build A/B counterfactual prompts with identical local suffix and matched actual token count;
4. measure direction-specific signed margins:
   - prompt A: `logit(A) - logit(B)`
   - prompt B: `logit(B) - logit(A)`;
5. use new seeds `[65, 66, 67, 68]`;
6. at approximately 256 actual tokens require both signed margins to be positive for all valid seeds before accepting the pair as the primary memory probe.

Do not choose a candidate pair using long-context outcomes.

Because the checkpoint is instruction-tuned, do not automatically call `apply_chat_template()`. The default protocol is the same plain causal-completion format used by prior StateFuzz rounds. If the model only yields a valid probe under a chat template, record that as a protocol deviation and do not compare raw margins directly with historical base-model margins.

If no predeclared pair is valid, stop with `hybrid_task_invalid`.

---

## Task 5 - Frozen Hybrid stress fingerprint

Only execute after Tasks 0-4 succeed.

### Frozen model

```text
VM checkpoint path: /202532803004/models/Zamba2-1.2B-Instruct-v2
```

### Frozen seeds

```text
65, 66, 67, 68
```

### Frozen stress families

1. `structured_repetitive`
2. `periodic_pattern`
3. `interleaved_distractor`
4. `semantic_distractor`
5. `lexically_diverse` negative control

### Target actual-token budgets

```text
256, 768, 1280, 1792, 2560, 3584
```

Clip only if the model's actual supported context limit is lower. Always store the achieved tokenizer count rather than assuming nominal generator length.

### Required per-record fields

```json
{
  "seed": 65,
  "stress_family": "structured_repetitive",
  "target_budget": 1792,
  "actual_input_tokens": 0,
  "value_pair": [" red", " blue"],
  "signed_margin_a": 0.0,
  "signed_margin_b": 0.0,
  "min_signed_margin": 0.0,
  "memory_signal": 0.0,
  "lexical_bias": 0.0,
  "failure": false,
  "runtime_status": "ok"
}
```

Definitions:

```text
memory_signal = (dA - dB) / 2
lexical_bias = (dA + dB) / 2
failure = min(dA, dB) <= 0
```

Write raw and summarized results to:

```text
results/hybrid_stress_round_031.json
```

No new stress-family discovery and no post-hoc seed replacement in this round.

---

## Task 6 - Build a normalized stress fingerprint

Create:

- `src/statefuzz/analyzer/stress_fingerprint.py`
- `tests/analyzer/test_stress_fingerprint.py`

Interfaces:

```python
def build_stress_fingerprint(records): ...

def compare_stress_fingerprints(reference, candidate): ...
```

Per stress family summarize:

- short-context validity;
- failure rate by actual token budget;
- mean/median signed margin;
- normalized margin retention relative to the model's own shortest valid budget;
- first observed risk region or lower bound;
- monotonicity violations;
- negative-control behavior.

### Cross-model rule

Compare against historical Mamba-130M/Pythia artifacts using **within-model normalized degradation**. Do not compare raw logits as if models shared calibration.

Allowed descriptive labels:

- `hybrid_more_robust_candidate`
- `hybrid_similar_stress_profile`
- `hybrid_distinct_stress_profile`
- `hybrid_more_sensitive_candidate`
- `inconclusive_due_to_scale_training_tuning_confounds`

Every comparison must include:

```json
{
  "architecture_causality_confirmed": false,
  "scale_confound": true,
  "training_data_confound": true,
  "instruction_tuning_confound": true
}
```

---

## Task 7 - Inspect Hybrid memory-path observability without making mechanism claims

Use one short valid prompt and one long valid prompt.

Inspect actual output/cache objects and record separately:

```json
{
  "attention_cache_observed": false,
  "attention_cache_type": null,
  "attention_cache_shapes": null,
  "ssm_recurrent_state_observed": false,
  "ssm_state_type": null,
  "ssm_state_shapes": null
}
```

Rules:

- generic `past_key_values` is not automatically an SSM recurrent state;
- a mixed cache object must be inspected field-by-field;
- do not flatten Attention and SSM memory into one tensor metric;
- do not call observed degradation `routing failure`, `state collision`, or `attention dilution` without a controlled intervention.

If both memory paths are observable and copy/restore semantics are clear, leave the intervention experiment for Round 032 rather than expanding Round 031.

---

## Task 8 - Optional realistic-workload transfer spot check

Run only if the synthetic fingerprint completes.

Use one existing task only:

```text
code_context_dependency
```

Compare:

- strongest descriptively observed Hybrid stress family;
- `lexically_diverse` negative control.

This is a transfer spot check, not a new benchmark search. Record actual token counts and same candidate-scoring protocol.

---

## Task 9 - Result artifact and handoff

Write:

```text
results/result_round_031.json
results/hybrid_stress_round_031.json
```

and update:

```text
status.json
```

with:

```json
{
  "last_result": "results/result_round_031.json",
  "next": "gpt",
  "previous_actor": "codex",
  "round": 31
}
```

### Required top-level result fields

```json
{
  "round": 31,
  "status": "...",
  "execution_environment": {
    "expected_location": "user_experiment_vm",
    "hostname": "...",
    "python_executable": "...",
    "python_version": "...",
    "gpu": "..."
  },
  "hybrid_model": "Zyphra/Zamba2-1.2B-Instruct-v2",
  "local_checkpoint": "/202532803004/models/Zamba2-1.2B-Instruct-v2",
  "checkpoint_visible_from_executor": true,
  "checkpoint_source": "user_downloaded_modelscope_directory",
  "offline_loading": true,
  "instruction_tuning_confound": true,
  "architecture_causality_confirmed": false,
  "runtime_versions": {},
  "task_validity": {},
  "stress_fingerprint": {},
  "comparison_to_existing_models": {},
  "cache_state_observability": {},
  "realistic_spot_check": {},
  "claim_scope": "...",
  "failures": [],
  "tests": {}
}
```

Allowed statuses:

- `hybrid_validation_complete`
- `vm_checkpoint_path_not_visible`
- `hybrid_runtime_incompatible`
- `hybrid_task_invalid`

Do not use a scientific failure status for an infrastructure/runtime problem.

---

## Verification commands

Run focused tests first:

```bash
python -m pytest tests/runner/test_hybrid_causal_lm_runner.py -q
python -m pytest tests/analyzer/test_stress_fingerprint.py -q
```

Then full regression suite:

```bash
python -m pytest -q -o addopts=''
```

Then validate artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path

for name in [
    'results/result_round_031.json',
    'results/hybrid_stress_round_031.json',
]:
    p = Path(name)
    assert p.exists(), name
    data = json.loads(p.read_text())
    print(name, 'OK', type(data).__name__)
PY
```

If Round 031 stops before the stress sweep due to `vm_checkpoint_path_not_visible`, `hybrid_runtime_incompatible`, or `hybrid_task_invalid`, `results/hybrid_stress_round_031.json` may be absent; document why in `results/result_round_031.json`.

---

## Scientific success criteria

Round 031 is successful if:

1. Codex proves it is executing in an environment that can see `/202532803004/models/Zamba2-1.2B-Instruct-v2` on the user's experiment VM.
2. The checkpoint is loaded offline without silently reaching Hugging Face or ModelScope.
3. Runtime/model incompatibilities are separated from behavioral failures.
4. A valid short-context remote-memory probe is established before long-context testing.
5. All five frozen stress/control families are evaluated if the probe is valid.
6. The Hybrid model receives a normalized stress fingerprint based on its own baseline.
7. Attention-cache and SSM-state observability are reported separately.
8. Instruction-tuning, parameter-scale, and training-data confounds remain explicit.
9. No architecture-causal claim is made from this unmatched checkpoint.
10. All new and regression tests pass, or any blocking failure is reported with an exact reproducible cause.

## Claim boundary

A positive Round 031 result may support:

> StateFuzz can transfer its controlled remote-memory stress methodology to an instruction-tuned Hybrid Mamba2+Attention checkpoint and characterize whether its stress-response fingerprint is similar to or different from previously tested model families.

It does **not** by itself support:

- Hybrid architecture is inherently better/worse than pure Mamba;
- Attention fixes SSM memory failure;
- Mamba2 fixes first-generation Mamba weaknesses;
- the observed difference is caused by routing;
- the result generalizes to all Hybrid models or all SSMs.
