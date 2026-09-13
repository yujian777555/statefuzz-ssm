# Plan 031 - Hybrid Architecture Memory-Stress Validation

> **For agentic workers:** REQUIRED SUB-SKILL: use the repository's normal agentic execution workflow and execute this plan task-by-task on the **user-provided experiment VM/container**. Follow `docs/PLANNER_EXECUTION_CONTRACT.md`. Do not substitute checkpoints post-hoc. Do not interpret VM-local paths as paths on the Planner side, GitHub, or another Codex machine.

**Goal:** Test whether StateFuzz memory-stress fingerprints transfer to a hybrid SSM-attention model using the checkpoint already downloaded on the user's experiment VM, with exact metric semantics, exact token-budget gates, and explicit instruction-tuning/scale/training-data confounds.

**Architecture:** Use a dedicated Hybrid causal-LM runner, but reuse the repository's already validated remote-memory metric implementation rather than redefining the signed-margin decomposition. Every nominal token budget must be fitted against the **Zamba2 tokenizer's actual token count**. Runtime compatibility, short-context validity, long-context stress behavior, and cache/state observability remain separate gates.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, pytest, existing StateFuzz remote-memory generator/analyzer modules.

**Spec:** `docs/PLANNER_EXECUTION_CONTRACT.md`, `src/statefuzz/analyzer/memory_dependence.py`, `src/statefuzz/generator/remote_memory.py`, `results/result_round_030.json`.

## Revision note - metric and short-context gate correction

This revision supersedes the earlier Round 031 wording in two places.

1. The previous plan accidentally wrote the decomposition formulas as if `dB` were a raw A-minus-B preference on Prompt B. It is not. In StateFuzz, both directional margins are **direction-correct signed margins**:

```text
dA = logit(A | Prompt A) - logit(B | Prompt A)
dB = logit(B | Prompt B) - logit(A | Prompt B)
```

Therefore the correct decomposition is:

```text
memory_signal = (dA + dB) / 2
lexical_bias  = (dA - dB) / 2
```

This is algebraically identical to the existing implementation in `decompose_pairwise_preference()`, which converts Prompt B back to raw A-minus-B preference before decomposition.

2. A previous exploratory gate used about **850 actual tokens**. That does **not** satisfy the Round 031 short-context gate and must not be cited as compliant completion evidence. The primary short-context gate is now frozen to:

```text
target_tokens = 256
tolerance_tokens = 8
acceptable actual-token interval = [248, 264]
```

Any 850-token scan may be retained only as `exploratory_noncompliant_scan`; it cannot replace the required 256-token gate.

---

## Global Constraints

- **Execution machine:** user-provided experiment VM/container where Codex runs experiment commands.
- **Checkpoint:** `Zyphra/Zamba2-1.2B-Instruct-v2`.
- **Exact VM path:** `/202532803004/models/Zamba2-1.2B-Instruct-v2`.
- **Checkpoint location meaning:** path is local to the experiment VM/container, not Planner/ChatGPT, GitHub, or an unrelated Codex environment.
- **Remote checkpoint download during scientific execution:** forbidden.
- **Offline flags:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`.
- **Loading:** `local_files_only=True` wherever supported.
- **Instruction-tuning confound:** `true`.
- **Architecture causality confirmed:** `false`.
- **Frozen seeds:** `[65, 66, 67, 68]`.
- **Frozen stress families:** `structured_repetitive`, `periodic_pattern`, `interleaved_distractor`, `semantic_distractor`.
- **Negative control:** `lexically_diverse`.
- **Frozen target budgets:** `[256, 768, 1280, 1792, 2560, 3584]` actual tokenizer tokens.
- **Budget tolerance:** `±8` tokens for every target. A target outside tolerance is `budget_unreachable`, not an approximate success.
- **Frozen candidate-pair order:** `(" red", " blue")`, `(" one", " two")`, `(" cat", " dog")`.
- Candidate pair selection must occur from tokenizer/short-context validity only, never from favorable long-context results.
- Attention KV cache and Mamba2 recurrent state are separate memory systems.
- Runtime/path/OOM/tokenizer failures are not model memory failures.
- Do not claim architecture superiority/inferiority from unmatched checkpoints.

---

## Task 0 - Prove Codex is executing on the correct experiment VM

This is a hard gate before code changes or model execution.

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

echo "=== checkpoint ==="
CHECKPOINT=/202532803004/models/Zamba2-1.2B-Instruct-v2
ls -lah "$CHECKPOINT"
test -d "$CHECKPOINT"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
test -f "$CHECKPOINT/tokenizer.json"
du -sh "$CHECKPOINT"
```

Record in `results/result_round_031.json`:

```json
{
  "execution_environment": {
    "expected_location": "user_experiment_vm",
    "hostname": "...",
    "python_executable": "...",
    "python_version": "...",
    "gpu": "..."
  },
  "local_checkpoint": "/202532803004/models/Zamba2-1.2B-Instruct-v2",
  "checkpoint_visible_from_executor": true
}
```

If the path is not visible from the actual Codex process, stop with:

```text
vm_checkpoint_path_not_visible
```

Do not download or substitute another checkpoint.

Before model loading:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

---

## Task 1 - Lock metric semantics with regression tests before Hybrid execution

**Files:**
- Existing implementation: `src/statefuzz/analyzer/memory_dependence.py`
- Modify tests if necessary: existing memory-dependence analyzer test file
- Hybrid runner must consume these existing functions rather than copying formulas.

**Canonical primitives:**

For the two candidate IDs `A` and `B`, store all four raw logits:

```text
l_AA = logit(A | Prompt A)
l_BA = logit(B | Prompt A)
l_AB = logit(A | Prompt B)
l_BB = logit(B | Prompt B)
```

Then define:

```text
dA = l_AA - l_BA
dB = l_BB - l_AB
```

Both `dA` and `dB` are direction-correct signed margins: positive means the model prefers the counterfactually correct answer on that prompt.

Derived metrics:

```text
memory_signal = (dA + dB) / 2
lexical_bias  = (dA - dB) / 2
min_signed_margin = min(dA, dB)
failure = min_signed_margin <= 0
```

Equivalent raw-preference formulation, for cross-checking only:

```text
raw_preference_prompt_a = dA                    # A - B on Prompt A
raw_preference_prompt_b = -dB                   # A - B on Prompt B
memory_signal = (raw_preference_prompt_a - raw_preference_prompt_b) / 2
lexical_bias  = (raw_preference_prompt_a + raw_preference_prompt_b) / 2
```

**Lexical-bias sign convention:** positive `lexical_bias` means shared A-favoring bias; negative means shared B-favoring bias.

### Required sanity tests

- [ ] **Symmetric correct memory test**

Synthetic directional margins:

```text
dA = +2
dB = +2
```

Expected:

```text
memory_signal = +2
lexical_bias = 0
```

This exact case guards against the erroneous previous plan formula.

- [ ] **Memory plus A lexical bias test**

```text
dA = +3
dB = +1
```

Expected:

```text
memory_signal = +2
lexical_bias = +1
```

- [ ] **Memory plus B lexical bias test**

```text
dA = +1
dB = +3
```

Expected:

```text
memory_signal = +2
lexical_bias = -1
```

- [ ] Verify the tests exercise `compute_pairwise_memory_metrics()` and `decompose_pairwise_preference()` rather than a second ad-hoc implementation.

- [ ] Run the focused analyzer tests and confirm PASS before continuing.

**Important:** historical artifacts that used the repository's existing `decompose_pairwise_preference()` do not need reinterpretation merely because the earlier *plan text* was wrong. If any Round 031 prototype code copied the wrong formula, discard/recompute those derived fields from stored raw logits.

---

## Task 2 - Implement a dedicated Hybrid runner without changing Transformer-control semantics

**Existing files to inspect:**
- `src/statefuzz/runner/hf_causal_lm_runner.py`
- `src/statefuzz/runner/mamba_runner.py`
- `src/statefuzz/runner/model_matrix.py`
- `tests/runner/test_hf_causal_lm_runner.py`
- `tests/runner/test_mamba_runner.py`

**Preferred files:**
- Create: `src/statefuzz/runner/hybrid_causal_lm_runner.py`
- Create: `tests/runner/test_hybrid_causal_lm_runner.py`
- Modify: `src/statefuzz/runner/__init__.py`
- Modify `src/statefuzz/runner/model_matrix.py` only if required for an explicit Hybrid role.

Do not relabel the existing `HFCausalLMRunner`, which is explicitly a Transformer control runner.

Suggested config:

```python
@dataclass(frozen=True)
class HybridCausalLMExperimentConfig:
    model_id: str = "Zyphra/Zamba2-1.2B-Instruct-v2"
    checkpoint_path: str = "/202532803004/models/Zamba2-1.2B-Instruct-v2"
    device: str = "cuda"
    dtype: str = "float16"
    seed: int = 0
    local_files_only: bool = True
    trust_remote_code: bool = False
```

Required behavior methods:

```python
single_token_id(text: str) -> int | None
count_tokens(prompt: str) -> int
score_candidate_tokens(prompt: str, candidate_token_ids: list[int]) -> dict
score_remote_memory_pair(pair, ...) -> dict
model_metadata() -> dict
```

For memory metrics, call/reuse canonical analyzer functions from `memory_dependence.py`; do not reimplement the decomposition inside the runner.

---

## Task 3 - Offline config/model runtime gate

Run on the experiment VM:

```bash
CHECKPOINT=/202532803004/models/Zamba2-1.2B-Instruct-v2
python - <<'PY'
import json
from pathlib import Path
p = Path('/202532803004/models/Zamba2-1.2B-Instruct-v2')
cfg = json.loads((p/'config.json').read_text())
for key in (
    'model_type', 'architectures', 'auto_map', 'max_position_embeddings',
    'num_hidden_layers', 'hybrid_layer_ids'
):
    print(key, '=', cfg.get(key))
PY
```

Record versions:

```bash
python - <<'PY'
import torch, transformers
print('torch', torch.__version__)
print('transformers', transformers.__version__)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0))
PY
```

Primary local load:

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

Use `trust_remote_code=False` first. If the installed Transformers version cannot instantiate the architecture, inspect the exact exception and local files. Do not enable network access. `trust_remote_code=True` is permitted only if every required custom Python implementation file already exists physically inside the mounted checkpoint/repository environment; record the deviation.

Then run a deterministic very-short forward pass and verify finite logits.

Stop with `hybrid_runtime_incompatible` if the checkpoint cannot be instantiated offline. This is an infrastructure/runtime result, not a memory failure.

---

## Task 4 - Record Hybrid capability metadata from evidence only

Required schema:

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

Populate only from actual config/runtime evidence. Preserve raw config field names if normalized interpretation requires inference.

---

## Task 5 - Hard 256-token short-context validity gate

No long-context stress result is admissible until this gate passes.

### Candidate-pair order

Try only:

1. `(" red", " blue")`
2. `(" one", " two")`
3. `(" cat", " dog")`

Do not reorder based on outcomes.

### Token-budget fitting

Use the existing:

```python
fit_remote_memory_pair_to_token_budget(
    token_counter=runner.count_tokens,
    target_tokens=256,
    tolerance_tokens=8,
    ...
)
```

Hard acceptance conditions for **every seed** `[65,66,67,68]`:

```text
fit.status == "ok"
248 <= fit.actual_tokens <= 264
count_tokens(prompt_a) == count_tokens(prompt_b) == fit.actual_tokens
candidate A is exactly one token
candidate B is exactly one token
candidate token IDs are distinct
dA > 0
dB > 0
all logits/margins finite
```

If the fitter returns `budget_unreachable`, do not substitute 850 tokens or another convenient length. Record the exact fit failure.

### Prior ~850-token evidence

Any already generated ~850-token gate/scan must be labeled:

```json
{
  "protocol_role": "exploratory_noncompliant_scan",
  "counts_as_round031_short_context_gate": false
}
```

It may help debug runtime behavior but cannot establish Round 031 short-context validity.

### Candidate selection

Select the **first** predeclared pair for which all four seeds satisfy the 256±8-token gate. Do not inspect long-context results before pair selection.

Because the checkpoint is instruction-tuned, default to the same plain causal-completion format used by earlier StateFuzz rounds; do not automatically apply a chat template. If only a chat-formatted probe is valid, record `prompt_format_protocol_deviation=true` and do not compare raw margins with base-model margins.

If no candidate pair passes, stop with `hybrid_task_invalid`. If failure is specifically inability to realize the 256±8 token budget, distinguish it in the artifact as `short_context_budget_unreachable` rather than model memory failure.

---

## Task 6 - Frozen Hybrid stress-family sweep with exact token-budget fitting

Only execute after Tasks 0-5 succeed.

### Frozen model

```text
/202532803004/models/Zamba2-1.2B-Instruct-v2
```

### Frozen seeds

```text
65, 66, 67, 68
```

### Families

```text
structured_repetitive
periodic_pattern
interleaved_distractor
semantic_distractor
lexically_diverse  # negative control
```

### Target budgets

```text
256, 768, 1280, 1792, 2560, 3584
```

For **each family × budget × seed**, call the actual-token fitter with:

```text
tolerance_tokens = 8
```

Accept a record only when the achieved actual token count lies within target±8 and Prompt A/B counts match exactly. Otherwise record `budget_unreachable` for that cell; do not silently use a materially different length.

If a target exceeds the model's verified supported context, record `unsupported_by_model_context_limit` rather than clipping it into a different target and pretending equivalence.

### Required raw per-record fields

```json
{
  "seed": 65,
  "stress_family": "structured_repetitive",
  "target_budget": 1792,
  "budget_tolerance": 8,
  "actual_input_tokens": 1790,
  "value_pair": [" red", " blue"],
  "candidate_token_ids": [0, 0],
  "logit_prompt_a_candidate_a": 0.0,
  "logit_prompt_a_candidate_b": 0.0,
  "logit_prompt_b_candidate_a": 0.0,
  "logit_prompt_b_candidate_b": 0.0,
  "direction_a_margin": 0.0,
  "direction_b_margin": 0.0,
  "raw_preference_prompt_a": 0.0,
  "raw_preference_prompt_b": 0.0,
  "memory_signal": 0.0,
  "lexical_bias": 0.0,
  "min_signed_margin": 0.0,
  "failure": false,
  "runtime_status": "ok"
}
```

Canonical computation:

```text
dA = logit_prompt_a_candidate_a - logit_prompt_a_candidate_b
dB = logit_prompt_b_candidate_b - logit_prompt_b_candidate_a
memory_signal = (dA + dB) / 2
lexical_bias = (dA - dB) / 2
min_signed_margin = min(dA, dB)
failure = min_signed_margin <= 0
```

All four raw logits must be preserved so every derived metric can be recomputed independently.

Write:

```text
results/hybrid_stress_round_031.json
```

No new stress-family discovery, no post-hoc seed replacement, and no substitution of previously scanned noncompliant token lengths.

---

## Task 7 - Build normalized stress fingerprints

**Files:**
- Create: `src/statefuzz/analyzer/stress_fingerprint.py`
- Create: `tests/analyzer/test_stress_fingerprint.py`

Suggested interfaces:

```python
def build_stress_fingerprint(records): ...
def compare_stress_fingerprints(reference, candidate): ...
```

Per family summarize:

- short-context validity;
- exact available budget cells and any unreachable cells;
- failure rate by actual token budget;
- mean/median `memory_signal`;
- mean/median `min_signed_margin`;
- mean `lexical_bias` as a diagnostic, not memory performance;
- normalized memory-signal/margin retention relative to this model's own shortest compliant budget;
- first observed risk region or lower bound;
- monotonicity violations;
- negative-control behavior.

Cross-model comparisons against historical Mamba-130M/Pythia artifacts must use within-model normalized degradation. Do not compare raw logits as calibrated cross-model quantities.

Allowed descriptive labels:

```text
hybrid_more_robust_candidate
hybrid_similar_stress_profile
hybrid_distinct_stress_profile
hybrid_more_sensitive_candidate
inconclusive_due_to_scale_training_tuning_confounds
```

Every comparison must contain:

```json
{
  "architecture_causality_confirmed": false,
  "scale_confound": true,
  "training_data_confound": true,
  "instruction_tuning_confound": true
}
```

---

## Task 8 - Inspect Hybrid memory paths separately

Use one **compliant short-budget** example and one **compliant long-budget** example.

Record:

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
- mixed cache objects must be inspected field-by-field;
- do not flatten Attention and SSM memory into one metric;
- no mechanism label such as `routing failure`, `state collision`, or `attention dilution` without controlled intervention;
- if both paths are observable and safe copy/restore semantics exist, leave causal path localization to Round 032.

---

## Task 9 - Optional realistic-workload spot check

Run only if the synthetic fingerprint completes.

Use one existing Round 027 workload:

```text
code_context_dependency
```

Compare the strongest descriptively observed Hybrid stress family with `lexically_diverse`. This is a descriptive transfer check, not new benchmark search. Record actual tokenizer counts and the same raw-logit/signed-margin schema.

---

## Task 10 - Result artifact, verification, and handoff

Write:

```text
results/result_round_031.json
results/hybrid_stress_round_031.json
```

Required top-level fields:

```json
{
  "round": 31,
  "status": "...",
  "execution_environment": {},
  "hybrid_model": "Zyphra/Zamba2-1.2B-Instruct-v2",
  "local_checkpoint": "/202532803004/models/Zamba2-1.2B-Instruct-v2",
  "checkpoint_visible_from_executor": true,
  "offline_loading": true,
  "instruction_tuning_confound": true,
  "metric_definition": {
    "direction_a_margin": "logit(A|PromptA)-logit(B|PromptA)",
    "direction_b_margin": "logit(B|PromptB)-logit(A|PromptB)",
    "memory_signal": "(dA+dB)/2",
    "lexical_bias": "(dA-dB)/2"
  },
  "short_context_gate": {
    "target_tokens": 256,
    "tolerance_tokens": 8,
    "acceptable_interval": [248, 264],
    "passed": false
  },
  "selected_value_pair": null,
  "stress_fingerprint": {},
  "comparison_to_existing_models": {},
  "cache_state_observability": {},
  "realistic_spot_check": {},
  "claim_scope": "...",
  "tests": {}
}
```

Allowed final statuses include:

```text
hybrid_validation_complete
vm_checkpoint_path_not_visible
hybrid_runtime_incompatible
hybrid_task_invalid
short_context_budget_unreachable
```

Update `status.json` only after producing the result artifact:

```json
{
  "last_plan": "plans/latest_plan.md",
  "last_result": "results/result_round_031.json",
  "next": "gpt",
  "previous_actor": "codex",
  "round": 31
}
```

### Verification commands

Run focused metric and runner tests first, then full suite:

```bash
python -m pytest tests/runner/test_hybrid_causal_lm_runner.py -q
# Run the exact existing/new analyzer test file containing the metric regression cases.
python -m pytest tests/analyzer -q
python -m pytest -q -o addopts=''
```

Also validate artifacts:

```bash
python - <<'PY'
import json
from pathlib import Path
for name in ['results/result_round_031.json', 'results/hybrid_stress_round_031.json']:
    p = Path(name)
    print(name, 'exists=', p.exists())
    if p.exists():
        json.loads(p.read_text())
        print(name, 'json=ok')
PY
```

## Success Criteria

Round 031 is compliant only if:

1. Codex proves it is running where the VM checkpoint path is visible.
2. Zamba2 loads offline from the exact VM path.
3. Metric semantics are verified by regression tests, including `dA=dB=+2 -> memory_signal=2, lexical_bias=0`.
4. The selected candidate pair passes the **256±8 actual-token** short-context gate on all four seeds.
5. Any prior ~850-token scan is excluded from compliant short-context evidence.
6. Every stress-budget record either lies within target±8 tokens or is explicitly marked unreachable/unsupported.
7. All four raw logits, both directional margins, memory signal, lexical bias, and minimum signed margin are retained.
8. Cross-model comparison uses within-model normalized degradation rather than raw model quality.
9. Attention-cache and SSM-state observations remain separate.
10. No architecture-causal claim is made from unmatched model scale/training/tuning.
11. Tests pass and artifacts are valid JSON.

## Claim Boundary

A successful Round 031 can support:

> StateFuzz's controlled remote-memory stress protocol transfers to the tested `Zyphra/Zamba2-1.2B-Instruct-v2` hybrid checkpoint, and its within-model stress-response fingerprint can be characterized under compliant actual-token budgets.

It does **not** establish that Hybrid architecture itself causes the observed robustness/sensitivity difference, because checkpoint size, training data, and instruction tuning are unmatched.
