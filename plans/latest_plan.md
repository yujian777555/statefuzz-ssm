# Latest Plan

See `plans/plan_031.md` for the full executable plan.

Also follow:

```text
docs/PLANNER_EXECUTION_CONTRACT.md
```

# Round 031 - Authoritative Corrections

## 1. Metric semantics

The canonical StateFuzz directional margins are both **direction-correct signed margins**:

```text
dA = logit(A | Prompt A) - logit(B | Prompt A)
dB = logit(B | Prompt B) - logit(A | Prompt B)
```

Therefore:

```text
memory_signal = (dA + dB) / 2
lexical_bias  = (dA - dB) / 2
min_signed_margin = min(dA, dB)
failure = min_signed_margin <= 0
```

Sanity check that must be tested:

```text
dA = +2, dB = +2
=> memory_signal = +2
=> lexical_bias = 0
```

The previous Round 031 plan text had these two decomposition formulas reversed. That wording is superseded. The repository's existing `src/statefuzz/analyzer/memory_dependence.py` implementation already uses the correct equivalent decomposition by first converting Prompt B back to raw A-minus-B preference.

All Round 031 records must preserve the four primitive logits:

```text
logit(A | Prompt A)
logit(B | Prompt A)
logit(A | Prompt B)
logit(B | Prompt B)
```

plus `dA`, `dB`, `memory_signal`, `lexical_bias`, and `min_signed_margin` so derived metrics can be independently recomputed.

## 2. Short-context gate is strictly 256±8 actual tokens

A previously observed ~850-token scan is **not compliant evidence** for the Round 031 short-context gate.

Required gate:

```text
target_tokens = 256
tolerance_tokens = 8
acceptable actual tokens = 248..264 inclusive
```

Use the existing `fit_remote_memory_pair_to_token_budget()` with the **actual Zamba2 tokenizer**.

For every seed `[65,66,67,68]`, the selected pair must satisfy:

```text
fit.status == "ok"
248 <= actual_input_tokens <= 264
Prompt A token count == Prompt B token count
candidate A = one token
candidate B = one token
candidate IDs distinct
dA > 0
dB > 0
all logits finite
```

If 256±8 cannot be reached, record `short_context_budget_unreachable`; do not substitute 850 or another convenient length.

The old ~850-token result may be retained only as:

```json
{
  "protocol_role": "exploratory_noncompliant_scan",
  "counts_as_round031_short_context_gate": false
}
```

## 3. All long-context budgets also use actual-token fitting

Frozen targets:

```text
256, 768, 1280, 1792, 2560, 3584
```

Each target has tolerance `±8` actual tokenizer tokens. Any cell outside tolerance is `budget_unreachable`; do not silently accept materially different lengths.

# Execution VM / checkpoint

Codex must execute on the user-provided experiment VM/container where this path is visible:

```text
/202532803004/models/Zamba2-1.2B-Instruct-v2
```

This means the filesystem inside that VM/container, not Planner/ChatGPT, GitHub, or another Codex machine.

Before execution:

```bash
set -e
hostname
whoami
pwd
python --version
which python
nvidia-smi || true

CHECKPOINT=/202532803004/models/Zamba2-1.2B-Instruct-v2
ls -lah "$CHECKPOINT"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
test -f "$CHECKPOINT/tokenizer.json"
du -sh "$CHECKPOINT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

Use `local_files_only=True` where supported. No model download or checkpoint substitution is allowed during the scientific run.

# Scientific scope

Checkpoint:

```text
Zyphra/Zamba2-1.2B-Instruct-v2
```

Because this is instruction tuned and historical Mamba/Pythia controls are unmatched base checkpoints:

```text
instruction_tuning_confound = true
architecture_causality_confirmed = false
scale_confound = true
training_data_confound = true
```

Round 031 is a Hybrid stress-transfer/fingerprint experiment, not clean architecture causality.

# Required execution order

```text
VM/path visibility
-> offline runtime compatibility
-> metric-regression tests
-> exact 256±8 short-context validity
-> exact-budget frozen stress sweep
-> normalized within-model fingerprint
-> separate Attention-cache / SSM-state observability
-> optional code_context_dependency spot check
-> result JSON + full tests + handoff
```

Do not continue past a failed hard gate.

# Required artifacts

```text
results/result_round_031.json
results/hybrid_stress_round_031.json
```

The result must explicitly include:

- actual VM/runtime identity;
- exact checkpoint path;
- metric definitions;
- short-gate target/tolerance/actual counts;
- whether any old 850-token scan was excluded from compliant evidence;
- selected pair;
- raw logits and directional margins;
- stress fingerprint;
- cache/state observability;
- runtime failures separate from scientific failures;
- tests.

Next executor: codex
