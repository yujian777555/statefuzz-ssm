# Latest Plan

See `plans/plan_031.md` for the full executable plan.

Also follow the persistent planning standard in:

```text
docs/PLANNER_EXECUTION_CONTRACT.md
```

# Round 031 Execution Context - IMPORTANT

The Executor/Codex must run the scientific experiment on the **user-provided experiment VM/container** that contains the downloaded model checkpoint.

The model path:

```text
/202532803004/models/Zamba2-1.2B-Instruct-v2
```

means the filesystem path **inside that experiment VM/container**.

It does **not** mean:

- the Planner/ChatGPT runtime;
- GitHub storage;
- Codex cloud-local storage on another machine;
- an arbitrary local workstation.

Before any experiment, Codex must prove from its own execution shell that this exact VM path is visible.

Required preflight commands:

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
test -d "$CHECKPOINT"
test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
test -f "$CHECKPOINT/tokenizer.json"
du -sh "$CHECKPOINT"
```

The user has already verified the directory is approximately `2.3G` and contains:

- `config.json`
- `configuration.json`
- `generation_config.json`
- `model.safetensors`
- `tokenizer.json`
- tokenizer metadata.

If Codex cannot see this path from its actual execution process, it must stop with:

```text
vm_checkpoint_path_not_visible
```

and must **not** download another checkpoint or silently run elsewhere.

# Offline Loading Requirement

For the Round 031 scientific run, set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

The primary execution path must use:

```text
/202532803004/models/Zamba2-1.2B-Instruct-v2
```

with `local_files_only=True` where supported.

No Hugging Face/ModelScope model download is allowed during the experiment.

# Implementation Direction

Do not repurpose the existing `HFCausalLMRunner` as if Hybrid were a Transformer control. Current `HFCausalLMRunner` is explicitly a Transformer behavior-control runner and reports `architecture_role="transformer_control"`.

Preferred new files:

```text
src/statefuzz/runner/hybrid_causal_lm_runner.py
tests/runner/test_hybrid_causal_lm_runner.py
src/statefuzz/analyzer/stress_fingerprint.py
tests/analyzer/test_stress_fingerprint.py
```

Reuse prior remote-memory generators/scoring logic, including direction-specific signed margins.

# Scientific Protocol

Checkpoint:

```text
Zyphra/Zamba2-1.2B-Instruct-v2
VM path: /202532803004/models/Zamba2-1.2B-Instruct-v2
```

This is an instruction-tuned Hybrid checkpoint. Therefore:

```text
instruction_tuning_confound = true
architecture_causality_confirmed = false
```

Frozen seeds:

```text
65, 66, 67, 68
```

Frozen stress families:

```text
structured_repetitive
periodic_pattern
interleaved_distractor
semantic_distractor
lexically_diverse   # negative control
```

Frozen target budgets:

```text
256, 768, 1280, 1792, 2560, 3584 actual tokens
```

Predeclared value-pair order:

```text
(" red", " blue")
(" one", " two")
(" cat", " dog")
```

The first pair that passes tokenizer and short-context validity is used. Do not select based on long-context outcomes.

# Execution Gates

Codex must pass these gates in order:

1. **VM path visibility**
2. **offline config/tokenizer/model load**
3. **short forward-pass runtime compatibility**
4. **short-context remote-memory validity**
5. **frozen stress-family sweep**
6. **normalized fingerprint comparison**
7. **separate Attention-cache vs SSM-state observability inspection**
8. optional `code_context_dependency` realistic spot check

Do not continue to later gates when an earlier gate fails.

# Required Artifacts

If evaluation completes:

```text
results/result_round_031.json
results/hybrid_stress_round_031.json
```

`results/result_round_031.json` must record at minimum:

- actual VM hostname;
- Python executable/version;
- GPU/runtime information;
- exact checkpoint path actually used;
- checkpoint visible from Executor: true/false;
- offline loading status;
- checkpoint variant;
- instruction-tuning confound;
- selected value pair;
- short-context validity;
- stress fingerprint summary;
- comparison to historical models;
- Attention-cache observability;
- SSM recurrent-state observability;
- runtime failures separately from scientific failures;
- tests.

# Claim Boundary

Round 031 may establish that StateFuzz transfers to this Hybrid checkpoint and characterize its stress-response fingerprint.

Round 031 must **not** claim that Hybrid architecture itself causes any robustness difference, because model scale, training data, and instruction tuning are unmatched.

# Persistent Planner Requirement

All future Planner rounds must follow `docs/PLANNER_EXECUTION_CONTRACT.md` and include, at minimum:

- explicit execution machine/environment;
- exact absolute model/data paths;
- exact files/functions to modify;
- exact frozen seeds/tasks/budgets/controls;
- exact commands to execute;
- result artifact schemas;
- infrastructure-vs-scientific failure separation;
- verification commands;
- success criteria;
- scientific claim boundaries.

Future plans should not rely on vague wording such as "load locally", "run on GPU", "test larger models", or "validate Hybrid" without specifying where, how, with what path, and how success/failure is decided.

Next executor: codex
