# Latest Plan

See `plans/plan_031.md`.

# Round 030 Review

Round 030 did not produce larger-model scientific results because checkpoint access was blocked. That remains an infrastructure limitation, not evidence that scale removes the phenomenon.

# Round 031 Local Checkpoint Update

A hybrid checkpoint is now available locally on the execution VM:

```text
Zyphra/Zamba2-1.2B-Instruct-v2
/202532803004/models/Zamba2-1.2B-Instruct-v2
```

The user verified the directory contains:
- `config.json`
- `model.safetensors`
- `tokenizer.json`
- tokenizer metadata
- generation/configuration files

Executor requirements:

1. Use the exact local path above; do not download from Hugging Face or ModelScope.
2. Prefer offline/local loading (`local_files_only=True` where supported).
3. First perform a runtime compatibility and short forward-pass gate.
4. Because this is an **instruction-tuned** Zamba2 variant rather than the originally planned base checkpoint, record `instruction_tuning_confound=true`.
5. Treat Round 031 as a hybrid stress-transfer probe, not an architecture-causal comparison against base Mamba/Pythia checkpoints.
6. Establish short-context remote-memory validity before any long-context sweep.
7. Run the frozen stress families and negative control only after validity passes.
8. Compare within-model normalized degradation, not raw cross-model quality.
9. Keep Attention KV cache and Mamba2 recurrent state separate in diagnostics.
10. If both memory paths are observable, leave causal path-localization for a possible Round 032.

The detailed execution contract is in `plans/plan_031.md`.

Next executor: codex
