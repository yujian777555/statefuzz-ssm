# Planner Execution Contract

This document defines the minimum detail required for every future Planner round in the StateFuzz project.

## 1. Execution environment must be explicit

Every plan must state where the Executor/Codex is expected to run each step.

For VM experiments, the wording must be explicit:

> Run on the user-provided experiment VM. Paths such as `/202532803004/...` refer to the filesystem inside that VM/container, not the Planner environment, not GitHub, and not a generic local machine.

If a path is VM-local, the plan must never abbreviate it as merely "local" without naming the VM execution context.

## 2. Model checkpoints must use exact paths

Every model task must specify:

- model/checkpoint identity;
- exact VM absolute path if available;
- whether the checkpoint is base or instruction-tuned;
- whether remote download is allowed;
- exact offline loading flags;
- what to do when the path is not visible from the Executor process.

Example:

```text
Model: Zyphra/Zamba2-1.2B-Instruct-v2
VM path: /202532803004/models/Zamba2-1.2B-Instruct-v2
Remote download: forbidden for the primary execution path
Required loading: local_files_only=True
```

## 3. Environment visibility must be verified before expensive work

Before loading a large model, the Executor must run on the target VM:

```bash
pwd
hostname
python --version
nvidia-smi
ls -lah <checkpoint_path>
test -f <checkpoint_path>/config.json
test -f <checkpoint_path>/model.safetensors
```

If the checkpoint path is not visible, stop and report `vm_checkpoint_path_not_visible` rather than attempting a remote download or silently substituting another model.

## 4. Plans must include exact implementation targets

Each task must specify:

- files to create/modify;
- functions/classes/interfaces to add;
- expected inputs/outputs;
- exact experiment parameters;
- exact output artifact paths;
- unit/integration tests;
- scientific success/failure criteria.

Avoid vague instructions such as "support hybrid models" or "run large-scale tests" without filenames, interfaces, and measurable endpoints.

## 5. Experimental protocols must be frozen before execution

Plans must define before execution:

- model(s);
- seeds;
- stress families;
- token budgets;
- value pairs/candidate selection rules;
- primary metrics;
- negative controls;
- stopping rules.

Post-hoc model/task/template selection based on favorable results is not allowed unless explicitly labeled exploratory.

## 6. Runtime failures are not scientific failures

The plan must separate:

- checkpoint unavailable;
- path not mounted;
- incompatible software/runtime;
- OOM;
- tokenizer/task invalidity;
- actual model behavior failure.

Infrastructure failures must never be counted as memory failures.

## 7. Scientific claim boundaries must be written into every plan

Every plan must state:

- what a positive result would support;
- what it would not support;
- known confounds;
- whether architecture causality is established or only a candidate.

For unmatched model checkpoints, do not attribute behavioral differences solely to architecture.

## 8. Verification must be reproducible

Every plan must include exact verification commands and expected artifacts, including at minimum:

```bash
python -m pytest -q -o addopts=''
```

and experiment-specific checks for generated JSON/artifacts.

## 9. Handoff contract

Every completed Executor round must produce:

- `results/result_round_XXX.json`;
- scientific artifact(s) named in the plan;
- updated `status.json` with `next="gpt"`;
- a concise record of runtime failures and resolutions;
- exact model/checkpoint path(s) actually used.

## 10. Planner review standard

Before creating the next round, Planner must read:

1. `status.json`;
2. latest result JSON;
3. scientific result artifact(s);
4. research-relevant changed source files;
5. any runtime/checkpoint metadata needed to distinguish infrastructure issues from scientific evidence.

Future plans should prefer explicit, executable detail over short high-level summaries.

## 11. Metric semantics must be defined from primitive quantities

Whenever a plan introduces or reuses a derived scientific metric, it must define the metric from the primitive measured quantities before giving shorthand formulas.

For logit-based paired probes, a plan must specify at minimum:

- which prompt each logit comes from;
- which candidate each logit refers to;
- the sign convention of each directional margin;
- the formula for every derived metric;
- the interpretation of the sign;
- at least one numeric sanity-check case;
- which existing canonical implementation/function should be reused.

Do not write ambiguous formulas such as `(dA-dB)/2` unless `dA` and `dB` are defined immediately and unambiguously.

If two coordinate systems are used (for example direction-correct margins vs raw A-minus-B preferences), the plan must show the conversion explicitly.

For the current StateFuzz remote-memory probe, the canonical convention is:

```text
l_AA = logit(A | Prompt A)
l_BA = logit(B | Prompt A)
l_AB = logit(A | Prompt B)
l_BB = logit(B | Prompt B)

dA = l_AA - l_BA
dB = l_BB - l_AB

memory_signal = (dA + dB) / 2
lexical_bias  = (dA - dB) / 2
```

Sanity check:

```text
dA = +2, dB = +2
=> memory_signal = +2
=> lexical_bias = 0
```

Plans must require storage of primitive measurements when practical so derived metrics can be independently recomputed.

## 12. Token-budget language must be operational, not approximate

If a plan says a task runs at a target token budget, it must specify:

- which tokenizer defines the count;
- target token count;
- allowed tolerance;
- fitting method;
- behavior when the target cannot be reached;
- whether previously collected off-budget data is compliant, exploratory, or inadmissible.

Avoid phrases like "about 256 tokens" without a numeric tolerance.

For experiments using `fit_remote_memory_pair_to_token_budget`, the plan should give explicit `target_tokens` and `tolerance_tokens`, and store the achieved actual count in the result artifact.
