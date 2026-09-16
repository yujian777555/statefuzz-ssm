# Round 37-C Plan: Scope Freeze and Paper Transition

## Purpose

Round37-A completed model discovery on the user-provided experiment VM and found only one currently available local checkpoint under `/202532803004/models`:

- `Zyphra/Zamba2-1.2B-Instruct-v2`
- `/202532803004/models/Zamba2-1.2B-Instruct-v2`

No Mamba2 checkpoint and no new modern pure-Attention checkpoint were available. Therefore Round37-B expanded evaluation MUST NOT run, because there is no new Planner-frozen model asset to evaluate.

Round37-C closes model-expansion work without inventing, downloading, or substituting checkpoints. It freezes the final paper scope and produces machine-readable claim boundaries for the paper-writing phase.

Follow `docs/PLANNER_EXECUTION_CONTRACT.md`.

---

## Execution environment

This round is repository analysis only. It does not require GPU inference.

Use the repository Python environment already used by Codex for tests. If `python` is not on PATH in the execution shell, use the same known working interpreter used in Round37-A when available:

```bash
/202532803004/conda_envs/amber/bin/python
```

Do not modify VM checkpoints.

---

## Frozen source evidence

Use only these repository artifacts as source evidence:

- `results/model_inventory_round_037.json`
- `results/result_round_036.json`
- `results/evidence_table_round_036.json`
- `results/figure_data_round_036.json`
- `results/paper_evidence_summary_round_036.json`
- `results/result_round_035.json`
- `results/stress_surface_round_035.json`
- `results/result_round_034.json`
- `configs/model_paths.json`

Historical Mamba/Pythia evidence may be referenced only through the already-packaged Round35/36 artifacts. Do not rerun or reinterpret raw historical experiments outside those frozen artifacts in this round.

---

## Frozen model scope

The final paper evidence scope for this repository version is:

1. `state-spaces/mamba-130m-hf` — historical pure-SSM reference already included in Round35/36 evidence.
2. `EleutherAI/pythia-160m` — historical pure-Attention reference already included in Round35/36 evidence.
3. `Zyphra/Zamba2-1.2B-Instruct-v2` — current Hybrid SSM+Attention checkpoint evaluated in Round31/34/35/36.

Planned but unavailable in Round37-A:

- Mamba2 checkpoint: unavailable on VM scan root.
- additional modern pure-Attention checkpoint: unavailable on VM scan root.

Do not fabricate model paths, revisions, parameter counts, or results for unavailable models.

---

## Claim boundary to freeze

### Supported paper-level claim

Use this as the principal supported scope:

> StateFuzz automatically characterizes architecture- and stress-dependent memory robustness surfaces in stateful sequence models, revealing non-uniform and potentially non-monotonic responses across evaluated architectures and stress families.

### Supported descriptive findings

- StateFuzz transfers to the tested Hybrid Zamba2 checkpoint as a behavioral stress-surface evaluation method.
- Different evaluated architectures show different descriptive margin-response profiles under shared stress families.
- The observed robustness response is not well described by a universal monotonic context-length degradation rule.
- Absence of a failure boundary in a tested range is itself valid negative evidence and must remain distinguishable from untested regions.

### Claims that remain unsupported

The following MUST be recorded as unsupported:

- universal failure boundaries across SSMs;
- general superiority or inferiority of Mamba, Pythia, or Zamba2;
- architecture-causal claims from cross-model comparisons;
- Mamba2 generalization;
- modern-Attention generalization beyond the historical Pythia reference;
- universal claims about all SSMs or all Hybrid architectures;
- Hybrid SSM-vs-Attention internal causal attribution;
- any claim that no failure exists outside evaluated token budgets.

---

## Task 1: Create scope-freeze artifact

Create:

`results/final_scope_round_037.json`

Required top-level schema:

```json
{
  "round": 37,
  "phase": "scope_freeze",
  "status": "complete",
  "evaluated_models": [],
  "planned_but_unavailable": [],
  "principal_claim": "",
  "supported_findings": [],
  "unsupported_claims": [],
  "evidence_sources": [],
  "paper_transition": {
    "ready": true,
    "next_phase": "paper_draft"
  }
}
```

For each evaluated model include, where available from frozen artifacts:

- `model_id`
- `architecture_role`
- `checkpoint_reference`
- `evidence_source`
- `evidence_scope`
- `known_limitations`

For each unavailable model include:

- `requested_role`
- `availability`: `unavailable_in_round037_inventory`
- `reason`
- `inventory_source`: `results/model_inventory_round_037.json`

Do not write null fields when a field is not required by this plan; where the source explicitly reports unknown revision, preserve `revision: null` rather than inventing one.

---

## Task 2: Create a final Round37 result summary

Create:

`results/result_round_037.json`

Required schema:

```json
{
  "round": 37,
  "status": "scope_frozen_paper_transition_ready",
  "model_expansion_attempted": true,
  "new_models_added": 0,
  "reason_no_new_models_added": "",
  "final_scope_artifact": "results/final_scope_round_037.json",
  "next_recommended_phase": "paper_draft",
  "tests": {}
}
```

`reason_no_new_models_added` must state that Round37-A found no Mamba2 or new pure-Attention checkpoint under the frozen VM scan root; do not describe this as a model-quality conclusion.

---

## Task 3: Add verification script

Create:

`scripts/round037_scope_freeze.py`

Supported command:

```bash
python scripts/round037_scope_freeze.py --verify
```

If `python` is unavailable, use the active repository interpreter or `/202532803004/conda_envs/amber/bin/python`.

`--verify` must fail with non-zero exit code if any of the following is true:

1. `results/final_scope_round_037.json` is missing or invalid JSON.
2. `results/result_round_037.json` is missing or invalid JSON.
3. `evaluated_models` does not contain the three frozen evidence roles: Mamba reference, Pythia reference, Zamba2 Hybrid.
4. `planned_but_unavailable` does not explicitly include Mamba2 and a modern pure-Attention baseline.
5. `paper_transition.ready` is not `true`.
6. unsupported claims omit any of the prohibited overclaims listed above.
7. the artifact falsely reports a newly evaluated Round37 model.
8. `new_models_added` is not `0`.

The verifier must only read files. It must not perform model inference, network access, checkpoint download, or GitHub writes.

---

## Task 4: Tests

Add focused tests for the verifier under the existing test layout, following repository conventions.

At minimum test:

- valid frozen-scope artifact passes;
- missing Mamba2 unavailable entry fails;
- unsupported universal-failure claim omission fails;
- fabricated newly evaluated model fails;
- `new_models_added != 0` fails.

Do not refactor unrelated code.

---

## Verify commands

Before completion run:

```bash
python scripts/round037_scope_freeze.py --verify
python -m pytest
```

If `python` is unavailable in the shell, substitute the active repository interpreter consistently for both commands.

Expected conditions before marking Round37 complete:

- verifier exits 0;
- full test suite passes;
- no network download occurred;
- no model inference occurred;
- no checkpoint path was fabricated;
- only scope-freeze / verifier / test / result files for this round were changed, plus `status.json` as the final handoff.

---

## Final status handoff

After successful verification, update `status.json` to exactly reflect:

```json
{
  "last_plan": "plans/latest_plan.md",
  "last_result": "results/result_round_037.json",
  "next": "gpt",
  "previous_actor": "codex",
  "round": 37,
  "phase": "scope_freeze_complete"
}
```

Do not advance to Round38 yourself.

---

## Success criterion

Round37-C succeeds only if the repository has a reproducible, machine-readable final scope stating:

- what was actually evaluated;
- what model expansion was attempted but unavailable;
- what claims are supported;
- what claims are not supported;
- that the next step is paper drafting rather than more unplanned experimentation.
