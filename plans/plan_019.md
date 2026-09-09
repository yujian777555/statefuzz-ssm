# Plan 019 - Causal State Validation After Prospective Risk Confirmation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether the observed structured-repetition remote-memory failure is causally linked to Mamba recurrent state representations after the Round 018 risk analysis.

**Architecture:** Round 019 must not search for a new failure condition. It freezes the validated Round 018 stress condition and performs minimal causal intervention on the already identified transition region. Behavioral differences remain primary; state analysis is used only to test mechanism.

**Tech Stack:** PyTorch, Hugging Face Transformers, existing StateFuzz runners/analyzers, pytest.

**Spec:** `results/result_round_018.json`, `plans/plan_018.md`.

## Global Constraints

- Do not claim all SSMs or all Transformers from one model pair.
- Freeze the stress condition from Round 018.
- Do not introduce new value pairs or filler styles.
- Separate behavioral confirmation from causal intervention.
- Preserve negative controls.

---

## Task 1: Validate Round 018 statistical result artifact

Files:
- Modify: `src/statefuzz/analyzer/failure_risk.py`
- Test: `tests/analyzer/test_failure_risk.py`

Implement helpers:

```python
summarize_seed_transition(records)
compare_paired_architecture_risk(mamba_records, transformer_records)
```

Required outputs:
- failure rate;
- Wilson interval;
- first-failure distribution;
- paired discordant counts;
- exact test result.

Verify:

```bash
pytest tests/analyzer/test_failure_risk.py -q
```

---

## Task 2: Add state intervention interface

Files:
- Modify: `src/statefuzz/runner/mamba_runner.py`
- Test: `tests/runner/test_mamba_runner.py`

Add a controlled interface:

```python
run_with_state_override(prompt, state_override=None)
```

Requirements:
- default path unchanged;
- intervention path explicitly records `state_source`;
- no silent state mutation.

---

## Task 3: State swap experiment

Frozen condition:

```
template_id=1
value_pair=(" red", " blue")
filler_style=structured_repetitive
```

Experiment:

A/B prompts:
- one short-context successful state;
- one long-context failure-risk state.

Interventions:

1. normal long-context state;
2. swap selected recurrent state from short-context run;
3. shuffled control state.

Measure:
- signed margin;
- pairwise memory score;
- recovery rate.

Do not interpret improvement unless behavioral recovery occurs.

---

## Task 4: Layer localization

Analyze only existing captured states.

Report:
- layers with largest A/B divergence;
- whether intervention-sensitive layers overlap;
- effect size.

Avoid claiming mechanism without intervention evidence.

---

## Task 5: Produce result artifact

Write:

`results/result_round_019.json`

Required:

```json
{
 "round":19,
 "frozen_condition":{},
 "risk_summary":{},
 "state_intervention":{},
 "causal_status":"..."
}
```

Allowed statuses:

- `state_intervention_supports_causal_role`
- `state_intervention_negative`
- `intervention_inconclusive`

---

## Verify

```bash
pytest -q
```

## Success

Round 019 succeeds when:

1. No condition drift occurs.
2. Intervention is explicitly controlled.
3. Behavioral recovery or non-recovery is measured.
4. The final claim remains limited to Mamba-130M under the frozen stress condition.
