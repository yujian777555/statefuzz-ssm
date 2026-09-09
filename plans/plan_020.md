# Plan 020 - Disambiguate Recurrent-State Causality After Intervention Recovery

## Goal

Round 019 shows that overriding Mamba recurrent cache can recover behavior, but randomized cache also recovers behavior. The next step is not to claim causal memory restoration. The goal is to determine whether the recovered behavior is caused by preserving remote-memory information or merely by perturbing the model into a different predictive regime.

## Tasks

### Task 1: Implement state-information preserving controls

Files:
- src/statefuzz/runner/mamba_runner.py
- tests/runner/test_mamba_runner.py

Add controlled cache interventions:

- short-context correct cache
- long-context failed cache
- value-A short cache
- value-B short cache
- randomized cache with matched statistics

Required:
- preserve DynamicCache format
- record state_source
- record whether intervention changes only state or also token processing

Verify:

```bash
python -m pytest tests/runner/test_mamba_runner.py -q
```

Success:

Intervention modes are explicitly distinguishable.

---

### Task 2: Add causal discrimination metrics

Files:
- src/statefuzz/analyzer/failure_risk.py
- tests/analyzer/test_failure_risk.py

Add metrics:

- memory-restoration effect:
  swapped_memory_state - randomized_state
- state specificity ratio
- behavior recovery gap

Do not call recovery causal unless:

1. memory-consistent state recovers behavior;
2. randomized state does not recover equally;
3. recovery is replicated.

---

### Task 3: Repeat frozen stress condition

Condition remains:

- model: state-spaces/mamba-130m-hf
- template_id: 1
- value pair: red/blue
- filler: structured_repetitive

Use fresh seeds:

[45,46,47,48,49,50,51,52]

No new stress search.

Measure:

- normal failure
- correct-memory-state injection
- wrong-memory-state injection
- randomized-state injection

---

### Task 4: Mechanism localization

Analyze:

- SSM state layers
- conv states
- intervention-sensitive layers

Output:

```json
{
 "causal_candidate": true/false,
 "strongest_layers": [],
 "memory_specific_recovery": true/false
}
```

Avoid naming mechanisms such as collision/forgetting unless evidence supports it.

---

### Task 5: Paper claim update

Produce:

results/result_round_020.json

Allowed conclusions:

- recurrent_state_causal_candidate_confirmed
- recurrent_state_intervention_effect_but_not_memory_specific
- intervention_inconclusive

## Verify

Run:

```bash
python -m pytest -q
```

## Success

Round 020 succeeds when:

1. intervention effects are replicated;
2. memory-consistent and randomized interventions are separated;
3. causal language is limited to evidence;
4. full tests pass.
