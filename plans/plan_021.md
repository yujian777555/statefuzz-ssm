# Plan 021 - Generalize Memory-Specific Causal Evidence Beyond One Frozen Intervention

## Tasks

### Task 1: Preserve Round 020 causal result as the baseline claim

Files:
- results/result_round_020.json
- plans/latest_plan.md

Conclusion to preserve:

- Round 019 showed cache perturbation can change behavior.
- Round 020 disambiguated this: memory-consistent state restored behavior 8/8, wrong-memory state restored 0/8, randomized matched state restored 1/8.
- The current evidence supports `recurrent_state_causal_candidate_confirmed`.

Do not upgrade beyond:

> Under the frozen structured-repetition red/blue stress condition on Mamba-130M, recurrent state content participates causally in remote-memory behavior.

Do not claim all SSMs or all long-context failures.

---

### Task 2: Test causal specificity across additional memory values

Files:
- Modify: src/statefuzz/runner/mamba_runner.py if intervention API needs generalization
- Modify: src/statefuzz/analyzer/failure_risk.py
- Tests: tests/runner/test_mamba_runner.py
- Tests: tests/analyzer/test_failure_risk.py

Add a reusable intervention evaluator:

```python
compare_memory_state_interventions(
    failed_prompt,
    memory_a_state,
    memory_b_state,
    randomized_states,
) -> dict
```

Required outputs:

- correct_memory_recovery_rate
- wrong_memory_recovery_rate
- randomized_recovery_rate
- specificity_gap
- seed_count

Acceptance:

Correct memory recovery must be separated from arbitrary state perturbation.

---

### Task 3: Fresh seed causal replication

Frozen:

```text
model=state-spaces/mamba-130m-hf
filler=structured_repetitive
template=1
value_pair=(red, blue)
```

New seeds:

```text
53-60
```

Run:

1. failed long context original cache
2. value-B short memory-consistent cache
3. value-A wrong cache
4. randomized matched cache

Write:

```text
results/result_round_021.json
```

---

### Task 4: Avoid overfitting to red/blue

Evaluate one additional predeclared value pair:

```text
cat/dog
```

Do not search pairs after seeing results.

Purpose:

Determine whether the mechanism is:

A. generic remote-memory state dependence

or

B. a red/blue artifact.

---

### Task 5: Prepare paper-level mechanism summary

If successful, update interpretation from:

"candidate"

to:

"memory-content-specific recurrent-state causal evidence under controlled SSM stress conditions"

Do not claim:

"SSM stores all memories in recurrent state"

because only one architecture/checkpoint is tested.

---

## Verify

Run:

```bash
python -m pytest -q -o addopts=''
```

Required:

- all tests pass
- result_round_021.json generated
- intervention APIs remain deterministic

## Success

Round 021 succeeds when:

1. fresh seeds replicate memory-specific recovery;
2. wrong-memory and random state controls remain separated;
3. at least one additional value pair provides supporting evidence or clearly limits the claim;
4. mechanism claim remains scoped to tested Mamba condition.
