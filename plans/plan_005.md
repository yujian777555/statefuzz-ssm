# Plan 005 - StateFuzz Research Core Transition

## Analysis

Codex has completed the infrastructure phase successfully.

The project should now transition from a testing framework into a research platform for discovering SSM long-context capability boundaries.

Current foundation:
- execution state protocol exists
- deterministic evaluation contract exists
- probe compilation exists

Next phase focuses on scientific contribution rather than additional infrastructure.

## Tasks

### Task 1: Add SSM-specific stress case generators

Files:
- src/statefuzz/generator/memory_decay.py
- src/statefuzz/generator/state_collision.py
- src/statefuzz/generator/state_pollution.py

Functions:
- generate_memory_decay_probe()
- generate_collision_probe()
- generate_pollution_probe()

Goal:
Generate inputs that specifically test SSM hidden-state limitations.

---

### Task 2: Add model execution abstraction

Files:
- src/statefuzz/runner/base.py
- src/statefuzz/runner/mamba_runner.py

Functions:
- run_probe()
- capture_hidden_state()

Goal:
Create a unified interface for Mamba/Mamba2 style models.

---

### Task 3: Add hidden-state analysis prototype

Files:
- src/statefuzz/analyzer/hidden_state.py
- src/statefuzz/analyzer/failure_classifier.py

Functions:
- compute_state_similarity()
- detect_state_collapse()
- classify_failure()

Goal:
Move from detecting failure to explaining failure mechanism.

---

## Verify

Run:

python -m pytest -q

Required:
- generator tests pass
- runner interface tests pass
- analyzer tests pass

## Success

Round 005 succeeds when:

1. StateFuzz can automatically generate at least one SSM-specific stress pattern.
2. Execution produces deterministic evaluation artifacts.
3. Hidden-state analysis can classify at least one failure category.
4. results/result_round_005.json is generated.
