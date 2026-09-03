# Plan 002

## Analysis

Codex round 001 completed successfully.

Evidence:
- results/result_round_001.json reports `python -m pytest -q` passed 30/30 tests.
- pytest pass rate is 1.0.
- Atomic write checks and status atomic update checks passed.

Implemented in round 001:
- src/statefuzz/execution_state.py
  - load_status()
  - write_status()
  - update_status()
  - _validate_status()
- src/statefuzz/models/registry.py
  - ModelSpec
  - load_registry()
- extended atomic JSON infrastructure in src/statefuzz/io_atomic.py

Problems found:

1. Execution state protocol lacks integration tests for failure recovery.

Target:
- tests/test_execution_state.py
- src/statefuzz/execution_state.py

Need verify:
- corrupted JSON handling
- missing required fields
- invalid actor transitions
- concurrent update behavior

2. Model registry validation is incomplete.

Target:
- src/statefuzz/models/registry.py

Issues:
- role/family semantic validation is absent.
- trust_remote_code policy is only typed, not enforced.

3. Evaluation and probes are only inspected, not connected into an end-to-end fuzz loop.

Target:
- src/statefuzz/probes/
- src/statefuzz/evaluation/

Need establish execution pipeline contracts.

## Tasks

### Task 1
Improve execution state robustness.

Files:
- src/statefuzz/execution_state.py
- tests/test_execution_state.py

Functions:
- load_status()
- write_status()
- update_status()

Add tests for malformed state transitions and recovery.

### Task 2
Strengthen model registry validation.

Files:
- src/statefuzz/models/registry.py

Function:
- load_registry()

Add validation for required semantic fields.

### Task 3
Create initial pipeline contract between probes and evaluation.

Files:
- src/statefuzz/probes/*
- src/statefuzz/evaluation/*

Define minimal interfaces without changing behavior.

## Verify

Run:

python -m pytest -q

Required:
- all existing tests pass
- new execution_state tests pass
- registry invalid configuration tests pass

## Success

Round 002 succeeds when:

1. state transition failures are detected automatically.
2. model registry rejects invalid configurations.
3. probe/evaluation interface has executable tests.
4. results/result_round_002.json is generated.
5. status.json points back to planner after Codex completion.
