# Plan 004

## Tasks

### Task 1: Complete deterministic probe execution pipeline

Files:
- src/statefuzz/probes/compiler.py
- src/statefuzz/evaluation/contract.py
- src/statefuzz/evaluation/pipeline.py (new)

Functions:
- compile_probe()
- evaluate_compiled_probe()
- add a deterministic pipeline entrypoint

Goal:
Connect ProbeSpec -> CompiledProbe -> EvaluationOutcome -> result artifact preparation.

### Task 2: Strengthen execution state adversarial coverage

Files:
- src/statefuzz/execution_state.py
- tests/test_execution_state.py

Functions:
- load_status()
- update_status()
- write_status()

Add verification for corrupted JSON, invalid transitions, and recovery paths.

### Task 3: Add result artifact contract

Files:
- src/statefuzz/evaluation/
- results/

Define the minimal structure required for future result_round_N.json generation.

## Verify

Run:

```bash
python -m pytest -q
```

Expected:
- Existing tests remain passing.
- New pipeline tests validate deterministic behavior.
- Result artifact schema is stable.

## Success

Round 004 succeeds when:

- Probe execution and evaluation are connected by a stable API.
- State recovery cases are tested.
- Codex produces results/result_round_004.json.
