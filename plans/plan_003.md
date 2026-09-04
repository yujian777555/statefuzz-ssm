# Plan 003

## Tasks

### Task 1: Extend evaluation contract from single prediction scoring to executable evaluation flow

Files:
- `src/statefuzz/evaluation/contract.py`
- `src/statefuzz/evaluation/`
- `tests/evaluation/`

Required:
- Keep `EvaluationOutcome` as immutable result object.
- Add explicit validation for invalid probe/prediction combinations.
- Define stable input/output contract for future fuzz execution pipeline.

### Task 2: Connect probes and evaluation modules

Files:
- `src/statefuzz/probes/`
- `src/statefuzz/evaluation/contract.py`

Required:
- Add minimal adapter layer only.
- Do not implement full fuzz engine yet.
- Preserve existing probe hash and deterministic evaluation behavior.

### Task 3: Improve state transition observability

Files:
- `src/statefuzz/execution_state.py`
- `tests/test_execution_state.py`

Required:
- Add tests for corrupted status JSON.
- Add tests for invalid state transitions.
- Verify atomic update behavior remains unchanged.

## Verify

Run:

```bash
python -m pytest -q
```

Expected:
- Existing 45 tests continue passing.
- New contract tests pass.
- No regression in status atomic writes.

## Success

Codex must produce:

- `results/result_round_003.json`
- Updated tests
- Updated source files only under approved paths
- `status.json` with `next: gpt`
