# Plan 001

## Analysis

Current repository inspection:
- `status.json`: not found. Execution state tracking has not been initialized.
- `results/`: not found. No Codex execution result artifact is available.
- `src/`: current implementation contains atomic JSON persistence foundation.

Observed implementation:
- `src/statefuzz/io_atomic.py`
  - `canonical_json_bytes(payload)` provides deterministic JSON serialization.
  - `atomic_write_json(path, payload)` provides temp-file write, fsync, and atomic replacement.

Current issues:
1. No execution loop state file exists, so Codex cannot know current round ownership.
2. No `results/` artifact protocol exists, so planner cannot verify whether previous execution succeeded.
3. Core package only has persistence primitive; fuzzing/state-machine workflow needs explicit implementation and verification paths.

## Tasks

1. Add execution state tracking.
   - File: `status.json`
   - Define fields:
     - round
     - previous_actor
     - next
     - last_plan
     - last_result
   - Use `src/statefuzz/io_atomic.py::atomic_write_json` for future state writes.

2. Add result artifact protocol.
   - Directory: `results/`
   - Add schema documentation for Codex result reports.
   - Required fields:
     - round
     - changed_files
     - tests
     - failures
     - metrics

3. Extend fuzzing core structure.
   - Inspect and implement missing modules under:
     - `src/statefuzz/probes/`
     - `src/statefuzz/evaluation/`
   - Ensure each module exposes testable functions rather than only scripts.

## Verify

Codex must verify:

1. `status.json` can be loaded and updated atomically.
2. A sample `results/result_001.json` can be generated.
3. Existing atomic write behavior remains correct:
   - deterministic JSON bytes
   - crash-safe replacement

## Success

Round 001 succeeds when:

- Planner/Executor handoff files exist.
- Codex produces `results/result_001.json`.
- `status.json.next` is returned to `planner` after execution.
- All modified functions have tests or verification commands.

