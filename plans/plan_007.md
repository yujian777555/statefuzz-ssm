# Plan 007

## Round 006 Review

Codex successfully moved StateFuzz from a prototype framework toward a capability analysis system.

Completed:
- capability measurement module (`src/statefuzz/analyzer/capability.py`)
- hidden-state evidence integration improvements
- expanded stress pattern tests
- 71 tests passing

However, the current implementation still measures capability boundaries from predefined observations. It does not yet perform true automatic discovery.

## Research Goal

Continue building:

**StateFuzz: Automatically discovering and diagnosing long-context capability boundaries of State Space Models.**

The next milestone is to transform measurements into an experimental discovery engine.

## Tasks

### 1. Implement capability search engine

Create:

`src/statefuzz/search/`

Required:
- search over context length
- search over target position
- search over distractor/interference strength
- record discovered failure boundary

Functions:
- `search_boundary()`
- `rank_failure_cases()`

Output should identify the smallest context/interference setting causing degradation.

### 2. Upgrade failure diagnosis

Modify:

`src/statefuzz/analyzer/failure_classifier.py`

Do not rely only on output string rules.

Use evidence:
- hidden state similarity
- state norm change
- layer behavior

Return an explanation artifact.

### 3. Generate paper-quality artifacts

Create results containing:

- model id
- effective memory boundary
- failure type
- trigger configuration
- hidden-state evidence

## Verify

Run:

`python -m pytest -q`

Expected:
- all tests pass
- capability search produces reproducible boundaries
- artifacts are JSON serializable

## Success

Round 007 succeeds when StateFuzz can automatically discover at least one long-context failure boundary and provide evidence explaining the mechanism.
