# Plan 006

## Research Objective

Continue transforming StateFuzz from a testing framework into an automated capability-boundary discovery platform for SSM long-context behavior.

## Tasks

### Task 1: Upgrade stress generators into search-oriented generators

Files:
- src/statefuzz/generator/memory_decay.py
- src/statefuzz/generator/state_collision.py
- src/statefuzz/generator/state_pollution.py

Required:
- Add parameterized generation space.
- Expose controllable variables such as context length, distractor density, target position, and interference strength.
- Prepare interfaces for automatic boundary search.

Success:
A generator can produce a family of related probes rather than one fixed example.

---

### Task 2: Add capability measurement layer

Files:
- src/statefuzz/analyzer/
- create capability measurement module if needed.

Required:
- Convert failures into measurable boundaries.
- Record effective memory length.
- Produce structured artifacts describing where performance degrades.

Success:
Generate machine-readable capability results.

---

### Task 3: Improve hidden-state diagnosis

Files:
- src/statefuzz/analyzer/hidden_state.py
- src/statefuzz/analyzer/failure_classifier.py

Required:
- Separate failure categories using evidence.
- Avoid relying only on output string heuristics.
- Add state similarity and state norm based diagnostics.

Success:
Failure reports include mechanism evidence, not only labels.

---

## Verify

Run:

```bash
python -m pytest -q
```

Expected:
All tests pass.

## Success Criteria

Round 006 succeeds when StateFuzz can:

1. Generate controlled SSM stress families.
2. Measure an effective memory boundary.
3. Explain failures through hidden-state evidence.

Do not spend this round only on generic infrastructure.
