# Round 38 Plan: Paper Draft Preparation

## Purpose

Transition StateFuzz from experimental exploration to paper construction.

Round37 scope freeze completed the experimental boundary. This round must not add models, download checkpoints, or modify evaluation protocols.

## Frozen Evidence Scope

Use only existing artifacts:

- results/result_round_037.json
- results/final_scope_round_037.json
- results/result_round_036.json
- results/evidence_table_round_036.json
- results/figure_data_round_036.json
- results/paper_evidence_summary_round_036.json

Do not rerun historical experiments.

## Objective

Create a paper draft package based on supported evidence only.

## Task 1: Paper structure

Create:

paper/
- abstract.md
- introduction.md
- related_work.md
- method.md
- experiment.md
- limitation.md
- conclusion.md

The writing must preserve claim boundaries from Round37-C.

## Task 2: Method description

Describe StateFuzz as:

stress generation -> behavioral probing -> margin measurement -> stress surface characterization

Do not describe it as a universal failure detector.

## Task 3: Experiment narrative

Structure experiments around:

architecture × stress family × context budget

Focus on robustness surface differences, not model ranking.

## Task 4: Figure planning

Create:

results/paper_figure_plan_round_038.json

Include planned figures:

1. StateFuzz pipeline overview.
2. Stress surface visualization.
3. Architecture-dependent response comparison.
4. Limitations and evaluation scope.

## Task 5: Verify

Run:

python -m pytest

Expected:
- no new model assets;
- no inference required;
- no unsupported claims added;
- paper artifacts generated successfully.

## Handoff

After completion update status:

next: gpt
phase: paper_draft_complete

Do not start additional experiments without a new Planner round.
