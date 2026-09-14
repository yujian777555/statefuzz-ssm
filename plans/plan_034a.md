# Round 34-A Plan: Model Environment Discovery

## Motivation

Round34 execution is blocked because model_paths.json does not exist. This phase only discovers available execution assets. It does not select checkpoints or modify the scientific protocol.

## Goal

Create a reproducible inventory of available model checkpoints on the execution VM.

## Rules

- Do not download models.
- Do not choose preferred checkpoints.
- Do not run stress experiments.
- Do not infer missing paths.
- Only record assets that are directly observable.

## Task 1: Discover checkpoints

Run:

```bash
find /202532803004/models -name config.json
```

If no results:

```bash
find / -name config.json 2>/dev/null | grep models
```

## Task 2: Inspect discovered models

For every discovered checkpoint collect:

- absolute path
- model identifier if available
- architecture/model_type
- parameter size if available
- revision or commit hash if available
- tokenizer availability

## Task 3: Create inventory

Output:

```
results/model_inventory_round_034.json
```

Schema:

```json
{
  "models": [
    {
      "path": "",
      "model_id": "",
      "architecture": "",
      "model_type": "",
      "revision": ""
    }
  ]
}
```

## Verify

```bash
python -m pytest
```

Verify that inventory only contains observed files.

## Success

Round34-A succeeds when:

1. All visible model checkpoints are recorded.
2. No checkpoint is selected by the executor.
3. No experimental result is generated.

After this phase, Planner will freeze model_paths.json for Round34-B.
