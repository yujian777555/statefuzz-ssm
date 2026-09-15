# Round37-A: Model Inventory Discovery

## Objective

Only discover available model assets in the VM. Do not run stress evaluation.

## Allowed actions

1. Scan existing model directories.
2. Read local config/tokenizer metadata.
3. Generate model inventory.

## Forbidden actions

- Do not download models.
- Do not select final checkpoints.
- Do not modify experiment protocol.
- Do not start Round37-B evaluation.

## Scan root

Use the known VM project environment. If `/202532803004/models` exists, scan it first. If no required model is found, report missing assets instead of guessing.

## Output

Create:

`results/model_inventory_round_037.json`

Required fields:

```json
{
  "model_id": "",
  "checkpoint_path": "",
  "architecture": "",
  "model_type": "",
  "revision": null,
  "availability": ""
}
```

## Completion condition

Round37-A is complete only when the inventory file is generated and verified. Planner will then freeze `configs/model_paths_round_037.json` before any evaluation.
