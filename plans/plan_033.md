# Round 033 — Hybrid Cache Reconstruction Validation

## Goal

Fix the prerequisite problem discovered in Round 032: long-context cache intervention cannot be interpreted until body→cache→tail replay is validated against direct forward.

The objective is NOT to claim a memory mechanism result. The objective is to validate or reject the intervention protocol.

## Tasks

### Task 1 — Debug reconstruction mismatch

Create:

- `scripts/round033_cache_reconstruction_debug.py`

Inspect:

- full DynamicCache replay
- SSM state components
  - conv_states
  - recurrent_states
  - ssm_states
- Attention KV components
  - keys
  - values

Record:

- `max_logit_diff`
- `mean_logit_diff`
- top changed token
- first mismatch location

### Task 2 — Use real decoding semantics

Do not use simplified tail injection.

Required flow:

```
prefill body
        ↓
obtain cache/state
        ↓
continue tail token-by-token
        ↓
compare logits with direct full prompt
```

The implementation must match the model generation update path.

### Task 3 — Validate target lengths

Run only:

- 256 tokens
- 1792 tokens
- 3584 tokens

Seeds:

- 69
- 70
- 71
- 72
- 73
- 74
- 75
- 76

## Verify

A reconstruction cell is valid only when:

```
max_logit_diff <= 1e-3
```

Success requirement:

```
reconstruction_valid_rate >= 90%
```

If this fails:

```
hybrid_cache_protocol_invalid
```

Do not perform SSM/Attention causal interpretation.

## Success

Round 033 succeeds when:

1. Reconstruction is validated at long context.
2. The state intervention protocol is trustworthy.
3. Round 034 can repeat SSM-only and Attention-only donor swap experiments.

Round 033 must not make architectural superiority claims.