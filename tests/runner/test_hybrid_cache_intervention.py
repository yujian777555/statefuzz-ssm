from types import SimpleNamespace

import pytest
import torch

from statefuzz.runner.hybrid_cache_intervention import (
    assert_no_tensor_alias,
    cache_structure_signature,
    clone_cache_independent,
    swap_cache_path,
)


def cache(value: float, key_length: int = 3):
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                keys=torch.full((1, 2, key_length, 4), value),
                values=torch.full((1, 2, key_length, 4), value + 1),
                conv_states=torch.full((1, 4, 2), value + 2),
                recurrent_states=torch.full((1, 2, 2, 4), value + 3),
                marker="recipient-metadata",
            )
        ]
    )


def test_clone_has_no_alias_and_signature_separates_paths():
    source = cache(1)
    clone = clone_cache_independent(source)
    assert_no_tensor_alias(source, clone)
    clone.layers[0].keys.add_(10)
    assert not torch.equal(source.layers[0].keys, clone.layers[0].keys)
    signature = cache_structure_signature(source)
    assert set(signature["layers"][0]["attention"]) == {"keys", "values"}
    assert set(signature["layers"][0]["ssm"]) == {"conv_states", "recurrent_states"}


def test_path_swaps_preserve_non_target_and_full_matches_donor():
    recipient, donor = cache(1), cache(9)
    ssm = swap_cache_path(recipient, donor, "ssm")
    assert torch.equal(ssm.layers[0].keys, recipient.layers[0].keys)
    assert torch.equal(ssm.layers[0].recurrent_states, donor.layers[0].recurrent_states)
    attention = swap_cache_path(recipient, donor, "attention")
    assert torch.equal(attention.layers[0].keys, donor.layers[0].keys)
    assert torch.equal(attention.layers[0].conv_states, recipient.layers[0].conv_states)
    full = swap_cache_path(recipient, donor, "full")
    assert torch.equal(full.layers[0].keys, donor.layers[0].keys)
    assert torch.equal(full.layers[0].recurrent_states, donor.layers[0].recurrent_states)
    assert_no_tensor_alias(donor, full)


def test_shape_mismatch_and_unknown_path_fail_deterministically():
    with pytest.raises(ValueError, match="形状"):
        swap_cache_path(cache(1), cache(2, key_length=4), "attention")
    with pytest.raises(ValueError, match="path"):
        swap_cache_path(cache(1), cache(2), "other")
