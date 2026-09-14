"""混合模型缓存的独立复制与路径定向交换。"""

from __future__ import annotations

import copy
from typing import Any


SSM_FIELDS = ("conv_states", "recurrent_states", "ssm_states")
ATTENTION_FIELDS = ("keys", "values")


def _tensor_fields(layer: Any, names: tuple[str, ...]) -> dict[str, Any]:
    import torch

    return {
        name: value
        for name in names
        if isinstance((value := getattr(layer, name, None)), torch.Tensor)
    }


def cache_structure_signature(cache: Any) -> dict[str, Any]:
    """记录每层实际存在的注意力与SSM张量字段。"""
    layers = list(getattr(cache, "layers", []))
    result = []
    for index, layer in enumerate(layers):
        tensors = _tensor_fields(layer, ATTENTION_FIELDS + SSM_FIELDS)
        result.append(
            {
                "layer": index,
                "type": type(layer).__name__,
                "attention": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in tensors.items()
                    if name in ATTENTION_FIELDS
                },
                "ssm": {
                    name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                    for name, value in tensors.items()
                    if name in SSM_FIELDS
                },
            }
        )
    return {
        "cache_type": type(cache).__name__,
        "layer_count": len(layers),
        "layers": result,
        "attention_cache_observed": any(item["attention"] for item in result),
        "ssm_recurrent_state_observed": any(item["ssm"] for item in result),
    }


def clone_cache_independent(cache: Any) -> Any:
    """深复制缓存，并显式克隆所有已知路径张量。"""
    clone = copy.deepcopy(cache)
    for layer in getattr(clone, "layers", []):
        for name, value in _tensor_fields(layer, ATTENTION_FIELDS + SSM_FIELDS).items():
            setattr(layer, name, value.detach().clone())
    assert_no_tensor_alias(cache, clone)
    return clone


def assert_no_tensor_alias(source: Any, clone: Any) -> None:
    """确保两份缓存没有共享任何已知路径张量存储。"""
    source_layers = list(getattr(source, "layers", []))
    clone_layers = list(getattr(clone, "layers", []))
    if len(source_layers) != len(clone_layers):
        raise AssertionError("cache层数不一致")
    for index, (source_layer, clone_layer) in enumerate(
        zip(source_layers, clone_layers, strict=True)
    ):
        for name in ATTENTION_FIELDS + SSM_FIELDS:
            source_value = getattr(source_layer, name, None)
            clone_value = getattr(clone_layer, name, None)
            if hasattr(source_value, "data_ptr") and hasattr(clone_value, "data_ptr"):
                if source_value.data_ptr() == clone_value.data_ptr():
                    raise AssertionError(f"layer {index}字段{name}仍共享张量")


def _validate_shapes(recipient: Any, donor: Any, fields: tuple[str, ...]) -> None:
    recipient_layers = list(getattr(recipient, "layers", []))
    donor_layers = list(getattr(donor, "layers", []))
    if len(recipient_layers) != len(donor_layers):
        raise ValueError("donor与recipient cache层数不一致")
    observed = False
    for index, (recipient_layer, donor_layer) in enumerate(
        zip(recipient_layers, donor_layers, strict=True)
    ):
        recipient_fields = _tensor_fields(recipient_layer, fields)
        donor_fields = _tensor_fields(donor_layer, fields)
        if set(recipient_fields) != set(donor_fields):
            raise ValueError(f"layer {index}路径字段不一致")
        for name in recipient_fields:
            observed = True
            if recipient_fields[name].shape != donor_fields[name].shape:
                raise ValueError(f"layer {index}字段{name}形状不一致")
    if not observed:
        raise ValueError("指定cache路径没有可交换张量")


def swap_cache_path(recipient_cache: Any, donor_cache: Any, path: str) -> Any:
    """从独立recipient副本出发，仅替换指定的缓存路径。"""
    if path not in {"ssm", "attention", "full"}:
        raise ValueError("path必须是ssm、attention或full")
    fields = (
        SSM_FIELDS
        if path == "ssm"
        else ATTENTION_FIELDS
        if path == "attention"
        else ATTENTION_FIELDS + SSM_FIELDS
    )
    _validate_shapes(recipient_cache, donor_cache, fields)
    if path == "full":
        return clone_cache_independent(donor_cache)
    result = clone_cache_independent(recipient_cache)
    for result_layer, donor_layer in zip(
        result.layers, donor_cache.layers, strict=True
    ):
        for name, value in _tensor_fields(donor_layer, fields).items():
            setattr(result_layer, name, value.detach().clone())
    assert_no_tensor_alias(recipient_cache, result)
    assert_no_tensor_alias(donor_cache, result)
    return result
