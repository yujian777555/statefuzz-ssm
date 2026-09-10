"""跨架构冻结实验矩阵。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelMatrixEntry:
    """一个不混淆模型规模与架构角色的实验条目。"""

    model_id: str
    architecture: str
    role: str
    runner_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "architecture": self.architecture,
            "role": self.role,
            "runner_kind": self.runner_kind,
        }


FROZEN_MODEL_MATRIX = (
    ModelMatrixEntry("state-spaces/mamba-130m-hf", "Mamba", "ssm_primary", "mamba"),
    ModelMatrixEntry("EleutherAI/pythia-160m", "GPT-NeoX", "transformer_control", "hf_causal_lm"),
)


def validate_model_matrix(entries=FROZEN_MODEL_MATRIX) -> tuple[ModelMatrixEntry, ...]:
    """验证矩阵包含至少两种架构且没有重复模型。"""
    values = tuple(entries)
    if len(values) < 2:
        raise ValueError("模型矩阵必须包含至少两种架构")
    if len({entry.model_id for entry in values}) != len(values):
        raise ValueError("模型矩阵不能包含重复模型")
    if len({entry.architecture for entry in values}) < 2:
        raise ValueError("模型矩阵必须包含至少两种架构")
    return values
