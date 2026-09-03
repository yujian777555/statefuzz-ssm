from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    family: str
    trust_remote_code: bool = False


def load_registry(path: Path) -> list[ModelSpec]:
    """加载并验证有序模型清单。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("模型配置必须包含models列表")
    specs: list[ModelSpec] = []
    for item in data["models"]:
        if not isinstance(item, dict):
            raise ValueError("模型条目必须是对象")
        normalized = dict(item)
        if "id" in normalized and "model_id" not in normalized:
            normalized["model_id"] = normalized.pop("id")
        try:
            specs.append(ModelSpec(**normalized))
        except TypeError as exc:
            raise ValueError("模型条目字段非法") from exc
    ids = [item.model_id for item in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("模型ID重复")
    return specs

