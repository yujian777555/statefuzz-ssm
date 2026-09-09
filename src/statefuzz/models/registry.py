from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    role: str
    family: str
    trust_remote_code: bool = False


_SEMANTIC_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


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
        required = ("id", "role", "family")
        missing = [field for field in required if field not in normalized]
        if missing and "model_id" in normalized:
            missing.remove("id")
        if missing:
            raise ValueError(f"模型条目缺少必填字段: {','.join(missing)}")
        if "id" in normalized and "model_id" in normalized:
            raise ValueError("模型条目不能同时使用id和model_id")
        if "id" in normalized and "model_id" not in normalized:
            normalized["model_id"] = normalized.pop("id")
        for field in ("model_id", "role", "family"):
            value = normalized.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field}必须是非空字符串")
        for field in ("role", "family"):
            if _SEMANTIC_NAME.fullmatch(normalized[field]) is None:
                raise ValueError(f"{field}必须使用小写字母、数字或下划线")
        trust_remote_code = normalized.get("trust_remote_code", False)
        if not isinstance(trust_remote_code, bool) or trust_remote_code:
            raise ValueError("trust_remote_code策略要求为false")
        normalized["trust_remote_code"] = trust_remote_code
        try:
            specs.append(ModelSpec(**normalized))
        except TypeError as exc:
            raise ValueError("模型条目字段非法") from exc
    ids = [item.model_id for item in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("模型ID重复")
    return specs

