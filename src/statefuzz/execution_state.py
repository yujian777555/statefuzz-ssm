"""Codex规划器与执行器之间的原子状态文件协议。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from statefuzz.io_atomic import atomic_write_json


REQUIRED_FIELDS = ("round", "previous_actor", "next", "last_plan", "last_result")


def _validate_status(status: dict[str, Any]) -> dict[str, Any]:
    """校验状态字段，返回可安全写入的副本。"""
    missing = [field for field in REQUIRED_FIELDS if field not in status]
    if missing:
        raise ValueError(f"状态缺少字段: {','.join(missing)}")
    if isinstance(status["round"], bool) or not isinstance(status["round"], int):
        raise ValueError("round必须是非负整数")
    if status["round"] < 0:
        raise ValueError("round必须是非负整数")
    for field in ("previous_actor", "next", "last_plan"):
        if not isinstance(status[field], str) or not status[field]:
            raise ValueError(f"{field}必须是非空字符串")
    if status["last_result"] is not None and not isinstance(status["last_result"], str):
        raise ValueError("last_result必须是字符串或null")
    return dict(status)


def load_status(path: Path) -> dict[str, Any]:
    """加载并校验状态文件。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("状态文件必须是JSON对象")
    return _validate_status(data)


def write_status(path: Path, status: dict[str, Any]) -> None:
    """通过项目原子JSON写入协议状态。"""
    atomic_write_json(Path(path), _validate_status(status))


def update_status(path: Path, **updates: Any) -> dict[str, Any]:
    """合并字段并原子更新状态；文件不存在时从round 0开始。"""
    target = Path(path)
    if target.exists():
        current = load_status(target)
    else:
        current = {
            "round": 0,
            "previous_actor": "planner",
            "next": "codex",
            "last_plan": "plans/latest_plan.md",
            "last_result": None,
        }
    current.update(updates)
    write_status(target, current)
    return current

