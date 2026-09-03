from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from statefuzz.io_atomic import atomic_write_json, canonical_json_bytes


_KEY_PATTERN = re.compile(r"^[a-z0-9]{3,128}$")


@dataclass(frozen=True)
class EvaluationRecord:
    key: str
    score: float
    payload: dict[str, Any]

    def body(self) -> dict[str, Any]:
        return {"key": self.key, "payload": self.payload, "score": self.score}

    def to_dict(self) -> dict[str, Any]:
        digest = hashlib.sha256(canonical_json_bytes(self.body())).hexdigest()
        return {**self.body(), "record_sha256": digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRecord":
        body = {
            "key": data["key"],
            "payload": data["payload"],
            "score": data["score"],
        }
        actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        if actual != data.get("record_sha256"):
            raise ValueError("评价记录哈希不匹配")
        return cls(
            key=str(body["key"]),
            score=float(body["score"]),
            payload=dict(body["payload"]),
        )


class EvaluationCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if _KEY_PATTERN.fullmatch(key) is None:
            raise ValueError("缓存键格式非法")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> EvaluationRecord | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return EvaluationRecord.from_dict(data)

    def put(self, record: EvaluationRecord) -> None:
        path = self._path(record.key)
        existing = self.get(record.key)
        if existing is not None:
            if existing != record:
                raise ValueError("同一缓存键存在冲突记录")
            return
        atomic_write_json(path, record.to_dict())
