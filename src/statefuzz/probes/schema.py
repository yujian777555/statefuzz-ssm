from __future__ import annotations

import hashlib
import json
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class TaskFamily(str, Enum):
    SINGLE_KEY = "single_key"
    MULTI_KEY = "multi_key"


class ProbeSpec(BaseModel):
    """可执行长上下文探针的类型化规范。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    task: TaskFamily = TaskFamily.SINGLE_KEY
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    context_tokens: int = Field(default=1024, ge=64, le=131072)
    n_items: int = Field(default=16, ge=1, le=4096)
    query_fanout: int = Field(default=1, ge=1, le=64)
    target_position: float = Field(default=0.5, ge=0.0, le=1.0)
    template_id: int = Field(default=0, ge=0, le=7)

    @model_validator(mode="after")
    def validate_task_shape(self) -> "ProbeSpec":
        if self.task is TaskFamily.SINGLE_KEY and self.query_fanout != 1:
            raise ValueError("single_key的query_fanout必须为1")
        if self.query_fanout > self.n_items:
            raise ValueError("query_fanout不能超过n_items")
        return self

    def canonical_payload(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"config_hash"})

    @computed_field
    @property
    def config_hash(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
