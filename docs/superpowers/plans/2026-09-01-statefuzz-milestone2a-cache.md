# StateFuzz-SSM Milestone 2A Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为StateFuzz评价结果实现原子、确定性、防篡改且拒绝冲突覆盖的文件缓存。

**Architecture:** `io_atomic.py`负责同目录临时文件、fsync和原子替换；`EvaluationRecord`负责规范化序列化和记录哈希；`EvaluationCache`按键前两字符分片，每个键一个JSON文件。缓存缺失返回`None`，相同记录重复写幂等，不同记录写同一键立即失败。

**Tech Stack:** Python 3.10.20、pytest 9.1.1；无新增依赖。

---

## Task 1：原子JSON写入

**Files:**
- Create: `src/statefuzz/io_atomic.py`
- Create: `tests/test_io_atomic.py`

- [ ] **Step 1：写失败测试**

```python
import importlib.util
import json

import pytest


IO_SPEC = importlib.util.find_spec("statefuzz.io_atomic")
requires_io = pytest.mark.skipif(IO_SPEC is None, reason="io_atomic尚未实现")


def test_io_atomic_module_exists() -> None:
    assert IO_SPEC is not None


@requires_io
def test_atomic_write_replaces_without_tmp_residue(tmp_path) -> None:
    from statefuzz.io_atomic import atomic_write_json

    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"value": 1})
    atomic_write_json(target, {"value": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2}
    assert list(tmp_path.glob("*.tmp")) == []


@requires_io
def test_atomic_write_uses_stable_key_order(tmp_path) -> None:
    from statefuzz.io_atomic import atomic_write_json

    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"z": 1, "a": 2})
    text = target.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"z"')
```

Run: `python -m pytest tests/test_io_atomic.py -q`  
Expected: 1 failed, 2 skipped。

- [ ] **Step 2：实现完整原子写入**

```python
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def canonical_json_bytes(payload: Any) -> bytes:
    """返回稳定、无多余空白的UTF-8 JSON。"""
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (text + "\n").encode("utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    """同目录写临时文件，fsync后原子替换目标。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
```

- [ ] **Step 3：验证并提交**

Run:

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/test_io_atomic.py -q
git add src/statefuzz/io_atomic.py tests/test_io_atomic.py
git commit -m "feat: add atomic deterministic JSON writes"
```

Expected: 3 passed。

## Task 2：防篡改评价缓存

**Files:**
- Create: `src/statefuzz/evaluation/__init__.py`
- Create: `src/statefuzz/evaluation/cache.py`
- Create: `tests/evaluation/test_cache.py`

- [ ] **Step 1：写失败测试**

```python
import importlib.util
import json

import pytest


CACHE_SPEC = importlib.util.find_spec("statefuzz.evaluation.cache")
requires_cache = pytest.mark.skipif(CACHE_SPEC is None, reason="cache尚未实现")


def test_cache_module_exists() -> None:
    assert CACHE_SPEC is not None


@requires_cache
def test_cache_round_trip_is_idempotent(tmp_path) -> None:
    from statefuzz.evaluation.cache import EvaluationCache, EvaluationRecord

    cache = EvaluationCache(tmp_path)
    record = EvaluationRecord(key="abc123", score=1.0, payload={"model": "fake"})
    cache.put(record)
    cache.put(record)
    assert cache.get("abc123") == record


@requires_cache
def test_cache_rejects_tampered_record(tmp_path) -> None:
    from statefuzz.evaluation.cache import EvaluationCache, EvaluationRecord

    cache = EvaluationCache(tmp_path)
    cache.put(EvaluationRecord(key="abc123", score=1.0, payload={"model": "fake"}))
    target = tmp_path / "ab" / "abc123.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["score"] = 0.0
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        cache.get("abc123")


@requires_cache
def test_cache_refuses_conflicting_overwrite(tmp_path) -> None:
    from statefuzz.evaluation.cache import EvaluationCache, EvaluationRecord

    cache = EvaluationCache(tmp_path)
    cache.put(EvaluationRecord(key="abc123", score=1.0, payload={"model": "fake"}))
    with pytest.raises(ValueError, match="冲突"):
        cache.put(EvaluationRecord(key="abc123", score=0.0, payload={"model": "fake"}))
```

Run: `python -m pytest tests/evaluation/test_cache.py -q`  
Expected: 1 failed, 3 skipped。

- [ ] **Step 2：实现完整缓存**

Create `src/statefuzz/evaluation/__init__.py`:

```python
"""模型评价、评分和可恢复缓存。"""
```

Create `src/statefuzz/evaluation/cache.py`:

```python
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
        body = self.body()
        digest = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        return {**body, "record_sha256": digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationRecord":
        body = {"key": data["key"], "payload": data["payload"], "score": data["score"]}
        actual = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        if actual != data.get("record_sha256"):
            raise ValueError("评价记录哈希不匹配")
        return cls(key=body["key"], score=float(body["score"]), payload=body["payload"])


class EvaluationCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        if not _KEY_PATTERN.fullmatch(key):
            raise ValueError("缓存键格式非法")
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> EvaluationRecord | None:
        path = self._path(key)
        if not path.exists():
            return None
        return EvaluationRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def put(self, record: EvaluationRecord) -> None:
        path = self._path(record.key)
        existing = self.get(record.key)
        if existing is not None:
            if existing != record:
                raise ValueError("同一缓存键存在冲突记录")
            return
        atomic_write_json(path, record.to_dict())
```

- [ ] **Step 3：验证并提交**

Run:

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/evaluation/test_cache.py -q
/202532803004/conda_envs/amber/bin/python -m pytest -q
git add src/statefuzz/evaluation tests/evaluation/test_cache.py
git commit -m "feat: add tamper-evident evaluation cache"
```

Expected: 缓存4项通过，全套25项通过。

## Task 3：验收

- [ ] 运行全套测试并保存JUnit：

```bash
mkdir -p runs/milestone2a
/202532803004/conda_envs/amber/bin/python -m pytest -q --junitxml=runs/milestone2a/pytest.xml
```

Expected: 25 passed。

- [ ] 创建`docs/MILESTONE2A_REPORT.md`：

```markdown
# StateFuzz-SSM Milestone 2A验收报告

## 范围

本阶段交付确定性原子JSON写入和防篡改评价缓存。未下载模型、未运行GPU、未实现搜索算法。

## 验收证据

- pytest：25 passed，0 failed，0 errors；
- 同键同记录重复写入幂等；
- 文件内容被修改后读取必然报哈希错误；
- 同键不同记录拒绝覆盖；
- 原子替换后没有残留`.tmp`文件。

## 下一步

Milestone 2B实现冻结参数域、fake/HF后端、状态hook和Gate 0 GPU冒烟。
```
- [ ] 提交报告并打标签：

```bash
git add docs/MILESTONE2A_REPORT.md
git commit -m "docs: validate atomic evaluation cache"
git tag -a milestone2a-cache -m "Atomic evaluation cache validated"
```

## 硬停止点

- 任一缓存篡改未被检测：停止。
- 冲突记录被静默覆盖：停止。
- 产生残留`.tmp`文件：停止。
- 全部通过后才规划Milestone 2B。

## 执行方式

使用当前会话内联TDD执行，不自动派发子代理。
