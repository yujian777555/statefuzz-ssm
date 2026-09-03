# StateFuzz-SSM Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在独立新仓库中交付可测试、确定性、可审计的 StateProbeIR CPU核心，包括严格schema、规范化哈希、精确oracle、提示编译器和语义保持变形。

**Architecture:** `ProbeSpec`是唯一输入事实源，经Pydantic严格验证后由`compile_probe`使用局部随机源生成提示、精确答案和provenance。Milestone 1不连接模型、不下载权重、不占用GPU、不实现搜索器或StatePolicyIR。

`context_tokens`在本阶段表示模型无关的目标长度预算；Milestone 2接入具体tokenizer后才执行精确token计数、裁剪和填充。Milestone 1不得把字符单元数报告为真实模型token数。

**Tech Stack:** Python 3.10.20、Pydantic 2.13.3、pytest 9.1.1、Git 2.25.1；解释器固定为`/202532803004/conda_envs/amber/bin/python`，仓库固定为`/202532803004/statefuzz_ssm_20260901`。

---

## 执行约束

- 执行前须得到用户对测试驱动开发的明确同意。
- 禁止读取、复制、导入或依赖HybridKV及其他旧项目代码和结果。
- 不修改共享`amber`环境，不安装依赖。
- Git只使用仓库本地身份；身份缺失时停止并询问用户。
- 注释、docstring和文档使用中文。

## Task 1：初始化独立仓库和最小包

**Files:**
- Create: `.gitignore`
- Create: `AGENTS.md`
- Create: `README.md`
- Create: `pyproject.toml`
- Create: `src/statefuzz/__init__.py`
- Create: `tests/test_import.py`

- [ ] **Step 1：验证路径和Git身份**

```bash
cd /202532803004/statefuzz_ssm_20260901
test "$(pwd -P)" = "/202532803004/statefuzz_ssm_20260901"
test "$(find . -maxdepth 1 -mindepth 1 -printf '%f\n' | sort)" = ".claude_resources.json"
test ! -e .git
git config --get user.name
git config --get user.email
```

Expected: 目录中只有`.claude_resources.json`，Git身份两项均非空；否则停止。

- [ ] **Step 2：初始化Git并写失败测试**

```bash
git init
mkdir -p tests
```

Create `tests/test_import.py`:

```python
def test_package_exposes_version() -> None:
    import statefuzz

    assert statefuzz.__version__ == "0.1.0"
```

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/test_import.py -q
```

Expected: FAIL，包含`ModuleNotFoundError`。

- [ ] **Step 3：创建骨架文件**

Create `.gitignore`:

```gitignore
__pycache__/
.pytest_cache/
*.py[cod]
*.egg-info/
.env
.env.*
runs/
models/
checkpoints/
*.pt
*.pth
*.safetensors
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "statefuzz-ssm"
version = "0.1.0"
description = "面向状态空间语言模型的可执行长上下文压力测试"
requires-python = ">=3.10,<3.11"
dependencies = ["pydantic==2.13.3"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
pythonpath = ["src"]
```

Create `src/statefuzz/__init__.py`:

```python
"""StateFuzz-SSM：面向状态空间语言模型的可执行压力测试。"""

__version__ = "0.1.0"
```

Create `README.md`:

```markdown
# StateFuzz-SSM

本项目研究具有精确程序化答案的长上下文压力测试，用于发现Mamba/SSM特有的状态容量、衰减和干扰失效。

本仓库是独立新项目，不读取、复制、导入或依赖HybridKV及其他旧项目资产。

## 测试

运行：`/202532803004/conda_envs/amber/bin/python -m pytest -q`
```

Create `AGENTS.md`:

```markdown
# StateFuzz-SSM 仓库规则

## 隔离

- 禁止读取、复制、导入或依赖HybridKV及其他旧项目代码和结果。
- 公共模型、数据和论文必须独立记录版本、许可证和哈希。

## 工程纪律

- Python遵循PEP 8，使用4空格和UTF-8。
- 注释、docstring和文档使用中文。
- 功能变更先写失败测试，再写最小实现。
- 随机性必须使用显式局部随机源。
- 超过10分钟的任务必须原子检查点、支持resume并可外部观察。
```

- [ ] **Step 4：验证并提交**

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/test_import.py -q
git add .claude_resources.json .gitignore AGENTS.md README.md pyproject.toml src/statefuzz/__init__.py tests/test_import.py
git commit -m "chore: initialize isolated StateFuzz-SSM repository"
git status --short
```

Expected: `1 passed`，提交成功，工作树无输出。

## Task 2：实现ProbeSpec和规范化哈希

**Files:**
- Create: `src/statefuzz/probes/__init__.py`
- Create: `src/statefuzz/probes/schema.py`
- Create: `tests/probes/test_schema.py`

- [ ] **Step 1：写失败测试**

Create `tests/probes/test_schema.py`:

```python
import pytest
from pydantic import ValidationError

from statefuzz.probes.schema import ProbeSpec, TaskFamily


def test_hash_is_canonical_and_seed_sensitive() -> None:
    left = ProbeSpec(task=TaskFamily.SINGLE_KEY, seed=7)
    same = ProbeSpec(seed=7, task="single_key")
    other = ProbeSpec(task=TaskFamily.SINGLE_KEY, seed=8)
    assert left.config_hash == same.config_hash
    assert left.config_hash != other.config_hash


@pytest.mark.parametrize(
    "field,value", [("context_tokens", 63), ("n_items", 0), ("query_fanout", 0)]
)
def test_rejects_out_of_range_values(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ProbeSpec(**{field: value})


def test_single_key_rejects_multiple_queries() -> None:
    with pytest.raises(ValidationError, match="single_key"):
        ProbeSpec(task="single_key", query_fanout=2)


def test_fanout_cannot_exceed_items() -> None:
    with pytest.raises(ValidationError, match="n_items"):
        ProbeSpec(task="multi_key", n_items=2, query_fanout=3)
```

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/probes/test_schema.py -q
```

Expected: FAIL，无法导入schema。

- [ ] **Step 2：实现完整schema**

Create `src/statefuzz/probes/__init__.py`:

```python
"""StateProbeIR schema、编译器、oracle和变形关系。"""
```

Create `src/statefuzz/probes/schema.py`:

```python
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
            self.canonical_payload(), ensure_ascii=False,
            sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
```

- [ ] **Step 3：验证并提交**

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/probes/test_schema.py -q
git add src/statefuzz/probes tests/probes/test_schema.py
git commit -m "feat: define canonical StateProbeIR schema"
```

Expected: `6 passed`，提交成功。

## Task 3：实现精确oracle和确定性编译器

**Files:**
- Create: `src/statefuzz/probes/oracle.py`
- Create: `src/statefuzz/probes/compiler.py`
- Create: `tests/probes/test_compiler.py`

- [ ] **Step 1：写失败测试**

Create `tests/probes/test_compiler.py`:

```python
import random

from statefuzz.probes.compiler import compile_probe
from statefuzz.probes.schema import ProbeSpec


def test_probe_is_byte_deterministic() -> None:
    spec = ProbeSpec(seed=11, n_items=8)
    assert compile_probe(spec) == compile_probe(spec)


def test_multi_key_answer_matches_provenance() -> None:
    probe = compile_probe(
        ProbeSpec(task="multi_key", seed=23, n_items=12, query_fanout=3)
    )
    assert probe.answer == "|".join(probe.provenance["queried_values"])


def test_compiler_does_not_modify_global_rng() -> None:
    random.seed(99)
    expected = random.random()
    random.seed(99)
    compile_probe(ProbeSpec(seed=7))
    assert random.random() == expected


def test_target_position_controls_primary_index() -> None:
    early = compile_probe(ProbeSpec(seed=7, n_items=10, target_position=0.0))
    late = compile_probe(ProbeSpec(seed=7, n_items=10, target_position=1.0))
    assert early.provenance["primary_index"] == 0
    assert late.provenance["primary_index"] == 9
```

Run:

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/probes/test_compiler.py -q
```

Expected: FAIL，无法导入compiler。

- [ ] **Step 2：实现oracle**

Create `src/statefuzz/probes/oracle.py`:

```python
def join_values(values: list[str]) -> str:
    """按查询顺序连接精确值。"""
    if not values:
        raise ValueError("oracle至少需要一个值")
    if any("|" in value for value in values):
        raise ValueError("值中不得包含答案分隔符")
    return "|".join(values)
```

- [ ] **Step 3：实现完整编译器**

Create `src/statefuzz/probes/compiler.py`:

```python
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from statefuzz.probes.oracle import join_values
from statefuzz.probes.schema import ProbeSpec, TaskFamily


@dataclass(frozen=True)
class CompiledProbe:
    spec: ProbeSpec
    prompt: str
    answer: str
    provenance: dict[str, Any]
    probe_hash: str


_RECORDS = (
    "记录 {key} = {value}",
    "映射 {key} -> {value}",
    "条目[{key}]的值为{value}",
    "保存键{key}，对应值{value}",
    "数据 {key}:{value}",
    "关系 {key} 等于 {value}",
    "缓存项 {key} 取值 {value}",
    "事实({key},{value})",
)
_QUERIES = (
    "请按键顺序仅输出对应值并用竖线连接：{keys}",
    "依序返回这些键的值，不要输出其他文字：{keys}",
    "恢复这些键的值，格式为值1|值2：{keys}",
    "严格按顺序回答并使用|：{keys}",
    "检索键{keys}，只返回组合答案。",
    "读取{keys}对应的数据并用|连接。",
    "找出{keys}的值，只输出最终序列。",
    "执行精确检索：{keys}。只返回值。",
)


def _make_pairs(rng: random.Random, n_items: int) -> list[tuple[str, str]]:
    numbers = rng.sample(range(10000, 100000), n_items)
    pairs = [(f"K{i:04d}", f"V{n:05d}") for i, n in enumerate(numbers)]
    rng.shuffle(pairs)
    return pairs


def compile_probe(spec: ProbeSpec) -> CompiledProbe:
    """将规范编译为提示、精确答案和可审计来源。"""
    rng = random.Random(spec.seed)
    pairs = _make_pairs(rng, spec.n_items)
    primary = round(spec.target_position * (spec.n_items - 1))
    indices = [primary]
    if spec.task is TaskFamily.MULTI_KEY:
        pool = [i for i in range(spec.n_items) if i != primary]
        indices.extend(sorted(rng.sample(pool, spec.query_fanout - 1)))
    queried = [pairs[i] for i in indices]
    lines = [_RECORDS[spec.template_id].format(key=k, value=v) for k, v in pairs]
    filler_count = max(0, spec.context_tokens - len(lines) * 4)
    filler = " ".join(f"中性词{i % 97:02d}" for i in range(filler_count))
    keys = [k for k, _ in queried]
    values = [v for _, v in queried]
    query = _QUERIES[spec.template_id].format(keys=",".join(keys))
    prompt = "\n".join([*lines, filler, query]).strip() + "\n"
    answer = join_values(values)
    provenance: dict[str, Any] = {
        "config_hash": spec.config_hash,
        "primary_index": primary,
        "queried_indices": indices,
        "queried_keys": keys,
        "queried_values": values,
        "pairs": pairs,
    }
    payload = json.dumps(
        {"spec": spec.canonical_payload(), "prompt": prompt, "answer": answer},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return CompiledProbe(
        spec=spec,
        prompt=prompt,
        answer=answer,
        provenance=provenance,
        probe_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
```

- [ ] **Step 4：验证并提交**

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/probes/test_compiler.py -q
/202532803004/conda_envs/amber/bin/python -c "from statefuzz.probes.compiler import compile_probe; from statefuzz.probes.schema import ProbeSpec; x=[compile_probe(ProbeSpec(seed=i)).probe_hash for i in range(100)]; assert len(set(x))==100; print('COMPILE_100_PASS')"
git add src/statefuzz/probes/oracle.py src/statefuzz/probes/compiler.py tests/probes/test_compiler.py
git commit -m "feat: compile deterministic executable probes"
```

Expected: `4 passed`和`COMPILE_100_PASS`。

## Task 4：实现语义保持变形并验收

**Files:**
- Create: `src/statefuzz/probes/metamorphic.py`
- Create: `tests/probes/test_metamorphic.py`
- Create: `docs/MILESTONE1_REPORT.md`

- [ ] **Step 1：写失败测试**

Create `tests/probes/test_metamorphic.py`:

```python
import pytest

from statefuzz.probes.compiler import compile_probe
from statefuzz.probes.metamorphic import rename_symbols, switch_template
from statefuzz.probes.schema import ProbeSpec


def test_rename_preserves_answer_and_is_deterministic() -> None:
    original = compile_probe(ProbeSpec(seed=5))
    left = rename_symbols(original, "audit")
    right = rename_symbols(original, "audit")
    assert left == right
    assert left.answer == original.answer
    assert left.prompt != original.prompt


def test_template_switch_preserves_answer() -> None:
    original = ProbeSpec(seed=5, template_id=0)
    switched = switch_template(original, 1)
    assert compile_probe(switched).answer == compile_probe(original).answer


@pytest.mark.parametrize("template_id", [0, 8])
def test_template_switch_rejects_same_or_invalid(template_id: int) -> None:
    with pytest.raises(ValueError):
        switch_template(ProbeSpec(template_id=0), template_id)
```

- [ ] **Step 2：实现完整变形模块**

Create `src/statefuzz/probes/metamorphic.py`:

```python
from __future__ import annotations

import hashlib
import json

from statefuzz.probes.compiler import CompiledProbe
from statefuzz.probes.schema import ProbeSpec


def switch_template(spec: ProbeSpec, template_id: int) -> ProbeSpec:
    """切换提示模板，同时保持探针语义参数不变。"""
    if template_id == spec.template_id:
        raise ValueError("新模板必须不同于原模板")
    if not 0 <= template_id <= 7:
        raise ValueError("template_id必须位于0到7")
    return ProbeSpec(**{**spec.canonical_payload(), "template_id": template_id})


def rename_symbols(compiled: CompiledProbe, salt: str) -> CompiledProbe:
    """对全部键执行确定性双射重命名，保持答案不变。"""
    if not salt:
        raise ValueError("salt不能为空")
    old_keys = [key for key, _ in compiled.provenance["pairs"]]
    new_keys = [
        "R" + hashlib.sha256(f"{salt}:{key}".encode("utf-8")).hexdigest()[:8]
        for key in old_keys
    ]
    if len(set(new_keys)) != len(new_keys):
        raise RuntimeError("键重命名发生碰撞")
    mapping = dict(zip(old_keys, new_keys, strict=True))
    prompt = compiled.prompt
    for key in old_keys:
        prompt = prompt.replace(key, mapping[key])
    provenance = dict(compiled.provenance)
    provenance["pairs"] = [(mapping[k], v) for k, v in provenance["pairs"]]
    provenance["queried_keys"] = [mapping[k] for k in provenance["queried_keys"]]
    provenance["metamorphic_parent"] = compiled.probe_hash
    payload = json.dumps(
        {"prompt": prompt, "answer": compiled.answer, "parent": compiled.probe_hash},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return CompiledProbe(
        spec=compiled.spec,
        prompt=prompt,
        answer=compiled.answer,
        provenance=provenance,
        probe_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )
```

- [ ] **Step 3：运行全量验证**

```bash
mkdir -p runs/milestone1
/202532803004/conda_envs/amber/bin/python -m pytest -q --junitxml=runs/milestone1/pytest.xml
```

Expected: 15项测试通过，0失败、0错误。

- [ ] **Step 4：创建验收报告**

Create `docs/MILESTONE1_REPORT.md`:

```markdown
# StateFuzz-SSM Milestone 1验收报告

## 范围

本阶段仅交付CPU侧StateProbeIR核心：严格schema、规范化哈希、精确oracle、确定性编译器、符号重命名和模板切换。未下载模型、未占用GPU、未实现搜索器或StatePolicyIR。

## 验收条件

- pytest：15 passed，0 failed，0 errors；
- 100个不同seed生成100个不同probe hash；
- 编译器不修改Python全局RNG；
- 相同ProbeSpec完全确定；
- 两种变形保持精确答案；
- 仓库没有旧项目资产。

## 下一步

只有本报告与测试证据一致后，才编写Milestone 2计划：原子缓存、参数域隔离、fake/HF后端、Mamba2Mixer/FalconH1Mixer状态hook和Gate 0 GPU冒烟。
```

- [ ] **Step 5：隔离检查、提交和标签**

```bash
rg -n "HybridKV|hybridkv" . --glob '!AGENTS.md' --glob '!README.md' --glob '!docs/MILESTONE1_REPORT.md' --glob '!docs/superpowers/**'
git add src/statefuzz/probes/metamorphic.py tests/probes/test_metamorphic.py docs/MILESTONE1_REPORT.md
git commit -m "feat: validate StateProbeIR milestone one"
git tag -a milestone1-probeir -m "StateProbeIR CPU core validated"
git status --short
```

Expected: `rg`无输出，提交和标签成功，工作树无输出。

## 硬停止点

- 任一测试失败或测试数不是15：停止，不创建Milestone 2计划。
- 确定性、全局RNG或变形不变量失败：视为关键失败。
- 发现旧项目资产：先报告路径，不擅自清理。
- 全部通过：回传报告到`D:\小论文2`，再单独规划Milestone 2。

## 计划自审

- 单一子系统，可独立测试和提交。
- 所有代码文件均给出完整内容。
- 类型名称一致：`ProbeSpec`、`TaskFamily`、`CompiledProbe`。
- 测试先于实现，命令和预期结果明确。
- 不安装依赖、不用GPU、不触碰旧项目。
- 无占位符、无省略实现或“同上”。

## 执行方式

默认在当前会话内联执行。只有用户明确要求子代理时才切换；当前不自动派发。
