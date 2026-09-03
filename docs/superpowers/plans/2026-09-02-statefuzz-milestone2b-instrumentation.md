# StateFuzz-SSM Milestone 2B Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended; current执行采用内联) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改模型权重的前提下，交付模型注册清单、状态张量统计和可逆白名单hook，为Gate 0真实GPU冒烟准备接口。

**Architecture:** `ModelSpec`从YAML读取模型ID、角色和家族；注册脚本通过镜像静态`config.json`请求记录可达性和内容哈希。`state_metrics`只处理显式tensor并在CPU float64计算，`StateCapture`仅挂载`Mamba2Mixer`与`FalconH1Mixer`，以forward hook捕获cache中的`ssm_states`/`conv_states`副本，不改变前向返回值。

**Tech Stack:** Python 3.10.20、PyTorch 2.5.1+cu121、PyYAML 6.0.3、pytest 9.1.1；固定解释器`/202532803004/conda_envs/amber/bin/python`，不新增依赖。

---

## 执行约束

- 只在`feat/milestone2b-instrumentation`隔离工作树开发，完成后再合并。
- 不读取、复制、导入或依赖HybridKV。
- 不修改共享amber环境，不访问模型权重正文以外的私密信息。
- 模型下载前必须先通过静态config请求并记录SHA-256。
- 当前阶段不启动长任务；若后续冒烟超过10分钟，必须原子checkpoint和resume。

## Task 1：模型注册清单和静态镜像验证

**Files:**
- Create: `configs/models/gate0.yaml`
- Create: `src/statefuzz/models/__init__.py`
- Create: `src/statefuzz/models/registry.py`
- Create: `scripts/validate_models.py`
- Create: `tests/models/test_registry.py`

- [ ] **Step 1：写失败测试**

```python
import importlib.util

import pytest


SPEC = importlib.util.find_spec("statefuzz.models.registry")
requires_registry = pytest.mark.skipif(SPEC is None, reason="registry尚未实现")


def test_registry_module_exists() -> None:
    assert SPEC is not None


@requires_registry
def test_load_registry_rejects_duplicate_ids(tmp_path) -> None:
    from statefuzz.models.registry import load_registry

    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  - id: a\n    role: x\n    family: y\n  - id: a\n    role: z\n    family: y\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        load_registry(config)


@requires_registry
def test_load_registry_preserves_order(tmp_path) -> None:
    from statefuzz.models.registry import load_registry

    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  - id: first\n    role: ssm\n    family: mamba\n  - id: second\n    role: control\n    family: transformer\n",
        encoding="utf-8",
    )
    assert [item.model_id for item in load_registry(config)] == ["first", "second"]
```

Run: `/202532803004/conda_envs/amber/bin/python -m pytest tests/models/test_registry.py -q`  
Expected: 1 failed, 2 skipped。

- [ ] **Step 2：实现完整registry和配置**

Create `configs/models/gate0.yaml`:

```yaml
models:
  - id: state-spaces/mamba-130m-hf
    role: development_ssm
    family: mamba
    trust_remote_code: false
  - id: tiiuae/Falcon-H1-0.5B-Instruct
    role: development_hybrid
    family: falcon_h1
    trust_remote_code: false
  - id: EleutherAI/pythia-410m
    role: transformer_control
    family: transformer
    trust_remote_code: false
```

Create `src/statefuzz/models/__init__.py`:

```python
"""模型注册与版本审计。"""
```

Create `src/statefuzz/models/registry.py`:

```python
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
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("models"), list):
        raise ValueError("模型配置必须包含models列表")
    specs = [ModelSpec(**item) for item in data["models"]]
    ids = [item.model_id for item in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("模型ID重复")
    return specs
```

Create `scripts/validate_models.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from statefuzz.io_atomic import atomic_write_json
from statefuzz.models.registry import load_registry


def fetch_config(model_id: str, endpoint: str) -> dict[str, object]:
    url = f"{endpoint.rstrip('/')}/{model_id}/resolve/main/config.json"
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
        return {
            "url": url,
            "http_status": response.status,
            "config_sha256": hashlib.sha256(body).hexdigest(),
            "config_bytes": len(body),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="验证Gate 0模型静态配置")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for spec in load_registry(args.config):
        record = {"model_id": spec.model_id, "role": spec.role, "family": spec.family}
        try:
            record.update({"available": True, **fetch_config(spec.model_id, args.endpoint)})
        except Exception as exc:
            record.update({"available": False, "error_type": type(exc).__name__})
        records.append(record)
    atomic_write_json(args.output, {"endpoint": args.endpoint, "models": records})
    print(f"MODEL_REGISTRY={args.output}")
    return 0 if all(item["available"] for item in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3：运行并提交**

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/models/test_registry.py -q
HF_ENDPOINT=https://hf-mirror.com /202532803004/conda_envs/amber/bin/python scripts/validate_models.py --config configs/models/gate0.yaml --output runs/milestone2b/model_registry.json
git add configs/models src/statefuzz/models scripts/validate_models.py tests/models/test_registry.py
git commit -m "feat: add audited Gate 0 model registry"
```

Expected: 3 passed；三个模型均`available=true`且有config SHA-256。

## Task 2：状态统计量

**Files:**
- Create: `src/statefuzz/instrumentation/__init__.py`
- Create: `src/statefuzz/instrumentation/state_metrics.py`
- Create: `tests/instrumentation/test_state_metrics.py`

- [ ] **Step 1：写失败测试**

```python
import importlib.util

import pytest
import torch


SPEC = importlib.util.find_spec("statefuzz.instrumentation.state_metrics")
requires_metrics = pytest.mark.skipif(SPEC is None, reason="metrics尚未实现")


def test_metrics_module_exists() -> None:
    assert SPEC is not None


@requires_metrics
def test_metrics_report_finite_norm_and_rank() -> None:
    from statefuzz.instrumentation.state_metrics import summarize_state

    state = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    metrics = summarize_state(state)
    assert metrics["finite_fraction"] == 1.0
    assert metrics["l2_norm"] > 0.0
    assert metrics["effective_rank"] == 2.0


@requires_metrics
def test_metrics_report_nonfinite_fraction() -> None:
    from statefuzz.instrumentation.state_metrics import summarize_state

    metrics = summarize_state(torch.tensor([1.0, float("nan")]))
    assert metrics["finite_fraction"] == 0.5
```

Run: `python -m pytest tests/instrumentation/test_state_metrics.py -q`  
Expected: 1 failed, 2 skipped。

- [ ] **Step 2：实现完整指标**

Create `src/statefuzz/instrumentation/__init__.py`:

```python
"""SSM内部状态捕获与统计。"""
```

Create `src/statefuzz/instrumentation/state_metrics.py`:

```python
from __future__ import annotations

import math

import torch


def summarize_state(state: torch.Tensor) -> dict[str, float]:
    """在CPU float64上计算不改变原tensor的状态摘要。"""
    value = state.detach().to(device="cpu", dtype=torch.float64)
    flat = value.reshape(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    if finite_values.numel() == 0:
        return {
            "finite_fraction": 0.0,
            "l2_norm": math.inf,
            "max_abs": math.inf,
            "mean_abs": math.inf,
            "effective_rank": 0.0,
        }
    matrix = value.reshape(value.shape[0], -1)
    singular = torch.linalg.svdvals(matrix[torch.isfinite(matrix).all(dim=1)])
    if singular.numel() == 0 or float(singular.sum()) == 0.0:
        effective_rank = 0.0
    else:
        prob = singular / singular.sum()
        entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum()
        effective_rank = float(torch.exp(entropy))
    return {
        "finite_fraction": float(finite.float().mean()),
        "l2_norm": float(torch.linalg.vector_norm(finite_values)),
        "max_abs": float(finite_values.abs().max()),
        "mean_abs": float(finite_values.abs().mean()),
        "effective_rank": effective_rank,
    }
```

- [ ] **Step 3：运行并提交**

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/instrumentation/test_state_metrics.py -q
git add src/statefuzz/instrumentation tests/instrumentation/test_state_metrics.py
git commit -m "feat: add deterministic SSM state metrics"
```

Expected: 3 passed。

## Task 3：可逆白名单StateCapture hook

**Files:**
- Create: `src/statefuzz/instrumentation/state_hooks.py`
- Create: `tests/instrumentation/test_state_hooks.py`

- [ ] **Step 1：写失败测试**

```python
import importlib.util

import pytest
import torch


SPEC = importlib.util.find_spec("statefuzz.instrumentation.state_hooks")
requires_hooks = pytest.mark.skipif(SPEC is None, reason="state_hooks尚未实现")


def test_hooks_module_exists() -> None:
    assert SPEC is not None


class FakeCache:
    def __init__(self) -> None:
        self.ssm_states = torch.ones(1, 2, 2)


class Mamba2Mixer(torch.nn.Module):
    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        cache_params.ssm_states.mul_(2.0)
        return hidden_states + 1.0


class OtherMixer(torch.nn.Module):
    def forward(self, hidden_states, cache_params=None, attention_mask=None):
        return hidden_states


@requires_hooks
def test_capture_copies_state_and_preserves_output() -> None:
    from statefuzz.instrumentation.state_hooks import StateCapture

    module = Mamba2Mixer()
    capture = StateCapture(module)
    cache = FakeCache()
    output = module(torch.zeros(1, 2, 2), cache_params=cache)
    capture.close()
    assert torch.equal(output, torch.ones(1, 2, 2))
    assert capture.snapshots[0]["ssm_states"].equal(torch.full((1, 2, 2), 2.0))


@requires_hooks
def test_capture_rejects_unapproved_class() -> None:
    from statefuzz.instrumentation.state_hooks import StateCapture

    with pytest.raises(ValueError, match="白名单"):
        StateCapture(OtherMixer())
```

Run: `python -m pytest tests/instrumentation/test_state_hooks.py -q`  
Expected: 1 failed, 2 skipped。

- [ ] **Step 2：实现完整hook**

Create `src/statefuzz/instrumentation/state_hooks.py`:

```python
from __future__ import annotations

from typing import Any

import torch


_ALLOWED = {"Mamba2Mixer", "FalconH1Mixer"}


class StateCapture:
    """捕获SSM cache副本，不改变模块输出，关闭时移除hook。"""

    def __init__(self, module: torch.nn.Module) -> None:
        if module.__class__.__name__ not in _ALLOWED:
            raise ValueError("模块类不在StateCapture白名单")
        self.snapshots: list[dict[str, torch.Tensor]] = []
        self._handle = module.register_forward_hook(self._after_forward, with_kwargs=True)

    def _after_forward(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        cache = kwargs.get("cache_params")
        if cache is None:
            return
        snapshot: dict[str, torch.Tensor] = {}
        for name in ("ssm_states", "conv_states"):
            value = getattr(cache, name, None)
            if isinstance(value, torch.Tensor):
                snapshot[name] = value.detach().cpu().clone()
        if snapshot:
            self.snapshots.append(snapshot)

    def close(self) -> None:
        self._handle.remove()
```

- [ ] **Step 3：运行并提交**

```bash
/202532803004/conda_envs/amber/bin/python -m pytest tests/instrumentation -q
git add src/statefuzz/instrumentation/state_hooks.py tests/instrumentation/test_state_hooks.py
git commit -m "feat: add reversible SSM state capture hook"
```

Expected: 7项instrumentation测试通过。

## Task 4：2B验收报告

**Files:**
- Create: `docs/MILESTONE2B_REPORT.md`

- [ ] **Step 1：运行全量测试和模型静态验证**

```bash
/202532803004/conda_envs/amber/bin/python -m pytest -q --junitxml=runs/milestone2b/pytest.xml
HF_ENDPOINT=https://hf-mirror.com /202532803004/conda_envs/amber/bin/python scripts/validate_models.py --config configs/models/gate0.yaml --output runs/milestone2b/model_registry.json
```

Expected: 当前全仓库34项测试通过；模型静态配置三项HTTP 200并有SHA-256。

- [ ] **Step 2：写入报告并提交标签**

报告必须准确记录：测试总数、模型config哈希、镜像端点、状态hook捕获字段、失败/跳过项、无模型权重下载或GPU长任务的事实，以及HfApi请求超时但静态请求成功的环境限制。

```bash
git add docs/MILESTONE2B_REPORT.md
git commit -m "docs: validate StateFuzz-SSM instrumentation"
git tag -a milestone2b-instrumentation -m "Model audit and state instrumentation validated"
git status --short
```

Expected: 工作树干净，标签指向最新提交。

## 硬停止点

- 任一模型config不是HTTP 200：停止，不下载权重。
- 状态hook改变模块输出或无法移除：停止。
- 任一状态指标产生NaN而未被明确报告：停止。
- 全部通过后才编写Milestone 2C HF GPU冒烟计划。

## 计划自审

- 这是单一可测试子系统，未包含搜索器或StatePolicyIR。
- 所有代码文件和关键测试均给出完整内容。
- 仅使用已安装依赖，不安装Hypothesis或其他包。
- 计划不引用HybridKV实现。
