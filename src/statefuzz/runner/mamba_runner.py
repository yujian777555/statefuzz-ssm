"""Mamba/Mamba2风格模型的可注入执行器。"""

from __future__ import annotations

import copy
from typing import Any, Callable

from statefuzz.probes.compiler import CompiledProbe
from statefuzz.runner.base import ProbeRunner


def _copy_hidden_state(value: Any) -> Any:
    """对张量执行脱离计算图的CPU复制，对其他对象执行深复制。"""
    if hasattr(value, "detach") and hasattr(value, "clone"):
        detached = value.detach().clone()
        return detached.cpu() if hasattr(detached, "cpu") else detached
    return copy.deepcopy(value)


class MambaRunner:
    """通过预测函数注入模型，避免在协议层绑定具体权重或框架。"""

    def __init__(
        self,
        predictor: Callable[[str], str],
        hidden_state_provider: Callable[[], Any] | None = None,
    ) -> None:
        self._predictor = predictor
        self._hidden_state_provider = hidden_state_provider
        self._last_hidden_state: Any = None

    def run_probe(self, probe: CompiledProbe) -> str:
        """执行探针并在预测完成后捕获隐藏状态。"""
        if not isinstance(probe, CompiledProbe):
            raise TypeError("probe必须是CompiledProbe")
        prediction = self._predictor(probe.prompt)
        if not isinstance(prediction, str):
            raise TypeError("模型预测必须是字符串")
        if self._hidden_state_provider is not None:
            self._last_hidden_state = _copy_hidden_state(self._hidden_state_provider())
        else:
            self._last_hidden_state = None
        return prediction

    def capture_hidden_state(self) -> Any:
        """返回最近一次捕获结果的再次复制，防止调用方修改内部状态。"""
        return _copy_hidden_state(self._last_hidden_state)


assert isinstance(MambaRunner(lambda _: ""), ProbeRunner)

