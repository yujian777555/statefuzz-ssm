"""模型执行器的最小可替换接口。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from statefuzz.probes.compiler import CompiledProbe


@runtime_checkable
class ProbeRunner(Protocol):
    """能够执行探针并读取最近隐藏状态的运行器协议。"""

    def run_probe(self, probe: CompiledProbe) -> str:
        """执行一个已编译探针并返回字符串预测。"""

    def capture_hidden_state(self) -> Any:
        """返回最近一次执行捕获的隐藏状态副本。"""

