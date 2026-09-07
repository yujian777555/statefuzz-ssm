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


@runtime_checkable
class RemoteMemoryRunner(Protocol):
    """远程记忆行为评估的最小协议，不包含架构特定状态语义。"""

    def single_token_id(self, text: str) -> int | None:
        """返回单token候选的id。"""

    def count_tokens(self, prompt: str) -> int:
        """返回实际token数量。"""

    def score_candidate_tokens(
        self, prompt: str, candidate_token_ids: list[int]
    ) -> dict[str, Any]:
        """返回显式候选行为评分。"""

    def score_remote_memory_pair(self, pair: Any, **kwargs: Any) -> dict[str, Any]:
        """返回长度匹配的远程记忆行为证据。"""
