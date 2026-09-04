"""探针编译结果到评价结果之间的最小执行契约。"""

from __future__ import annotations

from dataclasses import dataclass

from statefuzz.probes.compiler import CompiledProbe, compile_probe
from statefuzz.probes.schema import ProbeSpec


@dataclass(frozen=True)
class EvaluationOutcome:
    """一次探针预测的可审计评价结果。"""

    probe_hash: str
    expected: str
    prediction: str
    score: float

    def to_dict(self) -> dict[str, str | float]:
        """返回可写入结果缓存的基础字段。"""
        return {
            "probe_hash": self.probe_hash,
            "expected": self.expected,
            "prediction": self.prediction,
            "score": self.score,
        }


def evaluate_compiled_probe(
    probe: CompiledProbe, prediction: str
) -> EvaluationOutcome:
    """对编译探针执行严格字符串匹配，不修改探针对象。"""
    if not isinstance(probe, CompiledProbe):
        raise TypeError("probe必须是CompiledProbe")
    if not isinstance(prediction, str):
        raise TypeError("prediction必须是字符串")
    if not isinstance(probe.answer, str) or not probe.answer:
        raise ValueError("CompiledProbe的答案非法")
    if not isinstance(probe.probe_hash, str) or not probe.probe_hash:
        raise ValueError("CompiledProbe的哈希非法")
    score = 1.0 if prediction == probe.answer else 0.0
    return EvaluationOutcome(
        probe_hash=probe.probe_hash,
        expected=probe.answer,
        prediction=prediction,
        score=score,
    )


def evaluate_probe_spec(spec: ProbeSpec, prediction: str) -> EvaluationOutcome:
    """编译规范并立即评价预测，作为未来执行器的稳定适配入口。"""
    if not isinstance(spec, ProbeSpec):
        raise TypeError("spec必须是ProbeSpec")
    return evaluate_compiled_probe(compile_probe(spec), prediction)

