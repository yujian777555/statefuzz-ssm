"""探针编译结果到评价结果之间的最小执行契约。"""

from __future__ import annotations

from dataclasses import dataclass

from statefuzz.probes.compiler import CompiledProbe


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
    if not isinstance(prediction, str):
        raise TypeError("prediction必须是字符串")
    score = 1.0 if prediction == probe.answer else 0.0
    return EvaluationOutcome(
        probe_hash=probe.probe_hash,
        expected=probe.answer,
        prediction=prediction,
        score=score,
    )

