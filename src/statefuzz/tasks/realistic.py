"""受控、可复现的现实工作负载启发任务。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealisticMemoryTask:
    task_type: str
    family: str
    seed: int
    prompt: str
    hidden_memory_fact: str
    expected_answer: str
    distractor_context: str
    query_position: float
    success_metric: str = "expected_answer_is_top1_candidate"

    def to_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type,
            "family": self.family,
            "seed": self.seed,
            "hidden_memory_fact": self.hidden_memory_fact,
            "expected_answer": self.expected_answer,
            "query_position": self.query_position,
            "success_metric": self.success_metric,
        }


_TASK_TYPES = {"long_document_retrieval", "code_context_dependency", "agent_conversation_memory"}
_FAMILIES = {"structured_repetitive", "periodic_pattern", "interleaved_distractor", "semantic_distractor"}


def _distractors(family: str, seed: int, count: int) -> list[str]:
    if family == "structured_repetitive":
        return [f"Document note {seed:04d}-{index:03d} remains stable and unrelated.\n" for index in range(count)]
    if family == "periodic_pattern":
        pattern = ("alpha beta gamma", "delta epsilon zeta", "eta theta iota")
        return [f"Pattern {pattern[index % 3]} cycle {index % 3}.\n" for index in range(count)]
    if family == "interleaved_distractor":
        return [f"Distractor event {seed:04d}-{index:03d} has marker {index % 2}.\n" for index in range(count)]
    return [f"A realistic report {seed:04d}-{index:03d} discusses a laboratory, a bridge, and a quiet archive.\n" for index in range(count)]


def generate_realistic_memory_task(
    task_type: str,
    family: str,
    *,
    seed: int = 0,
    distractor_count: int = 32,
    query_position: float = 0.0,
) -> RealisticMemoryTask:
    """生成现实工作负载启发任务；不在生成后按结果挑选任务。"""
    if task_type not in _TASK_TYPES:
        raise ValueError("未知现实任务类型")
    if family not in _FAMILIES:
        raise ValueError("未知stress family")
    if isinstance(distractor_count, bool) or not isinstance(distractor_count, int) or distractor_count < 2:
        raise ValueError("distractor_count必须是不小于2的整数")
    if not 0.0 <= float(query_position) <= 1.0:
        raise ValueError("query_position必须位于0到1之间")
    answer = " red" if seed % 2 == 0 else " blue"
    alternative = " blue" if answer == " red" else " red"
    if task_type == "long_document_retrieval":
        fact = f"The report's confidential classification is{answer}.\n"
        query = "According to the confidential classification, the answer is"
    elif task_type == "code_context_dependency":
        fact = f"const SERVICE_COLOR = '{answer.strip()}';\n"
        query = "What value should SERVICE_COLOR return? It returns"
    else:
        fact = f"User memory: the preferred response color is{answer}.\n"
        query = "What color did the user prefer? The answer is"
    distractors = _distractors(family, seed, distractor_count)
    insertion = int(round(float(query_position) * len(distractors)))
    body = "".join(distractors[:insertion]) + fact + "".join(distractors[insertion:])
    prompt = body + query
    return RealisticMemoryTask(task_type, family, seed, prompt, fact, answer, "".join(distractors), float(query_position))
