"""不改权重的轻量记忆鲁棒性缓解策略。"""

from __future__ import annotations

MITIGATION_STRATEGIES = ("memory_reinjection", "context_anchor", "retrieval_reminder")


def apply_mitigation(
    prompt: str,
    expected_answer: str,
    strategy: str,
    *,
    query_suffix: str,
) -> str:
    """在原查询前增加固定格式提醒，并重新落到同一查询后缀。"""
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt必须是非空字符串")
    if expected_answer not in (" red", " blue"):
        raise ValueError("当前缓解协议只支持red/blue单token事实")
    if strategy not in MITIGATION_STRATEGIES:
        raise ValueError("未知缓解策略")
    if not isinstance(query_suffix, str) or not query_suffix:
        raise ValueError("query_suffix必须是非空字符串")
    if strategy == "memory_reinjection":
        reminder = f"Memory refresh: the stored answer is{expected_answer}.\n"
    elif strategy == "context_anchor":
        reminder = f"Important anchor — retain the answer{expected_answer} for the final query.\n"
    else:
        reminder = f"Retrieved reminder: answer={expected_answer.strip()}; use this retrieved fact.\n"
    return prompt + "\n" + reminder + query_suffix


def mitigation_overhead(token_counter, baseline_prompt: str, mitigated_prompt: str) -> dict[str, int]:
    """报告真实token开销，而非用字符数估计。"""
    if not callable(token_counter):
        raise TypeError("token_counter必须可调用")
    baseline = int(token_counter(baseline_prompt))
    mitigated = int(token_counter(mitigated_prompt))
    return {"baseline_tokens": baseline, "mitigated_tokens": mitigated, "overhead_tokens": mitigated - baseline}
