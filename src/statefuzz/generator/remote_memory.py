"""生成反事实远程记忆任务。

每个任务只替换远端记录的值，查询后缀和周围结构完全相同。值保留
前导空格，便于直接送入因果语言模型的单 token 候选校验。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteMemoryPair:
    """一个长度和结构匹配的远程记忆反事实对。"""

    prompt_a: str
    prompt_b: str
    value_a: str
    value_b: str
    template_id: int
    seed: int
    context_tokens: int
    target_position: float
    local_suffix: str
    filler_slots: int


_TEMPLATES: dict[int, tuple[str, str]] = {
    0: ("The stored color is", "The stored color is"),
    1: ("The remembered word is", "The remembered word is"),
    2: ("Memory record:", "Memory record:"),
}


def _validate_value(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name}必须是单行非空字符串")
    if not value.startswith(" "):
        raise ValueError(f"{name}必须保留一个前导空格")
    return value


def _filler_lines(seed: int, count: int, template_id: int) -> list[str]:
    """生成不包含候选值的确定性自然语言填充行。"""
    topic = ("context notes", "earlier remarks", "memory details")[template_id]
    return [
        f"Earlier {topic} {seed:04d}-{index:03d} remain unrelated and stable.\n"
        for index in range(count)
    ]


def generate_remote_memory_pair(
    context_tokens: int = 64,
    seed: int = 0,
    template_id: int = 0,
    value_a: str = " red",
    value_b: str = " blue",
    target_position: float = 0.0,
) -> RemoteMemoryPair:
    """生成只改变远端值的自然语言 A/B 提示对。

    ``context_tokens`` 用于控制填充规模；实际 tokenizer 长度由 runner
    再次测量。``target_position`` 在填充槽中选择远端记录的位置。
    """
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
        raise ValueError("context_tokens必须是正整数")
    if context_tokens <= 0:
        raise ValueError("context_tokens必须是正整数")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed必须是整数")
    if template_id not in _TEMPLATES:
        raise ValueError(f"未知template_id: {template_id}")
    if not isinstance(target_position, (int, float)) or not 0.0 <= float(
        target_position
    ) <= 1.0:
        raise ValueError("target_position必须位于0到1之间")
    value_a = _validate_value(value_a, "value_a")
    value_b = _validate_value(value_b, "value_b")
    if value_a == value_b:
        raise ValueError("value_a和value_b必须不同")

    local_suffix, _ = _TEMPLATES[template_id]
    # 每个填充槽约含六个自然语言token；实际长度由真实tokenizer复核。
    filler_slots = max(2, context_tokens // 6)
    insertion = int(round(float(target_position) * (filler_slots - 1)))
    fillers = _filler_lines(seed, filler_slots, template_id)
    record_a = f"{local_suffix}{value_a}\n"
    record_b = f"{local_suffix}{value_b}\n"
    prefix = "".join(fillers[:insertion])
    suffix = "".join(fillers[insertion:])
    prompt_a = prefix + record_a + suffix + local_suffix
    prompt_b = prefix + record_b + suffix + local_suffix
    return RemoteMemoryPair(
        prompt_a=prompt_a,
        prompt_b=prompt_b,
        value_a=value_a,
        value_b=value_b,
        template_id=template_id,
        seed=seed,
        context_tokens=context_tokens,
        target_position=float(target_position),
        local_suffix=local_suffix,
        filler_slots=filler_slots,
    )


def generate_remote_memory_family(
    context_tokens: int = 64,
    seeds: Iterable[int] = (0,),
    template_ids: Iterable[int] = (0, 1, 2),
    value_pairs: Iterable[tuple[str, str]] = ((" red", " blue"), (" cat", " dog")),
    target_positions: Sequence[float] = (0.0,),
) -> list[RemoteMemoryPair]:
    """生成校准/留出阶段使用的确定性任务族。"""
    seed_values = list(seeds)
    template_values = list(template_ids)
    pair_values = list(value_pairs)
    positions = list(target_positions)
    if not seed_values or not template_values or not pair_values or not positions:
        raise ValueError("seeds、template_ids、value_pairs和target_positions不能为空")
    return [
        generate_remote_memory_pair(
            context_tokens=context_tokens,
            seed=seed,
            template_id=template_id,
            value_a=values[0],
            value_b=values[1],
            target_position=target_position,
        )
        for seed in seed_values
        for template_id in template_values
        for values in pair_values
        for target_position in positions
    ]
