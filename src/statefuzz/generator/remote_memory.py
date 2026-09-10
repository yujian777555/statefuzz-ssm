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
    filler_style: str = "structured_repetitive"


@dataclass(frozen=True)
class RemoteMemoryBudgetFit:
    """实际token预算拟合结果。"""

    status: str
    pair: RemoteMemoryPair | None
    target_tokens: int
    actual_tokens: int | None
    absolute_error: int | None
    iterations: int


_TEMPLATES: dict[int, tuple[str, str]] = {
    0: ("The stored color is", "The stored color is"),
    1: ("The remembered word is", "The remembered word is"),
    2: ("Memory record:", "Memory record:"),
}

STRESS_FAMILIES = (
    "structured_repetitive",
    "periodic_pattern",
    "interleaved_distractor",
    "semantic_distractor",
)
NEGATIVE_CONTROL_FAMILIES = ("lexically_diverse",)


def _validate_value(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\n" in value:
        raise ValueError(f"{name}必须是单行非空字符串")
    if not value.startswith(" "):
        raise ValueError(f"{name}必须保留一个前导空格")
    return value


def _filler_lines(
    seed: int, count: int, template_id: int, filler_style: str
) -> list[str]:
    """生成不包含候选值的确定性填充行。"""
    if filler_style == "structured_repetitive":
        topic = ("context notes", "earlier remarks", "memory details")[template_id]
        return [
            f"Earlier {topic} {seed:04d}-{index:03d} remain unrelated and stable.\n"
            for index in range(count)
        ]
    if filler_style == "lexically_diverse":
        pool = (
            "A quiet archive entry mentions a river and a distant station.",
            "The afternoon report describes a small instrument near a window.",
            "A catalog note records weather, paper, and an unused signal.",
            "The laboratory diary mentions a lantern beside a wooden shelf.",
            "An old travel log describes hills, clouds, and a narrow bridge.",
            "The neutral memo discusses tools, fabric, and a silent clock.",
            "A museum label references clay, glass, and a covered doorway.",
            "The survey note describes a garden path beside a stone wall.",
        )
        return [
            f"{pool[(seed + index) % len(pool)]} Ref-{seed:04d}-{index:03d}.\n"
            for index in range(count)
        ]
    if filler_style == "periodic_pattern":
        patterns = ("alpha beta gamma", "delta epsilon zeta", "eta theta iota")
        return [f"Pattern {patterns[index % len(patterns)]} cycle {index % 3}.\n" for index in range(count)]
    if filler_style == "interleaved_distractor":
        return [
            f"Earlier note {seed:04d}-{index:03d} is stable; distractor marker {index % 2}.\n"
            for index in range(count)
        ]
    if filler_style == "semantic_distractor":
        topics = (
            "The archivist catalogued a brass compass beside a map.",
            "The gardener described a shaded path behind the library.",
            "The engineer inspected a quiet motor near the window.",
        )
        return [f"{topics[(seed + index) % len(topics)]} Ref-{seed:04d}-{index:03d}.\n" for index in range(count)]
    raise ValueError("未知filler_style")


def generate_stress_family_pair(
    family: str,
    *,
    context_tokens: int = 64,
    seed: int = 0,
    template_id: int = 1,
    value_a: str = " red",
    value_b: str = " blue",
    target_position: float = 0.0,
    filler_slots: int | None = None,
) -> RemoteMemoryPair:
    """通过统一接口生成一个可复现的压力族任务。"""
    if family not in STRESS_FAMILIES + NEGATIVE_CONTROL_FAMILIES:
        raise ValueError(f"未知stress family: {family}")
    return generate_remote_memory_pair(
        context_tokens=context_tokens,
        seed=seed,
        template_id=template_id,
        value_a=value_a,
        value_b=value_b,
        target_position=target_position,
        filler_style=family,
        filler_slots=filler_slots,
    )


def generate_remote_memory_pair(
    context_tokens: int = 64,
    seed: int = 0,
    template_id: int = 0,
    value_a: str = " red",
    value_b: str = " blue",
    target_position: float = 0.0,
    filler_style: str = "structured_repetitive",
    filler_slots: int | None = None,
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
    if filler_slots is None:
        # 每个填充槽约含六个自然语言token；实际长度由真实tokenizer复核。
        filler_slots = max(2, context_tokens // 6)
    if isinstance(filler_slots, bool) or not isinstance(filler_slots, int) or filler_slots < 2:
        raise ValueError("filler_slots必须是不小于2的整数")
    insertion = int(round(float(target_position) * (filler_slots - 1)))
    fillers = _filler_lines(seed, filler_slots, template_id, filler_style)
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
        filler_style=filler_style,
    )


def generate_remote_memory_family(
    context_tokens: int = 64,
    seeds: Iterable[int] = (0,),
    template_ids: Iterable[int] = (0, 1, 2),
    value_pairs: Iterable[tuple[str, str]] = ((" red", " blue"), (" cat", " dog")),
    target_positions: Sequence[float] = (0.0,),
    filler_style: str = "structured_repetitive",
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
            filler_style=filler_style,
        )
        for seed in seed_values
        for template_id in template_values
        for values in pair_values
        for target_position in positions
    ]


def fit_remote_memory_pair_to_token_budget(
    token_counter,
    target_tokens: int,
    tolerance_tokens: int = 8,
    **pair_kwargs,
) -> RemoteMemoryBudgetFit:
    """仅改变填充槽数量，把A/B提示拟合到实际token预算。"""
    if isinstance(target_tokens, bool) or not isinstance(target_tokens, int) or target_tokens <= 0:
        raise ValueError("target_tokens必须是正整数")
    if isinstance(tolerance_tokens, bool) or not isinstance(tolerance_tokens, int) or tolerance_tokens < 0:
        raise ValueError("tolerance_tokens必须是非负整数")
    if not callable(token_counter):
        raise TypeError("token_counter必须可调用")
    kwargs = dict(pair_kwargs)
    kwargs.pop("filler_slots", None)
    kwargs.setdefault("context_tokens", target_tokens)
    low, high = 2, max(4, target_tokens * 2)
    best: tuple[int, RemoteMemoryPair] | None = None
    iterations = 0
    max_calls = 128
    while low <= high and iterations < max_calls:
        iterations += 1
        slots = (low + high) // 2
        pair = generate_remote_memory_pair(**kwargs, filler_slots=slots)
        count_a = int(token_counter(pair.prompt_a))
        count_b = int(token_counter(pair.prompt_b))
        if count_a == count_b:
            error = abs(count_a - target_tokens)
            if best is None or error < best[0]:
                best = (error, pair)
            if count_a < target_tokens:
                low = slots + 1
            elif count_a > target_tokens:
                high = slots - 1
            else:
                break
        else:
            low = slots + 1
    if best is not None:
        center = best[1].filler_slots
        for slots in range(max(2, center - 32), center + 33):
            if iterations >= max_calls:
                break
            iterations += 1
            pair = generate_remote_memory_pair(**kwargs, filler_slots=slots)
            count_a = int(token_counter(pair.prompt_a))
            count_b = int(token_counter(pair.prompt_b))
            if count_a != count_b:
                continue
            error = abs(count_a - target_tokens)
            if error < best[0]:
                best = (error, pair)
    if best is None or best[0] > tolerance_tokens:
        return RemoteMemoryBudgetFit(
            status="budget_unreachable",
            pair=None,
            target_tokens=target_tokens,
            actual_tokens=None,
            absolute_error=None,
            iterations=iterations,
        )
    actual = int(token_counter(best[1].prompt_a))
    return RemoteMemoryBudgetFit(
        status="ok",
        pair=best[1],
        target_tokens=target_tokens,
        actual_tokens=actual,
        absolute_error=best[0],
        iterations=iterations,
    )
