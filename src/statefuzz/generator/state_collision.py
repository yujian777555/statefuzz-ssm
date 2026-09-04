"""生成多键干扰以观察状态表示碰撞的压力样例。"""

from __future__ import annotations

from collections.abc import Iterable

from statefuzz.generator import _tag_probe
from statefuzz.probes.compiler import CompiledProbe, compile_probe
from statefuzz.probes.schema import ProbeSpec, TaskFamily


def generate_collision_probe(
    seed: int = 0,
    context_tokens: int = 4096,
    n_items: int = 32,
    query_fanout: int | None = None,
    interference_strength: float = 0.25,
) -> CompiledProbe:
    """生成中部目标和多键查询，放大键间状态干扰。"""
    if not 0.0 < interference_strength <= 1.0:
        raise ValueError("interference_strength必须位于0到1之间")
    if query_fanout is None:
        query_fanout = max(2, min(n_items, round(n_items * interference_strength)))
    spec = ProbeSpec(
        task=TaskFamily.MULTI_KEY,
        seed=seed,
        context_tokens=context_tokens,
        n_items=n_items,
        query_fanout=query_fanout,
        target_position=0.5,
        template_id=4,
    )
    probe = compile_probe(spec)
    return _tag_probe(
        probe,
        "state_collision",
        query_fanout=query_fanout,
        interference_strength=interference_strength,
        interference_items=n_items - query_fanout,
    )


def generate_collision_family(
    seeds: Iterable[int] = (0,),
    interference_strengths: Iterable[float] = (0.25,),
    context_lengths: Iterable[int] = (4096,),
    n_items: int = 32,
) -> list[CompiledProbe]:
    """生成干扰强度与上下文长度的可搜索样例族。"""
    return [
        generate_collision_probe(
            seed=seed,
            context_tokens=context_tokens,
            n_items=n_items,
            interference_strength=strength,
        )
        for seed in seeds
        for context_tokens in context_lengths
        for strength in interference_strengths
    ]

