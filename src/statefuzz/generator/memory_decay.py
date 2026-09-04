"""生成早期记忆在长上下文中衰减的压力样例。"""

from __future__ import annotations

from collections.abc import Iterable

from statefuzz.generator import _tag_probe
from statefuzz.probes.compiler import CompiledProbe, compile_probe
from statefuzz.probes.schema import ProbeSpec


def generate_memory_decay_probe(
    seed: int = 0,
    context_tokens: int = 4096,
    n_items: int = 64,
    target_position: float = 0.0,
) -> CompiledProbe:
    """生成带可控目标位置的单键长上下文探针。"""
    spec = ProbeSpec(
        seed=seed,
        context_tokens=context_tokens,
        n_items=n_items,
        target_position=target_position,
        template_id=0,
    )
    probe = compile_probe(spec)
    return _tag_probe(probe, "memory_decay", target_position=target_position)


def generate_memory_decay_family(
    seeds: Iterable[int] = (0,),
    context_lengths: Iterable[int] = (4096,),
    target_positions: Iterable[float] = (0.0,),
    n_items: int = 64,
) -> list[CompiledProbe]:
    """生成上下文长度与目标位置的笛卡尔积样例族。"""
    return [
        generate_memory_decay_probe(
            seed=seed,
            context_tokens=context_tokens,
            n_items=n_items,
            target_position=target_position,
        )
        for seed in seeds
        for context_tokens in context_lengths
        for target_position in target_positions
    ]

