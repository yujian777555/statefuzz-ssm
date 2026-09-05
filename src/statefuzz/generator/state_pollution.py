"""生成高密度干扰内容以观察状态污染的压力样例。"""

from __future__ import annotations

from collections.abc import Iterable

from statefuzz.generator import _rebuild_probe, _tag_probe
from statefuzz.probes.compiler import CompiledProbe, compile_probe
from statefuzz.probes.schema import ProbeSpec


def generate_pollution_probe(
    seed: int = 0,
    context_tokens: int = 4096,
    n_items: int = 64,
    distractor_density: float = 1.0,
) -> CompiledProbe:
    """生成长中性填充和大量事实的单键检索探针。"""
    if not 0.0 < distractor_density <= 1.0:
        raise ValueError("distractor_density必须位于0到1之间")
    effective_items = max(1, round(n_items * distractor_density))
    spec = ProbeSpec(
        seed=seed,
        context_tokens=context_tokens,
        n_items=effective_items,
        target_position=0.5,
        template_id=7,
    )
    probe = compile_probe(spec)
    return _tag_probe(
        probe,
        "state_pollution",
        distractor_density=distractor_density,
        distractor_items=effective_items - 1,
        target_position=0.5,
    )


def generate_pollution_family(
    seeds: Iterable[int] = (0,),
    distractor_densities: Iterable[float] = (1.0,),
    context_lengths: Iterable[int] = (4096,),
    n_items: int = 64,
) -> list[CompiledProbe]:
    """生成干扰密度与上下文长度的可搜索样例族。"""
    return [
        generate_pollution_probe(
            seed=seed,
            context_tokens=context_tokens,
            n_items=n_items,
            distractor_density=density,
        )
        for seed in seeds
        for context_tokens in context_lengths
        for density in distractor_densities
    ]


def generate_pollution_recovery_probe(
    seed: int = 0, context_tokens: int = 4096
) -> CompiledProbe:
    """在高污染上下文后追加显式恢复标记。"""
    probe = generate_pollution_probe(
        seed=seed, context_tokens=context_tokens, n_items=64
    )
    prompt = probe.prompt + "\nRECOVER: return the original target only."
    return _rebuild_probe(
        probe,
        prompt,
        stress_pattern="state_pollution_recovery",
        recovery_protocol="explicit_marker",
    )

