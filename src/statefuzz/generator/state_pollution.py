"""生成高密度干扰内容以观察状态污染的压力样例。"""

from __future__ import annotations

from statefuzz.generator import _tag_probe
from statefuzz.probes.compiler import CompiledProbe, compile_probe
from statefuzz.probes.schema import ProbeSpec


def generate_pollution_probe(
    seed: int = 0, context_tokens: int = 4096, n_items: int = 64
) -> CompiledProbe:
    """生成长中性填充和大量事实的单键检索探针。"""
    spec = ProbeSpec(
        seed=seed,
        context_tokens=context_tokens,
        n_items=n_items,
        target_position=0.5,
        template_id=7,
    )
    probe = compile_probe(spec)
    return _tag_probe(
        probe,
        "state_pollution",
        distractor_items=n_items - 1,
        target_position=0.5,
    )

