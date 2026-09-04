"""生成早期记忆在长上下文中衰减的压力样例。"""

from __future__ import annotations

from statefuzz.generator import _tag_probe
from statefuzz.probes.compiler import CompiledProbe, compile_probe
from statefuzz.probes.schema import ProbeSpec


def generate_memory_decay_probe(
    seed: int = 0, context_tokens: int = 4096, n_items: int = 64
) -> CompiledProbe:
    """生成将目标放在上下文开头的单键长上下文探针。"""
    spec = ProbeSpec(
        seed=seed,
        context_tokens=context_tokens,
        n_items=n_items,
        target_position=0.0,
        template_id=0,
    )
    probe = compile_probe(spec)
    return _tag_probe(probe, "memory_decay", target_position=0.0)

