"""生成多键干扰以观察状态表示碰撞的压力样例。"""

from __future__ import annotations

from statefuzz.generator import _tag_probe
from statefuzz.probes.compiler import CompiledProbe, compile_probe
from statefuzz.probes.schema import ProbeSpec, TaskFamily


def generate_collision_probe(
    seed: int = 0,
    context_tokens: int = 4096,
    n_items: int = 32,
    query_fanout: int = 8,
) -> CompiledProbe:
    """生成中部目标和多键查询，放大键间状态干扰。"""
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
        interference_items=n_items - query_fanout,
    )

