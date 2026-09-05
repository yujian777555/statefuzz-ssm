"""长程needle检索压力任务。"""

from __future__ import annotations

from statefuzz.generator import _rebuild_probe, _tag_probe
from statefuzz.generator.memory_decay import generate_memory_decay_probe
from statefuzz.probes.compiler import CompiledProbe


def generate_long_range_retrieval_probe(
    seed: int = 0, context_tokens: int = 4096
) -> CompiledProbe:
    """把精确needle放在提示开头，测试长距离保持能力。"""
    probe = generate_memory_decay_probe(
        seed=seed, context_tokens=context_tokens, n_items=32, target_position=0.0
    )
    prompt = f"NEEDLE = {probe.answer}\n" + probe.prompt
    return _rebuild_probe(probe, prompt, stress_pattern="long_range_retrieval")

