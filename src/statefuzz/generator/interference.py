"""可控干扰注入压力任务。"""

from __future__ import annotations

from statefuzz.generator import _rebuild_probe
from statefuzz.generator.memory_decay import generate_memory_decay_probe
from statefuzz.probes.compiler import CompiledProbe


def generate_interference_probe(
    seed: int = 0,
    context_tokens: int = 4096,
    interference_strength: float = 0.5,
) -> CompiledProbe:
    """在查询前插入与目标无关的干扰事实。"""
    if not 0.0 <= interference_strength <= 1.0:
        raise ValueError("interference_strength必须位于0到1之间")
    probe = generate_memory_decay_probe(seed=seed, context_tokens=context_tokens, n_items=32)
    count = round(64 * interference_strength)
    distractors = "\n".join(
        f"DISTRACTOR_{i:03d} = NOISE_{(i * 17) % 997:03d}" for i in range(count)
    )
    prompt = probe.prompt.replace("请按键顺序", f"{distractors}\n请按键顺序", 1)
    return _rebuild_probe(
        probe,
        prompt,
        stress_pattern="interference_injection",
        interference_strength=interference_strength,
        interference_items=count,
    )

