import importlib.util

import pytest


GENERATOR_SPEC = importlib.util.find_spec("statefuzz.generator.memory_decay")
requires_generators = pytest.mark.skipif(
    GENERATOR_SPEC is None, reason="generator尚未实现"
)


def test_generator_modules_exist() -> None:
    assert GENERATOR_SPEC is not None
    assert importlib.util.find_spec("statefuzz.generator.state_collision") is not None
    assert importlib.util.find_spec("statefuzz.generator.state_pollution") is not None


@requires_generators
def test_memory_decay_probe_targets_early_memory_and_is_deterministic() -> None:
    from statefuzz.generator.memory_decay import generate_memory_decay_probe

    left = generate_memory_decay_probe(seed=7)
    right = generate_memory_decay_probe(seed=7)
    assert left == right
    assert left.spec.target_position == 0.0
    assert left.spec.context_tokens >= 4096
    assert left.provenance["stress_pattern"] == "memory_decay"


@requires_generators
def test_collision_probe_uses_multi_key_interference() -> None:
    from statefuzz.generator.state_collision import generate_collision_probe

    probe = generate_collision_probe(seed=11)
    assert probe.spec.task.value == "multi_key"
    assert probe.spec.query_fanout >= 2
    assert probe.provenance["stress_pattern"] == "state_collision"


@requires_generators
def test_pollution_probe_contains_long_distractor_context() -> None:
    from statefuzz.generator.state_pollution import generate_pollution_probe

    probe = generate_pollution_probe(seed=13)
    assert probe.spec.context_tokens >= 4096
    assert probe.spec.n_items >= 32
    assert probe.provenance["stress_pattern"] == "state_pollution"

