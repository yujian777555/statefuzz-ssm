import importlib.util
import random

import pytest

from statefuzz.probes.schema import ProbeSpec


COMPILER_SPEC = importlib.util.find_spec("statefuzz.probes.compiler")
requires_compiler = pytest.mark.skipif(
    COMPILER_SPEC is None, reason="compiler尚未实现"
)


def test_compiler_module_exists() -> None:
    assert COMPILER_SPEC is not None


@requires_compiler
def test_probe_is_byte_deterministic() -> None:
    from statefuzz.probes.compiler import compile_probe

    spec = ProbeSpec(seed=11, n_items=8)
    assert compile_probe(spec) == compile_probe(spec)


@requires_compiler
def test_multi_key_answer_matches_provenance() -> None:
    from statefuzz.probes.compiler import compile_probe

    probe = compile_probe(
        ProbeSpec(task="multi_key", seed=23, n_items=12, query_fanout=3)
    )
    assert probe.answer == "|".join(probe.provenance["queried_values"])


@requires_compiler
def test_compiler_does_not_modify_global_rng() -> None:
    from statefuzz.probes.compiler import compile_probe

    random.seed(99)
    expected = random.random()
    random.seed(99)
    compile_probe(ProbeSpec(seed=7))
    assert random.random() == expected


@requires_compiler
def test_target_position_controls_primary_index() -> None:
    from statefuzz.probes.compiler import compile_probe

    early = compile_probe(ProbeSpec(seed=7, n_items=10, target_position=0.0))
    late = compile_probe(ProbeSpec(seed=7, n_items=10, target_position=1.0))
    assert early.provenance["primary_index"] == 0
    assert late.provenance["primary_index"] == 9
