import importlib.util

import pytest

from statefuzz.probes.compiler import compile_probe
from statefuzz.probes.schema import ProbeSpec


METAMORPHIC_SPEC = importlib.util.find_spec("statefuzz.probes.metamorphic")
requires_metamorphic = pytest.mark.skipif(
    METAMORPHIC_SPEC is None, reason="metamorphic尚未实现"
)


def test_metamorphic_module_exists() -> None:
    assert METAMORPHIC_SPEC is not None


@requires_metamorphic
def test_rename_preserves_answer_and_is_deterministic() -> None:
    from statefuzz.probes.metamorphic import rename_symbols

    original = compile_probe(ProbeSpec(seed=5))
    left = rename_symbols(original, "audit")
    right = rename_symbols(original, "audit")
    assert left == right
    assert left.answer == original.answer
    assert left.prompt != original.prompt


@requires_metamorphic
def test_template_switch_preserves_answer() -> None:
    from statefuzz.probes.metamorphic import switch_template

    original = ProbeSpec(seed=5, template_id=0)
    switched = switch_template(original, 1)
    assert compile_probe(switched).answer == compile_probe(original).answer


@pytest.mark.parametrize("template_id", [0, 8])
@requires_metamorphic
def test_template_switch_rejects_same_or_invalid(template_id: int) -> None:
    from statefuzz.probes.metamorphic import switch_template

    with pytest.raises(ValueError):
        switch_template(ProbeSpec(template_id=0), template_id)
