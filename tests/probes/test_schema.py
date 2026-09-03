import importlib.util

import pytest


SCHEMA_SPEC = importlib.util.find_spec("statefuzz.probes.schema")


def test_schema_module_exists() -> None:
    assert SCHEMA_SPEC is not None


requires_schema = pytest.mark.skipif(SCHEMA_SPEC is None, reason="schema尚未实现")


def _types():
    from pydantic import ValidationError
    from statefuzz.probes.schema import ProbeSpec, TaskFamily

    return ValidationError, ProbeSpec, TaskFamily


@requires_schema
def test_hash_is_canonical_and_seed_sensitive() -> None:
    _, ProbeSpec, TaskFamily = _types()
    left = ProbeSpec(task=TaskFamily.SINGLE_KEY, seed=7)
    same = ProbeSpec(seed=7, task="single_key")
    other = ProbeSpec(task=TaskFamily.SINGLE_KEY, seed=8)
    assert left.config_hash == same.config_hash
    assert left.config_hash != other.config_hash


@pytest.mark.parametrize(
    "field,value", [("context_tokens", 63), ("n_items", 0), ("query_fanout", 0)]
)
@requires_schema
def test_rejects_out_of_range_values(field: str, value: int) -> None:
    ValidationError, ProbeSpec, _ = _types()
    with pytest.raises(ValidationError):
        ProbeSpec(**{field: value})


@requires_schema
def test_single_key_rejects_multiple_queries() -> None:
    ValidationError, ProbeSpec, _ = _types()
    with pytest.raises(ValidationError, match="single_key"):
        ProbeSpec(task="single_key", query_fanout=2)


@requires_schema
def test_fanout_cannot_exceed_items() -> None:
    ValidationError, ProbeSpec, _ = _types()
    with pytest.raises(ValidationError, match="n_items"):
        ProbeSpec(task="multi_key", n_items=2, query_fanout=3)
