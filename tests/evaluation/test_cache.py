import importlib.util
import json

import pytest


CACHE_SPEC = importlib.util.find_spec("statefuzz.evaluation.cache")
requires_cache = pytest.mark.skipif(CACHE_SPEC is None, reason="cache尚未实现")


def test_cache_module_exists() -> None:
    assert CACHE_SPEC is not None


@requires_cache
def test_cache_round_trip_is_idempotent(tmp_path) -> None:
    from statefuzz.evaluation.cache import EvaluationCache, EvaluationRecord

    cache = EvaluationCache(tmp_path)
    record = EvaluationRecord(key="abc123", score=1.0, payload={"model": "fake"})
    cache.put(record)
    cache.put(record)
    assert cache.get("abc123") == record


@requires_cache
def test_cache_rejects_tampered_record(tmp_path) -> None:
    from statefuzz.evaluation.cache import EvaluationCache, EvaluationRecord

    cache = EvaluationCache(tmp_path)
    cache.put(EvaluationRecord(key="abc123", score=1.0, payload={"model": "fake"}))
    target = tmp_path / "ab" / "abc123.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["score"] = 0.0
    target.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        cache.get("abc123")


@requires_cache
def test_cache_refuses_conflicting_overwrite(tmp_path) -> None:
    from statefuzz.evaluation.cache import EvaluationCache, EvaluationRecord

    cache = EvaluationCache(tmp_path)
    cache.put(EvaluationRecord(key="abc123", score=1.0, payload={"model": "fake"}))
    with pytest.raises(ValueError, match="冲突"):
        cache.put(EvaluationRecord(key="abc123", score=0.0, payload={"model": "fake"}))
