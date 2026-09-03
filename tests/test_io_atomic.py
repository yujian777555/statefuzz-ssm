import importlib.util
import json

import pytest


IO_SPEC = importlib.util.find_spec("statefuzz.io_atomic")
requires_io = pytest.mark.skipif(IO_SPEC is None, reason="io_atomic尚未实现")


def test_io_atomic_module_exists() -> None:
    assert IO_SPEC is not None


@requires_io
def test_atomic_write_replaces_without_tmp_residue(tmp_path) -> None:
    from statefuzz.io_atomic import atomic_write_json

    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"value": 1})
    atomic_write_json(target, {"value": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2}
    assert list(tmp_path.glob("*.tmp")) == []


@requires_io
def test_atomic_write_uses_stable_key_order(tmp_path) -> None:
    from statefuzz.io_atomic import atomic_write_json

    target = tmp_path / "manifest.json"
    atomic_write_json(target, {"z": 1, "a": 2})
    text = target.read_text(encoding="utf-8")
    assert text.index('"a"') < text.index('"z"')
