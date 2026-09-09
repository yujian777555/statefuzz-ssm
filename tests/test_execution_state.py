import json
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_status_can_be_loaded_and_updated_atomically(tmp_path) -> None:
    from statefuzz.execution_state import load_status, update_status, write_status

    target = tmp_path / "status.json"
    write_status(
        target,
        {
            "round": 0,
            "previous_actor": "planner",
            "next": "codex",
            "last_plan": "plans/latest_plan.md",
            "last_result": None,
        },
    )
    updated = update_status(
        target,
        round=1,
        previous_actor="codex",
        next="gpt",
        last_result="results/result_round_001.json",
    )
    assert updated["next"] == "gpt"
    assert load_status(target)["round"] == 1
    assert list(tmp_path.glob("*.tmp")) == []
    assert json.loads(target.read_text(encoding="utf-8"))["last_result"] == (
        "results/result_round_001.json"
    )


def test_status_rejects_missing_or_invalid_fields(tmp_path) -> None:
    from statefuzz.execution_state import write_status

    with pytest.raises(ValueError, match="缺少字段"):
        write_status(tmp_path / "status.json", {"round": 1})
    with pytest.raises(ValueError, match="非负整数"):
        write_status(
            tmp_path / "status.json",
            {
                "round": -1,
                "previous_actor": "planner",
                "next": "codex",
                "last_plan": "plans/latest_plan.md",
                "last_result": None,
            },
        )


def _valid_status() -> dict[str, object]:
    return {
        "round": 2,
        "previous_actor": "planner",
        "next": "codex",
        "last_plan": "plans/latest_plan.md",
        "last_result": "results/result_round_001.json",
    }


def test_corrupted_json_is_reported_and_can_be_recovered(tmp_path) -> None:
    from statefuzz.execution_state import load_status, write_status

    target = tmp_path / "status.json"
    target.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON损坏"):
        load_status(target)

    write_status(target, _valid_status())
    assert load_status(target) == _valid_status()
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    "previous_actor,next_actor",
    [("planner", "planner"), ("unknown", "codex"), ("planner", "unknown")],
)
def test_status_rejects_invalid_actor_transitions(
    tmp_path, previous_actor: str, next_actor: str
) -> None:
    from statefuzz.execution_state import write_status

    status = _valid_status()
    status["previous_actor"] = previous_actor
    status["next"] = next_actor
    with pytest.raises(ValueError, match="执行者转换非法"):
        write_status(tmp_path / "status.json", status)


def test_concurrent_updates_preserve_independent_fields(tmp_path) -> None:
    from statefuzz.execution_state import load_status, update_status, write_status

    target = tmp_path / "status.json"
    write_status(target, _valid_status())
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(update_status, target, last_plan="plans/plan_002.md")
        second = pool.submit(
            update_status, target, last_result="results/result_round_002.json"
        )
        first.result()
        second.result()

    status = load_status(target)
    assert status["last_plan"] == "plans/plan_002.md"
    assert status["last_result"] == "results/result_round_002.json"


def test_update_status_surfaces_corruption_without_overwriting(tmp_path) -> None:
    from statefuzz.execution_state import update_status

    target = tmp_path / "status.json"
    target.write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON损坏"):
        update_status(target, last_result="results/result_round_003.json")
    assert target.read_text(encoding="utf-8") == "not-json"


def test_update_status_surfaces_invalid_transition(tmp_path) -> None:
    from statefuzz.execution_state import update_status, write_status

    target = tmp_path / "status.json"
    write_status(target, _valid_status())
    with pytest.raises(ValueError, match="执行者转换非法"):
        update_status(target, next="planner")

