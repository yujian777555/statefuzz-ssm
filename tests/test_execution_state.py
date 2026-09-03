import json

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

