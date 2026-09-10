import pytest


def test_realistic_tasks_are_reproducible_and_measurable() -> None:
    from statefuzz.tasks import generate_realistic_memory_task

    first = generate_realistic_memory_task("long_document_retrieval", "periodic_pattern", seed=4)
    second = generate_realistic_memory_task("long_document_retrieval", "periodic_pattern", seed=4)
    assert first == second
    assert first.hidden_memory_fact in first.prompt
    assert first.expected_answer in (" red", " blue")
    assert first.success_metric == "expected_answer_is_top1_candidate"


def test_realistic_task_validation() -> None:
    from statefuzz.tasks import generate_realistic_memory_task

    with pytest.raises(ValueError, match="任务"):
        generate_realistic_memory_task("unknown", "periodic_pattern")
    with pytest.raises(ValueError, match="stress"):
        generate_realistic_memory_task("long_document_retrieval", "unknown")
