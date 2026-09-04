from statefuzz.probes.compiler import compile_probe
from statefuzz.probes.schema import ProbeSpec


def test_probe_evaluation_contract_scores_exact_prediction() -> None:
    from statefuzz.evaluation.contract import evaluate_compiled_probe

    probe = compile_probe(ProbeSpec(seed=17))
    outcome = evaluate_compiled_probe(probe, probe.answer)
    assert outcome.score == 1.0
    assert outcome.probe_hash == probe.probe_hash
    assert outcome.expected == probe.answer


def test_probe_evaluation_contract_scores_mismatch_without_mutation() -> None:
    from statefuzz.evaluation.contract import evaluate_compiled_probe

    probe = compile_probe(ProbeSpec(seed=17))
    original = (probe.prompt, probe.answer, probe.probe_hash)
    outcome = evaluate_compiled_probe(probe, "wrong")
    assert outcome.score == 0.0
    assert (probe.prompt, probe.answer, probe.probe_hash) == original


def test_probe_evaluation_contract_rejects_non_string_prediction() -> None:
    import pytest

    from statefuzz.evaluation.contract import evaluate_compiled_probe

    probe = compile_probe(ProbeSpec(seed=17))
    with pytest.raises(TypeError, match="字符串"):
        evaluate_compiled_probe(probe, None)

