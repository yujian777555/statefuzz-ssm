import importlib.util

import pytest

from statefuzz.probes.compiler import compile_probe
from statefuzz.probes.schema import ProbeSpec


RUNNER_SPEC = importlib.util.find_spec("statefuzz.runner.mamba_runner")
requires_runner = pytest.mark.skipif(RUNNER_SPEC is None, reason="runner尚未实现")


def test_runner_modules_exist() -> None:
    assert RUNNER_SPEC is not None
    assert importlib.util.find_spec("statefuzz.runner.base") is not None


@requires_runner
def test_mamba_runner_executes_probe_with_injected_predictor() -> None:
    from statefuzz.runner.mamba_runner import MambaRunner

    probe = compile_probe(ProbeSpec(seed=3))
    runner = MambaRunner(lambda prompt: probe.answer)
    assert runner.run_probe(probe) == probe.answer


@requires_runner
def test_mamba_runner_captures_a_copy_of_hidden_state() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaRunner

    state = torch.tensor([[1.0, 2.0]])
    runner = MambaRunner(lambda prompt: "answer", hidden_state_provider=lambda: state)
    runner.run_probe(compile_probe(ProbeSpec(seed=4)))
    captured = runner.capture_hidden_state()
    assert torch.equal(captured, state)
    assert captured is not state
    state.add_(1.0)
    assert torch.equal(captured, torch.tensor([[1.0, 2.0]]))


@requires_runner
def test_mamba_runner_rejects_non_string_prediction() -> None:
    from statefuzz.runner.mamba_runner import MambaRunner

    runner = MambaRunner(lambda prompt: None)
    with pytest.raises(TypeError, match="字符串"):
        runner.run_probe(compile_probe(ProbeSpec(seed=5)))


@requires_runner
def test_model_backed_path_collects_hidden_states() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaExperimentConfig, MambaRunner

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, tokens, skip_special_tokens=True):
            return "answer"

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            hidden = torch.ones(1, input_ids.shape[-1], 4)
            return type("Output", (), {"hidden_states": (hidden,)})()

        def generate(self, input_ids, **kwargs):
            return input_ids

    runner = MambaRunner(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        experiment_config=MambaExperimentConfig(model_id="fake/mamba"),
    )
    runner.run_probe(compile_probe(ProbeSpec(seed=6)))
    captured = runner.capture_hidden_state()
    assert tuple(captured.shape) == (1, 3, 4)
    assert runner.experiment_config.to_dict()["model_id"] == "fake/mamba"

