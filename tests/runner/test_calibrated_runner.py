def test_next_token_scoring_requires_model_backed_runner() -> None:
    import pytest

    from statefuzz.runner.mamba_runner import MambaRunner

    runner = MambaRunner(lambda prompt: "answer")
    with pytest.raises(RuntimeError, match="model-backed"):
        runner.score_next_token("prompt")


def test_next_token_scoring_returns_probability_and_hidden_state() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaRunner

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2]])}

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
            hidden = torch.ones(1, 2, 3)
            return type("Output", (), {"logits": logits, "hidden_states": (hidden,)})()

    runner = MambaRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    evidence = runner.score_next_token("prompt")
    assert evidence["predicted_token_id"] == 2
    assert evidence["target_token_id"] == 2
    assert evidence["target_probability"] > 0.9
    assert evidence["input_token_count"] == 2


def test_next_token_instance_scoring_returns_one_record_per_prompt() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaRunner

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2]])}

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
            hidden = torch.ones(1, 2, 3)
            return type("Output", (), {"logits": logits, "hidden_states": (hidden,)})()

    runner = MambaRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    records = runner.score_next_token_instances(["a", "b"], target_token_id=2)
    assert len(records) == 2
    assert all(record["target_token_id"] == 2 for record in records)


def test_next_token_scoring_captures_layer_states() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaRunner

    class FakeTokenizer:
        def __call__(self, prompt, return_tensors="pt"):
            return {"input_ids": torch.tensor([[1, 2]])}

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
            first = torch.ones(1, 2, 3)
            second = torch.full((1, 2, 3), 2.0)
            return type(
                "Output",
                (),
                {"logits": logits, "hidden_states": (first, second)},
            )()

    runner = MambaRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    runner.score_next_token("prompt")
    layers = runner.capture_layer_states()
    assert set(layers) == {"layer_0", "layer_1"}
    assert tuple(layers["layer_1"].shape) == (1, 3)

