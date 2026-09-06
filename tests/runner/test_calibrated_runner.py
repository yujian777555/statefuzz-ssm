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


def test_count_tokens_and_paired_scoring_report_length_match() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaRunner

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, prompt, return_tensors="pt", **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, **kwargs):
            return "tok"

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
            hidden = torch.ones(1, 2, 3)
            return type("Output", (), {"logits": logits, "hidden_states": (hidden,)})()

    runner = MambaRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    pair = runner.score_paired_next_token("a", "b")
    assert pair["matched"] is True
    assert pair["control_token_count"] == pair["stressed_token_count"] == 2
    assert pair["control"]["target_rank"] == 1
    assert pair["stressed"]["top1_margin"] == 0.0


def test_recurrent_state_is_explicitly_unavailable_when_model_does_not_expose_cache() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaRunner

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, prompt, return_tensors="pt", **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, **kwargs):
            return "tok"

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
            hidden = torch.ones(1, 2, 3)
            return type("Output", (), {"logits": logits, "hidden_states": (hidden,)})()

    runner = MambaRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    runner.score_next_token("a")
    assert runner.capture_recurrent_state() is None
    assert runner.capture_recurrent_state_summary()["state_source"] == "unavailable"


def test_recurrent_state_summary_identifies_dynamic_cache() -> None:
    import torch

    from statefuzz.runner.mamba_runner import MambaRunner

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, prompt, return_tensors="pt", **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, **kwargs):
            return "tok"

    class CacheLayer:
        conv_states = torch.ones(1, 2, 3)
        recurrent_states = torch.ones(1, 2, 4)

    class DynamicCache:
        layers = [CacheLayer(), CacheLayer()]

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            logits = torch.tensor([[[0.0, 0.0, 4.0], [0.0, 0.0, 4.0]]])
            hidden = torch.ones(1, 2, 3)
            return type(
                "Output",
                (),
                {"logits": logits, "hidden_states": (hidden,), "cache_params": DynamicCache()},
            )()

    runner = MambaRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    runner.score_next_token("a")
    summary = runner.capture_recurrent_state_summary()
    assert summary["state_source"] == "direct_recurrent_cache"
    assert len(summary["layers"]) == 2
    assert summary["layers"][0]["ssm_shape"] == [1, 2, 4]


def test_single_token_candidate_scoring_uses_explicit_values() -> None:
    import torch

    from statefuzz.generator.remote_memory import generate_remote_memory_pair
    from statefuzz.runner.mamba_runner import MambaRunner

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, text, return_tensors="pt", **kwargs):
            mapping = {" red": [1], " blue": [2], "two words": [1, 2]}
            ids = mapping.get(text, [4, 5, 6])
            return {"input_ids": torch.tensor([ids])}

        def decode(self, tokens, **kwargs):
            return {1: " red", 2: " blue", 3: " top"}.get(tokens[0], "?")

    class FakeModel(torch.nn.Module):
        device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            logits = torch.zeros(1, input_ids.shape[-1], 7)
            logits[0, -1, 1] = 2.0
            logits[0, -1, 2] = 1.0
            logits[0, -1, 3] = 3.0
            hidden = torch.ones(1, input_ids.shape[-1], 3)
            return type("Output", (), {"logits": logits, "hidden_states": (hidden,)})()

    runner = MambaRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    assert runner.single_token_id(" red") == 1
    assert runner.single_token_id("two words") is None
    scored = runner.score_candidate_tokens("prompt", [1, 2])
    assert scored["candidates"][0]["token_id"] == 1
    assert scored["candidates"][0]["rank"] == 2
    assert scored["top1_token_id"] == 3
    pair = runner.score_remote_memory_pair(
        generate_remote_memory_pair(64, seed=7, value_a=" red", value_b=" blue")
    )
    assert pair["candidate_token_ids"] == [1, 2]
    assert pair["matched"] is True
