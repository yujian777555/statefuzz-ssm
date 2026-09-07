import torch


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, return_tensors="pt", **kwargs):
        mapping = {" one": [1], " two": [2], "two words": [1, 2]}
        return {"input_ids": torch.tensor([mapping.get(text, [3, 4, 5])])}

    def decode(self, tokens, **kwargs):
        return {1: " one", 2: " two", 5: " top"}.get(tokens[0], "tok")


class FakeConfig:
    model_type = "gpt_neox"
    max_position_embeddings = 2048


class FakeModel(torch.nn.Module):
    device = torch.device("cpu")
    config = FakeConfig()

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros(1, input_ids.shape[-1], 8)
        logits[0, -1, 1] = 2.0
        logits[0, -1, 2] = 1.0
        logits[0, -1, 5] = 3.0
        return type("Output", (), {"logits": logits})()


def test_hf_causal_lm_runner_scores_behavior_without_recurrent_claims() -> None:
    from statefuzz.runner.hf_causal_lm_runner import (
        HFCausalLMExperimentConfig,
        HFCausalLMRunner,
    )

    config = HFCausalLMExperimentConfig(model_id="fake/transformer")
    runner = HFCausalLMRunner(
        model=FakeModel(), tokenizer=FakeTokenizer(), experiment_config=config
    )
    assert runner.count_tokens("prompt") == 3
    assert runner.single_token_id(" one") == 1
    assert runner.single_token_id("two words") is None
    result = runner.score_candidate_tokens("prompt", [1, 2])
    assert len(result["candidates"]) == 2
    assert result["state_source"] == "not_applicable"


def test_hf_causal_lm_runner_scores_matched_remote_memory_pair() -> None:
    from statefuzz.generator.remote_memory import generate_remote_memory_pair
    from statefuzz.runner.hf_causal_lm_runner import HFCausalLMRunner

    runner = HFCausalLMRunner(model=FakeModel(), tokenizer=FakeTokenizer())
    pair = generate_remote_memory_pair(
        64, seed=17, template_id=1, value_a=" one", value_b=" two"
    )
    result = runner.score_remote_memory_pair(pair)
    assert result["matched"] is True
    assert result["actual_input_tokens"] == 3
    assert result["state_source"] == "not_applicable"
    assert result["recurrent_state_a"] is None


def test_hf_causal_lm_metadata_identifies_transformer_control() -> None:
    from statefuzz.runner.hf_causal_lm_runner import (
        HFCausalLMExperimentConfig,
        HFCausalLMRunner,
    )

    runner = HFCausalLMRunner(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        experiment_config=HFCausalLMExperimentConfig(model_id="fake/transformer"),
    )
    metadata = runner.model_metadata()
    assert metadata["model_type"] == "gpt_neox"
    assert metadata["num_parameters"] == 4
    assert metadata["max_context_tokens"] == 2048
    assert metadata["architecture_role"] == "transformer_control"
