import pytest
import torch
from statefuzz.runner.hybrid_causal_lm_runner import HybridCausalLMRunner, HybridCausalLMExperimentConfig


def test_offline_policy_precedes_transformers_load():
    for options in ({'local_files_only': False}, {'trust_remote_code': True}):
        with pytest.raises(ValueError):
            HybridCausalLMRunner.from_pretrained(HybridCausalLMExperimentConfig(**options))


def test_missing_local_path_never_falls_back_to_remote(tmp_path):
    with pytest.raises(FileNotFoundError, match='vm_checkpoint_path_not_visible'):
        HybridCausalLMRunner.from_pretrained(HybridCausalLMExperimentConfig(checkpoint_path=str(tmp_path / 'missing')))


def test_incomplete_checkpoint_rejected_before_model_load(tmp_path):
    with pytest.raises(FileNotFoundError, match='config.json'):
        HybridCausalLMRunner.from_pretrained(HybridCausalLMExperimentConfig(checkpoint_path=str(tmp_path)))


def test_encode_body_cache_tail_helpers_use_independent_cache():
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.tensor([[1, 2, 3]])}

    class Layer:
        def __init__(self):
            self.keys = torch.ones(1, 1, 2, 1)
            self.values = torch.ones(1, 1, 2, 1)
            self.conv_states = torch.ones(1, 1, 1)
            self.recurrent_states = torch.ones(1, 1, 1, 1)

    class Cache:
        def __init__(self): self.layers = [Layer()]

    class Model:
        device = torch.device('cpu')
        def __init__(self): self.calls = 0
        def __call__(self, input_ids, past_key_values=None, **kwargs):
            self.calls += 1
            cache = past_key_values or Cache()
            logits = torch.zeros(1, input_ids.shape[1], 5)
            logits[0, -1, 1], logits[0, -1, 2] = 3, 1
            return type('Output', (), {'past_key_values': cache, 'logits': logits})()

    model = Model()
    runner = HybridCausalLMRunner(model=model, tokenizer=Tokenizer())
    ids = runner.encode_prompt_ids('prompt')
    cache = runner.run_body_to_cache(ids[:, :-1])
    result = runner.score_tail_from_cache(ids[:, -1:], cache, [1, 2])
    assert result['candidate_logits'] == {1: 3.0, 2: 1.0}
    assert result['past_key_values'] is not cache
    before = model.calls
    runner.score_tail_from_cache(torch.tensor([[1, 2, 3]]), cache, [1, 2])
    assert model.calls - before == 3
