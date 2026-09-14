import pytest
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
