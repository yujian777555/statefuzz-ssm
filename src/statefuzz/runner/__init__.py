"""模型执行器抽象与行为基线。"""

from statefuzz.runner.hf_causal_lm_runner import (
    HFCausalLMExperimentConfig,
    HFCausalLMRunner,
)

__all__ = ["HFCausalLMExperimentConfig", "HFCausalLMRunner"]

from .hybrid_causal_lm_runner import HybridCausalLMExperimentConfig, HybridCausalLMRunner
__all__ += ['HybridCausalLMExperimentConfig', 'HybridCausalLMRunner']
