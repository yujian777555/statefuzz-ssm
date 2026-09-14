"""离线混合模型行为执行器，保留独立架构元数据。"""
from dataclasses import dataclass
from pathlib import Path
import math

from .hf_causal_lm_runner import HFCausalLMRunner
from .hybrid_cache_intervention import clone_cache_independent
from statefuzz.analyzer.memory_dependence import compute_pairwise_memory_metrics, decompose_pairwise_preference


@dataclass(frozen=True)
class HybridCausalLMExperimentConfig:
    model_id: str = 'Zyphra/Zamba2-1.2B-Instruct-v2'
    checkpoint_path: str = '/202532803004/models/Zamba2-1.2B-Instruct-v2'
    device: str = 'cuda'
    dtype: str = 'float16'
    seed: int = 0
    local_files_only: bool = True
    trust_remote_code: bool = False


class HybridCausalLMRunner(HFCausalLMRunner):
    """复用候选评分接口，覆盖加载、指标与混合架构语义。"""

    @classmethod
    def from_pretrained(cls, config):
        if not config.local_files_only or config.trust_remote_code:
            raise ValueError('混合执行器要求离线加载且禁止远程代码')
        path = Path(config.checkpoint_path)
        if not path.is_absolute() or not path.is_dir():
            raise FileNotFoundError(f'vm_checkpoint_path_not_visible: {path}')
        for name in ('config.json', 'model.safetensors', 'tokenizer.json'):
            if not (path / name).is_file():
                raise FileNotFoundError(str(path / name))
        import torch
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
        options = dict(local_files_only=True, trust_remote_code=False)
        auto_config = AutoConfig.from_pretrained(str(path), **options)
        tokenizer = AutoTokenizer.from_pretrained(str(path), **options)
        torch.manual_seed(config.seed)
        model = AutoModelForCausalLM.from_pretrained(str(path), config=auto_config, dtype=getattr(torch, config.dtype), **options)
        model.to(config.device).eval()
        return cls(model=model, tokenizer=tokenizer, experiment_config=config)

    def score_candidate_tokens(self, prompt, candidate_token_ids):
        result = super().score_candidate_tokens(prompt, candidate_token_ids)
        if not all(math.isfinite(c['logit']) for c in result['candidates']):
            raise ValueError('候选logits不是有限数')
        result['state_source'] = 'hybrid_behavior_only'
        return result

    def encode_prompt_ids(self, prompt):
        """返回不含特殊token的单批次CPU token IDs。"""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError('prompt必须是非空字符串')
        encoded = self._tokenizer(
            prompt, return_tensors='pt', add_special_tokens=False
        )['input_ids']
        if encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError('tokenizer必须返回单批次input_ids')
        return encoded.detach().clone().cpu()

    def run_body_to_cache(self, body_input_ids):
        """运行body并返回与输出相互独立的混合DynamicCache。"""
        import torch

        ids = torch.as_tensor(body_input_ids)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError('body_input_ids必须是非空单批次token IDs')
        with torch.inference_mode():
            output = self._model(
                input_ids=ids.to(self._model.device),
                use_cache=True,
                return_dict=True,
            )
        cache = getattr(output, 'past_key_values', None)
        if cache is None:
            raise RuntimeError('模型输出没有past_key_values')
        return clone_cache_independent(cache)

    def score_tail_from_cache(
        self, tail_input_ids, past_key_values, candidate_token_ids
    ):
        """在独立cache副本上运行共同tail并返回原始候选logits。"""
        import torch

        ids = torch.as_tensor(tail_input_ids)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] == 0:
            raise ValueError('tail_input_ids必须是非空单批次token IDs')
        if not candidate_token_ids or any(
            isinstance(token, bool) or not isinstance(token, int)
            for token in candidate_token_ids
        ):
            raise ValueError('candidate_token_ids必须是非空整数列表')
        cache = clone_cache_independent(past_key_values)
        with torch.inference_mode():
            output = None
            device_ids = ids.to(self._model.device)
            # Zamba2的naive SSM增量路径在已有状态时采用单步解码语义；
            # 批量传入多个tail token不会重建完整forward，必须逐token推进。
            for index in range(device_ids.shape[1]):
                output = self._model(
                    input_ids=device_ids[:, index : index + 1],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                cache = getattr(output, 'past_key_values', cache)
            assert output is not None
            logits = output.logits[0, -1].float().detach().cpu()
        values = {int(token): float(logits[int(token)].item()) for token in candidate_token_ids}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('候选logits不是有限数')
        return {'candidate_logits': values, 'past_key_values': cache}

    def score_remote_memory_pair(self, pair, **kwargs):
        result = super().score_remote_memory_pair(pair, **kwargs)
        result['state_source'] = 'hybrid_behavior_only'
        if result['matched'] and result['candidate_valid']:
            result.update(compute_pairwise_memory_metrics(result))
            result.update(decompose_pairwise_preference(result))
        return result

    def model_metadata(self):
        cfg = self._model.config
        raw = cfg.to_dict()
        return {
            'architecture_family': 'hybrid_ssm_attention',
            'checkpoint_name': self.experiment_config.model_id,
            'checkpoint_path': self.experiment_config.checkpoint_path,
            'checkpoint_location': 'user_experiment_vm',
            'checkpoint_source': 'user_downloaded_modelscope_directory',
            'offline_loading': True, 'instruction_tuning_confound': True,
            'architecture_causality_confirmed': False,
            'model_type': raw.get('model_type'), 'architectures': raw.get('architectures'),
            'num_parameters': sum(p.numel() for p in self._model.parameters()),
            'num_layers': raw.get('num_hidden_layers'), 'layer_types': raw.get('layer_types'),
            'attention_layer_ids': raw.get('attention_layer_ids'), 'ssm_layer_ids': raw.get('ssm_layer_ids'),
            'max_context_tokens': raw.get('max_position_embeddings'),
            'tokenizer_model_max_length': self._tokenizer.model_max_length,
            'cache_semantics': 'hybrid', 'raw_config': raw,
        }
