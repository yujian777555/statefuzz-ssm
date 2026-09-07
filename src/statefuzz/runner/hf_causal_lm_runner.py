"""非Mamba Hugging Face因果语言模型的行为评估执行器。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([token_id], skip_special_tokens=False))
    except (AttributeError, TypeError, ValueError):
        return str(token_id)


@dataclass(frozen=True)
class HFCausalLMExperimentConfig:
    """行为型Transformer控制实验配置。"""

    model_id: str
    revision: str | None = None
    device: str = "cuda"
    dtype: str = "float16"
    seed: int = 0
    trust_remote_code: bool = False
    local_files_only: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "dtype": self.dtype,
            "seed": self.seed,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }


class HFCausalLMRunner:
    """只提供候选行为评分，不暴露Transformer KV为循环状态。"""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        experiment_config: HFCausalLMExperimentConfig | None = None,
    ) -> None:
        if model is None or tokenizer is None:
            raise ValueError("model和tokenizer必须同时提供")
        self._model = model
        self._tokenizer = tokenizer
        self.experiment_config = experiment_config

    @classmethod
    def from_pretrained(cls, config: HFCausalLMExperimentConfig) -> "HFCausalLMRunner":
        if config.trust_remote_code:
            raise ValueError("控制模型路径禁止trust_remote_code")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = getattr(torch, config.dtype, None)
        if dtype is None or not isinstance(dtype, torch.dtype):
            raise ValueError("dtype必须是torch支持的类型名")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            dtype=dtype,
            trust_remote_code=False,
            local_files_only=config.local_files_only,
        )
        model.to(config.device)
        model.eval()
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        if getattr(tokenizer, "pad_token_id", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(model=model, tokenizer=tokenizer, experiment_config=config)

    def single_token_id(self, text: str) -> int | None:
        if not isinstance(text, str) or not text:
            raise ValueError("text必须是非空字符串")
        encoded = self._tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"]
        if token_ids.ndim != 2 or token_ids.shape[0] != 1:
            raise ValueError("tokenizer必须返回单批次input_ids")
        if int(token_ids.shape[-1]) != 1:
            return None
        return int(token_ids[0, 0].item())

    def count_tokens(self, prompt: str) -> int:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt必须是非空字符串")
        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        return int(encoded["input_ids"].shape[-1])

    def score_candidate_tokens(
        self, prompt: str, candidate_token_ids: list[int]
    ) -> dict[str, Any]:
        if not isinstance(candidate_token_ids, list) or not candidate_token_ids:
            raise ValueError("candidate_token_ids必须是非空列表")
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in candidate_token_ids):
            raise ValueError("candidate_token_ids必须为整数")
        import torch

        encoded = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = getattr(self._model, "device", "cpu")
        inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in encoded.items()
        }
        with torch.inference_mode():
            outputs = self._model(**inputs, use_cache=True, return_dict=True)
            logits = outputs.logits[0, -1].float()
            probabilities = torch.softmax(logits, dim=-1)
            top1_token_id = int(torch.argmax(probabilities).item())
            candidates = []
            for token_id in candidate_token_ids:
                if not 0 <= token_id < probabilities.numel():
                    raise ValueError("candidate_token_id超出词表范围")
                probability = float(probabilities[token_id].item())
                candidates.append(
                    {
                        "token_id": token_id,
                        "token_text": _decode_token(self._tokenizer, token_id),
                        "probability": probability,
                        "logit": float(logits[token_id].item()),
                        "log_probability": float(torch.log(probabilities[token_id]).item()),
                        "rank": int((probabilities > probability).sum().item()) + 1,
                    }
                )
        return {
            "candidates": candidates,
            "top1_token_id": top1_token_id,
            "top1_token_text": _decode_token(self._tokenizer, top1_token_id),
            "top1_probability": float(probabilities[top1_token_id].item()),
            "input_token_count": int(inputs["input_ids"].shape[-1]),
            "state_source": "not_applicable",
        }

    def score_remote_memory_pair(
        self, pair: Any, *, include_state_copies: bool = False
    ) -> dict[str, Any]:
        value_a = str(pair.value_a)
        value_b = str(pair.value_b)
        token_a = self.single_token_id(value_a)
        token_b = self.single_token_id(value_b)
        counts = [self.count_tokens(pair.prompt_a), self.count_tokens(pair.prompt_b)]
        result: dict[str, Any] = {
            "seed": pair.seed,
            "template_id": pair.template_id,
            "target_position": pair.target_position,
            "candidate_values": [value_a, value_b],
            "candidate_token_ids": [token_a, token_b],
            "prompt_token_counts": counts,
            "actual_input_tokens": counts[0] if counts[0] == counts[1] else None,
            "matched": counts[0] == counts[1],
            "candidate_valid": token_a is not None and token_b is not None and token_a != token_b,
            "state_source": "not_applicable",
            "recurrent_state_a": None,
            "recurrent_state_b": None,
        }
        if not result["matched"]:
            result["reason"] = "token_length_mismatch"
            return result
        if not result["candidate_valid"]:
            result["reason"] = "candidate_not_single_token_or_duplicate"
            return result
        scores_a = self.score_candidate_tokens(pair.prompt_a, [int(token_a), int(token_b)])
        scores_b = self.score_candidate_tokens(pair.prompt_b, [int(token_a), int(token_b)])
        result.update({"prompt_a": scores_a, "prompt_b": scores_b, "score_a": scores_a, "score_b": scores_b})
        return result

    def model_metadata(self) -> dict[str, Any]:
        config = getattr(self._model, "config", None)
        model_type = getattr(config, "model_type", None)
        max_context = None
        for name in ("max_position_embeddings", "max_sequence_length", "n_positions"):
            value = getattr(config, name, None)
            if isinstance(value, int) and value > 0:
                max_context = value
                break
        num_parameters = sum(int(parameter.numel()) for parameter in self._model.parameters())
        return {
            "model_id": self.experiment_config.model_id if self.experiment_config else None,
            "model_type": model_type,
            "num_parameters": num_parameters,
            "max_context_tokens": max_context,
            "architecture_role": "transformer_control",
            "state_source": "not_applicable",
        }
